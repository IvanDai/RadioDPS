from __future__ import annotations

import json

import h5py
import numpy as np
import torch
from torch import nn

from rdps.checkpoint import load_checkpoint
from rdps.data import HDF5WaveformDataset, balanced_subset_indices, split_example_indices
from rdps.diffusion import DDPMDiffusion
from rdps.dps import dps_sample, measurement_loss
from rdps.evaluation import fixed_diffusion_validation
from rdps.model import UNet1D
from rdps.operators import ComplexFIRChannel
from rdps.training import train


def _iq(values: np.ndarray) -> np.ndarray:
    return np.stack((values.real, values.imag), axis=0).astype(np.float32)


def _minimal_dataset(path):
    rng = np.random.default_rng(4)
    shape = (2, 1, 8, 2, 16)
    channels = (2, 1, 8, 2, 3)
    with h5py.File(path, "w") as handle:
        handle.create_dataset("x_tx", data=rng.standard_normal(shape).astype(np.float32))
        handle.create_dataset("y_rx", data=rng.standard_normal(shape).astype(np.float32))
        handle.create_dataset("h", data=rng.standard_normal(channels).astype(np.float32))
        handle.create_dataset("modulation", data=np.asarray(["BPSK", "QPSK"], dtype=object), dtype=h5py.string_dtype())
        handle.create_dataset("noise_level_db", data=np.asarray([np.nan], dtype=np.float32))
        parameters = handle.create_group("parameters")
        parameters.create_dataset("noise_variance", data=np.zeros(shape[:3], dtype=np.float32))
        parameters.create_dataset("sample_seed", data=np.arange(np.prod(shape[:3]), dtype=np.uint64).reshape(shape[:3]))
        handle.create_dataset("metadata", data=json.dumps({"version": "rdps-rsig-hdf5-1"}), dtype=h5py.string_dtype())


def test_example_split_is_deterministic_disjoint_and_stratified(tmp_path):
    ratios = {"train": 0.5, "validation": 0.25, "test": 0.25}
    first = split_example_indices(8, seed=17, ratios=ratios)
    second = split_example_indices(8, seed=17, ratios=ratios)
    assert all(np.array_equal(first[name], second[name]) for name in first)
    assert sorted(np.concatenate(list(first.values())).tolist()) == list(range(8))

    path = tmp_path / "dataset.h5"
    _minimal_dataset(path)
    train = HDF5WaveformDataset(path, split="train", split_seed=17, split_ratios=ratios)
    assert len(train) == 8
    assert train[0]["x_tx"].shape == (2, 16)
    counts = np.bincount([coordinate[0] for coordinate in train.coordinates])
    np.testing.assert_array_equal(counts, [4, 4])
    selected, manifest = balanced_subset_indices(train, 4, seed=91)
    assert len(selected) == 4
    assert manifest["counts_by_modulation"] == {"BPSK": 2, "QPSK": 2}
    repeated, repeated_manifest = balanced_subset_indices(train, 4, seed=91)
    assert repeated == selected and repeated_manifest == manifest
    train.close()


def test_complex_fir_matches_numpy_and_has_exact_adjoint():
    rng = np.random.default_rng(8)
    x_complex = rng.standard_normal((2, 13)) + 1j * rng.standard_normal((2, 13))
    h_complex = rng.standard_normal((2, 4)) + 1j * rng.standard_normal((2, 4))
    x = torch.tensor(np.stack([_iq(value) for value in x_complex]), requires_grad=True)
    h = torch.tensor(np.stack([_iq(value) for value in h_complex]))
    operator = ComplexFIRChannel()
    actual = operator(x, h)
    expected = np.stack([_iq(np.convolve(a, b, mode="full")[:13]) for a, b in zip(x_complex, h_complex)])
    np.testing.assert_allclose(actual.detach().numpy(), expected, rtol=2e-6, atol=2e-6)

    probe = torch.randn_like(actual)
    left = torch.sum(actual * probe)
    right = torch.sum(x * operator.adjoint(probe, h))
    torch.testing.assert_close(left, right, rtol=2e-5, atol=2e-5)
    left.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_unet_diffusion_and_dps_shapes_are_finite():
    torch.manual_seed(2)
    model = UNet1D(base_channels=8, channel_multipliers=(1, 2), num_res_blocks=1)
    clean = torch.randn(2, 2, 32)
    timesteps = torch.tensor([0, 7])
    output = model(clean, timesteps)
    assert output.shape == clean.shape
    diffusion = DDPMDiffusion(steps=8)
    loss = diffusion.training_loss(model, clean, timesteps=timesteps)
    loss.backward()
    assert torch.isfinite(loss)

    class ZeroEpsilon(nn.Module):
        def forward(self, values, _timesteps):
            return torch.zeros_like(values)

    channel = torch.zeros(1, 2, 3)
    channel[:, 0, 0] = 1.0
    measurement = torch.randn(1, 2, 32)
    reconstruction, diagnostics = dps_sample(
        ZeroEpsilon(),
        diffusion,
        measurement,
        channel,
        noise_variance=0.0,
        sampling_steps=4,
        guidance_scale=0.01,
    )
    assert reconstruction.shape == measurement.shape
    assert torch.isfinite(reconstruction).all()
    assert len(diagnostics) == 4
    assert all(np.isfinite(item["gradient_norm_mean"]) for item in diagnostics)


def test_zero_noise_uses_normalized_loss_and_gaussian_rejects_it():
    measurement = torch.ones(1, 2, 4)
    prediction = torch.zeros_like(measurement)
    variance = torch.zeros(1)
    normalized = measurement_loss(
        measurement, prediction, noise_variance=variance, likelihood="auto", eps=1e-8
    )
    torch.testing.assert_close(normalized, torch.ones_like(normalized))
    try:
        measurement_loss(
            measurement, prediction, noise_variance=variance, likelihood="gaussian", eps=1e-8
        )
    except ValueError:
        pass
    else:
        raise AssertionError("zero-variance Gaussian likelihood must be rejected")

    mixed_measurement = torch.stack((measurement[0], 2 * measurement[0]))
    mixed_prediction = torch.zeros_like(mixed_measurement)
    mixed = measurement_loss(
        mixed_measurement,
        mixed_prediction,
        noise_variance=torch.tensor([0.0, 2.0]),
        likelihood="auto",
        eps=1e-8,
    )
    torch.testing.assert_close(mixed, torch.tensor([1.0, 8.0]))


def test_fixed_validation_is_independent_of_global_rng(tmp_path):
    path = tmp_path / "dataset.h5"
    _minimal_dataset(path)
    ratios = {"train": 0.5, "validation": 0.25, "test": 0.25}
    dataset = HDF5WaveformDataset(path, split="validation", split_seed=17, split_ratios=ratios)
    model = UNet1D(base_channels=4, channel_multipliers=(1,), num_res_blocks=1)
    model.train()
    diffusion = DDPMDiffusion(steps=4)
    indices, _ = balanced_subset_indices(dataset, None, seed=10)
    first = fixed_diffusion_validation(
        model, diffusion, dataset, indices, device=torch.device("cpu"), seed=22, noise_repeats=2, batch_size=2
    )
    torch.manual_seed(999)
    _ = torch.randn(100)
    second = fixed_diffusion_validation(
        model, diffusion, dataset, indices, device=torch.device("cpu"), seed=22, noise_repeats=2, batch_size=2
    )
    assert first == second
    assert model.training
    different_batching = fixed_diffusion_validation(
        model, diffusion, dataset, indices, device=torch.device("cpu"), seed=22, noise_repeats=2, batch_size=1
    )
    assert np.isclose(first, different_batching, rtol=1e-6)
    dataset.close()


def _training_config(dataset_path, output_root):
    return {
        "experiment": "test",
        "seed": 13,
        "device": "cpu",
        "dataset": {
            "path": str(dataset_path),
            "split_seed": 17,
            "split_ratios": {"train": 0.5, "validation": 0.25, "test": 0.25},
            "num_workers": 0,
        },
        "model": {
            "in_channels": 2,
            "out_channels": 2,
            "base_channels": 4,
            "channel_multipliers": [1],
            "num_res_blocks": 1,
            "dropout": 0.0,
        },
        "diffusion": {"steps": 4, "schedule": "linear", "beta_start": 0.0001, "beta_end": 0.02},
        "training": {
            "output_root": str(output_root),
            "resume_output_dir": None,
            "epochs": 1,
            "batch_size": 4,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "gradient_clip": 1.0,
            "save_every_epochs": 1,
            "early_stopping": {"enabled": True, "patience": 5, "min_delta": 0.0},
        },
        "validation": {"seed": 101, "sample_count": 4, "batch_size": 4, "noise_repeats": 1},
        "dps": {
            "sampling_steps": 2,
            "guidance_scale": 0.01,
            "likelihood": "normalized",
            "measurement_eps": 1e-8,
            "clip_denoised": False,
        },
        "dps_validation": {
            "enabled": False,
            "every_epochs": 1,
            "seed": 202,
            "sample_count": 2,
            "batch_size": 2,
            "sampling_steps": 2,
            "guidance_scale": 0.01,
            "plot_count": 0,
        },
        "evaluation": {
            "output_root": str(output_root / "dps"),
            "seed": 303,
            "sample_count": 2,
            "batch_size": 2,
            "sampling_steps": 2,
            "guidance_scale": 0.01,
            "plot_count": 0,
        },
    }


def test_epoch_resume_reuses_output_and_rejects_incompatible_config(tmp_path):
    dataset_path = tmp_path / "dataset.h5"
    _minimal_dataset(dataset_path)
    config = _training_config(dataset_path, tmp_path / "outputs")
    output = train(config)
    first = load_checkpoint(output / "checkpoints/last.pt")
    assert first["completed_epoch"] == 1 and first["global_step"] == 2

    resumed = _training_config(dataset_path, tmp_path / "unused")
    resumed["training"]["resume_output_dir"] = str(output)
    resumed["training"]["epochs"] = 2
    assert train(resumed) == output
    second = load_checkpoint(output / "checkpoints/last.pt")
    assert second["completed_epoch"] == 2 and second["global_step"] == 4
    assert len((output / "history.jsonl").read_text().splitlines()) == 2

    uninterrupted = _training_config(dataset_path, tmp_path / "uninterrupted")
    uninterrupted["training"]["epochs"] = 2
    uninterrupted_output = train(uninterrupted)
    uninterrupted_checkpoint = load_checkpoint(uninterrupted_output / "checkpoints/last.pt")
    for name, value in second["model"].items():
        torch.testing.assert_close(value, uninterrupted_checkpoint["model"][name], rtol=0, atol=0)
    assert second["best_validation_loss"] == uninterrupted_checkpoint["best_validation_loss"]

    incompatible = _training_config(dataset_path, tmp_path / "unused")
    incompatible["training"]["resume_output_dir"] = str(output)
    incompatible["training"]["epochs"] = 3
    incompatible["training"]["learning_rate"] = 0.5
    resume_config_dir = output / "resume_configs"
    config_files_before = set(resume_config_dir.iterdir())
    try:
        train(incompatible)
    except ValueError as error:
        assert "learning_rate" in str(error)
    else:
        raise AssertionError("incompatible resume must be rejected")
    assert set(resume_config_dir.iterdir()) == config_files_before


def test_early_stopping_restores_best_epoch_for_final_test(tmp_path):
    dataset_path = tmp_path / "dataset.h5"
    _minimal_dataset(dataset_path)
    config = _training_config(dataset_path, tmp_path / "outputs")
    config["training"]["epochs"] = 5
    config["training"]["early_stopping"] = {
        "enabled": True,
        "patience": 1,
        "min_delta": 1e6,
    }
    output = train(config)
    metrics = json.loads((output / "metrics.json").read_text())
    assert metrics["status"] == "early_stopped"
    assert metrics["completed_epoch"] == 2
    assert metrics["best_epoch"] == 1
    assert "test_best_epoch_0001_after_0002" in metrics["final_test_output"]
