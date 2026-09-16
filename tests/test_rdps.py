from __future__ import annotations

import json

import h5py
import numpy as np
import torch
from torch import nn

from rdps.data import HDF5WaveformDataset, split_example_indices
from rdps.diffusion import DDPMDiffusion
from rdps.dps import dps_sample, measurement_loss
from rdps.model import UNet1D
from rdps.operators import ComplexFIRChannel


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
        parameters = handle.create_group("parameters")
        parameters.create_dataset("noise_variance", data=np.zeros(shape[:3], dtype=np.float32))
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
