"""Shared deterministic diffusion validation and DPS evaluation."""

from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Callable

import h5py
import numpy as np
import torch
from torch import nn

from .checkpoint import load_checkpoint
from .data import HDF5WaveformDataset, balanced_subset_indices
from .diffusion import DDPMDiffusion
from .dps import dps_sample
from .metrics import measurement_residual, mse, nmse
from .model import UNet1D
from .operators import ComplexFIRChannel
from .utils import create_run_directory, resolve_device, save_yaml, write_json


METRIC_NAMES = ("mse", "nmse", "measurement_residual")
ProgressCallback = Callable[[int, int], None]


def _derived_seed(root_seed: int, coordinate: tuple[int, int, int], stream: int) -> int:
    sequence = np.random.SeedSequence([root_seed, *coordinate, stream])
    return int(sequence.generate_state(1, dtype=np.uint64)[0] & np.uint64(0x7FFFFFFFFFFFFFFF))


def _noise_for_coordinates(
    coordinates: list[tuple[int, int, int]],
    shape: tuple[int, int],
    *,
    seed: int,
    stream: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    values = []
    for coordinate in coordinates:
        generator = torch.Generator(device="cpu").manual_seed(
            _derived_seed(seed, coordinate, stream)
        )
        values.append(torch.randn(shape, generator=generator, dtype=dtype))
    return torch.stack(values)


def _collate(dataset: HDF5WaveformDataset, indices: list[int]) -> dict[str, Any]:
    samples = [dataset[index] for index in indices]
    return {
        "x_tx": torch.stack([sample["x_tx"] for sample in samples]),
        "h": torch.stack([sample["h"] for sample in samples]),
        "y_rx": torch.stack([sample["y_rx"] for sample in samples]),
        "noise_variance": torch.tensor(
            [sample["noise_variance"] for sample in samples], dtype=torch.float32
        ),
        "modulation": [str(sample["modulation"]) for sample in samples],
        "modulation_index": [int(sample["modulation_index"]) for sample in samples],
        "condition_index": [int(sample["condition_index"]) for sample in samples],
        "example_index": [int(sample["example_index"]) for sample in samples],
    }


@contextmanager
def _evaluation_model(model: nn.Module):
    was_training = model.training
    requires_grad = [parameter.requires_grad for parameter in model.parameters()]
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    try:
        yield
    finally:
        for parameter, required in zip(model.parameters(), requires_grad):
            parameter.requires_grad_(required)
        model.train(was_training)


@torch.no_grad()
def fixed_diffusion_validation(
    model: nn.Module,
    diffusion: DDPMDiffusion,
    dataset: HDF5WaveformDataset,
    indices: list[int],
    *,
    device: torch.device,
    seed: int,
    noise_repeats: int,
    batch_size: int,
    progress: ProgressCallback | None = None,
) -> float:
    """Evaluate identical timestep/noise draws at every epoch."""
    if not indices:
        raise ValueError("validation selection is empty")
    if noise_repeats < 1 or batch_size < 1:
        raise ValueError("noise_repeats and validation batch_size must be positive")
    squared_error = 0.0
    element_count = 0
    batches_per_repeat = (len(indices) + batch_size - 1) // batch_size
    total_batches = noise_repeats * batches_per_repeat
    completed_batches = 0
    was_training = model.training
    model.eval()
    try:
        for repeat in range(noise_repeats):
            for offset in range(0, len(indices), batch_size):
                batch_indices = indices[offset : offset + batch_size]
                batch = _collate(dataset, batch_indices)
                clean = batch["x_tx"].to(device)
                coordinates = [dataset.coordinates[index] for index in batch_indices]
                timesteps = torch.tensor(
                    [
                        _derived_seed(seed, coordinate, 10_000 + repeat) % diffusion.steps
                        for coordinate in coordinates
                    ],
                    dtype=torch.long,
                    device=device,
                )
                noise = _noise_for_coordinates(
                    coordinates,
                    tuple(clean.shape[1:]),
                    seed=seed,
                    stream=20_000 + repeat,
                    dtype=clean.dtype,
                ).to(device)
                noisy, _ = diffusion.q_sample(clean, timesteps, noise)
                prediction = model(noisy, timesteps)
                squared_error += float((prediction - noise).square().sum())
                element_count += noise.numel()
                completed_batches += 1
                if progress is not None:
                    progress(completed_batches, total_batches)
    finally:
        model.train(was_training)
    return squared_error / element_count


def evaluation_settings(config: dict[str, Any], section_name: str) -> dict[str, Any]:
    section = config[section_name]
    dps = config["dps"]
    return {
        "seed": int(section["seed"]),
        "sample_count": section.get("sample_count"),
        "batch_size": int(section.get("batch_size", 1)),
        "plot_count": int(section.get("plot_count", 0)),
        "sampling_steps": int(section.get("sampling_steps", dps["sampling_steps"])),
        "guidance_scale": float(section.get("guidance_scale", dps["guidance_scale"])),
        "likelihood": section.get("likelihood", dps.get("likelihood", "auto")),
        "measurement_eps": float(section.get("measurement_eps", dps.get("measurement_eps", 1e-8))),
        "clip_denoised": bool(section.get("clip_denoised", dps.get("clip_denoised", False))),
        "max_guidance_update_norm": section.get(
            "max_guidance_update_norm", dps.get("max_guidance_update_norm")
        ),
    }


def _aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    per_modulation: dict[str, dict[str, Any]] = {}
    for modulation in sorted({record["modulation"] for record in records}):
        selected = [record for record in records if record["modulation"] == modulation]
        per_modulation[modulation] = {
            "sample_count": len(selected),
            **{
                name: {
                    "mean": float(np.mean([record[name] for record in selected])),
                    "std": float(np.std([record[name] for record in selected])),
                }
                for name in METRIC_NAMES
            },
        }
    micro = {
        name: {
            "mean": float(np.mean([record[name] for record in records])),
            "std": float(np.std([record[name] for record in records])),
        }
        for name in METRIC_NAMES
    }
    macro = {
        name: float(np.mean([result[name]["mean"] for result in per_modulation.values()]))
        for name in METRIC_NAMES
    }
    return {"macro": macro, "micro": micro, "per_modulation": per_modulation}


def _complex(iq: np.ndarray) -> np.ndarray:
    return iq[0] + 1j * iq[1]


def _save_plot(path: Path, x_tx: np.ndarray, y_rx: np.ndarray, reconstructed: np.ndarray) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "rdps-matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    reference, observed, estimate = _complex(x_tx), _complex(y_rx), _complex(reconstructed)
    shown = min(256, len(reference))
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    axes[0].plot(reference.real[:shown], label="x_tx I", linewidth=0.9)
    axes[0].plot(reference.imag[:shown], label="x_tx Q", linewidth=0.9)
    axes[0].plot(estimate.real[:shown], label="DPS I", linewidth=0.8, alpha=0.8)
    axes[0].plot(estimate.imag[:shown], label="DPS Q", linewidth=0.8, alpha=0.8)
    axes[0].set_title("Time domain")
    axes[0].legend(fontsize=7, ncol=2)
    frequency = np.fft.fftshift(np.fft.fftfreq(len(reference)))
    for values, label in ((reference, "x_tx"), (observed, "y_rx"), (estimate, "DPS")):
        spectrum = np.abs(np.fft.fftshift(np.fft.fft(values)))
        normalized = spectrum / max(float(spectrum.max()), 1e-12)
        axes[1].plot(frequency, 20 * np.log10(np.maximum(normalized, 1e-8)), label=label)
    axes[1].set_ylim(-80, 5)
    axes[1].set_title("Normalized spectrum")
    axes[1].legend(fontsize=8)
    axes[2].scatter(reference.real, reference.imag, s=5, alpha=0.25, label="x_tx")
    axes[2].scatter(estimate.real, estimate.imag, s=5, alpha=0.25, label="DPS")
    axes[2].set_aspect("equal", adjustable="datalim")
    axes[2].set_title("IQ scatter")
    axes[2].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=140)
    plt.close(figure)


def run_dps_evaluation(
    model: nn.Module,
    diffusion: DDPMDiffusion,
    dataset: HDF5WaveformDataset,
    *,
    device: torch.device,
    settings: dict[str, Any],
    output_dir: str | Path | None = None,
    save_reconstructions: bool = False,
    context: dict[str, Any] | None = None,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    if float(settings["measurement_eps"]) <= 0:
        raise ValueError("DPS measurement_eps must be positive")
    max_update_norm = settings.get("max_guidance_update_norm")
    if max_update_norm is not None and float(max_update_norm) <= 0:
        raise ValueError("DPS max_guidance_update_norm must be positive or null")
    selected, selection = balanced_subset_indices(
        dataset, settings.get("sample_count"), seed=int(settings["seed"])
    )
    if not selection["complete_modulation_coverage"]:
        print(
            "Warning: DPS sample_count is smaller than the number of modulations; "
            "the reported macro metric has incomplete coverage.",
            flush=True,
        )
    batch_size = int(settings.get("batch_size", 1))
    if batch_size < 1:
        raise ValueError("DPS evaluation batch_size must be positive")
    operator = ComplexFIRChannel()
    records: list[dict[str, Any]] = []
    arrays: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    sequence = diffusion.sampling_timesteps(int(settings["sampling_steps"]))
    total_batches = (len(selected) + batch_size - 1) // batch_size
    total_progress_steps = total_batches * len(sequence)
    started = time.monotonic()
    with _evaluation_model(model):
        for offset in range(0, len(selected), batch_size):
            batch_indices = selected[offset : offset + batch_size]
            batch = _collate(dataset, batch_indices)
            coordinates = [dataset.coordinates[index] for index in batch_indices]
            y_rx = batch["y_rx"].to(device)
            channel = batch["h"].to(device)
            initial_noise = _noise_for_coordinates(
                coordinates,
                tuple(y_rx.shape[1:]),
                seed=int(settings["seed"]),
                stream=30_000,
                dtype=y_rx.dtype,
            ).to(device)
            sampling_noises = [
                _noise_for_coordinates(
                    coordinates,
                    tuple(y_rx.shape[1:]),
                    seed=int(settings["seed"]),
                    stream=40_000 + timestep,
                    dtype=y_rx.dtype,
                )
                for timestep in sequence
            ]
            reconstructed, diagnostics = dps_sample(
                model,
                diffusion,
                y_rx,
                channel,
                noise_variance=batch["noise_variance"].to(device),
                sampling_steps=int(settings["sampling_steps"]),
                guidance_scale=float(settings["guidance_scale"]),
                likelihood=settings["likelihood"],
                measurement_eps=float(settings["measurement_eps"]),
                clip_denoised=bool(settings["clip_denoised"]),
                max_guidance_update_norm=(
                    None if max_update_norm is None else float(max_update_norm)
                ),
                initial_noise=initial_noise,
                sampling_noises=sampling_noises,
                per_sample_diagnostics=True,
                progress=(
                    None
                    if progress is None
                    else lambda completed, _total, batch=offset // batch_size: progress(
                        batch * len(sequence) + completed,
                        total_progress_steps,
                    )
                ),
            )
            x_tx = batch["x_tx"].to(device)
            with torch.no_grad():
                predicted_y = operator(reconstructed, channel)
                batch_mse = mse(x_tx, reconstructed).cpu().tolist()
                batch_nmse = nmse(x_tx, reconstructed).cpu().tolist()
                batch_residual = measurement_residual(y_rx, predicted_y).cpu().tolist()
            for local_index, coordinate in enumerate(coordinates):
                gradient_norms = [item["gradient_norm"][local_index] for item in diagnostics]
                update_norms = [item["guidance_update_norm"][local_index] for item in diagnostics]
                clipped_steps = [item["guidance_clipped"][local_index] for item in diagnostics]
                record = {
                    "sample": len(records),
                    "modulation": batch["modulation"][local_index],
                    "modulation_index": coordinate[0],
                    "condition_index": coordinate[1],
                    "example_index": coordinate[2],
                    "mse": float(batch_mse[local_index]),
                    "nmse": float(batch_nmse[local_index]),
                    "measurement_residual": float(batch_residual[local_index]),
                    "final_measurement_gradient_norm": float(gradient_norms[-1]),
                    "mean_measurement_gradient_norm": float(np.mean(gradient_norms)),
                    "mean_guidance_update_norm": float(np.mean(update_norms)),
                    "max_guidance_update_norm": float(np.max(update_norms)),
                    "guidance_clipped_fraction": float(np.mean(clipped_steps)),
                }
                if not np.isfinite([record[name] for name in METRIC_NAMES]).all():
                    raise FloatingPointError(f"non-finite DPS metric for coordinate {coordinate}")
                records.append(record)
                if save_reconstructions:
                    arrays.append(
                        (
                            batch["x_tx"][local_index].numpy(),
                            batch["y_rx"][local_index].numpy(),
                            reconstructed[local_index].detach().cpu().numpy(),
                            batch["h"][local_index].numpy(),
                        )
                    )
    summary = {
        **(context or {}),
        "dataset": str(dataset.path),
        "split": dataset.split,
        "dataset_split": dataset.split_manifest(),
        "selection": selection,
        "settings": settings,
        "elapsed_seconds": time.monotonic() - started,
        "metrics": _aggregate(records),
        "measurement_gradient": {
            "all_finite": bool(
                np.isfinite([record["mean_measurement_gradient_norm"] for record in records]).all()
            ),
            "mean_norm": float(
                np.mean([record["mean_measurement_gradient_norm"] for record in records])
            ),
        },
        "guidance_update": {
            "all_finite": bool(
                np.isfinite([record["mean_guidance_update_norm"] for record in records]).all()
            ),
            "mean_norm": float(np.mean([record["mean_guidance_update_norm"] for record in records])),
            "max_norm": float(np.max([record["max_guidance_update_norm"] for record in records])),
            "clipped_fraction": float(
                np.mean([record["guidance_clipped_fraction"] for record in records])
            ),
        },
        "samples": records,
    }
    if output_dir is not None:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        save_yaml(output / "config.yaml", {**(context or {}), **settings, "selection": selection})
        write_json(output / "metrics.json", summary)
        if save_reconstructions:
            with h5py.File(output / "reconstructions.h5", "w") as handle:
                handle.create_dataset("x_tx", data=np.stack([item[0] for item in arrays]))
                handle.create_dataset("y_rx", data=np.stack([item[1] for item in arrays]))
                handle.create_dataset("x_reconstructed", data=np.stack([item[2] for item in arrays]))
                handle.create_dataset("h", data=np.stack([item[3] for item in arrays]))
                for name in METRIC_NAMES:
                    handle.create_dataset(
                        name, data=np.asarray([record[name] for record in records], dtype=np.float32)
                    )
                text_dtype = h5py.string_dtype(encoding="utf-8")
                handle.create_dataset(
                    "modulation",
                    data=np.asarray([record["modulation"] for record in records], dtype=object),
                    dtype=text_dtype,
                )
                for name in ("modulation_index", "condition_index", "example_index"):
                    handle.create_dataset(
                        name, data=np.asarray([record[name] for record in records], dtype=np.int64)
                    )
                handle.attrs["configuration"] = json.dumps(
                    {**(context or {}), **settings, "selection": selection}, sort_keys=True
                )
            plot_dir = output / "plots"
            plot_dir.mkdir(exist_ok=True)
            for index in range(min(int(settings.get("plot_count", 0)), len(arrays))):
                _save_plot(plot_dir / f"sample_{index:04d}.png", *arrays[index][:3])
    return summary


def run_checkpoint_evaluation(
    checkpoint_path: str | Path,
    *,
    dataset_path: str | Path | None = None,
    split: str = "test",
    output_root: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> Path:
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    checkpoint = load_checkpoint(checkpoint_path)
    config = checkpoint["config"]
    resolved_dataset = Path(dataset_path or config["dataset"]["path"]).expanduser().resolve()
    requested_device = (overrides or {}).get("device") or config.get("device", "auto")
    device = resolve_device(str(requested_device))
    dataset = HDF5WaveformDataset(
        resolved_dataset,
        split=split,
        split_seed=int(config["dataset"]["split_seed"]),
        split_ratios=config["dataset"]["split_ratios"],
    )
    try:
        model = UNet1D(**config["model"]).to(device)
        model.load_state_dict(checkpoint["model"])
        diffusion = DDPMDiffusion(**config["diffusion"]).to(device)
        settings = evaluation_settings(config, "evaluation")
        settings.update(
            {
                key: value
                for key, value in (overrides or {}).items()
                if key != "device" and value is not None
            }
        )
        root = output_root or config["evaluation"].get("output_root", "outputs/dps")
        output = create_run_directory(root, "dps")
        run_dps_evaluation(
            model,
            diffusion,
            dataset,
            device=device,
            settings=settings,
            output_dir=output,
            save_reconstructions=True,
            context={
                "checkpoint": str(checkpoint_path),
                "checkpoint_epoch": int(checkpoint.get("completed_epoch", 0)),
                "checkpoint_step": int(checkpoint["global_step"]),
                "device": str(device),
                "dataset": str(resolved_dataset),
                "split": split,
                "model": config["model"],
                "diffusion": config["diffusion"],
            },
        )
    finally:
        dataset.close()
    return output
