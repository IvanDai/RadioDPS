#!/usr/bin/env python3
"""Run known-channel DPS and evaluate waveform reconstruction."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
import time

import h5py
import numpy as np
import torch

from rdps.data import HDF5WaveformDataset, SPLIT_NAMES
from rdps.dps import dps_sample
from rdps.metrics import measurement_residual, mse, nmse
from rdps.operators import ComplexFIRChannel
from rdps.training import build_diffusion, build_model, load_checkpoint
from rdps.utils import create_run_directory, resolve_device, save_yaml, seed_everything, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, help="override checkpoint dataset path")
    parser.add_argument("--split", choices=SPLIT_NAMES, default="test")
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--sampling-steps", type=int, default=None)
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument("--likelihood", choices=("auto", "normalized", "gaussian"), default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps", "mlx"), default=None)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def _complex(iq: np.ndarray) -> np.ndarray:
    return iq[0] + 1j * iq[1]


def _save_plot(path: Path, x_tx: np.ndarray, y_rx: np.ndarray, reconstructed: np.ndarray) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "rdps-matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    reference = _complex(x_tx)
    observed = _complex(y_rx)
    estimate = _complex(reconstructed)
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
        spectrum_db = 20 * np.log10(np.maximum(spectrum / max(float(spectrum.max()), 1e-12), 1e-8))
        axes[1].plot(frequency, spectrum_db, label=label, linewidth=0.9)
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


def main() -> int:
    args = parse_args()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    checkpoint = load_checkpoint(checkpoint_path)
    config = checkpoint["config"]
    inference = config.get("dps", {})
    evaluation = config.get("evaluation", {})
    dataset_path = (
        args.dataset.expanduser().resolve()
        if args.dataset is not None
        else Path(config["dataset"]["path"]).expanduser().resolve()
    )
    sample_count = args.num_samples if args.num_samples is not None else int(evaluation.get("sample_count", 16))
    sampling_steps = (
        args.sampling_steps if args.sampling_steps is not None else int(inference.get("sampling_steps", config["diffusion"]["steps"]))
    )
    guidance_scale = (
        args.guidance_scale if args.guidance_scale is not None else float(inference.get("guidance_scale", 1.0))
    )
    likelihood = args.likelihood or inference.get("likelihood", "auto")
    requested_device = args.device or config.get("device", "auto")
    if sample_count < 1:
        raise ValueError("--num-samples must be positive")
    device = resolve_device(requested_device)
    seed = int(config["seed"])
    seed_everything(seed)
    ratios = config["dataset"]["split_ratios"]
    split_seed = int(config["dataset"]["split_seed"])
    dataset = HDF5WaveformDataset(
        dataset_path, split=args.split, split_seed=split_seed, split_ratios=ratios
    )
    sample_count = min(sample_count, len(dataset))
    if sample_count == 0:
        raise ValueError(f"{args.split} split is empty")

    model = build_model(config["model"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    diffusion = build_diffusion(config["diffusion"]).to(device)
    output_root = args.output if args.output is not None else evaluation.get("output_root", "outputs/dps")
    output = create_run_directory(output_root, "dps")
    plot_dir = output / "plots"
    plot_dir.mkdir()
    actual_config = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": int(checkpoint["global_step"]),
        "model": config["model"],
        "diffusion": config["diffusion"],
        "dataset": str(dataset_path),
        "dataset_split": dataset.split_manifest(),
        "seed": seed,
        "device": str(device),
        "sample_count": sample_count,
        "sampling_steps": sampling_steps,
        "guidance_scale": guidance_scale,
        "likelihood": likelihood,
        "measurement_eps": float(inference.get("measurement_eps", 1e-8)),
        "clip_denoised": bool(inference.get("clip_denoised", False)),
        "output_dir": str(output),
    }
    save_yaml(output / "config.yaml", actual_config)

    records = []
    arrays = []
    operator = ComplexFIRChannel()
    started = time.monotonic()
    plot_count = min(int(evaluation.get("plot_count", 4)), sample_count)
    for index in range(sample_count):
        sample = dataset[index]
        y_rx = sample["y_rx"].unsqueeze(0).to(device)
        channel = sample["h"].unsqueeze(0).to(device)
        variance = torch.tensor([sample["noise_variance"]], device=device, dtype=y_rx.dtype)
        torch.manual_seed(seed + index)
        reconstructed, diagnostics = dps_sample(
            model,
            diffusion,
            y_rx,
            channel,
            noise_variance=variance,
            sampling_steps=sampling_steps,
            guidance_scale=guidance_scale,
            likelihood=likelihood,
            measurement_eps=actual_config["measurement_eps"],
            clip_denoised=actual_config["clip_denoised"],
        )
        x_tx = sample["x_tx"].unsqueeze(0).to(device)
        with torch.no_grad():
            predicted_y = operator(reconstructed, channel)
            sample_mse = float(mse(x_tx, reconstructed)[0])
            sample_nmse = float(nmse(x_tx, reconstructed)[0])
            sample_residual = float(measurement_residual(y_rx, predicted_y)[0])
        if not np.isfinite((sample_mse, sample_nmse, sample_residual)).all():
            raise FloatingPointError(f"non-finite metric for sample {index}")
        record = {
            "sample": index,
            "modulation": sample["modulation"],
            "modulation_index": int(sample["modulation_index"]),
            "condition_index": int(sample["condition_index"]),
            "example_index": int(sample["example_index"]),
            "mse": sample_mse,
            "nmse": sample_nmse,
            "measurement_residual": sample_residual,
            "final_measurement_gradient_norm": diagnostics[-1]["gradient_norm_mean"],
            "mean_measurement_gradient_norm": float(np.mean([item["gradient_norm_mean"] for item in diagnostics])),
        }
        records.append(record)
        arrays.append(
            (
                sample["x_tx"].numpy(),
                sample["y_rx"].numpy(),
                reconstructed[0].cpu().numpy(),
                sample["h"].numpy(),
            )
        )
        if index < plot_count:
            _save_plot(plot_dir / f"sample_{index:04d}.png", arrays[-1][0], arrays[-1][1], arrays[-1][2])
        print(
            f"sample {index + 1:>4}/{sample_count} {sample['modulation']:<5} "
            f"NMSE={sample_nmse:.6g} residual={sample_residual:.6g}",
            flush=True,
        )

    with h5py.File(output / "reconstructions.h5", "w") as handle:
        handle.create_dataset("x_tx", data=np.stack([item[0] for item in arrays]))
        handle.create_dataset("y_rx", data=np.stack([item[1] for item in arrays]))
        handle.create_dataset("x_reconstructed", data=np.stack([item[2] for item in arrays]))
        handle.create_dataset("h", data=np.stack([item[3] for item in arrays]))
        for name in ("mse", "nmse", "measurement_residual"):
            handle.create_dataset(name, data=np.asarray([record[name] for record in records], dtype=np.float32))
        text_dtype = h5py.string_dtype(encoding="utf-8")
        handle.create_dataset(
            "modulation", data=np.asarray([record["modulation"] for record in records], dtype=object), dtype=text_dtype
        )
        for name in ("modulation_index", "condition_index", "example_index"):
            handle.create_dataset(name, data=np.asarray([record[name] for record in records], dtype=np.int64))
        handle.attrs["configuration"] = json.dumps(actual_config, sort_keys=True)

    summary = {
        **actual_config,
        "elapsed_seconds": time.monotonic() - started,
        "metrics": {
            name: {
                "mean": float(np.mean([record[name] for record in records])),
                "std": float(np.std([record[name] for record in records])),
            }
            for name in ("mse", "nmse", "measurement_residual")
        },
        "measurement_gradient": {
            "all_finite": bool(np.isfinite([record["mean_measurement_gradient_norm"] for record in records]).all()),
            "mean_norm": float(np.mean([record["mean_measurement_gradient_norm"] for record in records])),
        },
        "samples": records,
    }
    write_json(output / "metrics.json", summary)
    dataset.close()
    print(f"DPS output: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
