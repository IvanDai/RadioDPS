#!/usr/bin/env python3
"""Display one waveform before and after a full diffusion round trip."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

# This must be set before importing PyTorch when deterministic CUDA is enabled.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import matplotlib.pyplot as plt
import torch

from rdps.checkpoint import load_checkpoint
from rdps.data import HDF5WaveformDataset
from rdps.diffusion import DDPMDiffusion
from rdps.model import UNet1D
from rdps.utils import resolve_device, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", default="/home/ivan/MyWorkSpace/RadioDPS/outputs/train/train_20260917_ivan1/checkpoints/best.pt",type=Path)
    parser.add_argument("modulation", default="BPSK", help="modulation used to select the real reference, e.g. QPSK")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda", "mps"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = load_checkpoint(args.checkpoint)
    config = checkpoint["config"]
    device = resolve_device(args.device)
    seed_everything(int(config["seed"]))

    dataset = HDF5WaveformDataset(
        config["dataset"]["path"],
        split="test",
        split_seed=int(config["dataset"]["split_seed"]),
        split_ratios=config["dataset"]["split_ratios"],
    )
    try:
        modulation = args.modulation.upper()
        index = next(
            index
            for index, coordinate in enumerate(dataset.coordinates)
            if dataset.modulations[coordinate[0]].upper() == modulation
        )
        real = dataset[index]["x_tx"]

        model = UNet1D(**config["model"]).to(device)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        diffusion = DDPMDiffusion(**config["diffusion"]).to(device)

        clean = real.unsqueeze(0).to(device)
        last_timestep = diffusion.steps - 1
        timesteps = torch.tensor([last_timestep], device=device)
        current, _ = diffusion.q_sample(clean, timesteps)

        with torch.no_grad():
            sequence = diffusion.sampling_timesteps(diffusion.steps)
            for step, timestep in enumerate(sequence):
                previous = sequence[step + 1] if step + 1 < len(sequence) else -1
                batch_t = torch.tensor([timestep], device=device)
                epsilon = model(current, batch_t)
                x0 = diffusion.predict_x0(current, batch_t, epsilon)
                current = diffusion.posterior_sample_from_x0(
                    current, x0, timestep, previous
                )
        generated = current[0].cpu()
    finally:
        dataset.close()

    time = range(real.shape[-1])
    figure, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    for axis, channel, label in zip(axes, (0, 1), ("I", "Q")):
        axis.plot(time, real[channel].numpy(), label=f"Real {label}", linewidth=1.2)
        axis.plot(
            time,
            generated[channel].numpy(),
            label=f"Diffusion {label}",
            linewidth=1.0,
            alpha=0.8,
        )
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
        axis.legend()

    axes[-1].set_xlabel("Sample")
    figure.suptitle(f"{dataset.modulations[dataset.coordinates[index][0]]}: real vs diffusion")
    figure.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
