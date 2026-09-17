#!/usr/bin/env python3
"""Train an unconditional diffusion prior on rsig x_tx waveforms."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# PyTorch requires this before CUDA/cuBLAS initialization for deterministic GEMMs.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from rdps.training import train
from rdps.utils import load_yaml, optional_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/dps_v1.yaml"))
    parser.add_argument("--dataset", type=Path, help="override dataset.path")
    parser.add_argument("--output", type=Path, help="override training.output_root")
    parser.add_argument(
        "--resume-output",
        type=optional_path,
        default=argparse.SUPPRESS,
        metavar="DIR|none",
        help="resume DIR; use 'none' to force a new run (overrides config)",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps", "mlx"))
    parser.add_argument(
        "--epochs",
        type=int,
        help="final target epoch, not additional epochs when resuming",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_yaml(args.config)
    if args.dataset is not None:
        config["dataset"]["path"] = str(args.dataset)
    if hasattr(args, "resume_output"):
        config["training"]["resume_output_dir"] = (
            None if args.resume_output is None else str(args.resume_output.resolve())
        )
    if args.device is not None:
        config["device"] = args.device
    if args.epochs is not None:
        if args.epochs < 1:
            raise ValueError("--epochs must be positive")
        # This is the absolute target epoch, including work restored by resume.
        config["training"]["epochs"] = args.epochs
    output = train(config, output_root=args.output)
    print(f"Training output: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
