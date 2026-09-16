#!/usr/bin/env python3
"""Train an unconditional diffusion prior on rsig x_tx waveforms."""

from __future__ import annotations

import argparse
from pathlib import Path

from rdps.training import train
from rdps.utils import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/dps_v1.yaml"))
    parser.add_argument("--dataset", type=Path, help="override dataset.path")
    parser.add_argument("--output", type=Path, help="override training.output_root")
    parser.add_argument("--resume", type=Path, help="resume model and optimizer state")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps", "mlx"))
    parser.add_argument("--max-steps", type=int, help="override training.max_steps")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_yaml(args.config)
    if args.dataset is not None:
        config["dataset"]["path"] = str(args.dataset)
    if args.resume is not None:
        config["training"]["resume"] = str(args.resume.resolve())
    if args.device is not None:
        config["device"] = args.device
    if args.max_steps is not None:
        if args.max_steps < 1:
            raise ValueError("--max-steps must be positive")
        config["training"]["max_steps"] = args.max_steps
    output = train(config, output_root=args.output)
    print(f"Training output: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
