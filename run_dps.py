#!/usr/bin/env python3
"""Run reproducible, modulation-balanced known-channel DPS evaluation."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

# PyTorch requires this before CUDA/cuBLAS initialization for deterministic GEMMs.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

from rdps.data import SPLIT_NAMES
from rdps.evaluation import run_checkpoint_evaluation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, help="override checkpoint dataset path")
    parser.add_argument("--split", choices=SPLIT_NAMES, default="test")
    parser.add_argument("--num-samples", type=int)
    parser.add_argument("--sampling-steps", type=int)
    parser.add_argument("--guidance-scale", type=float)
    parser.add_argument(
        "--likelihood",
        choices=("auto", "normalized", "normalized_squared", "normalized_l2", "gaussian"),
    )
    parser.add_argument("--max-guidance-update-norm", type=float)
    parser.add_argument(
        "--ddim",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="enable deterministic DDIM sampling; --no-ddim uses ancestral DDPM",
    )
    parser.add_argument("--evaluation-seed", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps", "mlx"))
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = run_checkpoint_evaluation(
        args.checkpoint,
        dataset_path=args.dataset,
        split=args.split,
        output_root=args.output,
        overrides={
            "sample_count": args.num_samples,
            "sampling_steps": args.sampling_steps,
            "guidance_scale": args.guidance_scale,
            "likelihood": args.likelihood,
            "max_guidance_update_norm": args.max_guidance_update_norm,
            "use_ddim": args.ddim,
            "seed": args.evaluation_seed,
            "batch_size": args.batch_size,
            "device": args.device,
        },
    )
    print(f"DPS output: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
