"""Configuration, reproducibility, device, and experiment logging helpers."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError("configuration root must be a mapping")
    return value


def optional_path(value: str) -> Path | None:
    """Parse a CLI path while accepting explicit YAML-style null values."""
    if value.lower() in {"none", "null"}:
        return None
    return Path(value)


def save_yaml(path: str | Path, value: dict[str, Any]) -> None:
    with Path(path).open("w", encoding="utf-8") as stream:
        yaml.safe_dump(value, stream, sort_keys=False)


def json_default(value: Any):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def write_json(path: str | Path, value: Any) -> None:
    with Path(path).open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False, default=json_default)
        stream.write("\n")


def seed_everything(seed: int) -> None:
    if seed < 0:
        raise ValueError("seed must be non-negative")
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def resolve_device(requested: str) -> torch.device:
    name = requested.lower()
    if name == "auto":
        if torch.cuda.is_available():
            name = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            name = "mps"
        else:
            name = "cpu"
    if name == "mlx":
        name = "mps"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if name == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        raise RuntimeError("MLX/MPS was requested but PyTorch MPS is unavailable")
    if name not in {"cpu", "cuda", "mps"}:
        raise ValueError("device must be auto, cpu, cuda, mps, or mlx")
    return torch.device(name)


def create_run_directory(root: str | Path, prefix: str) -> Path:
    root = Path(root).expanduser().resolve()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    output = root / f"{prefix}_{timestamp}"
    output.mkdir(parents=True, exist_ok=False)
    return output


class JsonlLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def log(self, record: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True, allow_nan=False, default=json_default) + "\n")
