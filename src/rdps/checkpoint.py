"""Atomic checkpoints, RNG state, and resume compatibility checks."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch


CHECKPOINT_FORMAT_VERSION = 2


def capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(path: str | Path, *, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    return torch.load(Path(path), map_location=map_location, weights_only=False)


def invariant_configuration(config: dict[str, Any], dataset_signature: dict[str, Any]) -> dict[str, Any]:
    training = config["training"]
    validation = config["validation"]
    return {
        "seed": int(config["seed"]),
        "dataset": {
            "signature": dataset_signature["sha256"],
            "split_seed": int(config["dataset"]["split_seed"]),
            "split_ratios": config["dataset"]["split_ratios"],
        },
        "model": config["model"],
        "diffusion": config["diffusion"],
        "optimizer": {
            "name": "AdamW",
            "batch_size": int(training["batch_size"]),
            "learning_rate": float(training["learning_rate"]),
            "weight_decay": float(training.get("weight_decay", 0.0)),
            "gradient_clip": float(training["gradient_clip"]),
        },
        "validation": {
            "seed": int(validation["seed"]),
            "sample_count": validation.get("sample_count"),
            "noise_repeats": int(validation.get("noise_repeats", 1)),
        },
        "early_stopping": training["early_stopping"],
    }


def configuration_fingerprint(invariants: dict[str, Any]) -> str:
    encoded = json.dumps(invariants, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _differences(expected: Any, actual: Any, prefix: str = "") -> list[str]:
    if isinstance(expected, dict) and isinstance(actual, dict):
        differences: list[str] = []
        for key in sorted(set(expected) | set(actual)):
            path = f"{prefix}.{key}" if prefix else key
            if key not in expected:
                differences.append(f"{path}: unexpected {actual[key]!r}")
            elif key not in actual:
                differences.append(f"{path}: missing (expected {expected[key]!r})")
            else:
                differences.extend(_differences(expected[key], actual[key], path))
        return differences
    if expected != actual:
        return [f"{prefix}: checkpoint={expected!r}, requested={actual!r}"]
    return []


def validate_resume_checkpoint(
    checkpoint: dict[str, Any],
    *,
    requested_invariants: dict[str, Any],
    total_epochs: int,
) -> None:
    version = int(checkpoint.get("format_version", 1))
    if version != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"checkpoint format {version} cannot be resumed exactly by format "
            f"{CHECKPOINT_FORMAT_VERSION}; start a new epoch-based run"
        )
    stored = checkpoint.get("invariant_config")
    if not isinstance(stored, dict):
        raise ValueError("checkpoint does not contain invariant_config")
    stored_fingerprint = checkpoint.get("config_fingerprint")
    if stored_fingerprint != configuration_fingerprint(stored):
        raise ValueError("checkpoint invariant configuration fingerprint is invalid")
    differences = _differences(stored, requested_invariants)
    if differences:
        details = "\n  - ".join(differences)
        raise ValueError(f"resume configuration is incompatible:\n  - {details}")
    completed_epoch = int(checkpoint["completed_epoch"])
    if total_epochs <= completed_epoch:
        raise ValueError(
            f"training.epochs={total_epochs} must exceed completed_epoch={completed_epoch}"
        )


def move_optimizer_state(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)
