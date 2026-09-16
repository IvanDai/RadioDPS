"""PyTorch access to the fixed rsig HDF5 grid."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


SPLIT_ALGORITHM_VERSION = "example-permutation-v1"
SPLIT_NAMES = ("train", "validation", "test")


def _split_counts(size: int, ratios: Mapping[str, float]) -> dict[str, int]:
    if set(ratios) != set(SPLIT_NAMES):
        raise ValueError(f"split_ratios must contain exactly {SPLIT_NAMES}")
    values = np.asarray([ratios[name] for name in SPLIT_NAMES], dtype=np.float64)
    if not np.isfinite(values).all() or (values < 0).any() or not np.isclose(values.sum(), 1.0):
        raise ValueError("split ratios must be finite, non-negative, and sum to one")
    exact = values * size
    counts = np.floor(exact).astype(np.int64)
    remainder = size - int(counts.sum())
    order = np.argsort(-(exact - counts), kind="stable")
    counts[order[:remainder]] += 1
    return {name: int(count) for name, count in zip(SPLIT_NAMES, counts)}


def split_example_indices(
    examples_per_condition: int,
    *,
    seed: int,
    ratios: Mapping[str, float],
) -> dict[str, np.ndarray]:
    """Split example coordinates once, then reuse them for every stratum."""
    if examples_per_condition < 1 or seed < 0:
        raise ValueError("examples_per_condition must be positive and seed non-negative")
    counts = _split_counts(examples_per_condition, ratios)
    permutation = np.random.default_rng(seed).permutation(examples_per_condition)
    result: dict[str, np.ndarray] = {}
    offset = 0
    for name in SPLIT_NAMES:
        result[name] = np.sort(permutation[offset : offset + counts[name]])
        offset += counts[name]
    return result


class HDF5WaveformDataset(Dataset):
    """Flatten `(modulation, condition, example)` into PyTorch samples.

    The HDF5 handle is opened lazily so DataLoader worker processes never share
    an h5py handle inherited from their parent.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        split: str,
        split_seed: int,
        split_ratios: Mapping[str, float],
    ):
        if split not in SPLIT_NAMES:
            raise ValueError(f"split must be one of {SPLIT_NAMES}")
        self.path = Path(path).expanduser().resolve()
        self.split = split
        self.split_seed = int(split_seed)
        self.split_ratios = {name: float(split_ratios[name]) for name in SPLIT_NAMES}
        self._file: h5py.File | None = None

        with h5py.File(self.path, "r") as handle:
            self._validate_schema(handle)
            self.grid_shape = tuple(int(value) for value in handle["x_tx"].shape[:3])
            self.frame_length = int(handle["x_tx"].shape[-1])
            self.max_channel_taps = int(handle["h"].shape[-1])
            self.modulations = tuple(handle["modulation"].asstr()[:])
            raw_metadata = handle["metadata"][()]
            if isinstance(raw_metadata, bytes):
                raw_metadata = raw_metadata.decode("utf-8")
            self.metadata = json.loads(str(raw_metadata))
            if self.metadata.get("version") != "rdps-rsig-hdf5-1":
                raise ValueError("unsupported rsig HDF5 schema version")

        examples = split_example_indices(
            self.grid_shape[2], seed=self.split_seed, ratios=self.split_ratios
        )[split]
        self.example_indices = tuple(int(value) for value in examples)
        self.coordinates = tuple(
            (modulation, condition, example)
            for modulation in range(self.grid_shape[0])
            for condition in range(self.grid_shape[1])
            for example in self.example_indices
        )

    @staticmethod
    def _validate_schema(handle: h5py.File) -> None:
        required = {"x_tx", "h", "y_rx", "modulation", "parameters", "metadata"}
        missing = required.difference(handle.keys())
        if missing:
            raise ValueError(f"dataset is missing HDF5 entries: {sorted(missing)}")
        x_shape = handle["x_tx"].shape
        y_shape = handle["y_rx"].shape
        h_shape = handle["h"].shape
        if len(x_shape) != 5 or x_shape[-2] != 2 or y_shape != x_shape:
            raise ValueError("x_tx and y_rx must have matching shape [M,S,E,2,T]")
        if len(h_shape) != 5 or h_shape[:3] != x_shape[:3] or h_shape[-2] != 2:
            raise ValueError("h must have shape [M,S,E,2,L]")
        if "noise_variance" not in handle["parameters"]:
            raise ValueError("parameters/noise_variance is required")
        if handle["parameters/noise_variance"].shape != x_shape[:3]:
            raise ValueError("parameter coordinates must match [M,S,E]")

    def _handle(self) -> h5py.File:
        if self._file is None:
            self._file = h5py.File(self.path, "r")
        return self._file

    def __len__(self) -> int:
        return len(self.coordinates)

    def __getitem__(self, index: int) -> dict[str, object]:
        coordinate = self.coordinates[index]
        modulation, condition, example = coordinate
        handle = self._handle()
        return {
            "x_tx": torch.from_numpy(np.asarray(handle["x_tx"][coordinate], dtype=np.float32)),
            "h": torch.from_numpy(np.asarray(handle["h"][coordinate], dtype=np.float32)),
            "y_rx": torch.from_numpy(np.asarray(handle["y_rx"][coordinate], dtype=np.float32)),
            "modulation_index": modulation,
            "condition_index": condition,
            "example_index": example,
            "noise_variance": float(handle["parameters/noise_variance"][coordinate]),
            "modulation": self.modulations[modulation],
        }

    def split_manifest(self) -> dict[str, object]:
        counts = _split_counts(self.grid_shape[2], self.split_ratios)
        return {
            "algorithm": SPLIT_ALGORITHM_VERSION,
            "seed": self.split_seed,
            "ratios": self.split_ratios,
            "examples_per_stratum": counts,
            "strata": self.grid_shape[0] * self.grid_shape[1],
            "selected_split": self.split,
            "selected_examples": list(self.example_indices),
            "sample_count": len(self),
        }

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_file"] = None
        return state

    def __del__(self):
        self.close()
