from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Iterable, Iterator, Mapping

import h5py
import numpy as np

from .schema import Sample, complex_to_real, real_to_complex


FORMAT_VERSION = "rdps-rsig-hdf5-1"
PARAMETER_DTYPES = {
    "samples_per_symbol": np.uint16,
    "rrc_alpha": np.float32,
    "timing_offset": np.float32,
    "symbol_rate_offset": np.float32,
    "phase_offset": np.float32,
    "carrier_offset": np.float32,
    "delay_spread": np.float32,
    "channel_path_count": np.uint16,
    "channel_length": np.uint16,
    "noise_variance": np.float32,
    "noise_std": np.float32,
    "x_tx_scale": np.float32,
    "sample_seed": np.uint64,
}
DEFAULT_IMPAIRMENT_ORDER = (
    "timing_offset",
    "symbol_rate_offset",
    "carrier_frequency_offset",
    "phase_offset",
)


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _json_dumps(value: Mapping) -> str:
    return json.dumps(value, default=_json_default, ensure_ascii=True, sort_keys=True, allow_nan=False)


def _json_loads(value) -> dict:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, np.ndarray):
        value = value.item()
    return json.loads(str(value))


@dataclass(frozen=True)
class DatasetSpec:
    """Global dimensions and invariant metadata for the RDPS HDF5 schema."""

    modulations: tuple[str, ...]
    noise_levels_db: tuple[float, ...]
    examples_per_condition: int
    frame_length: int
    max_channel_taps: int
    generation_mode: str
    channel_type: str
    channel_normalization: str
    noise_calibration: str
    root_seed: int
    boundary: str = "causal_zero"
    impairment_order: tuple[str, ...] = DEFAULT_IMPAIRMENT_ORDER
    seed_derivation_rule: str = "SeedSequence(root_seed, spawn_key=(modulation_coordinate, noise_coordinate, example_coordinate))"
    extra_metadata: Mapping = field(default_factory=dict)

    def __post_init__(self):
        if not self.modulations or len(set(self.modulations)) != len(self.modulations):
            raise ValueError("modulations must be non-empty and unique")
        if not self.noise_levels_db:
            raise ValueError("noise_levels_db must contain at least one condition")
        if any(np.isinf(value) for value in self.noise_levels_db):
            raise ValueError("noise_levels_db values must be finite or NaN for disabled noise")
        if self.examples_per_condition < 1 or self.frame_length < 1 or self.max_channel_taps < 1:
            raise ValueError("E, T, and L dimensions must be positive")
        if self.max_channel_taps > np.iinfo(np.uint16).max:
            raise ValueError("max_channel_taps must fit the uint16 channel_length field")
        if self.root_seed < 0:
            raise ValueError("root_seed must be non-negative")
        if self.boundary != "causal_zero":
            raise ValueError("the RDPS schema currently requires boundary='causal_zero'")
        required_text = (
            self.generation_mode,
            self.channel_type,
            self.channel_normalization,
            self.noise_calibration,
            self.seed_derivation_rule,
        )
        if any(not str(value) for value in required_text):
            raise ValueError("global metadata string fields cannot be empty")

    @property
    def shape(self) -> tuple[int, int, int]:
        return len(self.modulations), len(self.noise_levels_db), self.examples_per_condition

    def metadata(self) -> dict:
        base = {
            "format": "HDF5",
            "version": FORMAT_VERSION,
            "generation_mode": self.generation_mode,
            "modulations": list(self.modulations),
            "noise_levels_db": [None if np.isnan(value) else float(value) for value in self.noise_levels_db],
            "dimensions": {
                "M": len(self.modulations),
                "S": len(self.noise_levels_db),
                "E": self.examples_per_condition,
                "T": self.frame_length,
                "L": self.max_channel_taps,
            },
            "examples_per_condition": self.examples_per_condition,
            "frame_length": self.frame_length,
            "max_channel_taps": self.max_channel_taps,
            "channel_type": self.channel_type,
            "boundary": self.boundary,
            "channel_normalization": self.channel_normalization,
            "noise_calibration": self.noise_calibration,
            "impairment_order": list(self.impairment_order),
            "iq_layout": ["I", "Q"],
            "root_seed": self.root_seed,
            "seed_derivation_rule": self.seed_derivation_rule,
            "array_coordinates": ["modulation", "condition", "example", "iq", "sample"],
        }
        overlap = set(base).intersection(self.extra_metadata)
        if overlap:
            raise ValueError(f"extra_metadata cannot override required fields: {sorted(overlap)}")
        return {**base, **dict(self.extra_metadata)}


def _noise_level(sample: Sample) -> float:
    return np.nan if sample.noise.kind == "none" else float(sample.noise.level_db)


def _optional_float(value) -> float:
    return 0.0 if value is None else float(value)


def _delay_spread(sample: Sample, taps: np.ndarray) -> float:
    delays = sample.channel.parameters.get("delays_samples")
    if delays:
        return float(max(delays) - min(delays))
    nonzero = np.flatnonzero(np.abs(taps) > np.finfo(np.float32).eps)
    return 0.0 if len(nonzero) < 2 else float(nonzero[-1] - nonzero[0])


def _path_count(sample: Sample, taps: np.ndarray) -> int:
    delays = sample.channel.parameters.get("delays_samples")
    if delays:
        return len(delays)
    return int(np.count_nonzero(np.abs(taps) > np.finfo(np.float32).eps))


def _sample_parameters(sample: Sample, taps: np.ndarray) -> dict:
    modulation_config = sample.config.get("modulation", {})
    waveform_family = sample.normalization.get("family")
    rrc_alpha = modulation_config.get("rolloff", 0.0) if waveform_family in {"linear", "oqpsk"} else 0.0
    return {
        "samples_per_symbol": int(modulation_config.get("samples_per_symbol", 1)),
        "rrc_alpha": float(rrc_alpha),
        "timing_offset": _optional_float(sample.impairments.timing_offset_samples),
        "symbol_rate_offset": _optional_float(sample.impairments.sample_rate_offset),
        "phase_offset": _optional_float(sample.impairments.phase_offset_rad),
        "carrier_offset": _optional_float(sample.impairments.carrier_frequency_offset_hz),
        "delay_spread": _delay_spread(sample, taps),
        "channel_path_count": _path_count(sample, taps),
        "channel_length": len(taps),
        "noise_variance": float(sample.noise.variance),
        "noise_std": float(sample.noise.component_std),
        "x_tx_scale": float(sample.normalization.get("transmit_scale", 1.0)),
        "sample_seed": int(sample.sample_seed),
    }


class HDF5DatasetWriter:
    """Write samples directly into the fixed [M,S,E,2,T] RDPS grid."""

    def __init__(self, path: str | Path, spec: DatasetSpec):
        self.path = Path(path)
        self.spec = spec
        self._metadata = spec.metadata()
        self.file = h5py.File(self.path, "w")
        self._written = np.zeros(spec.shape, dtype=np.bool_)
        self._next_example = np.zeros(spec.shape[:2], dtype=np.int64)
        self._closed = False
        self._create_schema()

    def _create_schema(self):
        m_count, s_count, e_count = self.spec.shape
        text_dtype = h5py.string_dtype(encoding="utf-8")
        self.file.create_dataset("modulation", data=np.asarray(self.spec.modulations, dtype=object), dtype=text_dtype)
        self.file.create_dataset("noise_level_db", data=np.asarray(self.spec.noise_levels_db, dtype=np.float32))
        waveform_shape = (m_count, s_count, e_count, 2, self.spec.frame_length)
        channel_shape = (m_count, s_count, e_count, 2, self.spec.max_channel_taps)
        self.file.create_dataset("x_tx", shape=waveform_shape, dtype=np.float32, chunks=(1, 1, 1, 2, self.spec.frame_length))
        self.file.create_dataset("h", shape=channel_shape, dtype=np.float32, chunks=(1, 1, 1, 2, self.spec.max_channel_taps))
        self.file.create_dataset("y_rx", shape=waveform_shape, dtype=np.float32, chunks=(1, 1, 1, 2, self.spec.frame_length))
        parameters = self.file.create_group("parameters")
        for name, dtype in PARAMETER_DTYPES.items():
            parameters.create_dataset(name, shape=self.spec.shape, dtype=dtype, chunks=(1, 1, 1))
        self.file.create_dataset("metadata", data=_json_dumps(self._metadata), dtype=text_dtype)

    def _resolve_coordinate(
        self,
        sample: Sample,
        condition_index: int,
        example_index: int | None,
        modulation_index: int | None,
    ) -> tuple[int, int, int]:
        if modulation_index is None:
            try:
                modulation_index = self.spec.modulations.index(sample.modulation)
            except ValueError as error:
                raise ValueError(f"sample modulation {sample.modulation!r} is not in DatasetSpec") from error
        if not 0 <= modulation_index < self.spec.shape[0]:
            raise IndexError("modulation_index is outside the dataset grid")
        if self.spec.modulations[modulation_index] != sample.modulation:
            raise ValueError("sample modulation does not match modulation_index")
        if not 0 <= condition_index < self.spec.shape[1]:
            raise IndexError("condition_index is outside the dataset grid")
        expected_level = self.spec.noise_levels_db[condition_index]
        actual_level = _noise_level(sample)
        if not (np.isnan(expected_level) and np.isnan(actual_level)) and not np.isclose(expected_level, actual_level):
            raise ValueError(
                f"sample noise level {actual_level} does not match noise_level_db[{condition_index}]={expected_level}"
            )
        if example_index is None:
            example_index = int(self._next_example[modulation_index, condition_index])
        if not 0 <= example_index < self.spec.shape[2]:
            raise IndexError("example_index is outside the dataset grid")
        return modulation_index, condition_index, example_index

    def append(
        self,
        sample: Sample,
        *,
        condition_index: int = 0,
        example_index: int | None = None,
        modulation_index: int | None = None,
    ) -> tuple[int, int, int]:
        if self._closed:
            raise RuntimeError("writer is closed")
        coordinate = self._resolve_coordinate(sample, condition_index, example_index, modulation_index)
        if self._written[coordinate]:
            raise ValueError(f"dataset coordinate {coordinate} has already been written")
        if len(sample.x_tx) != self.spec.frame_length or len(sample.y_rx) != self.spec.frame_length:
            raise ValueError("x_tx and y_rx must match DatasetSpec.frame_length")
        sample_channel_type = str(sample.config.get("channel", {}).get("kind", sample.channel.kind))
        if sample_channel_type != self.spec.channel_type:
            raise ValueError(
                f"sample channel type {sample_channel_type!r} does not match global "
                f"channel_type={self.spec.channel_type!r}"
            )
        if sample.noise.kind != "none" and sample.noise.calibration != self.spec.noise_calibration:
            raise ValueError(
                f"sample noise calibration {sample.noise.calibration!r} does not match global "
                f"noise_calibration={self.spec.noise_calibration!r}"
            )
        if sample.channel.taps is None:
            raise ValueError("the fixed known-channel schema requires explicit per-sample channel taps")
        taps = np.asarray(sample.channel.taps, dtype=np.complex64).reshape(-1)
        if len(taps) > self.spec.max_channel_taps:
            raise ValueError(
                f"sample has {len(taps)} channel taps, exceeding max_channel_taps={self.spec.max_channel_taps}"
            )
        parameters = _sample_parameters(sample, taps)
        if parameters["samples_per_symbol"] > np.iinfo(np.uint16).max:
            raise ValueError("samples_per_symbol does not fit the uint16 schema field")
        if parameters["sample_seed"] < 0 or parameters["sample_seed"] > np.iinfo(np.uint64).max:
            raise ValueError("sample_seed does not fit the uint64 schema field")

        self.file["x_tx"][coordinate] = np.moveaxis(complex_to_real(sample.x_tx), -1, 0)
        self.file["y_rx"][coordinate] = np.moveaxis(complex_to_real(sample.y_rx), -1, 0)
        padded_taps = np.zeros(self.spec.max_channel_taps, dtype=np.complex64)
        padded_taps[:len(taps)] = taps
        self.file["h"][coordinate] = np.moveaxis(complex_to_real(padded_taps), -1, 0)
        for name, value in parameters.items():
            self.file[f"parameters/{name}"][coordinate] = value

        self._written[coordinate] = True
        m_index, s_index, e_index = coordinate
        if e_index == self._next_example[m_index, s_index]:
            next_empty = np.flatnonzero(~self._written[m_index, s_index])
            self._next_example[m_index, s_index] = next_empty[0] if len(next_empty) else self.spec.shape[2]
        return coordinate

    def close(self, *, require_complete: bool = True):
        if self._closed:
            return
        missing = int(np.size(self._written) - np.count_nonzero(self._written))
        self.file.flush()
        self.file.close()
        self._closed = True
        if require_complete and missing:
            raise ValueError(f"dataset grid is incomplete: {missing} sample coordinates were not written")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None:
            self.close()
        else:
            self.file.close()
            self._closed = True


@dataclass(frozen=True)
class DatasetRecord:
    x_tx: np.ndarray
    h: np.ndarray
    y_rx: np.ndarray
    modulation: str
    noise_level_db: float
    modulation_index: int
    condition_index: int
    example_index: int
    parameters: dict


class SignalDataset:
    """Read records from the fixed HDF5 coordinate grid without repartitioning."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.file = h5py.File(self.path, "r")
        self.metadata = _json_loads(self.file["metadata"][()])
        if self.metadata.get("version") != FORMAT_VERSION:
            raise ValueError("unsupported rsig HDF5 schema version")
        self.modulations = tuple(self.file["modulation"].asstr()[:])
        self.noise_levels_db = np.asarray(self.file["noise_level_db"][:], dtype=np.float32)
        self.grid_shape = tuple(self.file["x_tx"].shape[:3])

    def _coordinate(self, item: int | tuple[int, int, int]) -> tuple[int, int, int]:
        if isinstance(item, tuple):
            if len(item) != 3:
                raise IndexError("coordinate must be (modulation_index, condition_index, example_index)")
            return tuple(int(value) for value in item)
        return tuple(int(value) for value in np.unravel_index(item, self.grid_shape))

    def __getitem__(self, item: int | tuple[int, int, int]) -> DatasetRecord:
        coordinate = self._coordinate(item)
        m_index, s_index, e_index = coordinate
        parameters = {
            name: dataset[coordinate].item()
            for name, dataset in self.file["parameters"].items()
        }
        return DatasetRecord(
            x_tx=real_to_complex(np.moveaxis(self.file["x_tx"][coordinate], 0, -1)),
            h=real_to_complex(np.moveaxis(self.file["h"][coordinate], 0, -1))[:int(parameters["channel_length"])],
            y_rx=real_to_complex(np.moveaxis(self.file["y_rx"][coordinate], 0, -1)),
            modulation=self.modulations[m_index],
            noise_level_db=float(self.noise_levels_db[s_index]),
            modulation_index=m_index,
            condition_index=s_index,
            example_index=e_index,
            parameters=parameters,
        )

    def __len__(self):
        return int(np.prod(self.grid_shape))

    def iter_condition(self, modulation_index: int, condition_index: int) -> Iterator[DatasetRecord]:
        for example_index in range(self.grid_shape[2]):
            yield self[modulation_index, condition_index, example_index]

    def close(self):
        self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()


def write_hdf5(path: str | Path, spec: DatasetSpec, samples: Iterable[tuple[Sample, int, int | None]]):
    """Write `(sample, condition_index, example_index)` records to an HDF5 grid."""
    with HDF5DatasetWriter(path, spec) as writer:
        for sample, condition_index, example_index in samples:
            writer.append(sample, condition_index=condition_index, example_index=example_index)


def write_dataset(path: str | Path, spec: DatasetSpec, samples: Iterable[tuple[Sample, int, int | None]]):
    if Path(path).suffix.lower() not in {".h5", ".hdf5"}:
        raise ValueError("the unified RDPS dataset format must use .h5 or .hdf5")
    return write_hdf5(path, spec, samples)
