from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class SourceData:
    bits: np.ndarray | None
    symbol_indices: np.ndarray | None
    message: np.ndarray | None
    metadata: dict


@dataclass(frozen=True)
class Waveform:
    reference: np.ndarray
    transmitted: np.ndarray
    source: SourceData
    symbols: np.ndarray | None
    samples_per_symbol: int
    metadata: dict


@dataclass(frozen=True)
class ChannelState:
    kind: str
    taps: np.ndarray | None
    parameters: dict


@dataclass(frozen=True)
class ImpairmentState:
    timing_offset_samples: float | None
    sample_rate_offset: float | None
    carrier_frequency_offset_hz: float | None
    phase_offset_rad: float | None
    xo_trajectory: np.ndarray | None
    parameters: dict


@dataclass(frozen=True)
class NoiseState:
    kind: str
    calibration: str
    level_db: float | None
    reference_power: float
    variance: float
    component_std: float
    parameters: dict


@dataclass(frozen=True)
class Sample:
    x_ref: np.ndarray
    x_tx: np.ndarray
    y_rx: np.ndarray
    modulation: str
    source_bits: np.ndarray | None
    source_symbol_indices: np.ndarray | None
    source_symbols: np.ndarray | None
    source_message: np.ndarray | None
    channel: ChannelState
    impairments: ImpairmentState
    noise: NoiseState
    sample_seed: int
    frame: dict
    normalization: dict
    config: dict


def complex_to_real(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    return np.stack((array.real, array.imag), axis=-1).astype(np.float32)


def real_to_complex(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim < 1 or array.shape[-1] != 2:
        raise ValueError("last dimension must have size 2")
    return (array[..., 0] + 1j * array[..., 1]).astype(np.complex64)
