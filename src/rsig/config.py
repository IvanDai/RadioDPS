from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np


CANONICAL_MODULATIONS = (
    "OOK", "4ASK", "8ASK", "BPSK", "QPSK", "8PSK", "16PSK", "32PSK",
    "16APSK", "32APSK", "64APSK", "128APSK", "16QAM", "32QAM",
    "64QAM", "128QAM", "256QAM", "AM-SSB-WC", "AM-SSB-SC",
    "AM-DSB-WC", "AM-DSB-SC", "FM", "GMSK", "OQPSK",
)
EXTRA_MODULATIONS = ("GFSK", "CPFSK")
MODULATION_ALIASES = {
    "PAM4": "4ASK",
    "QAM16": "16QAM",
    "QAM64": "64QAM",
    "WBFM": "FM",
    "AM-DSB": "AM-DSB-WC",
}


@dataclass(frozen=True)
class ModulationConfig:
    name: str = "QPSK"
    samples_per_symbol: int = 8
    rolloff: float = 0.35
    rrc_span_symbols: int = 11
    gmsk_bt: float = 0.3
    gfsk_bt: float = 0.3
    gfsk_sensitivity: float = np.pi / 2
    cpfsk_modulation_index: float = 0.5
    message_bandwidth: float = 0.2
    am_modulation_index: float = 1.0
    fm_deviation_hz: float = 75_000.0
    fm_preemphasis_tau: float = 75e-6

    def __post_init__(self):
        name = str(self.name).upper()
        name = MODULATION_ALIASES.get(name, name)
        if name not in CANONICAL_MODULATIONS + EXTRA_MODULATIONS:
            raise ValueError(f"unsupported modulation: {self.name!r}")
        if self.samples_per_symbol < 1:
            raise ValueError("samples_per_symbol must be positive")
        if name == "OQPSK" and self.samples_per_symbol % 2:
            raise ValueError("OQPSK requires even samples_per_symbol")
        if not 0 <= self.rolloff <= 1 or self.rrc_span_symbols < 1:
            raise ValueError("invalid RRC configuration")
        if self.gmsk_bt <= 0 or self.gfsk_bt <= 0 or self.gfsk_sensitivity <= 0:
            raise ValueError("continuous-phase modulation parameters must be positive")
        if self.cpfsk_modulation_index <= 0:
            raise ValueError("cpfsk_modulation_index must be positive")
        if not 0 < self.message_bandwidth < 1:
            raise ValueError("message_bandwidth must be relative to Nyquist and in (0, 1)")
        if not 0 <= self.am_modulation_index <= 1:
            raise ValueError("am_modulation_index must be in [0, 1]")
        if self.fm_deviation_hz <= 0 or self.fm_preemphasis_tau <= 0:
            raise ValueError("FM parameters must be positive")
        object.__setattr__(self, "name", name)


@dataclass(frozen=True)
class SourceConfig:
    kind: Literal["auto", "random", "synthetic_audio", "wav"] = "auto"
    wav_path: str | Path | None = None
    wav_channel: int = 0
    wav_start_sample: int = 0

    def __post_init__(self):
        if self.kind not in {"auto", "random", "synthetic_audio", "wav"}:
            raise ValueError("unsupported source kind")
        if self.kind == "wav" and self.wav_path is None:
            raise ValueError("wav source requires wav_path")
        if self.wav_channel < 0 or self.wav_start_sample < 0:
            raise ValueError("WAV channel and start sample cannot be negative")


@dataclass(frozen=True)
class FrameConfig:
    length: int = 1024
    guard_samples: int = 256

    def __post_init__(self):
        if self.length < 1 or self.guard_samples < 0:
            raise ValueError("frame length must be positive and guard non-negative")


@dataclass(frozen=True)
class ChannelConfig:
    kind: Literal["identity", "flat", "fir", "selective_fading"] = "identity"
    flat_coefficient: complex = 1 + 0j
    taps: tuple[complex, ...] = (1 + 0j,)
    delays_samples: tuple[float, ...] = (0.0, 0.0056, 0.0135, 0.0224, 0.0258, 0.0561, 0.1795, 0.2581, 0.5612)
    magnitudes_db: tuple[float, ...] = (-1, -1, -1, 0, 0, 0, -3, -5, -7)
    max_doppler_hz: float = 70.0
    num_sinusoids: int = 8
    los: bool = False
    k_factor: float = 4.0
    ntaps: int = 8
    normalize_taps: bool = True

    def __post_init__(self):
        if self.kind not in {"identity", "flat", "fir", "selective_fading"}:
            raise ValueError("unsupported channel kind")
        if not self.taps or not np.isfinite(self.taps).all():
            raise ValueError("FIR taps must be finite and non-empty")
        if len(self.delays_samples) != len(self.magnitudes_db) or not self.delays_samples:
            raise ValueError("selective-fading delays and magnitudes must have equal non-zero lengths")
        if min(self.delays_samples) < 0 or self.max_doppler_hz < 0:
            raise ValueError("channel delays and Doppler cannot be negative")
        if self.num_sinusoids < 1 or self.ntaps < 1 or self.k_factor < 0:
            raise ValueError("invalid selective-fading configuration")
        coefficient = complex(self.flat_coefficient)
        if coefficient == 0 or not np.isfinite((coefficient.real, coefficient.imag)).all():
            raise ValueError("flat_coefficient must be finite and non-zero")


@dataclass(frozen=True)
class XODriftConfig:
    standard_deviation: float = 1e-4
    maximum_deviation: float = 5.0
    oscillator_frequency_hz: float = 10e6
    clock_rate_hz: float = 100e6
    carrier_frequency_hz: float = 1e9

    def __post_init__(self):
        if self.standard_deviation < 0 or self.maximum_deviation <= 0:
            raise ValueError("invalid XO deviation configuration")
        if min(self.oscillator_frequency_hz, self.clock_rate_hz, self.carrier_frequency_hz) <= 0:
            raise ValueError("XO frequencies must be positive")


@dataclass(frozen=True)
class ImpairmentConfig:
    timing_offset_samples: float | None = None
    sample_rate_offset: float | None = None
    carrier_frequency_offset_hz: float | None = None
    phase_offset_rad: float | Literal["random"] | None = None
    xo_drift: XODriftConfig | None = None

    def __post_init__(self):
        values = tuple(value for value in (
            self.timing_offset_samples, self.sample_rate_offset,
            self.carrier_frequency_offset_hz, self.phase_offset_rad,
        ) if value is not None and value != "random")
        if values and not np.isfinite(values).all():
            raise ValueError("impairment values must be finite")
        if isinstance(self.phase_offset_rad, str) and self.phase_offset_rad != "random":
            raise ValueError("phase_offset_rad string value must be 'random'")
        if self.sample_rate_offset is not None and self.sample_rate_offset <= -1:
            raise ValueError("sample_rate_offset must be greater than -1")
        if self.xo_drift is not None and any(value is not None for value in (
            self.sample_rate_offset, self.carrier_frequency_offset_hz,
        )):
            raise ValueError("xo_drift cannot be combined with fixed SRO or CFO")


@dataclass(frozen=True)
class NoiseConfig:
    kind: Literal["none", "awgn"] = "none"
    calibration: Literal["sample_snr", "esn0", "rml22_nominal"] = "sample_snr"
    sample_snr_db: float = 10.0
    esn0_db: float = 10.0
    rml22_snr_db: float = 10.0

    def __post_init__(self):
        if self.kind not in {"none", "awgn"}:
            raise ValueError("noise kind must be 'none' or 'awgn'")
        if self.calibration not in {"sample_snr", "esn0", "rml22_nominal"}:
            raise ValueError("unsupported noise calibration")
        levels = (self.sample_snr_db, self.esn0_db, self.rml22_snr_db)
        if not np.isfinite(levels).all():
            raise ValueError("noise levels must be finite")


@dataclass(frozen=True)
class GeneratorConfig:
    sample_rate_hz: float = 1.0
    modulation: ModulationConfig = field(default_factory=ModulationConfig)
    source: SourceConfig = field(default_factory=SourceConfig)
    frame: FrameConfig = field(default_factory=FrameConfig)
    channel: ChannelConfig = field(default_factory=ChannelConfig)
    impairments: ImpairmentConfig = field(default_factory=ImpairmentConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)

    def __post_init__(self):
        if not np.isfinite(self.sample_rate_hz) or self.sample_rate_hz <= 0:
            raise ValueError("sample_rate_hz must be finite and positive")
