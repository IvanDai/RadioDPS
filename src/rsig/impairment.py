from __future__ import annotations

import numpy as np

from ._flowgraph import modules, run_chain
from .config import ImpairmentConfig, XODriftConfig
from .schema import ImpairmentState


def sample_xo_trajectory(length: int, config: XODriftConfig, rng: np.random.Generator) -> np.ndarray:
    if length < 1:
        return np.empty(0, dtype=np.float64)
    limit = config.maximum_deviation
    bias = rng.uniform(-limit + config.standard_deviation, limit - config.standard_deviation)
    result = np.empty(length, dtype=np.float64)
    result[0] = np.clip(bias + config.standard_deviation * rng.standard_normal(), -limit, limit)
    for index in range(1, length):
        proposal = result[index - 1] + config.standard_deviation * rng.standard_normal()
        while abs(proposal) > limit:
            proposal = result[index - 1] + config.standard_deviation * rng.standard_normal()
        result[index] = proposal
    return result


def apply_timing_offset(values: np.ndarray, offset_samples: float | None) -> np.ndarray:
    if offset_samples is None or offset_samples == 0:
        return np.asarray(values, dtype=np.complex64).copy()
    _, _, _, _, gr_filter, _, _ = modules()
    integer = int(np.floor(offset_samples))
    fractional = float(offset_samples - integer)
    source = np.asarray(values, dtype=np.complex64)
    if integer >= 0:
        source = source[integer:]
    else:
        source = np.pad(source, (-integer, 0))
    return run_chain(source, [gr_filter.mmse_resampler_cc(fractional, 1.0)])


def apply_sample_rate_offset(values: np.ndarray, offset: float | None) -> np.ndarray:
    if offset is None or offset == 0:
        return np.asarray(values, dtype=np.complex64).copy()
    _, _, _, _, gr_filter, _, _ = modules()
    return run_chain(values, [gr_filter.mmse_resampler_cc(0.0, 1.0 + float(offset))])


def apply_carrier_frequency_offset(values: np.ndarray, offset_hz: float | None, sample_rate_hz: float) -> np.ndarray:
    if offset_hz is None or offset_hz == 0:
        return np.asarray(values, dtype=np.complex64).copy()
    _, blocks, _, _, _, _, _ = modules()
    phase_increment = 2 * np.pi * float(offset_hz) / sample_rate_hz
    return run_chain(values, [blocks.rotator_cc(phase_increment)])


def apply_phase_offset(values: np.ndarray, offset_rad: float | None) -> np.ndarray:
    if offset_rad is None or offset_rad == 0:
        return np.asarray(values, dtype=np.complex64).copy()
    _, blocks, _, _, _, _, _ = modules()
    return run_chain(values, [blocks.multiply_const_cc(np.exp(1j * float(offset_rad)))])


def apply_xo_sample_rate_offset(values: np.ndarray, trajectory: np.ndarray | None, config: XODriftConfig | None) -> np.ndarray:
    if trajectory is None or config is None:
        return np.asarray(values, dtype=np.complex64).copy()
    source = np.asarray(values, dtype=np.complex64)
    scale = config.clock_rate_hz / config.oscillator_frequency_hz
    fractional_error = trajectory[:len(source)] * scale / config.clock_rate_hz
    positions = np.arange(len(source), dtype=np.float64) + np.cumsum(fractional_error)
    domain = np.arange(len(source), dtype=np.float64)
    real = np.interp(positions, domain, source.real, left=0.0, right=0.0)
    imag = np.interp(positions, domain, source.imag, left=0.0, right=0.0)
    return (real + 1j * imag).astype(np.complex64)


def apply_xo_carrier_frequency_offset(
    values: np.ndarray,
    trajectory: np.ndarray | None,
    config: XODriftConfig | None,
    sample_rate_hz: float,
) -> np.ndarray:
    if trajectory is None or config is None:
        return np.asarray(values, dtype=np.complex64).copy()
    source = np.asarray(values, dtype=np.complex64)
    scale = config.carrier_frequency_hz / config.oscillator_frequency_hz
    instantaneous_hz = trajectory[:len(source)] * scale
    phase = 2 * np.pi * np.cumsum(instantaneous_hz) / sample_rate_hz
    return (source * np.exp(1j * phase)).astype(np.complex64)


def resolve_impairments(config: ImpairmentConfig, length: int, rng: np.random.Generator) -> ImpairmentState:
    trajectory = sample_xo_trajectory(length, config.xo_drift, rng) if config.xo_drift else None
    phase_offset = rng.uniform(0.0, 2 * np.pi) if config.phase_offset_rad == "random" else config.phase_offset_rad
    parameters = {
        "order": ["timing_offset", "sample_rate_offset", "carrier_frequency_offset", "phase_offset"],
        "xo_couples_sro_and_cfo": config.xo_drift is not None,
    }
    return ImpairmentState(
        config.timing_offset_samples,
        config.sample_rate_offset,
        config.carrier_frequency_offset_hz,
        phase_offset,
        trajectory,
        parameters,
    )
