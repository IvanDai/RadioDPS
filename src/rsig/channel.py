from __future__ import annotations

import numpy as np

from ._flowgraph import modules, run_chain
from .config import ChannelConfig
from .schema import ChannelState


def _normalized_taps(taps: tuple[complex, ...], normalize: bool) -> np.ndarray:
    values = np.asarray(taps, dtype=np.complex64)
    if normalize:
        energy = float(np.sum(np.abs(values.astype(np.complex128)) ** 2))
        if energy <= np.finfo(np.float64).tiny:
            raise ValueError("channel taps have zero energy")
        values = values / np.sqrt(energy)
    return values.astype(np.complex64)


def apply_identity(values: np.ndarray) -> tuple[np.ndarray, ChannelState]:
    taps = np.array([1 + 0j], dtype=np.complex64)
    return np.asarray(values, dtype=np.complex64).copy(), ChannelState("identity", taps, {})


def apply_flat_fading(values: np.ndarray, coefficient: complex) -> tuple[np.ndarray, ChannelState]:
    _, blocks, _, _, _, _, _ = modules()
    output = run_chain(values, [blocks.multiply_const_cc(complex(coefficient))])
    taps = np.array([coefficient], dtype=np.complex64)
    return output, ChannelState("flat", taps, {"coefficient": complex(coefficient)})


def apply_fir_channel(
    values: np.ndarray,
    taps: tuple[complex, ...],
    *,
    normalize: bool = True,
) -> tuple[np.ndarray, ChannelState]:
    _, _, _, _, gr_filter, _, _ = modules()
    effective_taps = _normalized_taps(taps, normalize)
    output = run_chain(values, [gr_filter.fir_filter_ccc(1, effective_taps.tolist())])
    parameters = {
        "normalize_taps": normalize,
        "tap_count": len(effective_taps),
        "boundary": "causal_zero",
    }
    return output, ChannelState("fir", effective_taps, parameters)


def apply_selective_fading(
    values: np.ndarray,
    config: ChannelConfig,
    sample_rate_hz: float,
    seed: int,
) -> tuple[np.ndarray, ChannelState]:
    _, _, channels, _, _, _, _ = modules()
    magnitudes = np.power(10.0, np.asarray(config.magnitudes_db) / 20.0)
    block = channels.selective_fading_model(
        config.num_sinusoids,
        config.max_doppler_hz / sample_rate_hz,
        config.los,
        config.k_factor,
        int(seed & 0x7FFFFFFF),
        list(config.delays_samples),
        magnitudes.tolist(),
        config.ntaps,
    )
    output = run_chain(values, [block])
    parameters = {
        "delays_samples": list(config.delays_samples),
        "magnitudes_db": list(config.magnitudes_db),
        "max_doppler_hz": config.max_doppler_hz,
        "num_sinusoids": config.num_sinusoids,
        "los": config.los,
        "k_factor": config.k_factor,
        "ntaps": config.ntaps,
        "realization_seed": int(seed),
        "time_varying": True,
    }
    return output, ChannelState("selective_fading", None, parameters)


def apply_channel(
    values: np.ndarray,
    config: ChannelConfig,
    *,
    sample_rate_hz: float,
    seed: int,
) -> tuple[np.ndarray, ChannelState]:
    if config.kind == "identity":
        return apply_identity(values)
    if config.kind == "flat":
        return apply_flat_fading(values, config.flat_coefficient)
    if config.kind == "fir":
        return apply_fir_channel(values, config.taps, normalize=config.normalize_taps)
    if config.kind == "selective_fading":
        return apply_selective_fading(values, config, sample_rate_hz, seed)
    raise ValueError(f"unsupported channel kind: {config.kind!r}")
