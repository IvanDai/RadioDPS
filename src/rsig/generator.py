from __future__ import annotations

from dataclasses import asdict
import math

import numpy as np

from .channel import apply_channel
from .config import GeneratorConfig
from .impairment import (
    apply_carrier_frequency_offset,
    apply_phase_offset,
    apply_sample_rate_offset,
    apply_timing_offset,
    apply_xo_carrier_frequency_offset,
    apply_xo_sample_rate_offset,
    resolve_impairments,
)
from .modulation import modulate
from .noise import add_noise
from .schema import Sample
from .seed import component_seeds
from .sources import generate_source, is_analog


def _crop(values: np.ndarray, start: int, length: int, name: str) -> np.ndarray:
    stop = start + length
    if start < 0 or stop > len(values):
        raise ValueError(
            f"{name} has {len(values)} samples after processing, but crop [{start}:{stop}] "
            "was requested; increase frame.guard_samples or reduce the enabled offset"
        )
    return np.asarray(values[start:stop], dtype=np.complex64)


def _fit_processing_length(values: np.ndarray, length: int) -> np.ndarray:
    """Keep receiver stages finite and aligned without periodic wraparound."""
    array = np.asarray(values, dtype=np.complex64)
    if len(array) >= length:
        return array[:length]
    return np.pad(array, (0, length - len(array)))


def generate_sample(config: GeneratorConfig, seed: int = 0) -> Sample:
    """Generate one reproducible RF sample using the configured receive chain."""
    seeds = component_seeds(seed)
    working_length = config.frame.length + 2 * config.frame.guard_samples
    crop_start = config.frame.guard_samples
    margin_symbols = config.modulation.rrc_span_symbols + 2
    symbol_count = math.ceil(working_length / config.modulation.samples_per_symbol) + 2 * margin_symbols
    message_length = working_length + 512

    source_rng = np.random.default_rng(seeds.source)
    source = generate_source(
        config.modulation.name,
        config.source,
        symbol_count,
        message_length,
        source_rng,
        message_bandwidth=config.modulation.message_bandwidth,
    )
    waveform = modulate(
        source,
        config.modulation,
        config.sample_rate_hz,
        working_length,
        margin_symbols,
    )

    x_ref_full = waveform.reference
    x_tx_full = waveform.transmitted
    channel_input = x_tx_full.copy()
    if config.channel.kind == "fir":
        channel_input[:crop_start] = 0
    y_rx_full, channel_state = apply_channel(
        channel_input,
        config.channel,
        sample_rate_hz=config.sample_rate_hz,
        seed=seeds.channel,
    )

    impairment_rng = np.random.default_rng(seeds.impairment)
    impairment_state = resolve_impairments(config.impairments, len(y_rx_full), impairment_rng)
    y_rx_full = _fit_processing_length(apply_timing_offset(y_rx_full, impairment_state.timing_offset_samples), working_length)
    y_rx_full = _fit_processing_length(apply_sample_rate_offset(y_rx_full, impairment_state.sample_rate_offset), working_length)
    y_rx_full = _fit_processing_length(apply_xo_sample_rate_offset(y_rx_full, impairment_state.xo_trajectory, config.impairments.xo_drift), working_length)
    y_rx_full = _fit_processing_length(apply_carrier_frequency_offset(y_rx_full, impairment_state.carrier_frequency_offset_hz, config.sample_rate_hz), working_length)
    y_rx_full = _fit_processing_length(apply_xo_carrier_frequency_offset(y_rx_full, impairment_state.xo_trajectory, config.impairments.xo_drift, config.sample_rate_hz), working_length)
    y_rx_full = _fit_processing_length(apply_phase_offset(y_rx_full, impairment_state.phase_offset_rad), working_length)
    y_rx_full, noise_state = add_noise(
        y_rx_full,
        config.noise,
        samples_per_symbol=waveform.samples_per_symbol,
        seed=seeds.noise,
    )

    x_ref = _crop(x_ref_full, crop_start, config.frame.length, "x_ref")
    x_tx = _crop(x_tx_full, crop_start, config.frame.length, "x_tx")
    y_rx = _crop(y_rx_full, crop_start, config.frame.length, "y_rx")

    frame = {
        "length": config.frame.length,
        "guard_samples_before": config.frame.guard_samples,
        "guard_samples_after": config.frame.guard_samples,
        "crop_start": crop_start,
        "working_length": working_length,
        "no_periodic_wrap": True,
        "source_contains_generation_margins": True,
        "channel_boundary": "causal_zero" if config.channel.kind == "fir" else None,
    }
    normalization = {
        **waveform.metadata,
        "x_ref_sample_power": float(np.mean(np.abs(x_ref) ** 2)),
        "x_tx_sample_power": float(np.mean(np.abs(x_tx) ** 2)),
        "per_sample_waveform_normalization": False,
    }
    source_symbols = waveform.symbols if not is_analog(config.modulation.name) else None
    return Sample(
        x_ref=x_ref,
        x_tx=x_tx,
        y_rx=y_rx,
        modulation=config.modulation.name,
        source_bits=source.bits,
        source_symbol_indices=source.symbol_indices,
        source_symbols=source_symbols,
        source_message=source.message,
        channel=channel_state,
        impairments=impairment_state,
        noise=noise_state,
        sample_seed=int(seed),
        frame=frame,
        normalization=normalization,
        config=asdict(config),
    )
