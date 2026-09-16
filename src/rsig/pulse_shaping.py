from __future__ import annotations

import numpy as np

from ._flowgraph import modules, run_chain


def rrc_taps(samples_per_symbol: int, rolloff: float, span_symbols: int = 11):
    _, _, _, _, _, _, firdes = modules()
    count = span_symbols * samples_per_symbol + 1
    taps = np.asarray(
        firdes.root_raised_cosine(samples_per_symbol, samples_per_symbol, 1.0, rolloff, count),
        dtype=np.float32,
    )
    # Unit-energy symbols produce unit average sample power in steady state.
    energy = float(np.sum(taps.astype(np.float64) ** 2))
    return taps * np.sqrt(samples_per_symbol / energy)


def rectangular_pulse(symbols: np.ndarray, samples_per_symbol: int) -> np.ndarray:
    _, blocks, _, _, _, gr, _ = modules()
    return run_chain(
        symbols,
        [blocks.repeat(gr.sizeof_gr_complex, samples_per_symbol)],
        input_kind="complex",
    )


def rrc_pulse(symbols: np.ndarray, samples_per_symbol: int, rolloff: float, span_symbols: int = 11) -> np.ndarray:
    _, _, _, _, gr_filter, _, _ = modules()
    taps = rrc_taps(samples_per_symbol, rolloff, span_symbols)
    block = gr_filter.interp_fir_filter_ccf(samples_per_symbol, taps)
    return run_chain(symbols, [block], input_kind="complex")


def delay_samples(values: np.ndarray, count: int) -> np.ndarray:
    if count < 0:
        raise ValueError("delay cannot be negative")
    _, blocks, _, _, _, gr, _ = modules()
    return run_chain(values, [blocks.delay(gr.sizeof_gr_complex, count)], input_kind="complex")

