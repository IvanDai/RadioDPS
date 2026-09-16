from __future__ import annotations

import numpy as np

from ._flowgraph import modules, run_chain
from .config import CANONICAL_MODULATIONS, EXTRA_MODULATIONS, ModulationConfig
from .pulse_shaping import delay_samples, rectangular_pulse, rrc_pulse
from .schema import SourceData, Waveform


def _normalize(points):
    values = np.asarray(points, dtype=np.complex128)
    return (values / np.sqrt(np.mean(np.abs(values) ** 2))).astype(np.complex64)


def _gray_permute(points):
    values = np.asarray(points, dtype=np.complex64)
    result = np.empty_like(values)
    physical_indices = np.arange(len(values), dtype=np.uint64)
    labels = physical_indices ^ (physical_indices >> 1)
    result[labels] = values
    return result


def _gray_levels(count):
    levels = np.arange(-(count - 1), count, 2, dtype=np.float64)
    return _gray_permute(levels.astype(np.complex64)).real


def _rectangular_qam(order):
    rows = 2 ** (int(np.log2(order)) // 2)
    columns = order // rows
    i = _gray_levels(columns)
    q = _gray_levels(rows)
    return _normalize((i[:, None] + 1j * q[None, :]).reshape(-1))


def _cross_qam(order):
    side, corner = {32: (6, 1), 128: (12, 2)}[order]
    levels = np.arange(-(side - 1), side, 2)
    points = []
    for row, q in enumerate(levels):
        for column, i in enumerate(levels):
            if not ((column < corner or column >= side - corner) and (row < corner or row >= side - corner)):
                points.append(i + 1j * q)
    return _gray_permute(_normalize(points))


def _apsk(order):
    layouts = {
        16: ((4, 12), (1.0, 2.85)), 32: ((4, 12, 16), (1.0, 2.84, 5.27)),
        64: ((4, 12, 20, 28), (1.0, 2.73, 4.52, 6.31)),
        128: ((4, 12, 20, 28, 64), (1.0, 2.64, 4.64, 6.54, 8.41)),
    }
    populations, radii = layouts[order]
    points = []
    for index, (count, radius) in enumerate(zip(populations, radii, strict=True)):
        phase = 2 * np.pi * np.arange(count) / count + (index % 2) * np.pi / count
        points.extend(radius * np.exp(1j * phase))
    return _gray_permute(_normalize(points))


def constellation(name: str) -> np.ndarray:
    name = str(name).upper()
    if name == "OOK":
        return np.array([0, np.sqrt(2)], dtype=np.complex64)
    if name.endswith("ASK"):
        order = int(name[:-3])
        return _gray_permute(_normalize(np.arange(-(order - 1), order, 2)))
    if name.endswith("APSK"):
        return _apsk(int(name[:-4]))
    if name.endswith("PSK") and name != "OQPSK":
        order = {"BPSK": 2, "QPSK": 4}.get(name, int(name[:-3]) if name[:-3].isdigit() else 0)
        physical = np.exp(1j * 2 * np.pi * np.arange(order) / order).astype(np.complex64)
        return _gray_permute(physical)
    if name.endswith("QAM"):
        order = int(name[:-3])
        return _cross_qam(order) if order in {32, 128} else _rectangular_qam(order)
    raise ValueError(f"{name} does not use a memoryless constellation")


def map_symbols(indices: np.ndarray, name: str) -> np.ndarray:
    _, _, _, digital, _, _, _ = modules()
    points = constellation(name)
    if np.any(np.asarray(indices) >= len(points)):
        raise ValueError("symbol index is outside the constellation")
    return run_chain(indices, [digital.chunks_to_symbols_bc(points.tolist(), 1)], input_kind="byte")


def hard_demodulate(samples: np.ndarray, name: str) -> np.ndarray:
    points = constellation(name)
    return np.argmin(np.abs(np.asarray(samples)[:, None] - points[None, :]) ** 2, axis=1).astype(np.uint16)


def _fit(values, length, start=0):
    values = np.asarray(values)
    if start < 0 or start + length > len(values):
        raise RuntimeError("GNU Radio transmitter produced an unexpectedly short stream")
    return values[start:start + length].astype(np.complex64)


def _linear(source: SourceData, config: ModulationConfig, output_length: int, margin_symbols: int):
    symbols = map_symbols(source.symbol_indices, config.name)
    reference = rectangular_pulse(symbols, config.samples_per_symbol)
    transmitted = rrc_pulse(symbols, config.samples_per_symbol, config.rolloff, config.rrc_span_symbols)
    start = margin_symbols * config.samples_per_symbol
    metadata = {"family": "linear", "pulse_shape": "rrc", "rolloff": config.rolloff, "rrc_span_symbols": config.rrc_span_symbols}
    return _fit(reference, output_length, start), _fit(transmitted, output_length, start), symbols, metadata


def _oqpsk(source, config, output_length, margin_symbols):
    pairs = source.bits.reshape(-1, 2)
    i_symbols = map_symbols(pairs[:, 0], "BPSK")
    q_symbols = map_symbols(pairs[:, 1], "BPSK")
    i_ref = rectangular_pulse(i_symbols, config.samples_per_symbol)
    q_ref = delay_samples(rectangular_pulse(q_symbols, config.samples_per_symbol), config.samples_per_symbol // 2)
    i_tx = rrc_pulse(i_symbols, config.samples_per_symbol, config.rolloff, config.rrc_span_symbols)
    q_tx = delay_samples(rrc_pulse(q_symbols, config.samples_per_symbol, config.rolloff, config.rrc_span_symbols), config.samples_per_symbol // 2)
    start = margin_symbols * config.samples_per_symbol
    symbols = (i_symbols + 1j * q_symbols).astype(np.complex64)
    return _fit(i_ref + 1j * q_ref, output_length, start), _fit(i_tx + 1j * q_tx, output_length, start), symbols, {"family": "oqpsk", "q_delay_samples": config.samples_per_symbol // 2, "rolloff": config.rolloff}


def _continuous_phase(source, config, output_length, margin_symbols):
    analog, blocks, _, digital, _, gr, _ = modules()
    if config.name == "GMSK":
        block = digital.gmsk_mod(samples_per_symbol=config.samples_per_symbol, bt=config.gmsk_bt, verbose=False, log=False)
        metadata = {"family": "continuous_phase", "bt": config.gmsk_bt}
    elif config.name == "GFSK":
        block = digital.gfsk_mod(samples_per_symbol=config.samples_per_symbol, sensitivity=config.gfsk_sensitivity, bt=config.gfsk_bt)
        metadata = {"family": "continuous_phase", "bt": config.gfsk_bt, "sensitivity": config.gfsk_sensitivity}
    else:
        block = analog.cpfsk_bc(config.cpfsk_modulation_index, 1.0, config.samples_per_symbol)
        metadata = {"family": "continuous_phase", "modulation_index": config.cpfsk_modulation_index}
    chain = [block]
    if config.name in {"GMSK", "GFSK"}:
        chain.insert(0, blocks.unpacked_to_packed_bb(1, gr.GR_MSB_FIRST))
    transmitted = run_chain(source.bits, chain, input_kind="byte")
    start = margin_symbols * config.samples_per_symbol
    signal = _fit(transmitted, output_length, start)
    return signal.copy(), signal, np.where(source.bits > 0, 1, -1).astype(np.complex64), metadata


def _analog(source, config, sample_rate_hz, output_length):
    analog, blocks, _, _, gr_filter, _, _ = modules()
    message = np.asarray(source.message, np.float32)
    if config.name == "FM":
        sensitivity = 2 * np.pi * config.fm_deviation_hz / sample_rate_hz
        signal = run_chain(message, [analog.frequency_modulator_fc(sensitivity)], input_kind="float")
        metadata = {"family": "analog", "mode": "fm", "deviation_hz": config.fm_deviation_hz}
    elif "SSB" in config.name:
        chain = [blocks.multiply_const_ff(config.am_modulation_index)]
        if config.name.endswith("WC"):
            chain.append(blocks.add_const_ff(1.0))
        chain.append(gr_filter.hilbert_fc(401))
        signal = run_chain(message, chain, input_kind="float")
        metadata = {"family": "analog", "mode": "ssb", "carrier": config.name.endswith("WC")}
    else:
        chain = [blocks.multiply_const_ff(config.am_modulation_index)]
        if config.name.endswith("WC"):
            chain.append(blocks.add_const_ff(1.0))
        real = run_chain(message, chain, input_kind="float", output_kind="float")
        signal = real.astype(np.complex64)
        metadata = {"family": "analog", "mode": "dsb", "carrier": config.name.endswith("WC")}
    signal = _fit(signal, output_length)
    return signal.copy(), signal, None, metadata


def modulate(source: SourceData, config: ModulationConfig, sample_rate_hz: float, output_length: int, margin_symbols: int) -> Waveform:
    if config.name in CANONICAL_MODULATIONS[:17]:
        reference, transmitted, symbols, metadata = _linear(source, config, output_length, margin_symbols)
    elif config.name == "OQPSK":
        reference, transmitted, symbols, metadata = _oqpsk(source, config, output_length, margin_symbols)
    elif config.name in {"GMSK", "GFSK", "CPFSK"}:
        reference, transmitted, symbols, metadata = _continuous_phase(source, config, output_length, margin_symbols)
    else:
        reference, transmitted, symbols, metadata = _analog(source, config, sample_rate_hz, output_length)
    metadata.update({
        "constellation_average_power": 1.0 if symbols is not None else None,
        "reference_scale": 1.0,
        "transmit_scale": 1.0,
        "normalization": "deterministic constellation and pulse-tap normalization",
        "engine": "GNU Radio",
    })
    return Waveform(reference, transmitted, source, symbols, 1 if source.message is not None else config.samples_per_symbol, metadata)


generate_waveform = modulate
