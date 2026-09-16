from __future__ import annotations

import numpy as np

from ._flowgraph import modules
from .config import NoiseConfig
from .schema import NoiseState


def _gnu_radio_awgn(values: np.ndarray, component_std: float, seed: int) -> np.ndarray:
    analog, blocks, _, _, _, gr, _ = modules()
    top = gr.top_block()
    source = blocks.vector_source_c(np.asarray(values, dtype=np.complex64), False)
    noise = analog.noise_source_c(analog.GR_GAUSSIAN, component_std, int(seed & 0x7FFFFFFF))
    adder = blocks.add_vcc(1)
    head = blocks.head(gr.sizeof_gr_complex, len(values))
    sink = blocks.vector_sink_c()
    top.connect(source, (adder, 0))
    top.connect(noise, (adder, 1))
    top.connect(adder, head, sink)
    top.run()
    return np.asarray(sink.data(), dtype=np.complex64)


def no_noise(values: np.ndarray, calibration: str = "sample_snr") -> tuple[np.ndarray, NoiseState]:
    power = float(np.mean(np.abs(np.asarray(values)) ** 2))
    state = NoiseState("none", calibration, None, power, 0.0, 0.0, {})
    return np.asarray(values, dtype=np.complex64).copy(), state


def add_awgn_sample_snr(values: np.ndarray, snr_db: float, seed: int) -> tuple[np.ndarray, NoiseState]:
    power = float(np.mean(np.abs(np.asarray(values)) ** 2))
    variance = power / (10.0 ** (snr_db / 10.0))
    component_std = float(np.sqrt(variance / 2.0))
    output = _gnu_radio_awgn(values, component_std, seed)
    state = NoiseState("awgn", "sample_snr", snr_db, power, variance, component_std, {"sample_snr_db": snr_db})
    return output, state


def add_awgn_esn0(values: np.ndarray, esn0_db: float, samples_per_symbol: int, seed: int) -> tuple[np.ndarray, NoiseState]:
    sample_power = float(np.mean(np.abs(np.asarray(values)) ** 2))
    symbol_energy = sample_power * samples_per_symbol
    variance = symbol_energy / (10.0 ** (esn0_db / 10.0))
    component_std = float(np.sqrt(variance / 2.0))
    output = _gnu_radio_awgn(values, component_std, seed)
    parameters = {"esn0_db": esn0_db, "samples_per_symbol": samples_per_symbol}
    state = NoiseState("awgn", "esn0", esn0_db, symbol_energy, variance, component_std, parameters)
    return output, state


def add_awgn_rml22(values: np.ndarray, nominal_snr_db: float, seed: int) -> tuple[np.ndarray, NoiseState]:
    component_std = float(10.0 ** (-nominal_snr_db / 20.0))
    variance = 2.0 * component_std ** 2
    output = _gnu_radio_awgn(values, component_std, seed)
    power = float(np.mean(np.abs(np.asarray(values)) ** 2))
    parameters = {"rml22_snr_db": nominal_snr_db, "definition": "GNU Radio component amplitude = 10**(-level/20)"}
    state = NoiseState("awgn", "rml22_nominal", nominal_snr_db, power, variance, component_std, parameters)
    return output, state


def add_noise(values: np.ndarray, config: NoiseConfig, *, samples_per_symbol: int, seed: int) -> tuple[np.ndarray, NoiseState]:
    if config.kind == "none":
        return no_noise(values, config.calibration)
    if config.calibration == "sample_snr":
        return add_awgn_sample_snr(values, config.sample_snr_db, seed)
    if config.calibration == "esn0":
        return add_awgn_esn0(values, config.esn0_db, samples_per_symbol, seed)
    if config.calibration == "rml22_nominal":
        return add_awgn_rml22(values, config.rml22_snr_db, seed)
    raise ValueError(f"unsupported noise calibration: {config.calibration!r}")
