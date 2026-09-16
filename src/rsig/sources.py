from __future__ import annotations

from pathlib import Path
import numpy as np

from ._flowgraph import modules, run_chain
from .config import SourceConfig
from .schema import SourceData


ANALOG_MODULATIONS = {"AM-SSB-WC", "AM-SSB-SC", "AM-DSB-WC", "AM-DSB-SC", "FM"}
DIGITAL_MODULATION_ORDERS = {
    "OOK": 2,
    "4ASK": 4,
    "8ASK": 8,
    "BPSK": 2,
    "QPSK": 4,
    "8PSK": 8,
    "16PSK": 16,
    "32PSK": 32,
    "16APSK": 16,
    "32APSK": 32,
    "64APSK": 64,
    "128APSK": 128,
    "16QAM": 16,
    "32QAM": 32,
    "64QAM": 64,
    "128QAM": 128,
    "256QAM": 256,
    "GMSK": 2,
    "OQPSK": 4,
    "GFSK": 2,
    "CPFSK": 2,
}
DIGITAL_MODULATION_ALIASES = {
    "PAM4": "4ASK",
    "QAM16": "16QAM",
    "QAM64": "64QAM",
}


def is_analog(modulation: str) -> bool:
    return modulation in ANALOG_MODULATIONS


def bits_per_symbol(modulation: str) -> int:
    name = str(modulation).upper()
    name = DIGITAL_MODULATION_ALIASES.get(name, name)
    try:
        order = DIGITAL_MODULATION_ORDERS[name]
    except KeyError as error:
        raise ValueError(f"{modulation!r} is not a supported digital modulation") from error
    return int(np.log2(order))


def bits_to_indices(bits: np.ndarray, width: int) -> np.ndarray:
    values = np.asarray(bits, dtype=np.uint8)
    if values.ndim != 1 or len(values) % width or not np.all((values == 0) | (values == 1)):
        raise ValueError("bits must be binary and divisible by bits_per_symbol")
    weights = 1 << np.arange(width - 1, -1, -1)
    return values.reshape(-1, width).dot(weights).astype(np.uint16)


def indices_to_bits(indices: np.ndarray, width: int) -> np.ndarray:
    values = np.asarray(indices, dtype=np.uint64)
    shifts = np.arange(width - 1, -1, -1)
    return ((values[:, None] >> shifts) & 1).astype(np.uint8).reshape(-1)


def _synthetic_audio(length: int, bandwidth: float, rng: np.random.Generator) -> np.ndarray:
    _, _, _, _, gr_filter, _, firdes = modules()
    padding = 256
    white = rng.standard_normal(length + padding).astype(np.float32)
    taps = firdes.low_pass(1.0, 2.0, bandwidth, min(0.1, bandwidth / 4))
    filtered = run_chain(white, [gr_filter.fir_filter_fff(1, taps)], input_kind="float", output_kind="float")
    message = filtered[padding:padding + length]
    message -= np.mean(message)
    peak = float(np.max(np.abs(message)))
    if peak <= np.finfo(np.float32).tiny:
        raise RuntimeError("synthetic audio source has zero amplitude")
    return (message / peak).astype(np.float32)


def _wav_audio(config: SourceConfig, length: int) -> np.ndarray:
    _, blocks, _, _, _, gr, _ = modules()
    path = Path(config.wav_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(path)
    top = gr.top_block()
    source = blocks.wavfile_source(str(path), False)
    skip = blocks.skiphead(gr.sizeof_float, config.wav_start_sample)
    head = blocks.head(gr.sizeof_float, length)
    sink = blocks.vector_sink_f()
    top.connect((source, config.wav_channel), skip, head, sink)
    top.run()
    values = np.asarray(sink.data(), dtype=np.float32)
    if len(values) != length:
        raise ValueError("WAV source does not contain enough samples")
    return values


def generate_source(modulation, config: SourceConfig, symbol_count, message_length, rng, *, message_bandwidth=0.2) -> SourceData:
    if is_analog(modulation):
        kind = "synthetic_audio" if config.kind == "auto" else config.kind
        if kind == "wav":
            message = _wav_audio(config, message_length)
        elif kind == "synthetic_audio":
            message = _synthetic_audio(message_length, message_bandwidth, rng)
        else:
            raise ValueError("analog modulation requires WAV or synthetic audio")
        return SourceData(None, None, message, {"kind": kind, "message_bandwidth": message_bandwidth})
    if config.kind not in {"auto", "random"}:
        raise ValueError("digital modulation requires a random bit source")
    width = bits_per_symbol(modulation)
    bits = rng.integers(0, 2, symbol_count * width, dtype=np.uint8)
    indices = bits_to_indices(bits, width)
    return SourceData(bits, indices, None, {"kind": "random", "bits_per_symbol": width})
