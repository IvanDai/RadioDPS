from __future__ import annotations

import numpy as np


def modules():
    try:
        from gnuradio import analog, blocks, channels, digital, filter, gr
        from gnuradio.filter import firdes
    except ImportError as error:
        raise RuntimeError(
            "Signal generation requires GNU Radio. Install a supported GNU Radio "
            "release from conda-forge before calling rsig generation functions."
        ) from error
    return analog, blocks, channels, digital, filter, gr, firdes


def run_chain(values, chain, *, input_kind="complex", output_kind="complex"):
    _, blocks, _, _, _, gr, _ = modules()
    sources = {
        "byte": lambda: blocks.vector_source_b(np.asarray(values, np.uint8), False),
        "float": lambda: blocks.vector_source_f(np.asarray(values, np.float32), False),
        "complex": lambda: blocks.vector_source_c(np.asarray(values, np.complex64), False),
    }
    sinks = {"float": blocks.vector_sink_f, "complex": blocks.vector_sink_c}
    top = gr.top_block()
    sink = sinks[output_kind]()
    top.connect(sources[input_kind](), *chain, sink)
    top.run()
    dtype = np.float32 if output_kind == "float" else np.complex64
    return np.asarray(sink.data(), dtype=dtype)
