"""Config-driven communication signal and RF sample generation."""

from .channel import apply_channel, apply_fir_channel, apply_flat_fading, apply_identity, apply_selective_fading
from .config import (
    CANONICAL_MODULATIONS,
    EXTRA_MODULATIONS,
    ChannelConfig,
    FrameConfig,
    GeneratorConfig,
    ImpairmentConfig,
    ModulationConfig,
    NoiseConfig,
    SourceConfig,
    XODriftConfig,
)
from .generator import generate_sample
from .impairment import (
    apply_carrier_frequency_offset,
    apply_phase_offset,
    apply_sample_rate_offset,
    apply_timing_offset,
    sample_xo_trajectory,
)
from .modulation import constellation, generate_waveform, hard_demodulate, map_symbols
from .noise import add_awgn_esn0, add_awgn_rml22, add_awgn_sample_snr, add_noise, no_noise
from .pulse_shaping import rrc_taps
from .schema import Sample, complex_to_real, real_to_complex
from .seed import component_seeds, derive_seed
from .sources import bits_per_symbol, bits_to_indices, indices_to_bits
from .writer import DatasetRecord, DatasetSpec, HDF5DatasetWriter, SignalDataset, write_dataset, write_hdf5

__all__ = [
    "CANONICAL_MODULATIONS", "EXTRA_MODULATIONS", "ChannelConfig", "FrameConfig",
    "DatasetRecord", "DatasetSpec", "GeneratorConfig", "HDF5DatasetWriter", "ImpairmentConfig", "ModulationConfig",
    "NoiseConfig", "Sample", "SignalDataset", "SourceConfig", "XODriftConfig",
    "add_awgn_esn0", "add_awgn_rml22", "add_awgn_sample_snr", "add_noise",
    "apply_carrier_frequency_offset", "apply_channel", "apply_fir_channel",
    "apply_flat_fading", "apply_identity", "apply_phase_offset",
    "apply_sample_rate_offset", "apply_selective_fading", "apply_timing_offset",
    "bits_per_symbol", "bits_to_indices", "complex_to_real", "component_seeds", "constellation",
    "derive_seed", "generate_sample", "generate_waveform", "hard_demodulate",
    "indices_to_bits", "map_symbols", "no_noise",
    "real_to_complex", "rrc_taps", "sample_xo_trajectory", "write_dataset", "write_hdf5",
]
