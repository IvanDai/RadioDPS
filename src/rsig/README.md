# rsig

`rsig` is a config-driven communication-signal and RF simulation library. GNU Radio supplies the modulation, filtering, channel, oscillator, and noise blocks where equivalent blocks exist. The public API is organized by function; GNU Radio is an implementation detail.

`rsig` is independent of `rdps`. It does not train models, run reconstruction, choose dataset classes, or generate a dataset on import.

## Layout

```text
src/rsig/
  config.py          # all user-facing configuration
  generator.py       # readable one-sample pipeline
  sources.py         # random bits and analog messages
  modulation.py      # constellations, mapping, modulation, demodulation
  pulse_shaping.py   # rectangular and RRC transmit shaping
  channel.py         # identity, flat, FIR, selective fading
  impairment.py      # timing, SRO, CFO, phase, coupled XO drift
  noise.py           # none and AWGN calibrations
  schema.py          # returned sample objects and IQ conversion
  seed.py            # traversal-independent seed derivation
  writer.py          # fixed-grid HDF5 writer and reader
```

## Generate one sample

```python
from rsig import (
    ChannelConfig,
    FrameConfig,
    GeneratorConfig,
    ImpairmentConfig,
    ModulationConfig,
    NoiseConfig,
    generate_sample,
)

config = GeneratorConfig(
    sample_rate_hz=200_000,
    modulation=ModulationConfig(name="16QAM", samples_per_symbol=8),
    frame=FrameConfig(length=1024, guard_samples=256),
    channel=ChannelConfig(kind="flat", flat_coefficient=0.8 + 0.2j),
    impairments=ImpairmentConfig(
        timing_offset_samples=0.25,
        carrier_frequency_offset_hz=30,
        phase_offset_rad="random",
    ),
    noise=NoiseConfig(kind="awgn", calibration="esn0", esn0_db=12),
)

sample = generate_sample(config, seed=0)
```

The pipeline in `generator.py` is fixed and explicit:

```text
source -> modulation/transmit shaping -> channel
       -> timing offset -> sample-rate offset -> carrier-frequency offset
       -> phase offset -> noise -> aligned crop
```

Every stage can be disabled: identity channel, `None` impairment values, and `NoiseConfig(kind="none")` are exact bypasses. With all three choices, `y_rx == x_tx`.

## Modulations

The canonical catalog has 24 entries:

```text
OOK
4ASK, 8ASK
BPSK, QPSK, 8PSK, 16PSK, 32PSK
16APSK, 32APSK, 64APSK, 128APSK
16QAM, 32QAM, 64QAM, 128QAM, 256QAM
AM-SSB-WC, AM-SSB-SC, AM-DSB-WC, AM-DSB-SC
FM, GMSK, OQPSK
```

RML22's GFSK and CPFSK remain available as extra names. Its legacy aliases `PAM4`, `QAM16`, `QAM64`, `WBFM`, and `AM-DSB` are accepted and canonicalized.

Digital source bits are grouped MSB-first. The integer represented by each group is the `source_symbol_index`. ASK and PSK use binary-reflected Gray ordering around their natural amplitude/phase traversal. Rectangular QAM uses independent Gray ordering on I and Q. Cross QAM and APSK use a stable binary-reflected Gray permutation of their documented traversal. `indices_to_bits`, `bits_to_indices`, `map_symbols`, and `hard_demodulate` use the same convention, so BER and SER labels are reversible.

Linear digital modulation has an ideal rectangular-pulse reference and an RRC-shaped transmit waveform. OQPSK delays Q by half a symbol. GMSK/GFSK use Gaussian shaping, CPFSK is continuous phase, and AM/FM use a band-limited real message. Analog records use `None` for bits, indices, and symbols rather than fabricated empty arrays.

## Signal definitions

- `x_ref`: ideal modulation reference before practical transmit shaping.
- `x_tx`: transmitted waveform after transmit shaping, before the channel or receiver-side effects.
- `y_rx`: `x_tx` after channel, timing/SRO, CFO/phase, and noise.

The generator creates front and rear guards, processes the full guarded waveform, and crops `x_ref`, `x_tx`, and `y_rx` at the same indices without periodic wrap. All three returned arrays are `complex64` and have `FrameConfig.length` samples.

Constellations have deterministic unit average energy. RRC taps are deterministically normalized for unit steady-state sample power for unit-energy independent symbols. The generator never rescales an individual output frame. The normalization method, configured scales, and measured returned powers are stored with every sample.

## Channel and impairments

`ChannelConfig.kind` is one of:

- `identity`: exact bypass, taps `[1+0j]`.
- `flat`: configured known complex scalar.
- `fir`: configured complex taps, optionally unit-energy normalized.
- `selective_fading`: GNU Radio's RML22-style Rayleigh/Rician selective fading model with fractional sample delays, path magnitudes, Doppler, sinusoid count, LOS/K factor, and realization seed.

The selective-fading block is time varying, so it has no single per-frame FIR tap vector; its complete model parameters and seed are stored and `channel.taps` is `None`. Identity, flat, and FIR channels store their effective complex taps.

Fixed timing/SRO/CFO/phase effects use GNU Radio operations. `phase_offset_rad="random"` samples a reproducible uniform phase. `XODriftConfig` enables RML22's bounded oscillator random walk: one trajectory drives both time-varying SRO and CFO. That coupled time-varying operation has no equivalent GNU Radio block and is the only receiver-effect path implemented numerically.

## Noise definitions

`NoiseConfig(kind="none")` records zero complex variance. AWGN supports three deliberately separate calibrations:

- `sample_snr`: `variance = mean(|clean_rx|^2) / 10^(sample_snr_db/10)`.
- `esn0`: `Es = mean(|clean_rx|^2) * samples_per_symbol`, then `variance = Es / 10^(esn0_db/10)`.
- `rml22_nominal`: reproduces RML22's GNU Radio noise amplitude `10^(-rml22_snr_db/20)`. Since that amplitude is the standard deviation of each real component, complex variance is twice its square.

Each record stores the calibration name, relevant dB value, reference power/energy, complex variance, and real-component standard deviation.

## Sample schema

`generate_sample` returns a `Sample` with:

```text
x_ref, x_tx, y_rx                    complex64 [frame_length]
modulation                           canonical string
source_bits                          uint8 [variable] or None
source_symbol_indices                uint16 [variable] or None
source_symbols                       complex64 [variable] or None
source_message                       float32 [variable] or None
channel                              kind, effective taps or None, parameters
impairments                          realized values, optional XO trajectory
noise                                calibration, level, variance, std
sample_seed                          integer
frame                                guard and crop metadata
normalization                        deterministic scales and measured powers
config                               complete configuration snapshot
```

`complex_to_real` converts complex arrays to `float32[..., 2]`; `real_to_complex` performs the inverse for later reconstruction code.

## Dataset recipes and storage

A dataset recipe is an ordinary project-level Python file, for example `my_dataset.py`. It constructs a `DatasetSpec`, chooses configurations and seeds, calls `generate_sample`, and writes each result at its `(modulation, condition, example)` coordinate. Generation policy does not live in the writer:

```python
import numpy as np

from rsig import DatasetSpec, HDF5DatasetWriter, generate_sample

spec = DatasetSpec(
    modulations=("BPSK", "QPSK", "16QAM", "32QAM"),
    noise_levels_db=(np.nan,),
    examples_per_condition=1000,
    frame_length=1024,
    max_channel_taps=8,
    generation_mode="known-channel benchmark",
    channel_type="fir",
    channel_normalization="unit_energy",
    noise_calibration="sample_snr",
    root_seed=0,
)

with HDF5DatasetWriter("my_dataset.h5", spec) as writer:
    for modulation_index, modulation in enumerate(spec.modulations):
        for example_index in range(spec.examples_per_condition):
            config = config_for(modulation)
            sample = generate_sample(config, seed=seed_for(modulation_index, 0, example_index))
            writer.append(sample, condition_index=0, example_index=example_index)
```

The HDF5 hierarchy exactly follows the project-level dataset specification:

```text
dataset.h5
|-- modulation                  UTF-8   [M]
|-- noise_level_db              float32 [S]
|-- x_tx                        float32 [M,S,E,2,T]
|-- h                           float32 [M,S,E,2,L]
|-- y_rx                        float32 [M,S,E,2,T]
|-- parameters/
|   |-- samples_per_symbol      uint16  [M,S,E]
|   |-- rrc_alpha               float32 [M,S,E]
|   |-- timing_offset           float32 [M,S,E]
|   |-- symbol_rate_offset      float32 [M,S,E]
|   |-- phase_offset            float32 [M,S,E]
|   |-- carrier_offset          float32 [M,S,E]
|   |-- delay_spread            float32 [M,S,E]
|   |-- channel_path_count      uint16  [M,S,E]
|   |-- channel_length          uint16  [M,S,E]
|   |-- noise_variance          float32 [M,S,E]
|   |-- noise_std               float32 [M,S,E]
|   |-- x_tx_scale              float32 [M,S,E]
|   `-- sample_seed             uint64  [M,S,E]
`-- metadata                    scalar UTF-8 JSON
```

IQ is stored as real channels `[I,Q]`. Channel taps shorter than `L` are zero-padded and their effective length is stored in `parameters/channel_length`. `HDF5DatasetWriter.append()` rejects samples without explicit taps or taps longer than `L`; identity, flat, and fixed FIR known channels can therefore share the same representation. BPSK, QPSK, 16QAM, and 32QAM differ only along the `M` axis and can be stored in one file.

Disabled impairments are written as zero. With noise disabled, `noise_level_db[0]` is IEEE `NaN`, and per-sample `noise_variance` and `noise_std` are zero. The metadata JSON uses `null` when it repeats that disabled noise-axis value, keeping the JSON standards-compliant. Global invariant settings are stored once in `metadata`; no per-sample JSON is written.

In accordance with the root project specification, simulation files do not store splits, source bits/symbols/messages, Python objects, pickle data, or per-sample configuration JSON. Dataset splitting belongs to the loader or training experiment and is derived later from the three grid coordinates. `SignalDataset` reads records by either a flat integer or an explicit `(modulation_index, condition_index, example_index)` coordinate; it does not create or persist a split.

## Runtime

Install NumPy, SciPy, and h5py through the Python project environment. Signal generation additionally requires a GNU Radio runtime. GNU Radio is normally installed with conda-forge rather than PyPI, so it is intentionally not listed as an invalid pip dependency. Imports are lazy: configuration, schemas, and stored datasets remain usable without GNU Radio; waveform generation raises a clear runtime error until GNU Radio is installed.
