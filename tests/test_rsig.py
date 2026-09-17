import numpy as np

from rsig import (
    ChannelConfig,
    DatasetSpec,
    FrameConfig,
    GeneratorConfig,
    ImpairmentConfig,
    ModulationConfig,
    NoiseConfig,
    SignalDataset,
    complex_to_real,
    constellation,
    generate_sample,
    hard_demodulate,
    real_to_complex,
    write_dataset,
)
from rsig.schema import Waveform
from rsig.sources import bits_to_indices


MODULATIONS = ("BPSK", "QPSK", "16QAM", "32QAM")
FRAME_LENGTH = 64


def _stub_modulate(source, config, _sample_rate_hz, output_length, _margin_symbols):
    symbols = constellation(config.name)[source.symbol_indices]
    waveform = np.resize(symbols, output_length).astype(np.complex64)
    return Waveform(
        reference=waveform,
        transmitted=waveform.copy(),
        source=source,
        symbols=symbols,
        samples_per_symbol=config.samples_per_symbol,
        metadata={"family": "linear", "transmit_scale": 1.0},
    )


def _without_gnuradio(monkeypatch):
    monkeypatch.setattr("rsig.generator.modulate", _stub_modulate)


def _config(modulation="QPSK", *, channel=None):
    return GeneratorConfig(
        modulation=ModulationConfig(
            name=modulation,
            samples_per_symbol=4,
            rrc_span_symbols=5,
        ),
        frame=FrameConfig(length=FRAME_LENGTH, guard_samples=16),
        channel=channel or ChannelConfig(kind="identity"),
        impairments=ImpairmentConfig(),
        noise=NoiseConfig(kind="none"),
    )


def _spec(*, modulations=MODULATIONS, examples_per_condition=1):
    return DatasetSpec(
        modulations=tuple(modulations),
        noise_levels_db=(np.nan,),
        examples_per_condition=examples_per_condition,
        frame_length=FRAME_LENGTH,
        max_channel_taps=1,
        generation_mode="rsig unit test",
        channel_type="identity",
        channel_normalization="none",
        noise_calibration="none",
        root_seed=17,
    )


def test_mixed_modulations_round_trip_without_pickle(tmp_path, monkeypatch):
    _without_gnuradio(monkeypatch)
    samples = [
        generate_sample(_config(name), seed=index)
        for index, name in enumerate(MODULATIONS)
    ]
    path = tmp_path / "mixed.h5"
    records = [(sample, 0, 0) for sample in samples]
    write_dataset(path, _spec(), records)

    with SignalDataset(path) as dataset:
        assert len(dataset) == len(MODULATIONS)
        assert dataset.grid_shape == (len(MODULATIONS), 1, 1)
        assert dataset.metadata["version"] == "rdps-rsig-hdf5-1"
        for modulation_index, (expected, actual) in enumerate(
            zip(samples, (dataset[index, 0, 0] for index in range(len(MODULATIONS))))
        ):
            np.testing.assert_array_equal(actual.x_tx, expected.x_tx)
            np.testing.assert_array_equal(actual.y_rx, expected.y_rx)
            np.testing.assert_array_equal(actual.h, expected.channel.taps)
            assert actual.modulation == expected.modulation
            assert actual.modulation_index == modulation_index
            assert actual.condition_index == 0 and actual.example_index == 0
            assert actual.parameters["sample_seed"] == expected.sample_seed
            assert actual.parameters["noise_variance"] == 0.0


def test_none_noise_is_exact_and_mapping_round_trips(monkeypatch):
    _without_gnuradio(monkeypatch)
    sample = generate_sample(_config(), seed=7)
    np.testing.assert_array_equal(sample.y_rx, sample.x_tx)
    assert sample.noise.variance == 0.0 and sample.noise.component_std == 0.0
    recovered = hard_demodulate(sample.source_symbols, sample.modulation)
    np.testing.assert_array_equal(recovered, sample.source_symbol_indices)
    np.testing.assert_array_equal(
        bits_to_indices(sample.source_bits, 2),
        sample.source_symbol_indices,
    )


def test_complex_conversion_and_coordinate_reads_are_deterministic(tmp_path, monkeypatch):
    _without_gnuradio(monkeypatch)
    values = np.array([1 + 2j, -3 + 0.5j], np.complex64)
    np.testing.assert_array_equal(real_to_complex(complex_to_real(values)), values)

    samples = [generate_sample(_config("QPSK"), seed=index) for index in range(2)]
    path = tmp_path / "coordinates.h5"
    write_dataset(
        path,
        _spec(modulations=("QPSK",), examples_per_condition=2),
        [(sample, 0, index) for index, sample in enumerate(samples)],
    )

    with SignalDataset(path) as first, SignalDataset(path) as second:
        assert first.metadata == second.metadata
        for index in range(2):
            by_flat_index = first[index]
            by_coordinate = second[0, 0, index]
            np.testing.assert_array_equal(by_flat_index.x_tx, by_coordinate.x_tx)
            np.testing.assert_array_equal(by_flat_index.h, by_coordinate.h)
            np.testing.assert_array_equal(by_flat_index.y_rx, by_coordinate.y_rx)
            assert by_flat_index.parameters == by_coordinate.parameters
