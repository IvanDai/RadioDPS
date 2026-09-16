import numpy as np

from rsig import GenerationConfig, generate_sample
from rsig.channel import apply_channel
from rsig.data import SignalDataset, write_dataset
from rsig.generation import complex_to_real, real_to_complex
from rsig.signal import bits_to_indices, hard_demodulate, indices_to_bits


def test_mixed_modulations_round_trip_without_pickle(tmp_path):
    samples = [
        generate_sample(GenerationConfig(modulation=name, n_symbols=8, samples_per_symbol=4), seed=i)
        for i, name in enumerate(("BPSK", "QPSK", "16QAM", "32QAM"))
    ]
    path = tmp_path / "mixed.npz"
    write_dataset(samples, path, split_seed=17)
    dataset = SignalDataset(path)
    assert len(dataset) == 4
    for expected, actual in zip(samples, (dataset[i] for i in range(4))):
        np.testing.assert_array_equal(actual.bits, expected.bits)
        np.testing.assert_array_equal(actual.symbol_indices, expected.symbol_indices)
        np.testing.assert_array_equal(actual.h, expected.h)
        assert actual.channel_config == expected.channel_config
        assert actual.normalization == expected.normalization
    dataset.close()


def test_none_noise_is_exact_and_mapping_round_trips():
    sample = generate_sample(GenerationConfig(noise="none"), seed=7)
    np.testing.assert_array_equal(sample.y_rx, apply_channel(sample.x_tx, sample.h))
    assert sample.noise_variance == 0.0 and sample.noise_std == 0.0
    recovered = hard_demodulate(sample.symbols, sample.modulation)
    np.testing.assert_array_equal(recovered, sample.symbol_indices)
    np.testing.assert_array_equal(indices_to_bits(bits_to_indices(sample.bits, 2), 2), sample.bits)


def test_complex_conversion_and_persisted_split(tmp_path):
    x = np.array([1 + 2j, -3 + 0.5j], np.complex64)
    np.testing.assert_array_equal(real_to_complex(complex_to_real(x)), x)
    samples = [generate_sample(GenerationConfig(n_symbols=4), seed=i) for i in range(10)]
    path = tmp_path / "split.npz"
    write_dataset(samples, path, split_seed=3)
    first = SignalDataset(path)
    second = SignalDataset(path)
    for name in ("train", "validation", "test"):
        np.testing.assert_array_equal(first.splits[name], second.splits[name])
        split_view = first.for_split(name)
        assert len(split_view) == len(first.splits[name])
        split_view.close()
    first.close()
    second.close()
