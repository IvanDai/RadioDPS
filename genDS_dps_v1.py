#!/usr/bin/env python3
"""Generate and validate the first known-channel DPS dataset."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

import h5py
import numpy as np

from rsig import (
    ChannelConfig,
    DatasetSpec,
    FrameConfig,
    GeneratorConfig,
    HDF5DatasetWriter,
    ImpairmentConfig,
    ModulationConfig,
    NoiseConfig,
    component_seeds,
    constellation,
    derive_seed,
    generate_sample,
    real_to_complex,
    rrc_taps,
)


MODULATIONS = ("BPSK", "QPSK", "16QAM", "32QAM")
CONDITION_INDEX = 0
PARAMETER_DTYPES = {
    "samples_per_symbol": np.dtype("uint16"),
    "rrc_alpha": np.dtype("float32"),
    "timing_offset": np.dtype("float32"),
    "symbol_rate_offset": np.dtype("float32"),
    "phase_offset": np.dtype("float32"),
    "carrier_offset": np.dtype("float32"),
    "delay_spread": np.dtype("float32"),
    "channel_path_count": np.dtype("uint16"),
    "channel_length": np.dtype("uint16"),
    "noise_variance": np.dtype("float32"),
    "noise_std": np.dtype("float32"),
    "x_tx_scale": np.dtype("float32"),
    "sample_seed": np.dtype("uint64"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("datasets/dps_v1/dps_v1.h5"))
    parser.add_argument("--examples-per-modulation", type=int, default=16_384)
    parser.add_argument("--root-seed", type=int, default=233)
    parser.add_argument("--frame-length", type=int, default=1024)
    parser.add_argument("--samples-per-symbol", type=int, default=8)
    parser.add_argument("--rrc-rolloff", type=float, default=0.35)
    parser.add_argument("--rrc-span-symbols", type=int, default=11)
    parser.add_argument("--guard-samples", type=int, default=256)
    parser.add_argument("--channel-length", type=int, default=8)
    parser.add_argument("--channel-decay-samples", type=float, default=2.0)
    parser.add_argument("--validation-random-checks", type=int, default=64)
    parser.add_argument("--plots-per-modulation", type=int, default=3)
    parser.add_argument("--progress-every", type=int, default=128)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    positive_integer_names = (
        "examples_per_modulation", "frame_length", "samples_per_symbol",
        "rrc_span_symbols", "channel_length", "validation_random_checks",
        "plots_per_modulation", "progress_every",
    )
    for name in positive_integer_names:
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.root_seed < 0:
        parser.error("--root-seed must be non-negative")
    if args.guard_samples < args.channel_length - 1:
        parser.error("--guard-samples must be at least channel_length - 1")
    if not 0.0 <= args.rrc_rolloff <= 1.0:
        parser.error("--rrc-rolloff must be in [0, 1]")
    if args.channel_decay_samples <= 0:
        parser.error("--channel-decay-samples must be positive")
    return args


def sample_seed(root_seed: int, modulation_index: int, example_index: int) -> int:
    return derive_seed(root_seed, modulation_index, CONDITION_INDEX, example_index)


def sample_fir_taps(seed: int, length: int, decay_samples: float) -> np.ndarray:
    rng = np.random.default_rng(component_seeds(seed).channel)
    power_profile = np.exp(-np.arange(length, dtype=np.float64) / decay_samples)
    real = rng.standard_normal(length)
    imag = rng.standard_normal(length)
    taps = (real + 1j * imag) * np.sqrt(power_profile / 2.0)
    energy = float(np.sum(np.abs(taps) ** 2))
    if not np.isfinite(energy) or energy <= np.finfo(np.float64).tiny:
        raise RuntimeError("sampled FIR channel has invalid energy")
    return (taps / np.sqrt(energy)).astype(np.complex64)


def build_sample(args: argparse.Namespace, modulation_index: int, example_index: int):
    seed = sample_seed(args.root_seed, modulation_index, example_index)
    taps = sample_fir_taps(seed, args.channel_length, args.channel_decay_samples)
    config = GeneratorConfig(
        sample_rate_hz=1.0,
        modulation=ModulationConfig(
            name=MODULATIONS[modulation_index],
            samples_per_symbol=args.samples_per_symbol,
            rolloff=args.rrc_rolloff,
            rrc_span_symbols=args.rrc_span_symbols,
        ),
        frame=FrameConfig(length=args.frame_length, guard_samples=args.guard_samples),
        channel=ChannelConfig(kind="fir", taps=tuple(taps), normalize_taps=True),
        impairments=ImpairmentConfig(),
        noise=NoiseConfig(kind="none", calibration="sample_snr"),
    )
    return generate_sample(config, seed=seed)


def dataset_spec(args: argparse.Namespace) -> DatasetSpec:
    return DatasetSpec(
        modulations=MODULATIONS,
        noise_levels_db=(np.nan,),
        examples_per_condition=args.examples_per_modulation,
        frame_length=args.frame_length,
        max_channel_taps=args.channel_length,
        generation_mode="known-channel DPS v1",
        channel_type="fir",
        channel_normalization="unit_energy",
        noise_calibration="none",
        root_seed=args.root_seed,
        boundary="causal_zero",
        extra_metadata={
            "dataset_name": "dps_v1",
            "sample_rate_hz": 1.0,
            "samples_per_symbol": args.samples_per_symbol,
            "rrc_rolloff": args.rrc_rolloff,
            "rrc_span_symbols": args.rrc_span_symbols,
            "guard_samples": args.guard_samples,
            "noise": "none",
            "enabled_impairments": [],
            "channel_distribution": "zero-mean circular complex Gaussian",
            "channel_power_delay_profile": "exp(-tap_index / channel_decay_samples)",
            "channel_decay_samples": args.channel_decay_samples,
            "sample_seed_coordinates": ["root_seed", "modulation_index", "condition_index", "example_index"],
        },
    )


def generate_dataset(args: argparse.Namespace, output: Path) -> float:
    total = len(MODULATIONS) * args.examples_per_modulation
    completed = 0
    started = time.monotonic()
    with HDF5DatasetWriter(output, dataset_spec(args)) as writer:
        for modulation_index, modulation in enumerate(MODULATIONS):
            for example_index in range(args.examples_per_modulation):
                sample = build_sample(args, modulation_index, example_index)
                writer.append(
                    sample,
                    modulation_index=modulation_index,
                    condition_index=CONDITION_INDEX,
                    example_index=example_index,
                )
                completed += 1
                if completed == total or completed % args.progress_every == 0:
                    elapsed = time.monotonic() - started
                    rate = completed / elapsed if elapsed else 0.0
                    remaining = (total - completed) / rate if rate else float("inf")
                    print(
                        f"[{completed:>{len(str(total))}}/{total}] {100 * completed / total:6.2f}% "
                        f"{modulation:<5} {rate:7.2f} samples/s ETA {remaining:8.1f}s",
                        flush=True,
                    )
    return time.monotonic() - started


def _read_complex(dataset, coordinate) -> np.ndarray:
    return real_to_complex(np.moveaxis(dataset[coordinate], 0, -1))


def _array_digest(values: np.ndarray) -> str:
    return hashlib.blake2b(np.ascontiguousarray(values).view(np.uint8), digest_size=16).hexdigest()


def _matched_symbol_samples(x_tx: np.ndarray, args: argparse.Namespace, modulation: str) -> np.ndarray:
    taps = rrc_taps(args.samples_per_symbol, args.rrc_rolloff, args.rrc_span_symbols)
    matched = np.convolve(x_tx, taps, mode="same") / args.samples_per_symbol
    expected = constellation(modulation)
    best_samples = None
    best_error = float("inf")
    edge_symbols = args.rrc_span_symbols
    for phase in range(args.samples_per_symbol):
        candidates = matched[phase::args.samples_per_symbol]
        if len(candidates) > 2 * edge_symbols:
            candidates = candidates[edge_symbols:-edge_symbols]
        error = float(np.mean(np.min(np.abs(candidates[:, None] - expected[None, :]) ** 2, axis=1)))
        if error < best_error:
            best_error = error
            best_samples = candidates
    return np.asarray(best_samples, dtype=np.complex64)


def _record_check(checks: dict, name: str, passed: bool, **details):
    checks[name] = {"passed": bool(passed), **details}


def validate_dataset(args: argparse.Namespace, dataset_path: Path, validation_dir: Path, generation_seconds: float) -> dict:
    rng = np.random.default_rng(derive_seed(args.root_seed, 999))
    checks: dict[str, dict] = {}
    power_by_modulation: dict[str, list[float]] = {name: [] for name in MODULATIONS}
    channel_digests: set[str] = set()
    waveform_digests: set[str] = set()
    max_energy_error = 0.0
    finite = True

    with h5py.File(dataset_path, "r") as handle:
        expected_root = {"modulation", "noise_level_db", "x_tx", "h", "y_rx", "parameters", "metadata"}
        actual_root = set(handle.keys())
        expected_waveform_shape = (4, 1, args.examples_per_modulation, 2, args.frame_length)
        expected_channel_shape = (4, 1, args.examples_per_modulation, 2, args.channel_length)
        expected_parameter_shape = (4, 1, args.examples_per_modulation)
        schema_ok = actual_root == expected_root
        schema_ok &= handle["x_tx"].shape == expected_waveform_shape and handle["x_tx"].dtype == np.dtype("float32")
        schema_ok &= handle["y_rx"].shape == expected_waveform_shape and handle["y_rx"].dtype == np.dtype("float32")
        schema_ok &= handle["h"].shape == expected_channel_shape and handle["h"].dtype == np.dtype("float32")
        schema_ok &= handle["noise_level_db"].shape == (1,) and handle["noise_level_db"].dtype == np.dtype("float32")
        schema_ok &= handle["modulation"].shape == (4,)
        schema_ok &= set(handle["parameters"].keys()) == set(PARAMETER_DTYPES)
        for name, dtype in PARAMETER_DTYPES.items():
            schema_ok &= handle[f"parameters/{name}"].shape == expected_parameter_shape
            schema_ok &= handle[f"parameters/{name}"].dtype == dtype
        _record_check(checks, "schema", schema_ok, root_entries=sorted(actual_root))

        stored_modulations = tuple(handle["modulation"].asstr()[:])
        counts = {name: args.examples_per_modulation for name in stored_modulations}
        _record_check(
            checks,
            "balanced_modulation_counts",
            stored_modulations == MODULATIONS and len(set(counts.values())) == 1,
            counts=counts,
        )
        _record_check(
            checks,
            "noise_disabled_axis",
            bool(np.isnan(handle["noise_level_db"][0])),
            stored_value=None,
        )

        for modulation_index, modulation in enumerate(MODULATIONS):
            for example_index in range(args.examples_per_modulation):
                coordinate = (modulation_index, CONDITION_INDEX, example_index)
                x_tx = _read_complex(handle["x_tx"], coordinate)
                h = _read_complex(handle["h"], coordinate)
                y_rx = _read_complex(handle["y_rx"], coordinate)
                finite &= bool(np.isfinite(x_tx).all() and np.isfinite(h).all() and np.isfinite(y_rx).all())
                energy_error = abs(float(np.sum(np.abs(h) ** 2)) - 1.0)
                max_energy_error = max(max_energy_error, energy_error)
                power_by_modulation[modulation].append(float(np.mean(np.abs(x_tx) ** 2)))
                channel_digests.add(_array_digest(h))
                waveform_digests.add(_array_digest(x_tx))

        parameter_finite = all(np.isfinite(dataset[:]).all() for dataset in handle["parameters"].values())
        finite &= parameter_finite
        _record_check(checks, "finite_arrays", finite)
        _record_check(checks, "unit_energy_channels", max_energy_error <= 2e-5, max_absolute_error=max_energy_error)

        power_statistics = {}
        power_ok = True
        for modulation, values in power_by_modulation.items():
            array = np.asarray(values)
            statistics = {
                "min": float(array.min()),
                "mean": float(array.mean()),
                "std": float(array.std()),
                "max": float(array.max()),
            }
            power_statistics[modulation] = statistics
            power_ok &= 0.1 <= statistics["min"] and statistics["max"] <= 10.0
            power_ok &= 0.5 <= statistics["mean"] <= 2.0
        _record_check(checks, "x_tx_power", power_ok, by_modulation=power_statistics)

        total = len(MODULATIONS) * args.examples_per_modulation
        uniqueness_ok = len(channel_digests) == total and len(waveform_digests) == total
        _record_check(
            checks,
            "no_abnormal_duplicates",
            uniqueness_ok,
            unique_channels=len(channel_digests),
            unique_waveforms=len(waveform_digests),
            expected=total,
        )

        check_count = min(args.validation_random_checks, total)
        flat_indices = rng.choice(total, size=check_count, replace=False)
        convolution_errors = []
        reproducibility_errors = []
        checked_coordinates = []
        for flat_index in flat_indices:
            modulation_index, example_index = divmod(int(flat_index), args.examples_per_modulation)
            coordinate = (modulation_index, CONDITION_INDEX, example_index)
            checked_coordinates.append(list(coordinate))
            x_tx = _read_complex(handle["x_tx"], coordinate)
            h = _read_complex(handle["h"], coordinate)
            y_rx = _read_complex(handle["y_rx"], coordinate)
            reconstructed = np.convolve(x_tx, h, mode="full")[:len(x_tx)]
            convolution_errors.append(float(np.max(np.abs(reconstructed - y_rx))))

            repeated = build_sample(args, modulation_index, example_index)
            reproducibility_errors.append(float(max(
                np.max(np.abs(repeated.x_tx - x_tx)),
                np.max(np.abs(repeated.channel.taps - h)),
                np.max(np.abs(repeated.y_rx - y_rx)),
            )))
        max_convolution_error = max(convolution_errors, default=0.0)
        max_reproducibility_error = max(reproducibility_errors, default=0.0)
        _record_check(
            checks,
            "causal_zero_forward_model",
            max_convolution_error <= 2e-5,
            max_absolute_error=max_convolution_error,
            coordinates=checked_coordinates,
        )
        _record_check(
            checks,
            "seed_reproducibility",
            max_reproducibility_error == 0.0,
            max_absolute_error=max_reproducibility_error,
            coordinates=checked_coordinates,
        )

        disabled_parameters = (
            "timing_offset", "symbol_rate_offset", "phase_offset", "carrier_offset",
            "noise_variance", "noise_std",
        )
        disabled_ok = all(np.count_nonzero(handle[f"parameters/{name}"][:]) == 0 for name in disabled_parameters)
        _record_check(checks, "disabled_effects_are_zero", disabled_ok, fields=list(disabled_parameters))

    plot_files = create_validation_plots(args, dataset_path, validation_dir, rng)
    metrics = {
        "dataset": str(args.output),
        "validated_file": str(args.output.resolve()),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generation_seconds": generation_seconds,
        "configuration": {
            "modulations": list(MODULATIONS),
            "frame_length": args.frame_length,
            "samples_per_symbol": args.samples_per_symbol,
            "rrc_rolloff": args.rrc_rolloff,
            "channel_type": "fir",
            "channel_length": args.channel_length,
            "channel_decay_samples": args.channel_decay_samples,
            "examples_per_modulation": args.examples_per_modulation,
            "root_seed": args.root_seed,
            "noise": "none",
        },
        "checks": checks,
        "plots": plot_files,
    }
    metrics["passed"] = all(check["passed"] for check in checks.values())
    return metrics


def create_validation_plots(
    args: argparse.Namespace,
    dataset_path: Path,
    validation_dir: Path,
    rng: np.random.Generator,
) -> list[str]:
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "rsig-matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    validation_dir.mkdir(parents=True, exist_ok=True)
    plot_files = []
    with h5py.File(dataset_path, "r") as handle:
        count = min(args.plots_per_modulation, args.examples_per_modulation)
        for modulation_index, modulation in enumerate(MODULATIONS):
            examples = np.sort(rng.choice(args.examples_per_modulation, size=count, replace=False))
            figure, axes = plt.subplots(count, 4, figsize=(18, 4.2 * count), squeeze=False)
            for row, example_index in enumerate(examples):
                coordinate = (modulation_index, CONDITION_INDEX, int(example_index))
                x_tx = _read_complex(handle["x_tx"], coordinate)
                h = _read_complex(handle["h"], coordinate)
                y_rx = _read_complex(handle["y_rx"], coordinate)
                shown = min(256, len(x_tx))
                axes[row, 0].plot(x_tx.real[:shown], label="x_tx I", linewidth=0.9)
                axes[row, 0].plot(x_tx.imag[:shown], label="x_tx Q", linewidth=0.9)
                axes[row, 0].plot(y_rx.real[:shown], label="y_rx I", linewidth=0.7, alpha=0.7)
                axes[row, 0].plot(y_rx.imag[:shown], label="y_rx Q", linewidth=0.7, alpha=0.7)
                axes[row, 0].set_title(f"example {example_index}: time domain")
                axes[row, 0].legend(fontsize=7, ncol=2)

                frequency = np.fft.fftshift(np.fft.fftfreq(len(x_tx)))
                tx_spectrum = 20 * np.log10(np.maximum(np.abs(np.fft.fftshift(np.fft.fft(x_tx))), 1e-8))
                rx_spectrum = 20 * np.log10(np.maximum(np.abs(np.fft.fftshift(np.fft.fft(y_rx))), 1e-8))
                axes[row, 1].plot(frequency, tx_spectrum - tx_spectrum.max(), label="x_tx")
                axes[row, 1].plot(frequency, rx_spectrum - rx_spectrum.max(), label="y_rx", alpha=0.8)
                axes[row, 1].set_ylim(-80, 5)
                axes[row, 1].set_title("normalized spectrum")
                axes[row, 1].legend(fontsize=8)

                taps = np.arange(len(h))
                axes[row, 2].stem(taps, np.abs(h), linefmt="C0-", markerfmt="C0o", basefmt=" ")
                phase_axis = axes[row, 2].twinx()
                phase_axis.plot(taps, np.angle(h), "C1x--", label="phase")
                phase_axis.set_ylim(-np.pi, np.pi)
                axes[row, 2].set_title("FIR magnitude / phase")
                axes[row, 2].set_xlabel("tap")

                axes[row, 3].scatter(x_tx.real, x_tx.imag, s=3, alpha=0.12, label="x_tx")
                axes[row, 3].scatter(y_rx.real, y_rx.imag, s=3, alpha=0.10, label="y_rx")
                symbol_samples = _matched_symbol_samples(x_tx, args, modulation)
                axes[row, 3].scatter(symbol_samples.real, symbol_samples.imag, s=10, alpha=0.75, label="matched x_tx symbols")
                axes[row, 3].set_aspect("equal", adjustable="datalim")
                axes[row, 3].set_title("IQ scatter")
                axes[row, 3].legend(fontsize=7)
            figure.suptitle(f"{modulation} validation samples", fontsize=15)
            figure.tight_layout()
            destination = validation_dir / f"{modulation.lower()}_validation.png"
            figure.savefig(destination, dpi=150)
            plt.close(figure)
            plot_files.append(str(destination))
    return plot_files


def main() -> int:
    args = parse_args()
    output = args.output.resolve()
    validation_dir = output.parent / "validation"
    partial = output.with_name(f"{output.stem}.partial{output.suffix}")
    output.parent.mkdir(parents=True, exist_ok=True)
    validation_dir.mkdir(parents=True, exist_ok=True)
    if (output.exists() or partial.exists()) and not args.overwrite:
        raise FileExistsError("output or partial file exists; pass --overwrite to replace it")
    if args.overwrite:
        output.unlink(missing_ok=True)
        partial.unlink(missing_ok=True)

    print(f"Generating {len(MODULATIONS) * args.examples_per_modulation} samples -> {partial}")
    generation_seconds = generate_dataset(args, partial)
    print("Generation complete; validating HDF5 contents and numerical invariants...")
    metrics = validate_dataset(args, partial, validation_dir, generation_seconds)
    metrics_path = validation_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False) + "\n")
    if not metrics["passed"]:
        failed = [name for name, result in metrics["checks"].items() if not result["passed"]]
        raise RuntimeError(f"validation failed: {', '.join(failed)}; partial dataset retained at {partial}")
    partial.replace(output)
    print(f"Validated dataset: {output}")
    print(f"Validation metrics: {metrics_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
