"""Epoch-oriented diffusion training with reliable resume and evaluation."""

from __future__ import annotations

import copy
from pathlib import Path
import time
from typing import Any

import torch
from torch.utils.data import DataLoader

from .checkpoint import (
    CHECKPOINT_FORMAT_VERSION,
    capture_rng_state,
    configuration_fingerprint,
    invariant_configuration,
    load_checkpoint,
    move_optimizer_state,
    restore_rng_state,
    save_checkpoint,
    validate_resume_checkpoint,
)
from .data import HDF5WaveformDataset, balanced_subset_indices
from .diffusion import DDPMDiffusion
from .evaluation import evaluation_settings, fixed_diffusion_validation, run_dps_evaluation
from .model import UNet1D
from .progress import EpochProgress, format_duration
from .utils import JsonlLogger, create_run_directory, resolve_device, save_yaml, seed_everything, write_json


def _validate_config(config: dict[str, Any]) -> None:
    training = config["training"]
    validation = config["validation"]
    dps_validation = config["dps_validation"]
    positive = {
        "training.epochs": training["epochs"],
        "training.batch_size": training["batch_size"],
        "training.save_every_epochs": training["save_every_epochs"],
        "validation.batch_size": validation["batch_size"],
        "validation.noise_repeats": validation.get("noise_repeats", 1),
        "dps_validation.every_epochs": dps_validation["every_epochs"],
    }
    invalid = [name for name, value in positive.items() if int(value) < 1]
    if invalid:
        raise ValueError(f"configuration values must be positive: {invalid}")
    if float(training["learning_rate"]) <= 0:
        raise ValueError("training.learning_rate must be positive")
    if float(training.get("weight_decay", 0.0)) < 0:
        raise ValueError("training.weight_decay must be non-negative")
    if float(training["gradient_clip"]) <= 0:
        raise ValueError("training.gradient_clip must be positive")
    early = training["early_stopping"]
    if int(early["patience"]) < 1 or float(early.get("min_delta", 0.0)) < 0:
        raise ValueError("early stopping patience must be positive and min_delta non-negative")


def _prepare_output(config: dict[str, Any]) -> tuple[Path, Path | None]:
    resume = config["training"].get("resume_output_dir")
    if resume is None:
        output = create_run_directory(config["training"]["output_root"], "train")
        return output, None
    output = Path(resume).expanduser().resolve()
    checkpoint = output / "checkpoints" / "last.pt"
    if not output.is_dir() or not checkpoint.is_file():
        raise FileNotFoundError(f"resume output must contain checkpoints/last.pt: {output}")
    return output, checkpoint


def _checkpoint_payload(
    *,
    model: UNet1D,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    best_validation_loss: float,
    best_epoch: int,
    bad_epochs: int,
    train_loss: float,
    validation_loss: float,
    config: dict[str, Any],
    invariants: dict[str, Any],
) -> dict[str, Any]:
    return {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "completed_epoch": epoch,
        "global_step": global_step,
        "best_validation_loss": best_validation_loss,
        "best_epoch": best_epoch,
        "early_stopping_bad_epochs": bad_epochs,
        "train_loss": train_loss,
        "validation_loss": validation_loss,
        "config": config,
        "invariant_config": invariants,
        "config_fingerprint": configuration_fingerprint(invariants),
        "rng_state": capture_rng_state(),
    }


def _train_epoch(
    model: UNet1D,
    diffusion: DDPMDiffusion,
    optimizer: torch.optim.Optimizer,
    dataset: HDF5WaveformDataset,
    *,
    epoch: int,
    total_epochs: int,
    global_step: int,
    batch_size: int,
    num_workers: int,
    gradient_clip: float,
    seed: int,
    device: torch.device,
) -> tuple[float, float, int]:
    generator = torch.Generator(device="cpu").manual_seed(seed + epoch)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    progress = EpochProgress(epoch, total_epochs, len(loader))
    weighted_loss = 0.0
    sample_count = 0
    model.train()
    for batch_number, batch in enumerate(loader, start=1):
        clean = batch["x_tx"].to(device, non_blocking=device.type == "cuda")
        optimizer.zero_grad(set_to_none=True)
        loss = diffusion.training_loss(model, clean)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite training loss at global step {global_step + 1}")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError(f"non-finite gradient at global step {global_step + 1}")
        optimizer.step()
        global_step += 1
        batch_loss = float(loss.detach())
        weighted_loss += batch_loss * clean.shape[0]
        sample_count += clean.shape[0]
        progress.update(batch_number, batch_loss, weighted_loss / sample_count)
    average_loss = weighted_loss / sample_count
    elapsed = progress.finish(average_loss)
    return average_loss, elapsed, global_step


def train(config: dict[str, Any], *, output_root: str | Path | None = None) -> Path:
    config = copy.deepcopy(config)
    if output_root is not None:
        config["training"]["output_root"] = str(output_root)
    _validate_config(config)
    seed = int(config["seed"])
    seed_everything(seed)
    device = resolve_device(str(config.get("device", "auto")))
    dataset_path = Path(config["dataset"]["path"]).expanduser().resolve()
    split_seed = int(config["dataset"]["split_seed"])
    ratios = config["dataset"]["split_ratios"]
    train_dataset = HDF5WaveformDataset(dataset_path, split="train", split_seed=split_seed, split_ratios=ratios)
    validation_dataset = HDF5WaveformDataset(
        dataset_path, split="validation", split_seed=split_seed, split_ratios=ratios
    )
    if not train_dataset or not validation_dataset:
        raise ValueError("training and validation splits must both be non-empty")

    output, resume_checkpoint_path = _prepare_output(config)
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    evaluations_dir = output / "evaluations"
    evaluations_dir.mkdir(exist_ok=True)
    config["runtime"] = {
        "device": str(device),
        "dataset_path": str(dataset_path),
        "output_dir": str(output),
        "resume_output_dir": config["training"].get("resume_output_dir"),
    }
    invariants = invariant_configuration(config, train_dataset.signature)
    total_epochs = int(config["training"]["epochs"])
    resume_checkpoint = None
    if resume_checkpoint_path is not None:
        resume_checkpoint = load_checkpoint(resume_checkpoint_path, map_location="cpu")
        validate_resume_checkpoint(
            resume_checkpoint,
            requested_invariants=invariants,
            total_epochs=total_epochs,
        )
    if resume_checkpoint_path is None:
        save_yaml(output / "config.yaml", config)
        write_json(
            output / "split.json",
            {
                "train": train_dataset.split_manifest(),
                "validation": validation_dataset.split_manifest(),
                "dataset_signature": train_dataset.signature,
            },
        )
    else:
        resume_configs = output / "resume_configs"
        resume_configs.mkdir(exist_ok=True)
        save_yaml(resume_configs / f"resume_to_epoch_{total_epochs:04d}.yaml", config)

    validation_indices, validation_selection = balanced_subset_indices(
        validation_dataset,
        config["validation"].get("sample_count"),
        seed=int(config["validation"]["seed"]),
    )
    write_json(output / "validation_selection.json", validation_selection)
    model = UNet1D(**config["model"]).to(device)
    diffusion = DDPMDiffusion(**config["diffusion"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"].get("weight_decay", 0.0)),
    )
    completed_epoch = 0
    global_step = 0
    best_validation_loss = float("inf")
    best_epoch = 0
    bad_epochs = 0
    latest_train_loss = float("nan")
    latest_validation_loss = float("nan")
    if resume_checkpoint is not None:
        model.load_state_dict(resume_checkpoint["model"])
        optimizer.load_state_dict(resume_checkpoint["optimizer"])
        move_optimizer_state(optimizer, device)
        completed_epoch = int(resume_checkpoint["completed_epoch"])
        global_step = int(resume_checkpoint["global_step"])
        best_validation_loss = float(resume_checkpoint["best_validation_loss"])
        best_epoch = int(resume_checkpoint["best_epoch"])
        bad_epochs = int(resume_checkpoint["early_stopping_bad_epochs"])
        latest_train_loss = float(resume_checkpoint.get("train_loss", float("nan")))
        latest_validation_loss = float(resume_checkpoint.get("validation_loss", float("nan")))
        restore_rng_state(resume_checkpoint["rng_state"])
        print(
            f"Resume | epoch={completed_epoch} step={global_step} best={best_validation_loss:.5f} "
            f"patience={bad_epochs}/{config['training']['early_stopping']['patience']}",
            flush=True,
        )

    history = JsonlLogger(output / "history.jsonl")
    batch_size = int(config["training"]["batch_size"])
    num_workers = int(config["dataset"].get("num_workers", 0))
    gradient_clip = float(config["training"]["gradient_clip"])
    validation_config = config["validation"]
    early_config = config["training"]["early_stopping"]
    dps_validation_config = config["dps_validation"]
    save_every = int(config["training"]["save_every_epochs"])
    training_started = time.monotonic()
    stopped_early = bool(early_config.get("enabled", True)) and bad_epochs >= int(early_config["patience"])
    if stopped_early:
        print(
            f"Early stopping state already reached at epoch {completed_epoch}; "
            "skipping further optimization.",
            flush=True,
        )

    try:
        epoch_range = () if stopped_early else range(completed_epoch + 1, total_epochs + 1)
        for epoch in epoch_range:
            latest_train_loss, train_seconds, global_step = _train_epoch(
                model,
                diffusion,
                optimizer,
                train_dataset,
                epoch=epoch,
                total_epochs=total_epochs,
                global_step=global_step,
                batch_size=batch_size,
                num_workers=num_workers,
                gradient_clip=gradient_clip,
                seed=seed,
                device=device,
            )
            validation_started = time.monotonic()
            latest_validation_loss = fixed_diffusion_validation(
                model,
                diffusion,
                validation_dataset,
                validation_indices,
                device=device,
                seed=int(validation_config["seed"]),
                noise_repeats=int(validation_config.get("noise_repeats", 1)),
                batch_size=int(validation_config["batch_size"]),
            )
            validation_seconds = time.monotonic() - validation_started
            improved = latest_validation_loss < best_validation_loss - float(early_config.get("min_delta", 0.0))
            if improved:
                best_validation_loss = latest_validation_loss
                best_epoch = epoch
                bad_epochs = 0
            else:
                bad_epochs += 1
            print(
                f"Epoch {epoch:03d}/{total_epochs} | Valid | loss={latest_validation_loss:.5f} "
                f"| best={best_validation_loss:.5f} | patience={bad_epochs}/{early_config['patience']}",
                flush=True,
            )

            dps_summary = None
            if bool(dps_validation_config.get("enabled", True)) and epoch % int(dps_validation_config["every_epochs"]) == 0:
                dps_output = evaluations_dir / "validation" / f"epoch_{epoch:04d}"
                dps_summary = run_dps_evaluation(
                    model,
                    diffusion,
                    validation_dataset,
                    device=device,
                    settings=evaluation_settings(config, "dps_validation"),
                    output_dir=dps_output,
                    save_reconstructions=False,
                    context={"epoch": epoch, "global_step": global_step, "device": str(device)},
                )
                macro = dps_summary["metrics"]["macro"]
                print(
                    f"Epoch {epoch:03d}/{total_epochs} | DPS | NMSE={macro['nmse']:.4f} "
                    f"| residual={macro['measurement_residual']:.4f}",
                    flush=True,
                )

            payload = _checkpoint_payload(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                global_step=global_step,
                best_validation_loss=best_validation_loss,
                best_epoch=best_epoch,
                bad_epochs=bad_epochs,
                train_loss=latest_train_loss,
                validation_loss=latest_validation_loss,
                config=config,
                invariants=invariants,
            )
            if improved:
                save_checkpoint(checkpoint_dir / "best.pt", payload)
            save_checkpoint(checkpoint_dir / "last.pt", payload)
            if epoch % save_every == 0:
                save_checkpoint(checkpoint_dir / f"epoch_{epoch:04d}.pt", payload)
            history.log(
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "train_loss": latest_train_loss,
                    "validation_loss": latest_validation_loss,
                    "best_validation_loss": best_validation_loss,
                    "best_epoch": best_epoch,
                    "early_stopping_bad_epochs": bad_epochs,
                    "train_seconds": train_seconds,
                    "validation_seconds": validation_seconds,
                    "dps": None if dps_summary is None else dps_summary["metrics"],
                }
            )
            completed_epoch = epoch
            if bool(early_config.get("enabled", True)) and bad_epochs >= int(early_config["patience"]):
                stopped_early = True
                print(
                    f"Early stopping | epoch={epoch} best_epoch={best_epoch} best={best_validation_loss:.5f}",
                    flush=True,
                )
                break

        best_checkpoint_path = checkpoint_dir / "best.pt"
        best_checkpoint = load_checkpoint(best_checkpoint_path, map_location="cpu")
        best_checkpoint["config"] = config
        save_checkpoint(best_checkpoint_path, best_checkpoint)
        model.load_state_dict(best_checkpoint["model"])
        test_dataset = HDF5WaveformDataset(
            dataset_path,
            split="test",
            split_seed=split_seed,
            split_ratios=ratios,
        )
        try:
            test_output = evaluations_dir / (
                f"test_best_epoch_{best_epoch:04d}_after_{completed_epoch:04d}"
            )
            test_summary = run_dps_evaluation(
                model,
                diffusion,
                test_dataset,
                device=device,
                settings=evaluation_settings(config, "evaluation"),
                output_dir=test_output,
                save_reconstructions=True,
                context={
                    "checkpoint": str(best_checkpoint_path),
                    "checkpoint_epoch": best_epoch,
                    "checkpoint_step": int(best_checkpoint["global_step"]),
                    "device": str(device),
                },
            )
        finally:
            test_dataset.close()
        test_macro = test_summary["metrics"]["macro"]
        print(
            f"Final Test | best_epoch={best_epoch} | NMSE={test_macro['nmse']:.4f} "
            f"| residual={test_macro['measurement_residual']:.4f}",
            flush=True,
        )
        elapsed_seconds = time.monotonic() - training_started
        summary = {
            "status": "early_stopped" if stopped_early else "complete",
            "seed": seed,
            "device": str(device),
            "dataset": str(dataset_path),
            "completed_epoch": completed_epoch,
            "global_step": global_step,
            "train_loss": latest_train_loss,
            "validation_loss": latest_validation_loss,
            "best_validation_loss": best_validation_loss,
            "best_epoch": best_epoch,
            "early_stopping_bad_epochs": bad_epochs,
            "elapsed_seconds": elapsed_seconds,
            "elapsed": format_duration(elapsed_seconds),
            "output_dir": str(output),
            "last_checkpoint": str(checkpoint_dir / "last.pt"),
            "best_checkpoint": str(best_checkpoint_path),
            "final_test_output": str(test_output),
            "final_test_metrics": test_summary["metrics"],
        }
        write_json(output / "metrics.json", summary)
        return output
    finally:
        train_dataset.close()
        validation_dataset.close()
