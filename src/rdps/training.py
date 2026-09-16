"""Step-based training, validation, checkpointing, and resume support."""

from __future__ import annotations

import copy
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import HDF5WaveformDataset
from .diffusion import DDPMDiffusion
from .model import UNet1D
from .utils import JsonlLogger, create_run_directory, resolve_device, save_yaml, seed_everything, write_json


def build_model(config: dict[str, Any]) -> UNet1D:
    return UNet1D(**config)


def build_diffusion(config: dict[str, Any]) -> DDPMDiffusion:
    return DDPMDiffusion(**config)


def _rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(path: str | Path, *, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    return torch.load(Path(path), map_location=map_location, weights_only=False)


@torch.no_grad()
def validation_loss(
    model: nn.Module,
    diffusion: DDPMDiffusion,
    loader: DataLoader,
    *,
    device: torch.device,
    seed: int,
    max_batches: int | None,
) -> float:
    model.eval()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    weighted_loss = 0.0
    sample_count = 0
    for batch_index, batch in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        clean = batch["x_tx"].to(device)
        timesteps = torch.randint(diffusion.steps, (clean.shape[0],), generator=generator).to(device)
        noise = torch.randn(clean.shape, generator=generator, dtype=clean.dtype).to(device)
        loss = diffusion.training_loss(model, clean, timesteps=timesteps, noise=noise)
        weighted_loss += float(loss) * clean.shape[0]
        sample_count += clean.shape[0]
    if sample_count == 0:
        raise ValueError("validation split is empty")
    return weighted_loss / sample_count


def train(config: dict[str, Any], *, output_root: str | Path | None = None) -> Path:
    config = copy.deepcopy(config)
    seed = int(config["seed"])
    seed_everything(seed)
    device = resolve_device(str(config.get("device", "auto")))
    dataset_path = Path(config["dataset"]["path"]).expanduser().resolve()
    ratios = config["dataset"]["split_ratios"]
    split_seed = int(config["dataset"]["split_seed"])
    train_dataset = HDF5WaveformDataset(
        dataset_path, split="train", split_seed=split_seed, split_ratios=ratios
    )
    validation_dataset = HDF5WaveformDataset(
        dataset_path, split="validation", split_seed=split_seed, split_ratios=ratios
    )
    if not train_dataset or not validation_dataset:
        raise ValueError("training and validation splits must both be non-empty")

    training_config = config["training"]
    root = output_root if output_root is not None else training_config["output_root"]
    output = create_run_directory(root, "train")
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir()
    config["runtime"] = {
        "device": str(device),
        "dataset_path": str(dataset_path),
        "output_dir": str(output),
        "resume": training_config.get("resume"),
    }
    save_yaml(output / "config.yaml", config)
    write_json(
        output / "split.json",
        {
            "train": train_dataset.split_manifest(),
            "validation": validation_dataset.split_manifest(),
            "test": HDF5WaveformDataset(
                dataset_path, split="test", split_seed=split_seed, split_ratios=ratios
            ).split_manifest(),
        },
    )
    logger = JsonlLogger(output / "log.jsonl")

    model = build_model(config["model"]).to(device)
    diffusion = build_diffusion(config["diffusion"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config.get("weight_decay", 0.0)),
    )
    global_step = 0
    start_epoch = 0
    start_batch = 0
    best_validation = float("inf")
    resume_path = training_config.get("resume")
    if resume_path:
        checkpoint = load_checkpoint(resume_path, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        global_step = int(checkpoint["global_step"])
        start_epoch = int(checkpoint["next_epoch"])
        start_batch = int(checkpoint["next_batch"])
        best_validation = float(checkpoint.get("best_validation_loss", float("inf")))
        if "rng_state" in checkpoint:
            _restore_rng_state(checkpoint["rng_state"])
        print(f"Resumed {resume_path} at step {global_step}", flush=True)

    batch_size = int(training_config["batch_size"])
    num_workers = int(config["dataset"].get("num_workers", 0))
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    max_steps = int(training_config["max_steps"])
    if global_step >= max_steps:
        raise ValueError("checkpoint global_step is already at or beyond configured max_steps")
    validation_every = int(training_config["validation_every"])
    checkpoint_every = int(training_config["checkpoint_every"])
    log_every = int(training_config["log_every"])
    grad_clip = float(training_config["gradient_clip"])
    maximum_validation_batches = training_config.get("max_validation_batches")
    if maximum_validation_batches is not None:
        maximum_validation_batches = int(maximum_validation_batches)
    started = time.monotonic()
    latest_train_loss = float("nan")
    latest_validation_loss = float("nan")
    epoch = start_epoch

    while global_step < max_steps:
        shuffle_generator = torch.Generator().manual_seed(seed + epoch)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=shuffle_generator,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
        )
        batches_per_epoch = len(train_loader)
        for batch_index, batch in enumerate(train_loader):
            if epoch == start_epoch and batch_index < start_batch:
                continue
            model.train()
            clean = batch["x_tx"].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = diffusion.training_loss(model, clean)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite training loss at step {global_step + 1}")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError(f"non-finite gradient at step {global_step + 1}")
            optimizer.step()
            global_step += 1
            latest_train_loss = float(loss.detach())
            next_epoch = epoch + 1 if batch_index + 1 == batches_per_epoch else epoch
            next_batch = 0 if next_epoch != epoch else batch_index + 1

            should_validate = global_step % validation_every == 0 or global_step == max_steps
            if should_validate:
                latest_validation_loss = validation_loss(
                    model,
                    diffusion,
                    validation_loader,
                    device=device,
                    seed=seed + 1_000_000 + global_step,
                    max_batches=maximum_validation_batches,
                )
            record = {
                "step": global_step,
                "epoch": epoch,
                "batch": batch_index,
                "train_loss": latest_train_loss,
                "gradient_norm": float(gradient_norm),
                "elapsed_seconds": time.monotonic() - started,
            }
            if should_validate:
                record["validation_loss"] = latest_validation_loss
            logger.log(record)
            if global_step % log_every == 0 or global_step == 1 or global_step == max_steps:
                validation_text = (
                    f" val={latest_validation_loss:.6f}" if should_validate else ""
                )
                print(
                    f"step {global_step:>7}/{max_steps} train={latest_train_loss:.6f} "
                    f"grad={float(gradient_norm):.4f}{validation_text}",
                    flush=True,
                )

            payload = {
                "format_version": 1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "global_step": global_step,
                "next_epoch": next_epoch,
                "next_batch": next_batch,
                "best_validation_loss": min(best_validation, latest_validation_loss),
                "config": config,
                "rng_state": _rng_state(),
            }
            if should_validate and latest_validation_loss < best_validation:
                best_validation = latest_validation_loss
                payload["best_validation_loss"] = best_validation
                save_checkpoint(checkpoint_dir / "best.pt", payload)
            if global_step % checkpoint_every == 0 or global_step == max_steps:
                save_checkpoint(checkpoint_dir / "last.pt", payload)
            if global_step >= max_steps:
                break
        epoch += 1
        start_batch = 0

    summary = {
        "status": "complete",
        "seed": seed,
        "device": str(device),
        "dataset": str(dataset_path),
        "split": train_dataset.split_manifest(),
        "global_step": global_step,
        "train_loss": latest_train_loss,
        "validation_loss": latest_validation_loss,
        "best_validation_loss": best_validation,
        "elapsed_seconds": time.monotonic() - started,
        "output_dir": str(output),
        "last_checkpoint": str(checkpoint_dir / "last.pt"),
        "best_checkpoint": str(checkpoint_dir / "best.pt"),
    }
    write_json(output / "metrics.json", summary)
    train_dataset.close()
    validation_dataset.close()
    return output
