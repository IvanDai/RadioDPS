"""Known-channel diffusion posterior sampling."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Callable, Literal

import torch
from torch import nn

from .diffusion import DDPMDiffusion
from .metrics import measurement_residual
from .operators import ComplexFIRChannel


Likelihood = Literal[
    "auto",
    "normalized",
    "normalized_squared",
    "normalized_l2",
    "gaussian",
]


def measurement_loss(
    measurement: torch.Tensor,
    prediction: torch.Tensor,
    *,
    noise_variance: torch.Tensor,
    likelihood: Likelihood,
    eps: float,
) -> torch.Tensor:
    """Return one measurement loss per batch item."""
    residual_energy = (measurement - prediction).square().flatten(1).sum(1)
    measurement_energy = measurement.square().flatten(1).sum(1)
    normalized_squared = residual_energy / (measurement_energy + eps)
    normalized_l2 = torch.sqrt(residual_energy + eps) / torch.sqrt(measurement_energy + eps)
    if likelihood in {"normalized", "normalized_squared"}:
        return normalized_squared
    if likelihood == "normalized_l2":
        return normalized_l2
    if likelihood == "gaussian":
        if bool((noise_variance <= 0).any()):
            raise ValueError("Gaussian likelihood requires strictly positive noise_variance")
        return residual_energy / (2.0 * noise_variance)
    if likelihood == "auto":
        gaussian = residual_energy / (2.0 * noise_variance.clamp_min(eps))
        return torch.where(noise_variance > 0, gaussian, normalized_squared)
    raise ValueError(
        "likelihood must be auto, normalized, normalized_squared, normalized_l2, or gaussian"
    )


def _clip_guidance_update(
    update: torch.Tensor, max_norm: float | None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Clip each sample's complete guidance update and return pre/post norms."""
    raw_norms = update.flatten(1).norm(dim=1)
    if max_norm is None:
        return update, raw_norms, raw_norms
    scale = (max_norm / raw_norms.clamp_min(torch.finfo(update.dtype).tiny)).clamp(max=1.0)
    clipped = update * scale.view(-1, *([1] * (update.ndim - 1)))
    return clipped, raw_norms, clipped.flatten(1).norm(dim=1)


def dps_sample(
    model: nn.Module,
    diffusion: DDPMDiffusion,
    measurement: torch.Tensor,
    channel: torch.Tensor,
    *,
    noise_variance: torch.Tensor | float,
    sampling_steps: int,
    guidance_scale: float,
    likelihood: Likelihood = "auto",
    measurement_eps: float = 1e-8,
    clip_denoised: bool = False,
    max_guidance_update_norm: float | None = None,
    initial_noise: torch.Tensor | None = None,
    sampling_noises: Sequence[torch.Tensor] | None = None,
    per_sample_diagnostics: bool = False,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    """Run Algorithm 1-style DPS from Gaussian noise."""
    if measurement.ndim != 3 or measurement.shape[1] != 2:
        raise ValueError("measurement must have shape [B,2,T]")
    if guidance_scale < 0:
        raise ValueError("guidance_scale must be non-negative")
    if max_guidance_update_norm is not None and max_guidance_update_norm <= 0:
        raise ValueError("max_guidance_update_norm must be positive or None")
    device = measurement.device
    if isinstance(noise_variance, (float, int)):
        noise_variance = torch.full(
            (measurement.shape[0],), float(noise_variance), device=device, dtype=measurement.dtype
        )
    else:
        noise_variance = noise_variance.to(device=device, dtype=measurement.dtype).reshape(-1)
    if noise_variance.shape != (measurement.shape[0],):
        raise ValueError("noise_variance must be scalar or have shape [B]")
    current = torch.randn_like(measurement) if initial_noise is None else initial_noise.to(device).clone()
    operator = ComplexFIRChannel()
    diagnostics: list[dict[str, Any]] = []
    sequence = diffusion.sampling_timesteps(sampling_steps)
    if sampling_noises is not None and len(sampling_noises) != len(sequence):
        raise ValueError("sampling_noises must contain one tensor per sampling timestep")

    for index, timestep in enumerate(sequence):
        previous = sequence[index + 1] if index + 1 < len(sequence) else -1
        current = current.detach().requires_grad_(True)
        batch_t = torch.full((measurement.shape[0],), timestep, device=device, dtype=torch.long)
        epsilon = model(current, batch_t)
        x0 = diffusion.predict_x0(current, batch_t, epsilon)
        if clip_denoised:
            x0 = x0.clamp(-1.0, 1.0)
        predicted_measurement = operator(x0, channel)
        losses = measurement_loss(
            measurement,
            predicted_measurement,
            noise_variance=noise_variance,
            likelihood=likelihood,
            eps=measurement_eps,
        )
        gradient = torch.autograd.grad(losses.sum(), current, only_inputs=True)[0]
        if not torch.isfinite(gradient).all():
            raise FloatingPointError(f"non-finite measurement gradient at timestep {timestep}")

        with torch.no_grad():
            step_noise = None
            if sampling_noises is not None:
                step_noise = sampling_noises[index].to(device=device, dtype=current.dtype)
                if step_noise.shape != current.shape:
                    raise ValueError("each sampling noise tensor must match measurement shape")
            prior_sample = diffusion.posterior_sample_from_x0(
                current, x0, timestep, previous, noise=step_noise
            )
            guidance_update, raw_update_norms, update_norms = _clip_guidance_update(
                guidance_scale * gradient, max_guidance_update_norm
            )
            current = prior_sample - guidance_update
            if not torch.isfinite(current).all():
                raise FloatingPointError(f"non-finite DPS state after timestep {timestep}")
            residuals = measurement_residual(measurement, predicted_measurement, measurement_eps)
            gradient_norms = gradient.flatten(1).norm(dim=1)
            clipped = update_norms < raw_update_norms
            diagnostic = {
                "timestep": float(timestep),
                "measurement_loss_mean": float(losses.mean()),
                "measurement_residual_mean": float(residuals.mean()),
                "gradient_norm_mean": float(gradient_norms.mean()),
                "guidance_update_norm_mean": float(update_norms.mean()),
                "raw_guidance_update_norm_mean": float(raw_update_norms.mean()),
                "guidance_clipped_fraction": float(clipped.float().mean()),
            }
            if per_sample_diagnostics:
                diagnostic.update(
                    {
                        "measurement_loss": losses.detach().cpu().tolist(),
                        "measurement_residual": residuals.detach().cpu().tolist(),
                        "gradient_norm": gradient_norms.detach().cpu().tolist(),
                        "guidance_update_norm": update_norms.detach().cpu().tolist(),
                        "raw_guidance_update_norm": raw_update_norms.detach().cpu().tolist(),
                        "guidance_clipped": clipped.detach().cpu().tolist(),
                    }
                )
            diagnostics.append(diagnostic)
        if progress is not None:
            progress(index + 1, len(sequence))
    return current.detach(), diagnostics
