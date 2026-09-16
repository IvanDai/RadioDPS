"""Known-channel diffusion posterior sampling."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import torch
from torch import nn

from .diffusion import DDPMDiffusion
from .metrics import measurement_residual
from .operators import ComplexFIRChannel


Likelihood = Literal["auto", "normalized", "gaussian"]


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
    normalized = residual_energy / (measurement_energy + eps)
    if likelihood == "normalized":
        return normalized
    if likelihood == "gaussian":
        if bool((noise_variance <= 0).any()):
            raise ValueError("Gaussian likelihood requires strictly positive noise_variance")
        return residual_energy / (2.0 * noise_variance)
    if likelihood == "auto":
        gaussian = residual_energy / (2.0 * noise_variance.clamp_min(eps))
        return torch.where(noise_variance > 0, gaussian, normalized)
    raise ValueError("likelihood must be auto, normalized, or gaussian")


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
    initial_noise: torch.Tensor | None = None,
    sampling_noises: Sequence[torch.Tensor] | None = None,
    per_sample_diagnostics: bool = False,
) -> tuple[torch.Tensor, list[dict[str, Any]]]:
    """Run Algorithm 1-style DPS from Gaussian noise."""
    if measurement.ndim != 3 or measurement.shape[1] != 2:
        raise ValueError("measurement must have shape [B,2,T]")
    if guidance_scale < 0:
        raise ValueError("guidance_scale must be non-negative")
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
            current = prior_sample - guidance_scale * gradient
            residuals = measurement_residual(measurement, predicted_measurement, measurement_eps)
            gradient_norms = gradient.flatten(1).norm(dim=1)
            diagnostic = {
                "timestep": float(timestep),
                "measurement_loss_mean": float(losses.mean()),
                "measurement_residual_mean": float(residuals.mean()),
                "gradient_norm_mean": float(gradient_norms.mean()),
            }
            if per_sample_diagnostics:
                diagnostic.update(
                    {
                        "measurement_loss": losses.detach().cpu().tolist(),
                        "measurement_residual": residuals.detach().cpu().tolist(),
                        "gradient_norm": gradient_norms.detach().cpu().tolist(),
                    }
                )
            diagnostics.append(diagnostic)
    return current.detach(), diagnostics
