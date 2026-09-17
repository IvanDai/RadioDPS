"""DDPM noise schedules, epsilon training, and ancestral sampling."""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn


def make_beta_schedule(
    steps: int,
    *,
    schedule: str,
    beta_start: float,
    beta_end: float,
) -> torch.Tensor:
    if steps < 2:
        raise ValueError("diffusion steps must be at least two")
    if schedule == "linear":
        betas = torch.linspace(beta_start, beta_end, steps, dtype=torch.float64)
    elif schedule == "cosine":
        offset = 0.008
        times = torch.linspace(0, steps, steps + 1, dtype=torch.float64) / steps
        alpha_bar = torch.cos((times + offset) / (1 + offset) * math.pi / 2).square()
        alpha_bar = alpha_bar / alpha_bar[0]
        betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
        betas = betas.clamp(max=0.999)
    else:
        raise ValueError("schedule must be 'linear' or 'cosine'")
    if (betas <= 0).any() or (betas >= 1).any():
        raise ValueError("all diffusion betas must lie in (0, 1)")
    return betas.float()


def _extract(values: torch.Tensor, timesteps: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return values.gather(0, timesteps).reshape(timesteps.shape[0], *([1] * (target.ndim - 1)))


class DDPMDiffusion(nn.Module):
    def __init__(
        self,
        *,
        steps: int = 1000,
        schedule: str = "linear",
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
    ):
        super().__init__()
        betas = make_beta_schedule(
            steps, schedule=schedule, beta_start=beta_start, beta_end=beta_end
        )
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.steps = steps
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("sqrt_alpha_bars", alpha_bars.sqrt())
        self.register_buffer("sqrt_one_minus_alpha_bars", (1.0 - alpha_bars).sqrt())

    def q_sample(
        self, clean: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if noise is None:
            noise = torch.randn_like(clean)
        noisy = (
            _extract(self.sqrt_alpha_bars, timesteps, clean) * clean
            + _extract(self.sqrt_one_minus_alpha_bars, timesteps, clean) * noise
        )
        return noisy, noise

    def training_loss(
        self,
        model: nn.Module,
        clean: torch.Tensor,
        timesteps: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if timesteps is None:
            timesteps = torch.randint(self.steps, (clean.shape[0],), device=clean.device)
        noisy, target = self.q_sample(clean, timesteps, noise)
        prediction = model(noisy, timesteps)
        return torch.nn.functional.mse_loss(prediction, target)

    def predict_x0(self, noisy: torch.Tensor, timesteps: torch.Tensor, epsilon: torch.Tensor) -> torch.Tensor:
        alpha_bar = _extract(self.alpha_bars, timesteps, noisy)
        return (noisy - (1.0 - alpha_bar).sqrt() * epsilon) / alpha_bar.sqrt()

    def sampling_timesteps(self, sampling_steps: int | None = None) -> list[int]:
        if sampling_steps is None:
            sampling_steps = self.steps
        if not 2 <= sampling_steps <= self.steps:
            raise ValueError(f"sampling_steps must be in [2, {self.steps}]")
        ascending = np.rint(np.linspace(0, self.steps - 1, sampling_steps)).astype(np.int64)
        if len(np.unique(ascending)) != sampling_steps:
            raise RuntimeError("failed to construct unique sampling timesteps")
        return [int(value) for value in ascending[::-1]]

    def posterior_sample_from_x0(
        self,
        noisy: torch.Tensor,
        x0: torch.Tensor,
        timestep: int,
        previous_timestep: int,
        *,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sample q(x_s | x_t, x_0), supporting non-adjacent `s < t`."""
        alpha_bar_t = self.alpha_bars[timestep].to(dtype=noisy.dtype)
        alpha_bar_s = (
            torch.ones((), device=noisy.device, dtype=noisy.dtype)
            if previous_timestep < 0
            else self.alpha_bars[previous_timestep].to(dtype=noisy.dtype)
        )
        transition_beta = 1.0 - alpha_bar_t / alpha_bar_s
        denominator = 1.0 - alpha_bar_t
        coefficient_x0 = alpha_bar_s.sqrt() * transition_beta / denominator
        coefficient_xt = (alpha_bar_t / alpha_bar_s).sqrt() * (1.0 - alpha_bar_s) / denominator
        mean = coefficient_x0 * x0 + coefficient_xt * noisy
        variance = ((1.0 - alpha_bar_s) / denominator * transition_beta).clamp_min(0.0)
        if previous_timestep < 0:
            return mean
        if noise is None:
            noise = torch.randn_like(noisy)
        return mean + variance.sqrt() * noise

    def ddim_sample_from_x0(
        self,
        noisy: torch.Tensor,
        x0: torch.Tensor,
        timestep: int,
        previous_timestep: int,
    ) -> torch.Tensor:
        """Return the deterministic DDIM (eta=0) update for an arbitrary previous timestep."""
        if previous_timestep < 0:
            return x0
        alpha_bar_t = self.alpha_bars[timestep].to(dtype=noisy.dtype)
        alpha_bar_s = self.alpha_bars[previous_timestep].to(dtype=noisy.dtype)
        epsilon = (noisy - alpha_bar_t.sqrt() * x0) / (1.0 - alpha_bar_t).sqrt()
        return alpha_bar_s.sqrt() * x0 + (1.0 - alpha_bar_s).sqrt() * epsilon

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        shape: tuple[int, ...],
        *,
        sampling_steps: int | None = None,
        clip_denoised: bool = False,
        device: torch.device | str,
    ) -> torch.Tensor:
        current = torch.randn(shape, device=device)
        sequence = self.sampling_timesteps(sampling_steps)
        for index, timestep in enumerate(sequence):
            previous = sequence[index + 1] if index + 1 < len(sequence) else -1
            batch_t = torch.full((shape[0],), timestep, device=device, dtype=torch.long)
            epsilon = model(current, batch_t)
            x0 = self.predict_x0(current, batch_t, epsilon)
            if clip_denoised:
                x0 = x0.clamp(-1.0, 1.0)
            current = self.posterior_sample_from_x0(current, x0, timestep, previous)
        return current
