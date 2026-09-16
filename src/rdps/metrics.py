"""Waveform reconstruction metrics."""

from __future__ import annotations

import torch


def mse(reference: torch.Tensor, estimate: torch.Tensor) -> torch.Tensor:
    return (reference - estimate).square().flatten(1).mean(1)


def nmse(reference: torch.Tensor, estimate: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    numerator = (reference - estimate).square().flatten(1).sum(1)
    denominator = reference.square().flatten(1).sum(1).clamp_min(eps)
    return numerator / denominator


def measurement_residual(
    measurement: torch.Tensor, prediction: torch.Tensor, eps: float = 1e-12
) -> torch.Tensor:
    numerator = (measurement - prediction).square().flatten(1).sum(1)
    denominator = measurement.square().flatten(1).sum(1).clamp_min(eps)
    return numerator / denominator
