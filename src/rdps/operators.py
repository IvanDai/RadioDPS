"""Differentiable known-channel operators in two-channel real form."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _check_shapes(x: torch.Tensor, h: torch.Tensor) -> None:
    if x.ndim != 3 or h.ndim != 3 or x.shape[1] != 2 or h.shape[1] != 2:
        raise ValueError("x and h must have shapes [B,2,T] and [B,2,L]")
    if x.shape[0] != h.shape[0] or h.shape[-1] < 1:
        raise ValueError("x and h batch sizes must match and L must be positive")
    if x.device != h.device or x.dtype != h.dtype:
        raise ValueError("x and h must use the same device and dtype")


class ComplexFIRChannel(nn.Module):
    """Batched complex FIR with causal-zero boundary and exact adjoint."""

    def forward(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        _check_shapes(x, h)
        tap_count = h.shape[-1]
        windows = F.pad(x, (tap_count - 1, 0)).unfold(-1, tap_count, 1)
        x_real, x_imag = windows[:, 0], windows[:, 1]
        h_real, h_imag = h[:, 0].flip(-1), h[:, 1].flip(-1)
        y_real = torch.einsum("btl,bl->bt", x_real, h_real) - torch.einsum(
            "btl,bl->bt", x_imag, h_imag
        )
        y_imag = torch.einsum("btl,bl->bt", x_real, h_imag) + torch.einsum(
            "btl,bl->bt", x_imag, h_real
        )
        return torch.stack((y_real, y_imag), dim=1)

    def adjoint(self, y: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        _check_shapes(y, h)
        tap_count = h.shape[-1]
        windows = F.pad(y, (0, tap_count - 1)).unfold(-1, tap_count, 1)
        y_real, y_imag = windows[:, 0], windows[:, 1]
        h_real, h_imag = h[:, 0], h[:, 1]
        x_real = torch.einsum("btl,bl->bt", y_real, h_real) + torch.einsum(
            "btl,bl->bt", y_imag, h_imag
        )
        x_imag = -torch.einsum("btl,bl->bt", y_real, h_imag) + torch.einsum(
            "btl,bl->bt", y_imag, h_real
        )
        return torch.stack((x_real, x_imag), dim=1)
