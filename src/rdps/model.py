"""A configurable one-dimensional U-Net for I/Q epsilon prediction."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dimension: int):
        super().__init__()
        self.dimension = dimension

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.dimension // 2
        scale = math.log(10_000.0) / max(half - 1, 1)
        frequencies = torch.exp(
            -scale * torch.arange(half, device=timesteps.device, dtype=torch.float32)
        )
        angles = timesteps.float()[:, None] * frequencies[None]
        embedding = torch.cat((angles.sin(), angles.cos()), dim=1)
        if self.dimension % 2:
            embedding = torch.nn.functional.pad(embedding, (0, 1))
        return embedding


class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_channels: int, dropout: float):
        super().__init__()
        self.norm1 = nn.GroupNorm(_group_count(in_channels), in_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.time_projection = nn.Linear(time_channels, out_channels)
        self.norm2 = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.skip = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        hidden = self.conv1(torch.nn.functional.silu(self.norm1(x)))
        hidden = hidden + self.time_projection(time_embedding)[:, :, None]
        hidden = self.conv2(self.dropout(torch.nn.functional.silu(self.norm2(hidden))))
        return hidden + self.skip(x)


class _Level(nn.Module):
    def __init__(self, blocks: Sequence[ResidualBlock1D]):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x, time_embedding)
        return x


class UNet1D(nn.Module):
    """Multi-scale U-Net mapping `[B, 2, T]` to `[B, 2, T]`."""

    def __init__(
        self,
        *,
        in_channels: int = 2,
        out_channels: int = 2,
        base_channels: int = 64,
        channel_multipliers: Sequence[int] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        if in_channels < 1 or out_channels < 1 or base_channels < 2:
            raise ValueError("channel counts must be positive and base_channels at least two")
        if not channel_multipliers or any(value < 1 for value in channel_multipliers):
            raise ValueError("channel_multipliers must contain positive integers")
        if num_res_blocks < 1:
            raise ValueError("num_res_blocks must be positive")
        self.downsample_factor = 2 ** (len(channel_multipliers) - 1)
        time_channels = base_channels * 4
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(base_channels),
            nn.Linear(base_channels, time_channels),
            nn.SiLU(),
            nn.Linear(time_channels, time_channels),
        )
        self.input = nn.Conv1d(in_channels, base_channels, kernel_size=3, padding=1)

        level_channels = [base_channels * multiplier for multiplier in channel_multipliers]
        self.down_levels = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        current = base_channels
        for index, channels in enumerate(level_channels):
            blocks = [ResidualBlock1D(current, channels, time_channels, dropout)]
            blocks.extend(
                ResidualBlock1D(channels, channels, time_channels, dropout)
                for _ in range(num_res_blocks - 1)
            )
            self.down_levels.append(_Level(blocks))
            current = channels
            if index < len(level_channels) - 1:
                self.downsamples.append(nn.Conv1d(current, current, kernel_size=4, stride=2, padding=1))

        self.middle = _Level(
            [
                ResidualBlock1D(current, current, time_channels, dropout),
                ResidualBlock1D(current, current, time_channels, dropout),
            ]
        )
        self.up_levels = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for reverse_index, channels in enumerate(reversed(level_channels)):
            blocks = [ResidualBlock1D(current + channels, channels, time_channels, dropout)]
            blocks.extend(
                ResidualBlock1D(channels, channels, time_channels, dropout)
                for _ in range(num_res_blocks - 1)
            )
            self.up_levels.append(_Level(blocks))
            current = channels
            if reverse_index < len(level_channels) - 1:
                self.upsamples.append(nn.ConvTranspose1d(current, current, kernel_size=4, stride=2, padding=1))

        self.output_norm = nn.GroupNorm(_group_count(current), current)
        self.output = nn.Conv1d(current, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("model input must have shape [B,C,T]")
        if x.shape[-1] % self.downsample_factor:
            raise ValueError(f"waveform length must be divisible by {self.downsample_factor}")
        if timesteps.shape != (x.shape[0],):
            raise ValueError("timesteps must have shape [B]")
        time_embedding = self.time_mlp(timesteps)
        hidden = self.input(x)
        skips = []
        for index, level in enumerate(self.down_levels):
            hidden = level(hidden, time_embedding)
            skips.append(hidden)
            if index < len(self.downsamples):
                hidden = self.downsamples[index](hidden)
        hidden = self.middle(hidden, time_embedding)
        for index, level in enumerate(self.up_levels):
            hidden = torch.cat((hidden, skips.pop()), dim=1)
            hidden = level(hidden, time_embedding)
            if index < len(self.upsamples):
                hidden = self.upsamples[index](hidden)
        return self.output(torch.nn.functional.silu(self.output_norm(hidden)))
