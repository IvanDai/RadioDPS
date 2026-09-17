"""Compact terminal progress rendering."""

from __future__ import annotations

import sys
import time


def format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


class EpochProgress:
    def __init__(self, epoch: int, total_epochs: int, batches: int, *, width: int = 20):
        self.epoch = epoch
        self.total_epochs = total_epochs
        self.batches = batches
        self.width = width
        self.started = time.monotonic()
        self.dynamic = sys.stdout.isatty()

    def update(self, batch: int, loss: float, average_loss: float) -> None:
        if not self.dynamic:
            return
        fraction = batch / max(self.batches, 1)
        completed = min(self.width, int(self.width * fraction))
        bar = "█" * completed + "░" * (self.width - completed)
        elapsed = time.monotonic() - self.started
        eta = elapsed / batch * (self.batches - batch) if batch else 0.0
        message = (
            f"Epoch {self.epoch}/{self.total_epochs} Train [{bar}] "
            f"{batch}/{self.batches} loss={loss:.4f} | avg_loss={average_loss:.4f} "
            f"| ETA={format_duration(eta)}"
        )
        sys.stdout.write("\r\033[2K" + message)
        sys.stdout.flush()

    def finish(self, average_loss: float) -> float:
        elapsed = time.monotonic() - self.started
        if self.dynamic:
            sys.stdout.write("\r\033[2K")
        print(
            f"Epoch {self.epoch:03d}/{self.total_epochs} | Train | "
            f"loss={average_loss:.5f} | time={format_duration(elapsed)}",
            flush=True,
        )
        return elapsed


class PhaseProgress:
    """Render a compact work-unit progress line for validation phases."""

    def __init__(self, epoch: int, total_epochs: int, phase: str, *, width: int = 20):
        self.epoch = epoch
        self.total_epochs = total_epochs
        self.phase = phase
        self.width = width
        self.started = time.monotonic()
        self.dynamic = sys.stdout.isatty()

    def update(self, completed: int, total: int) -> None:
        if not self.dynamic:
            return
        fraction = completed / max(total, 1)
        filled = min(self.width, int(self.width * fraction))
        bar = "█" * filled + "░" * (self.width - filled)
        elapsed = time.monotonic() - self.started
        eta = elapsed / completed * (total - completed) if completed else 0.0
        message = (
            f"Epoch {self.epoch}/{self.total_epochs} {self.phase} [{bar}] "
            f"{completed}/{total} | ETA={format_duration(eta)}"
        )
        sys.stdout.write("\r\033[2K" + message)
        sys.stdout.flush()

    def finish(self) -> None:
        if self.dynamic:
            sys.stdout.write("\r\033[2K")
            sys.stdout.flush()
