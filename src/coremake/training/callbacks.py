"""Training callbacks."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from coremake.training.trainer import Trainer


class EarlyStoppingCallback:
    def __init__(self, patience: int = 5, min_delta: float = 1e-4) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0

    def __call__(self, trainer: "Trainer", batch_idx: int, loss: float) -> None:
        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.counter = 0
        else:
            self.counter += 1


class LoggingCallback:
    def __init__(self, log_every: int = 50) -> None:
        self.log_every = log_every
        self.losses = []

    def __call__(self, trainer: "Trainer", batch_idx: int, loss: float) -> None:
        self.losses.append(loss)
