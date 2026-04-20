"""Unified trainer for all CoRemake training phases."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from coremake.utils.logging import get_logger

logger = get_logger(__name__)


class Trainer:
    """General-purpose trainer with callback support."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        scheduler: Optional[Any] = None,
        device: str = "cuda",
        max_epochs: int = 5,
        max_steps: Optional[int] = None,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        log_every: int = 100,
        save_every: int = 1000,
        save_dir: str = "checkpoints",
        callbacks: Optional[List[Callable]] = None,
        bf16: bool = True,
    ) -> None:
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.max_epochs = max_epochs
        self.max_steps = max_steps
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.log_every = log_every
        self.save_every = save_every
        self.save_dir = Path(save_dir)
        self.callbacks = callbacks or []
        self.bf16 = bf16
        self.scaler = torch.amp.GradScaler(enabled=bf16)
        self.global_step = 0

    def train(
        self,
        train_loader: DataLoader,
        loss_fn: Callable,
        val_loader: Optional[DataLoader] = None,
    ) -> Dict[str, float]:
        self.model.train()
        best_loss = float("inf")

        for epoch in range(self.max_epochs):
            epoch_loss = 0.0
            for batch_idx, batch in enumerate(train_loader):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

                with torch.amp.autocast("cuda", enabled=self.bf16):
                    loss = loss_fn(self.model, batch)
                    loss = loss / self.gradient_accumulation_steps

                self.scaler.scale(loss).backward()

                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    if self.scheduler is not None:
                        self.scheduler.step()
                    self.global_step += 1

                epoch_loss += loss.item() * self.gradient_accumulation_steps

                if self.global_step % self.log_every == 0:
                    logger.info(f"Step {self.global_step} | Loss: {loss.item():.4f}")

                if self.global_step % self.save_every == 0:
                    self._save_checkpoint(f"step_{self.global_step}.pt")

                if self.max_steps and self.global_step >= self.max_steps:
                    break

                for cb in self.callbacks:
                    cb(self, batch_idx, loss.item())

            avg_loss = epoch_loss / max(len(train_loader), 1)
            logger.info(f"Epoch {epoch + 1}/{self.max_epochs} | Avg Loss: {avg_loss:.4f}")

            if avg_loss < best_loss:
                best_loss = avg_loss
                self._save_checkpoint("best.pt")

            if self.max_steps and self.global_step >= self.max_steps:
                break

        return {"best_loss": best_loss, "final_step": self.global_step}

    def _save_checkpoint(self, filename: str) -> None:
        self.save_dir.mkdir(parents=True, exist_ok=True)
        path = self.save_dir / filename
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_step": self.global_step,
        }, path)
        logger.info(f"Saved checkpoint: {path}")
