"""
Dark Experience Replay++ (DER++)
=================================
Buzzega et al., "Dark Experience for General Continual Learning" (NeurIPS 2020)

Key idea: maintain a replay buffer of past examples.  For each replayed sample,
match not only the target label but also the model's *logits* (dark knowledge)
from when that sample was first stored.

Loss = L_task + α * L_logit_replay + β * L_label_replay

where:
  - L_logit_replay = MSE between current logits and stored logits (soft targets)
  - L_label_replay = MSE between current predictions and stored labels (hard targets)
"""

from __future__ import annotations

import logging
import random
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.incremental_learning.base import ContinualLearner

logger = logging.getLogger(__name__)


class ReplayBuffer:
    """
    Fixed-size reservoir-sampling replay buffer.
    Stores (obs, act, logits) tuples.
    """

    def __init__(self, capacity: int = 5000):
        self.capacity = capacity
        self.obs: list[torch.Tensor] = []
        self.act: list[torch.Tensor] = []
        self.logits: list[torch.Tensor] = []
        self._n_seen = 0

    def __len__(self) -> int:
        return len(self.obs)

    def add(
        self,
        obs: torch.Tensor,
        act: torch.Tensor,
        logits: torch.Tensor,
    ) -> None:
        """Add samples using reservoir sampling to maintain uniform distribution."""
        batch_size = obs.size(0)
        for i in range(batch_size):
            self._n_seen += 1
            if len(self.obs) < self.capacity:
                self.obs.append(obs[i].cpu())
                self.act.append(act[i].cpu())
                self.logits.append(logits[i].cpu())
            else:
                # Reservoir sampling
                j = random.randint(0, self._n_seen - 1)
                if j < self.capacity:
                    self.obs[j] = obs[i].cpu()
                    self.act[j] = act[i].cpu()
                    self.logits[j] = logits[i].cpu()

    def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a random mini-batch from the buffer."""
        n = min(batch_size, len(self.obs))
        indices = random.sample(range(len(self.obs)), n)
        return (
            torch.stack([self.obs[i] for i in indices]),
            torch.stack([self.act[i] for i in indices]),
            torch.stack([self.logits[i] for i in indices]),
        )

    @property
    def is_empty(self) -> bool:
        return len(self.obs) == 0


class DERPlusPlus(ContinualLearner):
    """
    Dark Experience Replay++ (DER++).

    Parameters
    ----------
    model : nn.Module
        The policy network.
    buffer_size : int
        Maximum number of samples in the replay buffer.
    alpha : float
        Weight for the logit-match (dark knowledge) replay loss.
    beta : float
        Weight for the label-match (hard target) replay loss.
    replay_batch_ratio : float
        Ratio of replay samples vs. task samples per batch.
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        device: str = "cpu",
        buffer_size: int = 5000,
        alpha: float = 0.5,
        beta: float = 0.5,
        replay_batch_ratio: float = 1.0,
    ):
        super().__init__(model=model, lr=lr, device=device)
        self.buffer = ReplayBuffer(capacity=buffer_size)
        self.alpha = alpha
        self.beta = beta
        self.replay_batch_ratio = replay_batch_ratio

    def on_task_start(self, task_id: int, train_loader: DataLoader) -> None:
        logger.info(
            f"[DER++] Starting task {task_id} "
            f"(buffer={self.buffer.capacity}, α={self.alpha}, β={self.beta}, "
            f"buffer_used={len(self.buffer)})"
        )

    def compute_loss(
        self, obs: torch.Tensor, act: torch.Tensor, task_id: int
    ) -> torch.Tensor:
        """
        Compute DER++ loss:
          L = L_task + α * L_logit + β * L_label
        """
        # Current task loss
        pred = self.model(obs)
        task_loss = nn.functional.mse_loss(pred, act)

        # Store current batch in buffer with logits
        with torch.no_grad():
            logits = self.model(obs).detach()
        self.buffer.add(obs.detach(), act.detach(), logits)

        # Replay loss
        if self.buffer.is_empty or task_id == 0:
            return task_loss

        replay_size = max(1, int(obs.size(0) * self.replay_batch_ratio))
        buf_obs, buf_act, buf_logits = self.buffer.sample(replay_size)
        buf_obs = buf_obs.to(self.device)
        buf_act = buf_act.to(self.device)
        buf_logits = buf_logits.to(self.device)

        buf_pred = self.model(buf_obs)

        # Logit-matching loss (dark knowledge)
        logit_loss = nn.functional.mse_loss(buf_pred, buf_logits)

        # Label-matching loss (hard targets)
        label_loss = nn.functional.mse_loss(buf_pred, buf_act)

        total = task_loss + self.alpha * logit_loss + self.beta * label_loss
        return total

    def on_task_end(self, task_id: int, train_loader: DataLoader) -> None:
        """No additional consolidation needed — buffer is updated during training."""
        logger.info(f"[DER++] Task {task_id} done. Buffer size: {len(self.buffer)}")
