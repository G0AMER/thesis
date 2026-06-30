"""
Behavioral Cloning (BC) — Baseline Imitation Learning
======================================================
Trains a policy π(a|s) via supervised regression on expert demonstrations.

Two variants:
  - **NaiveFineTune**: trains on each task sequentially (no CL strategy).
    Serves as the lower bound — will exhibit catastrophic forgetting.
  - **JointTraining**: trains on accumulated data from all tasks.
    Serves as the upper bound (oracle) — not realistic in practice.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset

from src.incremental_learning.base import ContinualLearner

logger = logging.getLogger(__name__)


class NaiveFineTune(ContinualLearner):
    """
    Naive sequential fine-tuning (no continual learning).
    Trains on each new task's data only — catastrophic forgetting expected.
    This is the LOWER BOUND baseline.
    """

    def __init__(self, model: nn.Module, lr: float = 1e-3, device: str = "cpu"):
        super().__init__(model=model, lr=lr, device=device)

    def on_task_start(self, task_id: int, train_loader: DataLoader) -> None:
        """No special preparation."""
        logger.info(f"[NaiveFineTune] Starting task {task_id}")

    def compute_loss(
        self, obs: torch.Tensor, act: torch.Tensor, task_id: int
    ) -> torch.Tensor:
        pred = self.model(obs)
        return nn.functional.mse_loss(pred, act)

    def on_task_end(self, task_id: int, train_loader: DataLoader) -> None:
        """No consolidation."""
        pass


class JointTraining(ContinualLearner):
    """
    Joint/cumulative training on all tasks seen so far.
    Stores all data and retrains from scratch on the union.
    This is the UPPER BOUND baseline (oracle).
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        device: str = "cpu",
        reset_model: bool = True,
    ):
        super().__init__(model=model, lr=lr, device=device)
        self._task_datasets: dict[int, TensorDataset] = {}
        self._initial_state = model.state_dict().copy()
        self._reset_model = reset_model

    def on_task_start(self, task_id: int, train_loader: DataLoader) -> None:
        """Store this task's data. Optionally reset model weights."""
        logger.info(f"[JointTraining] Starting task {task_id}")

        # Collect all data from this loader
        all_obs, all_act = [], []
        for batch in train_loader:
            all_obs.append(batch[0])
            all_act.append(batch[1])
        obs = torch.cat(all_obs)
        act = torch.cat(all_act)
        self._task_datasets[task_id] = TensorDataset(obs, act)

        # Reset model to initial weights for fair joint training
        if self._reset_model:
            self.model.load_state_dict(self._initial_state)
            self.optimizer = torch.optim.Adam(
                self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )

    def compute_loss(
        self, obs: torch.Tensor, act: torch.Tensor, task_id: int
    ) -> torch.Tensor:
        pred = self.model(obs)
        return nn.functional.mse_loss(pred, act)

    def on_task_end(self, task_id: int, train_loader: DataLoader) -> None:
        pass

    def train_task(
        self,
        task_id: int,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 50,
        patience: int = 10,
        verbose: bool = True,
    ) -> dict:
        """Override: train on the union of all task data seen so far."""
        self._current_task = task_id
        self.on_task_start(task_id, train_loader)

        # Build joint dataset
        joint_ds = ConcatDataset(list(self._task_datasets.values()))
        joint_loader = DataLoader(
            joint_ds,
            batch_size=train_loader.batch_size if hasattr(train_loader, 'batch_size') else 256,
            shuffle=True,
            drop_last=False,
        )

        # Use parent train loop with the joint loader
        self.model.train()
        return super().train_task(
            task_id=task_id,
            train_loader=joint_loader,
            val_loader=val_loader,
            epochs=epochs,
            patience=patience,
            verbose=verbose,
        )
