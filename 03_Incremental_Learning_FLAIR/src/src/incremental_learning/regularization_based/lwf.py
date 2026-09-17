"""
Learning without Forgetting (LwF)
==================================
Li & Hoiem, "Learning without Forgetting" (TPAMI 2017)

Key idea: use knowledge distillation to preserve the model's output distribution
on the current task's data before training on new data. No replay buffer needed —
only the current task's data is used, but the teacher's soft targets constrain
the student to not deviate on old tasks' outputs.

Loss = L_task + α * L_distill

where L_distill = MSE(student_logits, teacher_logits) averaged over new task data.
The teacher is the model snapshot taken right before training on the new task.
"""

from __future__ import annotations

import copy
import logging
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.incremental_learning.base import ContinualLearner

logger = logging.getLogger(__name__)


class LwF(ContinualLearner):
    """
    Learning without Forgetting.

    Parameters
    ----------
    model : nn.Module
        The policy network.
    lwf_alpha : float
        Distillation loss weight. Higher = more retention of old behavior.
    temperature : float
        Softmax temperature for distillation (for classification). For regression
        (our case) this is not strictly needed, but kept for completeness.
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        device: str = "cpu",
        lwf_alpha: float = 1.0,
        temperature: float = 2.0,
    ):
        super().__init__(model=model, lr=lr, device=device)
        self.lwf_alpha = lwf_alpha
        self.temperature = temperature
        self._teacher: Optional[nn.Module] = None

    def on_task_start(self, task_id: int, train_loader: DataLoader) -> None:
        """Snapshot the current model as the teacher (frozen)."""
        logger.info(f"[LwF] Starting task {task_id} (α={self.lwf_alpha})")
        if task_id > 0:
            self._teacher = copy.deepcopy(self.model)
            self._teacher.eval()
            for p in self._teacher.parameters():
                p.requires_grad = False
            logger.info(f"[LwF] Teacher snapshot created for distillation")
        else:
            self._teacher = None

    def compute_loss(
        self, obs: torch.Tensor, act: torch.Tensor, task_id: int
    ) -> torch.Tensor:
        """MSE loss on current task + distillation loss from teacher."""
        pred = self.model(obs)
        task_loss = nn.functional.mse_loss(pred, act)

        if self._teacher is None or task_id == 0:
            return task_loss

        # Distillation: match teacher's output on the current data
        with torch.no_grad():
            teacher_pred = self._teacher(obs)

        distill_loss = nn.functional.mse_loss(pred, teacher_pred)

        return task_loss + self.lwf_alpha * distill_loss

    def on_task_end(self, task_id: int, train_loader: DataLoader) -> None:
        """Nothing to consolidate — teacher snapshot is updated at next task_start."""
        pass
