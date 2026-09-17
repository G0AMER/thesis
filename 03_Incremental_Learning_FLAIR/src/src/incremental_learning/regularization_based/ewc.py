"""
Elastic Weight Consolidation (EWC)
===================================
Kirkpatrick et al., "Overcoming catastrophic forgetting in neural networks" (PNAS 2017)

Key idea: penalize changes to parameters that are important for previous tasks,
measured by the diagonal of the Fisher Information Matrix.

Loss = L_task + (λ/2) Σ_i F_i (θ_i - θ*_i)²

where F_i is the Fisher importance of parameter i, θ*_i is the optimal value
after training the previous task.

Variants:
  - **EWC**: accumulates separate Fisher matrices per task
  - **Online EWC** (a.k.a. EWC++): maintains a running Fisher estimate
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


class EWC(ContinualLearner):
    """
    Elastic Weight Consolidation.

    Parameters
    ----------
    model : nn.Module
        The policy network.
    ewc_lambda : float
        Regularization strength. Higher = less forgetting but more rigidity.
    fisher_samples : int
        Number of samples used to estimate the Fisher Information Matrix.
    online : bool
        If True, use Online EWC (running Fisher average). If False, classical per-task EWC.
    gamma : float
        Decay factor for online EWC (only used if online=True). 0 < gamma ≤ 1.
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        device: str = "cpu",
        ewc_lambda: float = 5000.0,
        fisher_samples: int = 1000,
        online: bool = False,
        gamma: float = 0.95,
    ):
        super().__init__(model=model, lr=lr, device=device)
        self.ewc_lambda = ewc_lambda
        self.fisher_samples = fisher_samples
        self.online = online
        self.gamma = gamma

        # Storage for Fisher + optimal params per task (classical EWC)
        # or running estimates (Online EWC)
        self._fisher_diags: list[dict[str, torch.Tensor]] = []
        self._optimal_params: list[dict[str, torch.Tensor]] = []

        # Online EWC running estimates
        self._running_fisher: Optional[dict[str, torch.Tensor]] = None
        self._running_params: Optional[dict[str, torch.Tensor]] = None

    def on_task_start(self, task_id: int, train_loader: DataLoader) -> None:
        logger.info(
            f"[EWC{'++' if self.online else ''}] Starting task {task_id} "
            f"(λ={self.ewc_lambda}, fisher_samples={self.fisher_samples})"
        )

    def compute_loss(
        self, obs: torch.Tensor, act: torch.Tensor, task_id: int
    ) -> torch.Tensor:
        """MSE loss + EWC regularization penalty."""
        pred = self.model(obs)
        task_loss = nn.functional.mse_loss(pred, act)

        if task_id == 0:
            return task_loss  # No regularization for the first task

        ewc_penalty = self._compute_ewc_penalty()
        return task_loss + ewc_penalty

    def on_task_end(self, task_id: int, train_loader: DataLoader) -> None:
        """Compute Fisher Information Matrix and store optimal parameters."""
        logger.info(f"[EWC] Computing Fisher Information for task {task_id}...")
        fisher = self._compute_fisher(train_loader)

        if self.online:
            if self._running_fisher is None:
                self._running_fisher = fisher
                self._running_params = {
                    n: p.clone().detach()
                    for n, p in self.model.named_parameters()
                    if p.requires_grad
                }
            else:
                # Exponential moving average of Fisher
                for name in self._running_fisher:
                    self._running_fisher[name] = (
                        self.gamma * self._running_fisher[name] + fisher[name]
                    )
                self._running_params = {
                    n: p.clone().detach()
                    for n, p in self.model.named_parameters()
                    if p.requires_grad
                }
        else:
            self._fisher_diags.append(fisher)
            self._optimal_params.append(
                {
                    n: p.clone().detach()
                    for n, p in self.model.named_parameters()
                    if p.requires_grad
                }
            )

    def _compute_fisher(self, data_loader: DataLoader) -> dict[str, torch.Tensor]:
        """Estimate diagonal Fisher Information using empirical Fisher."""
        fisher = {
            n: torch.zeros_like(p)
            for n, p in self.model.named_parameters()
            if p.requires_grad
        }

        self.model.eval()
        n_samples = 0

        for batch in data_loader:
            obs = batch[0].to(self.device)
            act = batch[1].to(self.device)

            self.model.zero_grad()
            pred = self.model(obs)
            loss = nn.functional.mse_loss(pred, act)
            loss.backward()

            for n, p in self.model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += p.grad.data.pow(2) * obs.size(0)

            n_samples += obs.size(0)
            if n_samples >= self.fisher_samples:
                break

        # Normalize
        for n in fisher:
            fisher[n] /= max(n_samples, 1)

        self.model.train()
        return fisher

    def _compute_ewc_penalty(self) -> torch.Tensor:
        """Compute the EWC quadratic penalty across all previous tasks."""
        penalty = torch.tensor(0.0, device=self.device)

        if self.online and self._running_fisher is not None:
            for n, p in self.model.named_parameters():
                if n in self._running_fisher:
                    diff = p - self._running_params[n]
                    penalty += (self._running_fisher[n] * diff.pow(2)).sum()
        else:
            for fisher, opt_params in zip(self._fisher_diags, self._optimal_params):
                for n, p in self.model.named_parameters():
                    if n in fisher:
                        diff = p - opt_params[n]
                        penalty += (fisher[n] * diff.pow(2)).sum()

        return (self.ewc_lambda / 2.0) * penalty
