"""
Synaptic Intelligence (SI)
===========================
Zenke et al., "Continual Learning Through Synaptic Intelligence" (ICML 2017)

Key idea: track an online estimate of parameter importance based on the
*path integral* of the gradient contribution during training. Parameters
that contributed more to loss reduction on previous tasks are penalized
more when changed.

Loss = L_task + (c/2) Σ_i Ω_i (θ_i - θ*_i)²

where Ω_i is the online importance estimate (per-param path integral).
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.incremental_learning.base import ContinualLearner

logger = logging.getLogger(__name__)


class SI(ContinualLearner):
    """
    Synaptic Intelligence (SI).

    Parameters
    ----------
    model : nn.Module
        The policy network.
    si_c : float
        Regularization coefficient (like λ in EWC).
    xi : float
        Damping term to avoid division by zero in importance computation.
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        device: str = "cpu",
        si_c: float = 1.0,
        xi: float = 1e-3,
    ):
        super().__init__(model=model, lr=lr, device=device)
        self.si_c = si_c
        self.xi = xi

        # Per-parameter online importance (Ω)
        self._omega: dict[str, torch.Tensor] = {}
        # Stored optimal params after each task (θ*)
        self._prev_params: dict[str, torch.Tensor] = {}
        # Running product of grad * param_change (ω — unnormalized importance)
        self._w: dict[str, torch.Tensor] = {}
        # Params at start of current task
        self._task_start_params: dict[str, torch.Tensor] = {}

        # Initialize
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                self._omega[n] = torch.zeros_like(p)
                self._prev_params[n] = p.clone().detach()
                self._w[n] = torch.zeros_like(p)

    def on_task_start(self, task_id: int, train_loader: DataLoader) -> None:
        logger.info(f"[SI] Starting task {task_id} (c={self.si_c})")
        # Record starting parameters for this task
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                self._task_start_params[n] = p.clone().detach()
                self._w[n] = torch.zeros_like(p)

    def compute_loss(
        self, obs: torch.Tensor, act: torch.Tensor, task_id: int
    ) -> torch.Tensor:
        """MSE loss + SI surrogate loss."""
        pred = self.model(obs)
        task_loss = nn.functional.mse_loss(pred, act)

        # Accumulate online importance (before optimizer step)
        # We track grad * (θ - θ_prev) per parameter
        # This is done in the training loop via _update_w after backward

        # SI penalty
        if task_id > 0:
            si_penalty = self._compute_si_penalty()
            return task_loss + si_penalty

        return task_loss

    def on_task_end(self, task_id: int, train_loader: DataLoader) -> None:
        """Update Ω (importance) and store optimal params."""
        logger.info(f"[SI] Computing importance for task {task_id}")

        for n, p in self.model.named_parameters():
            if p.requires_grad:
                delta = p.detach() - self._task_start_params[n]
                # Ω += max(ω / (Δθ² + ξ), 0)
                importance = self._w[n] / (delta.pow(2) + self.xi)
                self._omega[n] += torch.clamp(importance, min=0)
                # Store optimal params
                self._prev_params[n] = p.clone().detach()

    def _compute_si_penalty(self) -> torch.Tensor:
        penalty = torch.tensor(0.0, device=self.device)
        for n, p in self.model.named_parameters():
            if n in self._omega:
                diff = p - self._prev_params[n]
                penalty += (self._omega[n] * diff.pow(2)).sum()
        return (self.si_c / 2.0) * penalty

    def train_task(
        self,
        task_id: int,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 50,
        patience: int = 10,
        verbose: bool = True,
    ) -> dict:
        """Override to hook into gradient tracking for importance estimation."""
        self._current_task = task_id
        self.on_task_start(task_id, train_loader)
        self.model.train()

        import copy

        best_val_loss = float("inf")
        best_state = None
        epochs_no_improve = 0
        history = {"train_loss": [], "val_loss": []}

        for epoch in range(epochs):
            epoch_loss = 0.0
            n_batches = 0

            for batch in train_loader:
                obs, act = batch[0].to(self.device), batch[1].to(self.device)

                # Store params before step
                prev_p = {
                    n: p.clone().detach()
                    for n, p in self.model.named_parameters()
                    if p.requires_grad
                }

                self.optimizer.zero_grad()
                loss = self.compute_loss(obs, act, task_id)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                # Track gradient × parameter change for SI
                for n, p in self.model.named_parameters():
                    if p.requires_grad and p.grad is not None:
                        # We use -grad (direction of descent) as surrogate for contribution
                        self._w[n] += (-p.grad.detach()) * (p.detach() - prev_p[n] + 1e-20)

                self.optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            avg_train = epoch_loss / max(n_batches, 1)
            history["train_loss"].append(avg_train)

            if val_loader is not None:
                val_loss = self._evaluate_loss(val_loader, task_id)
                history["val_loss"].append(val_loss)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = copy.deepcopy(self.model.state_dict())
                    epochs_no_improve = 0
                else:
                    epochs_no_improve += 1
                if verbose and (epoch + 1) % 10 == 0:
                    logger.info(
                        f"  Task {task_id} | Epoch {epoch+1}/{epochs} | "
                        f"Train: {avg_train:.6f} | Val: {val_loss:.6f}"
                    )
                if epochs_no_improve >= patience:
                    if verbose:
                        logger.info(f"  Early stopping at epoch {epoch+1}")
                    break
            else:
                if verbose and (epoch + 1) % 10 == 0:
                    logger.info(
                        f"  Task {task_id} | Epoch {epoch+1}/{epochs} | "
                        f"Train: {avg_train:.6f}"
                    )

        if best_state is not None:
            self.model.load_state_dict(best_state)

        self.on_task_end(task_id, train_loader)
        return history
