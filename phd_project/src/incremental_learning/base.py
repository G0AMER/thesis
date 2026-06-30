"""
Incremental Learning — Base Classes
====================================
Defines the abstract interfaces for:
  - ContinualLearner: wraps a PyTorch model with task-boundary hooks
  - ImitationPolicy: maps observations → actions (used with BC, DAgger, …)
  - Metrics helpers for backward/forward transfer, average accuracy

All concrete strategies (EWC, DER++, SI, LwF, PackNet) inherit from ContinualLearner.
"""

from __future__ import annotations

import abc
import copy
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@dataclass
class TaskResult:
    """Stores evaluation results for one task after training on a sequence."""
    task_id: int
    loss: float
    mse: float          # Mean Squared Error on action prediction
    mae: float          # Mean Absolute Error
    r2: float           # R² score
    n_samples: int = 0


@dataclass
class ContinualMetrics:
    """
    Tracks the full R[i,j] accuracy matrix where R[i,j] = performance on
    task j after training on task i.

    From this matrix we derive:
      - Average Accuracy (AA)
      - Backward Transfer (BWT)  — how much old tasks degrade
      - Forward Transfer (FWT)   — how well new tasks benefit from old knowledge
    """
    n_tasks: int = 0
    # R[i][j] = metric on task_j after training up to task_i
    accuracy_matrix: list[list[float]] = field(default_factory=list)
    task_names: list[str] = field(default_factory=list)

    def record(self, trained_up_to: int, task_results: list[TaskResult]):
        """Record evaluation results after training on task `trained_up_to`."""
        row = [r.mse for r in task_results]
        while len(self.accuracy_matrix) <= trained_up_to:
            self.accuracy_matrix.append([])
        self.accuracy_matrix[trained_up_to] = row

    @property
    def average_accuracy(self) -> float:
        """Average performance on all seen tasks after training on the last task."""
        if not self.accuracy_matrix:
            return float('nan')
        last_row = self.accuracy_matrix[-1]
        return float(np.mean(last_row)) if last_row else float('nan')

    @property
    def backward_transfer(self) -> float:
        """
        BWT = (1/T-1) * Σ_{j<T} [R[T,j] - R[j,j]]
        Negative BWT = forgetting. Closer to 0 = better.
        """
        T = len(self.accuracy_matrix)
        if T < 2:
            return 0.0
        bwt_sum = 0.0
        count = 0
        for j in range(T - 1):
            if j < len(self.accuracy_matrix[T - 1]) and j < len(self.accuracy_matrix[j]):
                # For MSE: lower is better, so forgetting = R[T,j] - R[j,j] > 0 means worse
                bwt_sum += self.accuracy_matrix[T - 1][j] - self.accuracy_matrix[j][j]
                count += 1
        return bwt_sum / count if count > 0 else 0.0

    @property
    def forward_transfer(self) -> float:
        """
        FWT = (1/T-1) * Σ_{j>0} [R[j-1,j] - R_baseline[j]]
        Where R_baseline is performance without any prior training.
        For now we approximate R_baseline as the diagonal (single-task).
        """
        T = len(self.accuracy_matrix)
        if T < 2:
            return 0.0
        fwt_sum = 0.0
        count = 0
        for j in range(1, T):
            if j < len(self.accuracy_matrix[j - 1]) and j < len(self.accuracy_matrix[j]):
                fwt_sum += self.accuracy_matrix[j - 1][j] - self.accuracy_matrix[j][j]
                count += 1
        return fwt_sum / count if count > 0 else 0.0

    def summary_dict(self) -> dict:
        return {
            "n_tasks": len(self.accuracy_matrix),
            "average_accuracy_mse": self.average_accuracy,
            "backward_transfer": self.backward_transfer,
            "forward_transfer": self.forward_transfer,
            "accuracy_matrix": self.accuracy_matrix,
            "task_names": self.task_names,
        }


# ---------------------------------------------------------------------------
# Base Policy Network
# ---------------------------------------------------------------------------

class MLPPolicy(nn.Module):
    """
    Simple MLP policy: observation → action (continuous).
    Used as the backbone for all IL methods.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dims: tuple[int, ...] = (256, 256),
        dropout: float = 0.1,
        activation: str = "relu",
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        layers = []
        in_dim = obs_dim
        act_fn = nn.ReLU if activation == "relu" else nn.Tanh
        for h in hidden_dims:
            layers.extend([nn.Linear(in_dim, h), act_fn(), nn.Dropout(dropout)])
            in_dim = h
        layers.append(nn.Linear(in_dim, act_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


# ---------------------------------------------------------------------------
# Abstract Continual Learner
# ---------------------------------------------------------------------------

class ContinualLearner(abc.ABC):
    """
    Base class for all continual/incremental learning strategies.

    Subclasses must implement:
      - on_task_start(task_id)   — called before training on a new task
      - compute_loss(batch)      — returns scalar loss (may include regularization)
      - on_task_end(task_id)     — called after training on a task (e.g. consolidate)
    """

    def __init__(
        self,
        model: nn.Module,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.device = device
        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        self._current_task: int = -1

    @abc.abstractmethod
    def on_task_start(self, task_id: int, train_loader: DataLoader) -> None:
        """Called before training begins on a new task."""
        ...

    @abc.abstractmethod
    def compute_loss(
        self, obs: torch.Tensor, act: torch.Tensor, task_id: int
    ) -> torch.Tensor:
        """Compute the training loss for one batch (including any CL regularization)."""
        ...

    @abc.abstractmethod
    def on_task_end(self, task_id: int, train_loader: DataLoader) -> None:
        """Called after training on a task ends (consolidation step)."""
        ...

    def train_task(
        self,
        task_id: int,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        epochs: int = 50,
        patience: int = 10,
        verbose: bool = True,
    ) -> dict:
        """
        Full training loop for one task with early stopping.

        Returns a dict with training history.
        """
        self._current_task = task_id
        self.on_task_start(task_id, train_loader)
        self.model.train()

        best_val_loss = float('inf')
        best_state = None
        epochs_no_improve = 0
        history = {"train_loss": [], "val_loss": []}

        for epoch in range(epochs):
            epoch_loss = 0.0
            n_batches = 0

            for batch in train_loader:
                obs, act = batch[0].to(self.device), batch[1].to(self.device)
                self.optimizer.zero_grad()
                loss = self.compute_loss(obs, act, task_id)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            avg_train = epoch_loss / max(n_batches, 1)
            history["train_loss"].append(avg_train)

            # Validation
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

        # Restore best model
        if best_state is not None:
            self.model.load_state_dict(best_state)

        self.on_task_end(task_id, train_loader)

        return history

    def _evaluate_loss(self, loader: DataLoader, task_id: int) -> float:
        self.model.eval()
        total_loss = 0.0
        n = 0
        with torch.no_grad():
            for batch in loader:
                obs, act = batch[0].to(self.device), batch[1].to(self.device)
                pred = self.model(obs)
                loss = nn.functional.mse_loss(pred, act)
                total_loss += loss.item() * obs.size(0)
                n += obs.size(0)
        self.model.train()
        return total_loss / max(n, 1)

    def evaluate_task(self, test_loader: DataLoader, task_id: int) -> TaskResult:
        """Evaluate current model on a specific task's test data."""
        self.model.eval()
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in test_loader:
                obs = batch[0].to(self.device)
                act = batch[1].to(self.device)
                pred = self.model(obs)
                all_preds.append(pred.cpu())
                all_targets.append(act.cpu())

        preds = torch.cat(all_preds)
        targets = torch.cat(all_targets)

        mse = nn.functional.mse_loss(preds, targets).item()
        mae = (preds - targets).abs().mean().item()

        # R² score
        ss_res = ((targets - preds) ** 2).sum().item()
        ss_tot = ((targets - targets.mean(dim=0)) ** 2).sum().item()
        r2 = 1.0 - ss_res / max(ss_tot, 1e-8)

        self.model.train()
        return TaskResult(
            task_id=task_id,
            loss=mse,
            mse=mse,
            mae=mae,
            r2=r2,
            n_samples=len(targets),
        )

    def save(self, path: Path):
        """Save model + optimizer state."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "current_task": self._current_task,
            },
            path,
        )

    def load(self, path: Path):
        """Load model + optimizer state."""
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self._current_task = ckpt.get("current_task", -1)
