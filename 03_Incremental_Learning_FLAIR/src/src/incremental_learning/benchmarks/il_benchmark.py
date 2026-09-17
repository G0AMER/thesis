"""
Incremental Learning Benchmark Runner
=======================================
Orchestrates running Experiment 1: sequential training across HARMONIC
participants, evaluating each CL strategy on backward/forward transfer.

The benchmark:
  1. Loads preprocessed HARMONIC data (Parquet files)
  2. Defines "tasks" as individual participants (each participant = 1 task)
  3. Creates state/action tensors (obs → joystick, state → joints+gaze+...)
  4. Trains each CL method sequentially: task_0, task_1, ..., task_N
  5. After each task, evaluates on ALL tasks to build the accuracy matrix
  6. Computes AA, BWT, FWT and generates comparison tables
"""

from __future__ import annotations

import copy
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from src.incremental_learning.base import (
    ContinualLearner,
    ContinualMetrics,
    MLPPolicy,
    TaskResult,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading for experiments
# ---------------------------------------------------------------------------

@dataclass
class TaskData:
    """A single task (participant) worth of state-action data."""
    task_id: int
    participant_id: str
    obs_train: torch.Tensor
    act_train: torch.Tensor
    obs_val: torch.Tensor
    act_val: torch.Tensor
    obs_test: torch.Tensor
    act_test: torch.Tensor
    n_train: int = 0
    n_val: int = 0
    n_test: int = 0

    def __post_init__(self):
        self.n_train = self.obs_train.shape[0]
        self.n_val = self.obs_val.shape[0]
        self.n_test = self.obs_test.shape[0]

    def train_loader(self, batch_size: int = 256, shuffle: bool = True) -> DataLoader:
        ds = TensorDataset(self.obs_train, self.act_train)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)

    def val_loader(self, batch_size: int = 512) -> DataLoader:
        ds = TensorDataset(self.obs_val, self.act_val)
        return DataLoader(ds, batch_size=batch_size, shuffle=False)

    def test_loader(self, batch_size: int = 512) -> DataLoader:
        ds = TensorDataset(self.obs_test, self.act_test)
        return DataLoader(ds, batch_size=batch_size, shuffle=False)


def load_participant_data(
    processed_dir: Path,
    participant: str,
    state_modalities: list[str] | None = None,
    action_modality: str = "ada_joy",
    val_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Load all preprocessed trials for one participant, concat into tensors.

    State = concat of all state_modalities columns → obs
    Action = action_modality columns → act

    Returns: (obs_train, act_train, obs_val, act_val, obs_test, act_test)
    """
    if state_modalities is None:
        state_modalities = ["joint_positions", "gaze_positions", "robot_position"]

    pdir = Path(processed_dir) / participant
    if not pdir.exists():
        raise FileNotFoundError(f"Participant directory not found: {pdir}")

    all_obs = []
    all_act = []

    for trial_dir in sorted(pdir.iterdir()):
        if not trial_dir.is_dir():
            continue
        # Load action
        act_path = trial_dir / f"{action_modality}.parquet"
        if not act_path.exists():
            continue

        act_df = pd.read_parquet(act_path)
        # Drop time column if present
        act_cols = [c for c in act_df.columns if c not in ("time_s", "timestamp", "rosbag_timestamp")]
        if not act_cols:
            continue
        act_arr = act_df[act_cols].values

        # Load state modalities
        state_frames = []
        skip = False
        for mod in state_modalities:
            mod_path = trial_dir / f"{mod}.parquet"
            if not mod_path.exists():
                skip = True
                break
            df = pd.read_parquet(mod_path)
            cols = [c for c in df.columns if c not in ("time_s", "timestamp", "rosbag_timestamp")]
            state_frames.append(df[cols].values)

        if skip or not state_frames:
            continue

        # The modalities are already aligned (same row count) by the pipeline
        obs_arr = np.concatenate(state_frames, axis=1)

        # Ensure same row count
        n_rows = min(obs_arr.shape[0], act_arr.shape[0])
        obs_arr = obs_arr[:n_rows]
        act_arr = act_arr[:n_rows]

        # Remove NaN rows
        valid = ~(np.isnan(obs_arr).any(axis=1) | np.isnan(act_arr).any(axis=1))
        obs_arr = obs_arr[valid]
        act_arr = act_arr[valid]

        if len(obs_arr) > 0:
            all_obs.append(obs_arr)
            all_act.append(act_arr)

    if not all_obs:
        raise ValueError(f"No valid data for participant {participant}")

    obs = np.concatenate(all_obs, axis=0).astype(np.float32)
    act = np.concatenate(all_act, axis=0).astype(np.float32)

    # Replace any remaining inf
    obs = np.nan_to_num(obs, nan=0.0, posinf=1e6, neginf=-1e6)
    act = np.nan_to_num(act, nan=0.0, posinf=1e6, neginf=-1e6)

    # Split into train / val / test
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(obs))
    n_val = int(len(obs) * val_ratio)
    n_test = n_val  # val_ratio for both val and test

    test_idx = indices[:n_test]
    val_idx = indices[n_test : n_test + n_val]
    train_idx = indices[n_test + n_val :]

    obs_t = torch.from_numpy(obs)
    act_t = torch.from_numpy(act)

    return (
        obs_t[train_idx], act_t[train_idx],
        obs_t[val_idx], act_t[val_idx],
        obs_t[test_idx], act_t[test_idx],
    )


def build_task_sequence(
    processed_dir: Path,
    participants: list[str],
    state_modalities: list[str] | None = None,
    action_modality: str = "ada_joy",
    normalize: bool = True,
    seed: int = 42,
) -> tuple[list[TaskData], dict]:
    """
    Build a sequence of tasks (one per participant).

    Parameters
    ----------
    processed_dir : path to data/processed/harmonic/
    participants : ordered list of participant IDs
    state_modalities : which parquet modalities to use as observation features
    action_modality : which modality is the action (target)
    normalize : whether to z-score normalize globally

    Returns
    -------
    tasks : list of TaskData
    info : dict with obs_dim, act_dim, normalization stats
    """
    tasks: list[TaskData] = []
    info: dict[str, Any] = {}

    for tid, pid in enumerate(participants):
        try:
            data = load_participant_data(
                processed_dir, pid, state_modalities, action_modality, seed=seed
            )
            td = TaskData(
                task_id=tid,
                participant_id=pid,
                obs_train=data[0], act_train=data[1],
                obs_val=data[2], act_val=data[3],
                obs_test=data[4], act_test=data[5],
            )
            tasks.append(td)
            logger.info(
                f"Task {tid} ({pid}): train={td.n_train}, val={td.n_val}, test={td.n_test}"
            )
        except (FileNotFoundError, ValueError) as e:
            logger.warning(f"Skipping participant {pid}: {e}")

    if not tasks:
        raise ValueError("No valid task data loaded")

    # Get dimensions
    obs_dim = tasks[0].obs_train.shape[1]
    act_dim = tasks[0].act_train.shape[1]
    info["obs_dim"] = obs_dim
    info["act_dim"] = act_dim
    info["n_tasks"] = len(tasks)
    info["participants"] = [t.participant_id for t in tasks]

    # Global normalization
    if normalize:
        all_obs = torch.cat([t.obs_train for t in tasks])
        all_act = torch.cat([t.act_train for t in tasks])

        obs_mean = all_obs.mean(dim=0)
        obs_std = all_obs.std(dim=0).clamp(min=1e-6)
        act_mean = all_act.mean(dim=0)
        act_std = all_act.std(dim=0).clamp(min=1e-6)

        info["obs_mean"] = obs_mean.tolist()
        info["obs_std"] = obs_std.tolist()
        info["act_mean"] = act_mean.tolist()
        info["act_std"] = act_std.tolist()

        for t in tasks:
            t.obs_train = (t.obs_train - obs_mean) / obs_std
            t.obs_val = (t.obs_val - obs_mean) / obs_std
            t.obs_test = (t.obs_test - obs_mean) / obs_std
            t.act_train = (t.act_train - act_mean) / act_std
            t.act_val = (t.act_val - act_mean) / act_std
            t.act_test = (t.act_test - act_mean) / act_std

    return tasks, info


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    """Stores results for one CL strategy across the full task sequence."""
    strategy_name: str
    metrics: ContinualMetrics
    training_time: float = 0.0
    per_task_history: list[dict] = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "strategy": self.strategy_name,
            "training_time_s": round(self.training_time, 1),
            **self.metrics.summary_dict(),
        }


def run_benchmark(
    strategy_name: str,
    learner: ContinualLearner,
    tasks: list[TaskData],
    epochs_per_task: int = 50,
    patience: int = 10,
    batch_size: int = 256,
    verbose: bool = True,
) -> BenchmarkResult:
    """
    Run a single CL strategy across the full task sequence.
    After each task, evaluate on ALL tasks (builds the accuracy matrix).
    """
    metrics = ContinualMetrics(
        n_tasks=len(tasks),
        task_names=[t.participant_id for t in tasks],
    )
    result = BenchmarkResult(strategy_name=strategy_name, metrics=metrics)

    t0 = time.time()

    for task in tasks:
        if verbose:
            print(f"\n{'='*50}")
            print(f"[{strategy_name}] Training on Task {task.task_id} ({task.participant_id})")
            print(f"  Samples: train={task.n_train}, val={task.n_val}")
            print(f"{'='*50}")

        train_loader = task.train_loader(batch_size=batch_size)
        val_loader = task.val_loader(batch_size=batch_size * 2)

        history = learner.train_task(
            task_id=task.task_id,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs_per_task,
            patience=patience,
            verbose=verbose,
        )
        result.per_task_history.append(history)

        # Evaluate on ALL tasks seen so far (+ future tasks if desired)
        task_results = []
        for eval_task in tasks:
            test_loader = eval_task.test_loader(batch_size=batch_size * 2)
            tr = learner.evaluate_task(test_loader, eval_task.task_id)
            task_results.append(tr)

            if verbose and eval_task.task_id <= task.task_id:
                status = "current" if eval_task.task_id == task.task_id else "previous"
                print(
                    f"  Eval Task {eval_task.task_id} ({eval_task.participant_id}) "
                    f"[{status}]: MSE={tr.mse:.6f}, MAE={tr.mae:.4f}, R²={tr.r2:.4f}"
                )

        metrics.record(task.task_id, task_results)

    result.training_time = time.time() - t0

    if verbose:
        print(f"\n{'='*50}")
        print(f"[{strategy_name}] Final Summary")
        print(f"  Average MSE: {metrics.average_accuracy:.6f}")
        print(f"  Backward Transfer: {metrics.backward_transfer:+.6f}")
        print(f"  Forward Transfer: {metrics.forward_transfer:+.6f}")
        print(f"  Total time: {result.training_time:.1f}s")
        print(f"{'='*50}")

    return result


def run_all_strategies(
    strategies: dict[str, ContinualLearner],
    tasks: list[TaskData],
    epochs_per_task: int = 50,
    patience: int = 10,
    batch_size: int = 256,
    verbose: bool = True,
) -> dict[str, BenchmarkResult]:
    """
    Run all CL strategies and return comparison results.
    Each learner gets a fresh copy of the model for fairness.
    """
    results = {}

    for name, learner in strategies.items():
        print(f"\n{'#'*60}")
        print(f"# Strategy: {name}")
        print(f"{'#'*60}")

        result = run_benchmark(
            strategy_name=name,
            learner=learner,
            tasks=tasks,
            epochs_per_task=epochs_per_task,
            patience=patience,
            batch_size=batch_size,
            verbose=verbose,
        )
        results[name] = result

    return results


def results_to_dataframe(results: dict[str, BenchmarkResult]) -> pd.DataFrame:
    """Convert benchmark results to a comparison DataFrame."""
    rows = []
    for name, res in results.items():
        rows.append({
            "Strategy": name,
            "Avg MSE (↓)": f"{res.metrics.average_accuracy:.6f}",
            "BWT (→0)": f"{res.metrics.backward_transfer:+.6f}",
            "FWT (↓)": f"{res.metrics.forward_transfer:+.6f}",
            "Time (s)": f"{res.training_time:.1f}",
        })
    return pd.DataFrame(rows)


def save_benchmark_results(
    results: dict[str, BenchmarkResult],
    output_dir: Path,
) -> None:
    """Save all benchmark results to JSON files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, res in results.items():
        path = output_dir / f"{name.lower().replace(' ', '_')}_results.json"
        with open(path, "w") as f:
            json.dump(res.summary(), f, indent=2)

    # Summary comparison
    comparison = {name: res.summary() for name, res in results.items()}
    with open(output_dir / "comparison_summary.json", "w") as f:
        json.dump(comparison, f, indent=2)
