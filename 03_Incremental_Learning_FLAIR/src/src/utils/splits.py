"""
Dataset Split Utilities
========================
Subject-level stratified train / validation / test splits for both
HARMONIC and DASIG datasets.

Design principles:
  - Splits are always at the **subject** level so no data leakage.
  - Reproducible via fixed random seeds.
  - Support leave-N-subjects-out for cross-validation.
  - Persist split definitions as JSON for reproducibility.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core splitting logic
# ---------------------------------------------------------------------------


def subject_stratified_split(
    subject_ids: list[str],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    stratify_labels: Optional[list[str]] = None,
) -> dict[str, list[str]]:
    """
    Split subjects into train / val / test sets.

    Parameters
    ----------
    subject_ids : list[str]
        List of subject identifiers.
    train_ratio, val_ratio, test_ratio : float
        Target proportions (should sum to 1.0).
    seed : int
        Random seed for reproducibility.
    stratify_labels : list[str], optional
        Labels for stratification (e.g., gender, age group).  If provided,
        each stratum is split proportionally.

    Returns
    -------
    dict with keys 'train', 'val', 'test', each a list of subject IDs.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, (
        f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"
    )

    rng = np.random.RandomState(seed)

    if stratify_labels is None:
        return _split_ids(subject_ids, train_ratio, val_ratio, rng)
    else:
        assert len(stratify_labels) == len(subject_ids)
        # Group by stratum
        strata: dict[str, list[str]] = {}
        for sid, label in zip(subject_ids, stratify_labels):
            strata.setdefault(label, []).append(sid)

        combined = {"train": [], "val": [], "test": []}
        for label, ids in sorted(strata.items()):
            sub_split = _split_ids(ids, train_ratio, val_ratio, rng)
            for key in combined:
                combined[key].extend(sub_split[key])

        return combined


def _split_ids(
    ids: list[str], train_ratio: float, val_ratio: float, rng: np.random.RandomState
) -> dict[str, list[str]]:
    """Split a list of IDs into train/val/test."""
    shuffled = list(ids)
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = max(1, round(n * train_ratio))
    n_val = max(1, round(n * val_ratio))
    # Test gets the remainder
    n_test = n - n_train - n_val
    if n_test < 1:
        n_val = max(1, n - n_train - 1)
        n_test = n - n_train - n_val

    return {
        "train": sorted(shuffled[:n_train]),
        "val": sorted(shuffled[n_train : n_train + n_val]),
        "test": sorted(shuffled[n_train + n_val :]),
    }


def leave_n_subjects_out(
    subject_ids: list[str],
    n_out: int = 1,
    seed: int = 42,
) -> list[dict[str, list[str]]]:
    """
    Generate leave-N-subjects-out folds.

    Each fold has N subjects in the 'test' set and the rest in 'train'.
    No validation set — use for final evaluation or small-N settings.

    Returns a list of fold dicts.
    """
    rng = np.random.RandomState(seed)
    shuffled = list(subject_ids)
    rng.shuffle(shuffled)

    folds = []
    for start in range(0, len(shuffled), n_out):
        test_ids = sorted(shuffled[start : start + n_out])
        train_ids = sorted([s for s in shuffled if s not in test_ids])
        if test_ids:
            folds.append({"train": train_ids, "test": test_ids})

    return folds


# ---------------------------------------------------------------------------
# HARMONIC-specific splits
# ---------------------------------------------------------------------------


def split_harmonic(
    harmonic_root: Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> dict[str, list[str]]:
    """
    Create subject-level split for HARMONIC dataset.

    Discovers participant directories (p100, p101, ..., p123) and
    splits them.
    """
    from src.utils.data_loading.harmonic_loader import discover_participants

    participants = discover_participants(harmonic_root)
    logger.info(f"HARMONIC: {len(participants)} participants found")

    return subject_stratified_split(
        subject_ids=participants,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )


def split_dasig(
    dasig_root: Path,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    seed: int = 42,
    stratify_by_gender: bool = True,
) -> dict[str, list[str]]:
    """
    Create subject-level split for DASIG dataset.

    With 60 subjects, uses 60/20/20 split by default.
    Can stratify by gender if subjects_info.csv is available.
    """
    from src.utils.data_loading.dasig_loader import discover_recordings, load_subjects_info

    recordings = discover_recordings(dasig_root)
    subject_ids = sorted(set(r.subject_id for r in recordings))
    logger.info(f"DASIG: {len(subject_ids)} subjects found")

    stratify_labels = None
    if stratify_by_gender:
        info_df = load_subjects_info(dasig_root)
        if not info_df.empty:
            # Try to extract gender info for stratification
            gender_col = None
            for col in info_df.columns:
                if "gender" in col.lower() or "sex" in col.lower() or "genre" in col.lower():
                    gender_col = col
                    break
            if gender_col:
                id_col = info_df.columns[0]
                gender_map = dict(zip(info_df[id_col].astype(str), info_df[gender_col].astype(str)))
                stratify_labels = [gender_map.get(sid, "unknown") for sid in subject_ids]
                logger.info(f"Stratifying by {gender_col}")

    return subject_stratified_split(
        subject_ids=subject_ids,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
        stratify_labels=stratify_labels,
    )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_split(split: dict, path: Path) -> None:
    """Save a split definition to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(split, f, indent=2)
    logger.info(f"Split saved to {path}")


def load_split(path: Path) -> dict:
    """Load a split definition from JSON."""
    with open(path, "r") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate dataset splits")
    parser.add_argument(
        "--dataset",
        choices=["harmonic", "dasig", "both"],
        default="both",
    )
    parser.add_argument("--harmonic-root", type=str, default=None)
    parser.add_argument("--dasig-root", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="configs/splits")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    output_dir = Path(args.output_dir)

    if args.dataset in ("harmonic", "both") and args.harmonic_root:
        split = split_harmonic(Path(args.harmonic_root), seed=args.seed)
        save_split(split, output_dir / "harmonic_split.json")
        print(f"HARMONIC split: train={len(split['train'])}, "
              f"val={len(split['val'])}, test={len(split['test'])}")

    if args.dataset in ("dasig", "both") and args.dasig_root:
        split = split_dasig(Path(args.dasig_root), seed=args.seed)
        save_split(split, output_dir / "dasig_split.json")
        print(f"DASIG split: train={len(split['train'])}, "
              f"val={len(split['val'])}, test={len(split['test'])}")


if __name__ == "__main__":
    main()
