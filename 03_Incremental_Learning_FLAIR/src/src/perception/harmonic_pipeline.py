"""
HARMONIC Preprocessing Pipeline
================================
End-to-end pipeline that reads raw HARMONIC trials, applies all preprocessing
steps, and saves the result as structured Parquet/HDF5 files ready for training.

Usage:
    python -m src.perception.harmonic_pipeline --harmonic-root /path/to/harmonic \\
           --output-dir data/processed/harmonic --target-rate 50

Pipeline steps per trial:
    1. Load all core modalities (ada_joy, gaze, joints, robot_position, EMG, IMU)
    2. Align timestamps to a common time grid (default 50 Hz)
    3. Preprocess EMG (filter → rectify → RMS → z-score)
    4. Preprocess gaze (confidence filter → interpolate → smooth)
    5. Preprocess IMU (low-pass filter)
    6. Save aligned & preprocessed data as Parquet files
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.perception.preprocessing import (
    PreprocessConfig,
    align_timestamps,
    compute_participant_stats,
    normalize_dataframe,
    preprocess_emg,
    preprocess_gaze,
    preprocess_imu,
)
from src.utils.data_loading.harmonic_loader import (
    CORE_MODALITIES,
    TrialData,
    TrialInfo,
    discover_participants,
    discover_trials,
    get_data_columns,
    load_trial,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


def preprocess_trial(
    trial_data: TrialData,
    config: PreprocessConfig,
) -> dict[str, pd.DataFrame]:
    """
    Apply the full preprocessing pipeline to a single trial.

    Returns a dict of modality name → preprocessed DataFrame,
    all aligned to a common time grid.
    """
    modalities = trial_data.modalities

    # Step 1: Align timestamps
    aligned = align_timestamps(
        modalities,
        target_rate_hz=config.target_rate_hz,
        timestamp_col="timestamp",
    )

    # Step 2: Preprocess EMG (if available)
    if "myo_emg" in aligned and not aligned["myo_emg"].empty:
        aligned["myo_emg"] = preprocess_emg(aligned["myo_emg"], config)

    # Step 3: Preprocess gaze
    if "gaze_positions" in aligned and not aligned["gaze_positions"].empty:
        aligned["gaze_positions"] = preprocess_gaze(aligned["gaze_positions"], config)

    # Step 4: Preprocess IMU
    if "myo_imu" in aligned and not aligned["myo_imu"].empty:
        aligned["myo_imu"] = preprocess_imu(aligned["myo_imu"], config)

    return aligned


def save_preprocessed_trial(
    aligned: dict[str, pd.DataFrame],
    trial_info: TrialInfo,
    output_dir: Path,
) -> Path:
    """
    Save preprocessed trial data as Parquet files.

    Directory structure:
        output_dir/
            <participant>/
                <trial_type>_<trial_id>/
                    ada_joy.parquet
                    gaze_positions.parquet
                    ...
                    metadata.json
    """
    trial_dir = output_dir / trial_info.participant / f"{trial_info.trial_type}_{trial_info.trial_id}"
    trial_dir.mkdir(parents=True, exist_ok=True)

    saved_modalities = []
    for name, df in aligned.items():
        if df.empty:
            continue
        out_path = trial_dir / f"{name}.parquet"
        df.to_parquet(out_path, index=False)
        saved_modalities.append(name)

    # Save metadata
    metadata = {
        "participant": trial_info.participant,
        "trial_type": trial_info.trial_type,
        "trial_id": trial_info.trial_id,
        "trial_key": trial_info.trial_key,
        "saved_modalities": saved_modalities,
        "has_emg": trial_info.has_emg,
        "n_samples": int(aligned.get("joint_positions", pd.DataFrame()).shape[0])
        if "joint_positions" in aligned
        else 0,
    }
    with open(trial_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    return trial_dir


def run_harmonic_pipeline(
    harmonic_root: Path,
    output_dir: Path,
    config: Optional[PreprocessConfig] = None,
    participants: Optional[list[str]] = None,
    trial_type: str = "run",
    modalities: Optional[list[str]] = None,
    dry_run: bool = False,
) -> dict:
    """
    Run the full HARMONIC preprocessing pipeline.

    Parameters
    ----------
    harmonic_root : Path
        Root directory of the HARMONIC dataset.
    output_dir : Path
        Where to save preprocessed data.
    config : PreprocessConfig, optional
    participants : list[str], optional
        Process only these participants. None = all.
    trial_type : str
        'run', 'check', or 'all'.
    modalities : list[str], optional
        Which modalities to load. None = CORE_MODALITIES.
    dry_run : bool
        If True, discover trials but don't process.

    Returns
    -------
    dict
        Summary statistics of the preprocessing run.
    """
    if config is None:
        config = PreprocessConfig()
    if modalities is None:
        modalities = CORE_MODALITIES

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover trials
    trials = discover_trials(
        harmonic_root,
        trial_type=trial_type,
        participants=participants,
    )

    logger.info(f"Pipeline: {len(trials)} trials to process")

    if dry_run:
        return {
            "n_trials": len(trials),
            "participants": sorted(set(t.participant for t in trials)),
            "trials_with_emg": sum(1 for t in trials if t.has_emg),
        }

    # Process each trial
    summary = {
        "n_trials": len(trials),
        "n_processed": 0,
        "n_errors": 0,
        "n_with_emg": 0,
        "participants": [],
        "errors": [],
    }

    t0 = time.time()
    for i, trial_info in enumerate(trials):
        try:
            # Load
            trial_data = load_trial(trial_info, modalities=modalities)

            # Preprocess
            aligned = preprocess_trial(trial_data, config)

            # Save
            save_preprocessed_trial(aligned, trial_info, output_dir)

            summary["n_processed"] += 1
            if trial_info.has_emg:
                summary["n_with_emg"] += 1

            if (i + 1) % 50 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                logger.info(
                    f"  [{i+1}/{len(trials)}] {rate:.1f} trials/s — "
                    f"{trial_info.trial_key}"
                )

        except Exception as e:
            summary["n_errors"] += 1
            summary["errors"].append(
                {"trial": trial_info.trial_key, "error": str(e)}
            )
            logger.error(f"Error processing {trial_info.trial_key}: {e}")

    elapsed = time.time() - t0
    summary["participants"] = sorted(set(t.participant for t in trials))
    summary["elapsed_seconds"] = round(elapsed, 1)

    # Save pipeline summary
    with open(output_dir / "pipeline_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(
        f"Pipeline complete: {summary['n_processed']}/{len(trials)} trials "
        f"in {elapsed:.1f}s ({summary['n_errors']} errors)"
    )

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="HARMONIC dataset preprocessing pipeline"
    )
    parser.add_argument(
        "--harmonic-root",
        type=Path,
        required=True,
        help="Root directory of the HARMONIC dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/harmonic"),
        help="Output directory for preprocessed data",
    )
    parser.add_argument(
        "--target-rate",
        type=float,
        default=50.0,
        help="Target resampling rate in Hz (default: 50)",
    )
    parser.add_argument(
        "--participants",
        nargs="+",
        default=None,
        help="Process only these participants (e.g., p100 p101)",
    )
    parser.add_argument(
        "--trial-type",
        choices=["run", "check", "all"],
        default="run",
        help="Trial type to process (default: run)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only discover trials, don't process",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose logging",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    config = PreprocessConfig(target_rate_hz=args.target_rate)

    summary = run_harmonic_pipeline(
        harmonic_root=args.harmonic_root,
        output_dir=args.output_dir,
        config=config,
        participants=args.participants,
        trial_type=args.trial_type,
        dry_run=args.dry_run,
    )

    print(f"\n{'='*60}")
    print(f"HARMONIC Preprocessing Summary")
    print(f"{'='*60}")
    print(f"  Trials processed: {summary.get('n_processed', summary.get('n_trials'))}")
    print(f"  Errors: {summary.get('n_errors', 0)}")
    print(f"  With EMG: {summary.get('n_with_emg', summary.get('trials_with_emg'))}")
    print(f"  Participants: {len(summary.get('participants', []))}")
    if "elapsed_seconds" in summary:
        print(f"  Time: {summary['elapsed_seconds']}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
