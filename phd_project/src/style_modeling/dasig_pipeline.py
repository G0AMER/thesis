"""
DASIG End-to-End Pipeline
==========================
Orchestrates loading, preprocessing, feature extraction and saving for the
full DASIG dataset.  Can be run as a CLI script or imported as a library.

Usage:
    python -m src.style_modeling.dasig_pipeline \
        --dasig-root /content/data/dasig \
        --output-dir data/processed/dasig \
        --subjects sub001 sub002 \
        --verbose
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

from src.utils.data_loading.dasig_loader import (
    CONDITIONS,
    RecordingData,
    RecordingInfo,
    discover_recordings,
    load_recording,
    load_subjects_info,
)
from src.style_modeling.dasig_preprocessing import (
    DASIGPreprocessConfig,
    compute_subject_stats,
    detect_abrupt_motions,
    extract_features_windowed,
    normalize_features,
    preprocess_mimu,
    segment_around_alarms,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------


def preprocess_recording(
    rec: RecordingData,
    config: DASIGPreprocessConfig,
) -> dict:
    """
    Full preprocessing for a single recording.

    Returns a dict with:
        - 'filtered': filtered MIMU DataFrame
        - 'alarm_segments': list of alarm-windowed DataFrames
        - 'features': windowed feature DataFrame
        - 'abrupt_events': abrupt motion events DataFrame
    """
    result = {"recording_key": rec.info.recording_key}

    if rec.mimu is None or rec.mimu.empty:
        logger.warning(f"No MIMU data for {rec.info.recording_key}")
        return result

    # 1. Estimate actual sampling rate from data
    dt = np.diff(rec.mimu["Time_s"].values)
    actual_fs = 1.0 / np.median(dt) if len(dt) > 0 else config.mimu_sampling_rate
    adjusted_config = DASIGPreprocessConfig(
        **{
            **config.__dict__,
            "mimu_sampling_rate": actual_fs,
        }
    )

    # 2. Filter
    filtered = preprocess_mimu(rec.mimu, adjusted_config)
    result["filtered"] = filtered

    # 3. Alarm-based segmentation (only for LA_L condition)
    if rec.info.condition == "LA_L" and rec.arduino is not None:
        segments = segment_around_alarms(filtered, rec.arduino, adjusted_config)
        result["alarm_segments"] = segments
    else:
        result["alarm_segments"] = []

    # 4. Extract sliding-window features from full recording
    features = extract_features_windowed(filtered, adjusted_config)
    result["features"] = features

    # 5. Detect abrupt motions
    abrupt = detect_abrupt_motions(filtered, adjusted_config)
    result["abrupt_events"] = abrupt

    return result


def save_preprocessed_recording(
    preprocessed: dict,
    output_dir: Path,
    rec_info: RecordingInfo,
) -> None:
    """Save preprocessed data to disk as Parquet files."""
    rec_dir = output_dir / rec_info.subject_id / rec_info.condition
    rec_dir.mkdir(parents=True, exist_ok=True)

    # Filtered MIMU
    if "filtered" in preprocessed and preprocessed["filtered"] is not None:
        preprocessed["filtered"].to_parquet(rec_dir / "mimu_filtered.parquet", index=False)

    # Features
    if "features" in preprocessed and isinstance(preprocessed["features"], pd.DataFrame):
        if not preprocessed["features"].empty:
            preprocessed["features"].to_parquet(rec_dir / "features.parquet", index=False)

    # Alarm segments
    if "alarm_segments" in preprocessed:
        seg_dir = rec_dir / "alarm_segments"
        seg_dir.mkdir(exist_ok=True)
        for i, seg_df in enumerate(preprocessed["alarm_segments"]):
            seg_df.to_parquet(seg_dir / f"segment_{i:03d}.parquet", index=False)

    # Abrupt events
    if "abrupt_events" in preprocessed and isinstance(
        preprocessed["abrupt_events"], pd.DataFrame
    ):
        if not preprocessed["abrupt_events"].empty:
            preprocessed["abrupt_events"].to_parquet(
                rec_dir / "abrupt_events.parquet", index=False
            )

    # Metadata
    metadata = {
        "subject_id": rec_info.subject_id,
        "condition": rec_info.condition,
        "n_features": (
            len(preprocessed["features"])
            if "features" in preprocessed
            and isinstance(preprocessed["features"], pd.DataFrame)
            else 0
        ),
        "n_alarm_segments": len(preprocessed.get("alarm_segments", [])),
        "n_abrupt_events": (
            len(preprocessed["abrupt_events"])
            if "abrupt_events" in preprocessed
            and isinstance(preprocessed["abrupt_events"], pd.DataFrame)
            else 0
        ),
    }
    with open(rec_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_dasig_pipeline(
    dasig_root: str | Path,
    output_dir: str | Path = "data/processed/dasig",
    subjects: Optional[list[str]] = None,
    conditions: Optional[list[str]] = None,
    config: Optional[DASIGPreprocessConfig] = None,
    dry_run: bool = False,
    verbose: bool = False,
) -> dict:
    """
    Run the full DASIG preprocessing pipeline.

    Parameters
    ----------
    dasig_root : Path
        Root of the DASIG dataset.
    output_dir : Path
        Where to save processed data.
    subjects : list[str], optional
        Filter to specific subjects.
    conditions : list[str], optional
        Filter to specific conditions.
    config : DASIGPreprocessConfig, optional
        Preprocessing configuration.
    dry_run : bool
        If True, discover and report but don't process.
    verbose : bool

    Returns
    -------
    dict
        Pipeline summary.
    """
    dasig_root = Path(dasig_root)
    output_dir = Path(output_dir)

    if config is None:
        config = DASIGPreprocessConfig()

    if verbose:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Discover recordings
    recordings = discover_recordings(dasig_root, subjects, conditions)
    logger.info(f"Found {len(recordings)} recordings to process")

    if dry_run:
        return {
            "mode": "dry_run",
            "n_recordings": len(recordings),
            "recordings": [r.recording_key for r in recordings],
        }

    # Process each recording
    output_dir.mkdir(parents=True, exist_ok=True)
    results_summary = {
        "n_recordings": len(recordings),
        "processed": 0,
        "failed": 0,
        "per_subject": {},
    }

    # Per-subject feature aggregation for normalization
    subject_features: dict[str, list[pd.DataFrame]] = {}

    for i, rec_info in enumerate(recordings):
        t0 = time.time()
        try:
            rec_data = load_recording(rec_info)
            preprocessed = preprocess_recording(rec_data, config)
            save_preprocessed_recording(preprocessed, output_dir, rec_info)

            # Track features per subject
            if "features" in preprocessed and isinstance(
                preprocessed["features"], pd.DataFrame
            ):
                sid = rec_info.subject_id
                if sid not in subject_features:
                    subject_features[sid] = []
                subject_features[sid].append(preprocessed["features"])

            dt = time.time() - t0
            results_summary["processed"] += 1
            logger.info(
                f"[{i + 1}/{len(recordings)}] {rec_info.recording_key} "
                f"processed in {dt:.1f}s"
            )
        except Exception as e:
            results_summary["failed"] += 1
            logger.error(f"[{i + 1}/{len(recordings)}] {rec_info.recording_key} FAILED: {e}")

    # Compute and save per-subject normalization stats
    for sid, feat_list in subject_features.items():
        stats = compute_subject_stats(feat_list)
        stats_path = output_dir / sid / "normalization_stats.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2, default=str)

        # Also save normalized features
        for condition_dir in (output_dir / sid).iterdir():
            if not condition_dir.is_dir() or condition_dir.name == "__pycache__":
                continue
            feat_path = condition_dir / "features.parquet"
            if feat_path.exists():
                feat_df = pd.read_parquet(feat_path)
                normalized = normalize_features(feat_df, stats, config.normalization_method)
                normalized.to_parquet(
                    condition_dir / "features_normalized.parquet", index=False
                )

        results_summary["per_subject"][sid] = {
            "n_conditions": len(feat_list),
            "n_feature_columns": len(stats),
        }

    # Load and attach subjects_info for reference
    subjects_df = load_subjects_info(dasig_root)
    if not subjects_df.empty:
        subjects_df.to_csv(output_dir / "subjects_info.csv", index=False)

    # Save summary
    summary_path = output_dir / "pipeline_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results_summary, f, indent=2)

    logger.info(
        f"Pipeline complete: {results_summary['processed']} OK, "
        f"{results_summary['failed']} failed"
    )
    return results_summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="DASIG preprocessing pipeline")
    parser.add_argument(
        "--dasig-root",
        type=str,
        required=True,
        help="Root directory of the DASIG dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed/dasig",
        help="Output directory for processed data",
    )
    parser.add_argument(
        "--subjects",
        nargs="+",
        default=None,
        help="Specific subjects to process (e.g., sub001 sub002)",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=None,
        choices=["FR_L", "FR_R", "LA_L"],
        help="Specific conditions to process",
    )
    parser.add_argument(
        "--accel-lowpass",
        type=float,
        default=20.0,
        help="Accelerometer low-pass cutoff (Hz)",
    )
    parser.add_argument(
        "--gyro-lowpass",
        type=float,
        default=15.0,
        help="Gyroscope low-pass cutoff (Hz)",
    )
    parser.add_argument(
        "--window-size",
        type=float,
        default=1.0,
        help="Feature extraction window size (seconds)",
    )
    parser.add_argument(
        "--window-step",
        type=float,
        default=0.5,
        help="Feature extraction window step (seconds)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Discover only, don't process")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    config = DASIGPreprocessConfig(
        accel_lowpass_hz=args.accel_lowpass,
        gyro_lowpass_hz=args.gyro_lowpass,
        feature_window_s=args.window_size,
        feature_step_s=args.window_step,
    )

    run_dasig_pipeline(
        dasig_root=args.dasig_root,
        output_dir=args.output_dir,
        subjects=args.subjects,
        conditions=args.conditions,
        config=config,
        dry_run=args.dry_run,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
