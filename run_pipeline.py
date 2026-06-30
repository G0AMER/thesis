#!/usr/bin/env python3
"""
Cobot Safety-State Detection — Full Pipeline
==============================================
End-to-end pipeline from raw DASIG data to trained models and evaluation.

Usage:
    python run_pipeline.py [--data-dir PATH] [--output-dir PATH] [--quick]

Options:
    --data-dir    Path to the DASIG directory (default: data/multiphysio_hrc/DASIG)
    --output-dir  Where to save results (default: outputs/cobot_safety)
    --quick       Use only 10 subjects for fast testing
"""

import os
import sys
import argparse
import time
import numpy as np

# Add parent to path so we can import the package
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cobot_safety_model.data_loader import (
    load_all_trials, split_by_subject, NUM_SUBJECTS,
)
from cobot_safety_model.features import (
    segment_all_trials, extract_features_bulk, normalize_features,
)
from cobot_safety_model.models import (
    train_random_forest, train_gradient_boosting, evaluate_model,
    to_binary_labels, CLASS_NAMES, BINARY_CLASS_NAMES,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Cobot Safety-State Detection Pipeline")
    parser.add_argument(
        "--data-dir",
        default="data/multiphysio_hrc/DASIG",
        help="Path to DASIG root directory",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/cobot_safety",
        help="Directory to save results",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: use only 10 subjects",
    )
    parser.add_argument(
        "--window-size",
        type=float,
        default=1.0,
        help="Sliding window size in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--step-size",
        type=float,
        default=0.5,
        help="Sliding window step in seconds (default: 0.5)",
    )
    parser.add_argument(
        "--binary",
        action="store_true",
        help="Use binary classification (SAFE vs ABRUPT) instead of 3-class",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("  COBOT SAFETY-STATE DETECTION PIPELINE")
    print("  Based on DASIG dataset (Digo et al., Robotics 2025)")
    print("=" * 70)

    # ── Step 1: Load Data ──────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("STEP 1: Loading DASIG dataset")
    print("─" * 70)

    subjects = None
    if args.quick:
        subjects = [f"sub{i:03d}" for i in range(1, 11)]
        print(f"  Quick mode: loading {len(subjects)} subjects only")

    t0 = time.time()
    all_trials = load_all_trials(
        args.data_dir,
        subjects=subjects,
        generate_labels=True,
    )
    print(f"  Data loading took {time.time() - t0:.1f}s")

    # ── Step 2: Split by Subject ───────────────────────────────────────
    print("\n" + "─" * 70)
    print("STEP 2: Train/Val/Test split (by subject)")
    print("─" * 70)

    train_trials, val_trials, test_trials = split_by_subject(
        all_trials, train_ratio=0.6, val_ratio=0.2, seed=42
    )

    # ── Step 3: Sliding Window Segmentation ────────────────────────────
    print("\n" + "─" * 70)
    print(f"STEP 3: Sliding window segmentation "
          f"(window={args.window_size}s, step={args.step_size}s)")
    print("─" * 70)

    t0 = time.time()
    X_train_raw, y_train = segment_all_trials(
        train_trials, args.window_size, args.step_size, label_strategy="any_danger"
    )
    X_val_raw, y_val = segment_all_trials(
        val_trials, args.window_size, args.step_size, label_strategy="any_danger"
    )
    X_test_raw, y_test = segment_all_trials(
        test_trials, args.window_size, args.step_size, label_strategy="any_danger"
    )
    print(f"  Segmentation took {time.time() - t0:.1f}s")

    # ── Step 4: Feature Extraction ─────────────────────────────────────
    print("\n" + "─" * 70)
    print("STEP 4: Feature extraction")
    print("─" * 70)

    t0 = time.time()
    X_train_feat, feature_names = extract_features_bulk(X_train_raw)
    X_val_feat, _ = extract_features_bulk(X_val_raw, verbose=False)
    X_test_feat, _ = extract_features_bulk(X_test_raw, verbose=False)
    print(f"  Feature extraction took {time.time() - t0:.1f}s")

    # Handle NaN/Inf values
    for arr in [X_train_feat, X_val_feat, X_test_feat]:
        arr[~np.isfinite(arr)] = 0.0

    # ── Step 5: Normalize ──────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("STEP 5: Feature normalization")
    print("─" * 70)

    X_train_norm, X_val_norm, X_test_norm, (feat_mean, feat_std) = normalize_features(
        X_train_feat, X_val_feat, X_test_feat
    )
    print(f"  Normalized {X_train_norm.shape[1]} features")

    # ── Step 6: Convert to binary if requested ─────────────────────────
    if args.binary:
        print("\n  Converting to binary labels (SAFE vs ABRUPT)...")
        y_train = to_binary_labels(y_train)
        y_val = to_binary_labels(y_val)
        y_test = to_binary_labels(y_test)
        class_names = BINARY_CLASS_NAMES
    else:
        class_names = CLASS_NAMES

    # ── Step 7: Train Models ───────────────────────────────────────────
    print("\n" + "─" * 70)
    print("STEP 6: Training models")
    print("─" * 70)

    # Random Forest
    print("\n[A] Random Forest")
    rf = train_random_forest(X_train_norm, y_train, n_estimators=300, max_depth=20)

    # Gradient Boosting
    print("\n[B] Gradient Boosting")
    gb = train_gradient_boosting(X_train_norm, y_train, n_estimators=200, max_depth=6)

    # ── Step 8: Evaluate ───────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("STEP 7: Evaluation on test set")
    print("─" * 70)

    rf_metrics = evaluate_model(
        rf, X_test_norm, y_test,
        class_names=class_names,
        output_dir=args.output_dir,
        model_name="random_forest",
    )

    gb_metrics = evaluate_model(
        gb, X_test_norm, y_test,
        class_names=class_names,
        output_dir=args.output_dir,
        model_name="gradient_boosting",
    )

    # Also evaluate on validation set
    print("\n  Validation set performance:")
    print(f"    RF  — F1 macro: {evaluate_model(rf, X_val_norm, y_val, class_names=class_names, model_name='RF_val')['f1_macro']:.4f}")
    print(f"    GB  — F1 macro: {evaluate_model(gb, X_val_norm, y_val, class_names=class_names, model_name='GB_val')['f1_macro']:.4f}")

    # ── Step 9: Feature Importance Analysis ────────────────────────────
    print("\n" + "─" * 70)
    print("STEP 8: Feature importance analysis")
    print("─" * 70)

    # Top features from Random Forest
    importances = rf.feature_importances_
    top_indices = np.argsort(importances)[::-1][:15]
    print("\n  Top 15 most important features (Random Forest):")
    for rank, idx in enumerate(top_indices, 1):
        print(f"    {rank:2d}. {feature_names[idx]:40s} {importances[idx]:.4f}")

    from cobot_safety_model.models import plot_feature_importance
    plot_feature_importance(
        importances,
        feature_names=feature_names,
        output_dir=args.output_dir,
        model_name="random_forest",
        top_n=25,
    )

    # ── Summary ────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE")
    print("=" * 70)
    print(f"\n  Results saved to: {os.path.abspath(args.output_dir)}")
    print(f"\n  Model Performance (Test Set, F1 Macro):")
    print(f"    Random Forest:      {rf_metrics['f1_macro']:.4f}")
    print(f"    Gradient Boosting:  {gb_metrics['f1_macro']:.4f}")
    print(f"\n  Output files:")
    for f in sorted(os.listdir(args.output_dir)):
        print(f"    • {f}")


if __name__ == "__main__":
    main()
