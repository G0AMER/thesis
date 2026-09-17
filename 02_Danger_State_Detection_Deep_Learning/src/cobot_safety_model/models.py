"""
Model Training & Evaluation for Cobot Safety-State Detection
==============================================================
Implements:
  1. Random Forest baseline
  2. Evaluation with per-class metrics, confusion matrix, and latency analysis
  3. Full pipeline orchestration
"""

import os
import json
import time as time_module
import numpy as np
import matplotlib
import os
if os.environ.get('DISPLAY') is None and 'inline' not in matplotlib.get_backend().lower():
    matplotlib.use('Agg')  # Non-interactive backend only when no display available
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score,
    precision_recall_fscore_support, ConfusionMatrixDisplay,
)
from sklearn.utils.class_weight import compute_class_weight
from typing import Optional

from .data_loader import (
    SAFE, WARNING, DANGER,
)

CLASS_NAMES = ["SAFE", "WARNING", "DANGER"]
BINARY_CLASS_NAMES = ["SAFE", "ABRUPT"]


# ─── Model Training ──────────────────────────────────────────────────────────

def train_random_forest(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 300,
    max_depth: Optional[int] = 20,
    class_weight: str = "balanced",
    random_state: int = 42,
    verbose: bool = True,
) -> RandomForestClassifier:
    """
    Train a Random Forest classifier with class balancing.

    Uses 'balanced' class weights by default to handle the severe
    class imbalance (SAFE >> DANGER >> WARNING).
    """
    if verbose:
        print(f"Training Random Forest (n_estimators={n_estimators}, "
              f"max_depth={max_depth}, class_weight={class_weight})...")

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=-1,
    )

    t0 = time_module.time()
    clf.fit(X_train, y_train)
    elapsed = time_module.time() - t0

    if verbose:
        train_acc = clf.score(X_train, y_train)
        print(f"  Training time: {elapsed:.1f}s")
        print(f"  Training accuracy: {train_acc:.4f}")

    return clf


def train_gradient_boosting(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_estimators: int = 200,
    max_depth: int = 6,
    learning_rate: float = 0.1,
    random_state: int = 42,
    verbose: bool = True,
) -> GradientBoostingClassifier:
    """
    Train a Gradient Boosting classifier.

    Note: sklearn GradientBoosting doesn't support class_weight directly,
    so we use sample_weight computed from class frequencies.
    """
    if verbose:
        print(f"Training Gradient Boosting (n_estimators={n_estimators}, "
              f"max_depth={max_depth}, lr={learning_rate})...")

    # Compute sample weights to handle imbalance
    classes = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes, y=y_train)
    weight_map = dict(zip(classes, weights))
    sample_weights = np.array([weight_map[y] for y in y_train])

    clf = GradientBoostingClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        random_state=random_state,
    )

    t0 = time_module.time()
    clf.fit(X_train, y_train, sample_weight=sample_weights)
    elapsed = time_module.time() - t0

    if verbose:
        train_acc = clf.score(X_train, y_train)
        print(f"  Training time: {elapsed:.1f}s")
        print(f"  Training accuracy: {train_acc:.4f}")

    return clf


# ─── Evaluation ───────────────────────────────────────────────────────────────

def evaluate_model(
    clf,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: Optional[list[str]] = None,
    output_dir: Optional[str] = None,
    model_name: str = "model",
) -> dict:
    """
    Full evaluation: classification report, confusion matrix, and feature importance.

    Args:
        clf: Trained classifier
        X_test: Test features
        y_test: Test labels
        class_names: Names for each class
        output_dir: Directory to save plots (None = don't save)
        model_name: Name prefix for saved files

    Returns:
        Dictionary with all metrics
    """
    if class_names is None:
        n_classes = len(np.unique(y_test))
        class_names = CLASS_NAMES[:n_classes] if n_classes <= 3 else [str(i) for i in range(n_classes)]

    y_pred = clf.predict(X_test)

    # Classification report
    report = classification_report(y_test, y_pred, target_names=class_names, digits=4)
    print(f"\n{'='*60}")
    print(f"  {model_name} — Test Evaluation")
    print(f"{'='*60}")
    print(report)

    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, average=None, labels=range(len(class_names))
    )

    # Macro and weighted F1
    f1_macro = f1_score(y_test, y_pred, average="macro")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")

    print(f"  Macro F1:    {f1_macro:.4f}")
    print(f"  Weighted F1: {f1_weighted:.4f}")
    print(f"{'='*60}\n")

    metrics = {
        "accuracy": float(np.mean(y_pred == y_test)),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "per_class": {
            name: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, name in enumerate(class_names)
        },
    }

    # Save plots if output_dir is provided
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

        # Confusion matrix plot
        fig, ax = plt.subplots(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred, labels=range(len(class_names)))
        disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
        disp.plot(ax=ax, cmap="Blues", values_format="d")
        ax.set_title(f"{model_name} — Confusion Matrix", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{model_name}_confusion_matrix.png"), dpi=150)
        plt.close()

        # Normalized confusion matrix
        fig, ax = plt.subplots(figsize=(8, 6))
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        disp = ConfusionMatrixDisplay(cm_norm, display_labels=class_names)
        disp.plot(ax=ax, cmap="Blues", values_format=".2%")
        ax.set_title(f"{model_name} — Normalized Confusion Matrix", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{model_name}_confusion_matrix_norm.png"), dpi=150)
        plt.close()

        # Feature importance (for tree-based models)
        if hasattr(clf, "feature_importances_"):
            plot_feature_importance(
                clf.feature_importances_,
                output_dir=output_dir,
                model_name=model_name,
                top_n=25,
            )

        # Save metrics JSON
        with open(os.path.join(output_dir, f"{model_name}_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)

    return metrics


def plot_feature_importance(
    importances: np.ndarray,
    feature_names: Optional[list[str]] = None,
    output_dir: str = ".",
    model_name: str = "model",
    top_n: int = 25,
):
    """Plot top-N most important features."""
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(len(importances))]

    # Sort by importance
    indices = np.argsort(importances)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(
        range(top_n),
        importances[indices][::-1],
        color="#2196F3",
        edgecolor="#1565C0",
    )
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_names[i] for i in indices][::-1], fontsize=9)
    ax.set_xlabel("Feature Importance", fontsize=12)
    ax.set_title(f"{model_name} — Top {top_n} Features", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{model_name}_feature_importance.png"), dpi=150)
    plt.close()


# ─── Binary Simplification ───────────────────────────────────────────────────

def to_binary_labels(y: np.ndarray) -> np.ndarray:
    """Convert 3-class labels to binary: SAFE=0, ABRUPT=1 (WARNING + DANGER → 1)."""
    return (y > 0).astype(np.int32)
