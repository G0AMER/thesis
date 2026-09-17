#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared estimator registry for the v3 rebuild.
Mirrors the model set / hyperparameters of final_detector_loso_10models.py so the
rebuild stays comparable with prior runs, but every pipeline is fit inside a single
training fold (imputer + optional scaler) — never across the held-out subject.
"""
from __future__ import annotations
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              GradientBoostingClassifier, HistGradientBoostingClassifier)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:                      # pragma: no cover
    HAS_XGB = False

RANDOM_STATE = 42


def make_models(seed: int = RANDOM_STATE) -> dict:
    """Return {model_name: sklearn Pipeline} with per-fold preprocessing inside."""
    models = {
        "logreg": Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced",
                                       random_state=seed, n_jobs=-1))]),
        "random_forest": Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("clf", RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                           random_state=seed, n_jobs=-1))]),
        "extra_trees": Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("clf", ExtraTreesClassifier(n_estimators=400, class_weight="balanced",
                                         random_state=seed, n_jobs=-1))]),
        "gradient_boosting": Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("clf", GradientBoostingClassifier(random_state=seed))]),
        "hist_gradient_boosting": Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("clf", HistGradientBoostingClassifier(random_state=seed,
                                                   class_weight="balanced"))]),
        "svm_rbf": Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=3.0, gamma="scale", class_weight="balanced",
                        probability=True, random_state=seed))]),
        "knn": Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=7, weights="distance"))]),
        "decision_tree": Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("clf", DecisionTreeClassifier(class_weight="balanced", random_state=seed))]),
        "gaussian_nb": Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("clf", GaussianNB())]),
        "mlp": Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
            ("clf", MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=600,
                                  random_state=seed, early_stopping=False))]),
    }
    if HAS_XGB:
        models["xgboost"] = Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("clf", XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05,
                                  subsample=0.9, colsample_bytree=0.9,
                                  objective="multi:softprob", eval_metric="mlogloss",
                                  random_state=seed, n_jobs=-1, verbosity=0))])
    return models


def expected_n_features_safe() -> int:
    return 128


def estimate_latency_us(model, X_sample: np.ndarray) -> float:
    """Mean inference time per window in microseconds (for the HRC-latency table)."""
    import time
    t0 = time.perf_counter()
    model.predict(X_sample)
    return (time.perf_counter() - t0) / len(X_sample) * 1e6
