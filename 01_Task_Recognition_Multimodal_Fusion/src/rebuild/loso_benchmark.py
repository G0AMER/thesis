#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v3_rebuild — Stage 2: subject-disjoint Leave-One-Subject-Out benchmark on the CLEAN dataset.

Protocol (answers Reviewer 3.2 / R1.5 / R3.6):
  * each held-out subject is a fold; preprocessing (impute + scale) is fitted ONLY on the
    other 51 subjects' training windows inside that fold;
  * all 52 subjects are used (not a 5-subject subset);
  * features exclude ID/Repetition leaks and duplicated EEG exports (see Stage 1);
  * per-fold metrics are reported, then summarised as mean +/- std with a t-CI over folds;
  * pooled (out-of-fold) confusion gives the overall class-level picture with RAW counts;
  * every model is compared against the best with a paired Wilcoxon signed-rank test;
  * a within-subject (stratified random 80/20) run on the SAME clean features is kept
    separately as an explicit upper bound, never as the headline.

Outputs: output/research_outputs/fusion_training/v3_rebuild/benchmark/{fold_metrics,summary,..}
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             confusion_matrix, classification_report)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from models_registry import make_models, estimate_latency_us

REPO = Path(__file__).resolve().parents[2]
OUT_ROOT = REPO / "output" / "research_outputs" / "fusion_training" / "v3_rebuild"
SEED = 42

FEATURE_GROUPS = {"EEG", "ECG", "EDA", "EMG", "RESP"}


def load_clean():
    df = pd.read_parquet(OUT_ROOT / "dataset_clean.parquet")
    manifest = pd.read_csv(OUT_ROOT / "feature_manifest.csv")
    feats = manifest.loc[manifest.decision == "keep", "column"].tolist()
    feats = [c for c in feats if c.split("__", 1)[0] in FEATURE_GROUPS or
             c.startswith("EEG_channel_")]
    feats = [c for c in df.columns if c in feats]
    X = df[feats].to_numpy(dtype=float)
    y_raw = df["pseudo_label"].astype(str).to_numpy()
    le = LabelEncoder().fit(y_raw)
    y = le.transform(y_raw)
    subjects = df["subject_id"].astype(int).to_numpy()
    return df, np.asarray(X, dtype=float), y, le, subjects, feats


def per_class_recall(y_true, y_pred, n_class):
    """Recall per class over a FIXED label space; absent class -> 0."""
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(n_class))
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.nan_to_num(np.diag(cm) / np.maximum(cm.sum(axis=1), 1), nan=0.0)
    return r


def fold_metrics(y_true, y_pred, n_class):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_acc": float(per_class_recall(y_true, y_pred, n_class).mean()),
        "macro_f1": f1_score(y_true, y_pred, labels=np.arange(n_class),
                             average="macro", zero_division=0),
    }


def t_ci95(x):
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 2:
        return float("nan"), float("nan")
    m, s = x.mean(), x.std(ddof=1)
    h = s / np.sqrt(n) * stats.t.ppf(0.975, df=n - 1)
    return float(m - h), float(m + h)


def ece_from_1hot(y_true, proba, n_bins=10):
    """Expected calibration error for the pooled OOF probabilities."""
    conf = proba.max(axis=1)
    acc = (proba.argmax(axis=1) == y_true).astype(float)
    bin_id = np.clip((conf * n_bins).astype(int), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = bin_id == b
        if m.sum() == 0:
            continue
        ece += m.sum() / len(conf) * abs(acc[m].mean() - conf[m].mean())
    return float(ece)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="all",
                    help="comma-separated subset, or 'all'")
    ap.add_argument("--no-within-subject", action="store_true")
    args = ap.parse_args()

    bench_dir = OUT_ROOT / "benchmark"
    bench_dir.mkdir(parents=True, exist_ok=True)

    df, X, y, le, subjects, feats = load_clean()
    n_class = len(le.classes_)
    print(f"rows={len(df)}  features={len(feats)}  subjects={len(np.unique(subjects))}  "
          f"classes={list(le.classes_)}")
    print("class counts:", {k: int(v) for k, v in
                            pd.Series(y).map(lambda i: le.classes_[i]).value_counts().items()})

    registry = make_models(seed=SEED)
    chosen = list(registry) if args.models == "all" else args.models.split(",")
    for m in chosen:
        if m not in registry:
            raise SystemExit(f"unknown model {m}; available: {sorted(registry)}")

    summary_rows, oof_store = [], {}
    for model_name in chosen:
        pipe = registry[model_name]
        fold_rows, oof_true, oof_pred, oof_proba = [], [], [], []
        n_test_per_fold = []
        for subj in sorted(np.unique(subjects)):
            tr = subjects != subj
            te = subjects == subj
            pipe.fit(X[tr], y[tr])
            yp = pipe.predict(X[te])
            yt = y[te]
            oof_true.append(yt)
            oof_pred.append(yp)
            if hasattr(pipe, "predict_proba"):
                oof_proba.append(pipe.predict_proba(X[te]))
            fold_rows.append({"model": model_name, "subject": int(subj),
                              "n_test": int(te.sum()), **fold_metrics(yt, yp, n_class)})
            n_test_per_fold.append(int(te.sum()))

        folds = pd.DataFrame(fold_rows)
        folds.to_csv(bench_dir / f"fold_metrics_{model_name}.csv", index=False)

        oof_true = np.concatenate(oof_true)
        oof_pred = np.concatenate(oof_pred)
        proba = (np.concatenate(oof_proba) if oof_proba and len(oof_proba) == len(np.unique(subjects))
                 else None)

        mf, ba, acc = folds.macro_f1, folds.balanced_acc, folds.accuracy
        lo_mf, hi_mf = t_ci95(mf)
        lo_ba, hi_ba = t_ci95(ba)
        summary_rows.append({
            "model": model_name,
            "folds": int(len(folds)),
            "macro_f1_mean": float(mf.mean()), "macro_f1_std": float(mf.std(ddof=1)),
            "macro_f1_ci95": [lo_mf, hi_mf],
            "balanced_acc_mean": float(ba.mean()), "balanced_acc_std": float(ba.std(ddof=1)),
            "balanced_acc_ci95": [lo_ba, hi_ba],
            "accuracy_mean": float(acc.mean()), "accuracy_std": float(acc.std(ddof=1)),
            "pooled_macro_f1": float(f1_score(oof_true, oof_pred, average="macro", zero_division=0)),
            "pooled_balanced_acc": float(balanced_accuracy_score(oof_true, oof_pred)),
            "pooled_accuracy": float(accuracy_score(oof_true, oof_pred)),
            "calib_ece": ece_from_1hot(oof_true, proba) if proba is not None else float("nan"),
        })
        np.savez_compressed(bench_dir / f"oof_{model_name}.npz",
                            y_true=oof_true, y_pred=oof_pred,
                            proba=proba if proba is not None else np.zeros((0, n_class)))
        print(f"[{model_name}] macro-F1 {mf.mean():.4f}±{mf.std(ddof=1):.4f}  "
              f"bal {ba.mean():.4f}  acc {acc.mean():.4f}  ECE {summary_rows[-1]['calib_ece']:.4f}")

    summary = pd.DataFrame(summary_rows).sort_values("macro_f1_mean", ascending=False)
    summary = summary.reset_index(drop=True)
    summary.insert(0, "rank", np.arange(1, len(summary) + 1))
    summary.to_csv(bench_dir / "benchmark_summary.csv", index=False)

    # ---- paired significance vs best (per-fold Wilcoxon) ------------------------
    best = summary.iloc[0].model
    sig_rows = []
    for model_name in summary.model:
        if model_name == best:
            continue
        a = pd.read_csv(bench_dir / f"fold_metrics_{best}.csv").sort_values("subject").macro_f1
        b = pd.read_csv(bench_dir / f"fold_metrics_{model_name}.csv").sort_values("subject").macro_f1
        d = a.to_numpy() - b.to_numpy()
        try:
            w, p = stats.wilcoxon(d, zero_method="wilcox")
        except ValueError:
            w, p = float("nan"), float("nan")
        dz = d.mean() / (d.std(ddof=1) + 1e-12)
        sig_rows.append({"model": model_name, "vs": best,
                         "wilcoxon_p": float(p), "paired_dz": float(dz),
                         "delta_macro_f1": float(a.mean() - b.mean())})
    sig = pd.DataFrame(sig_rows).sort_values("wilcoxon_p")
    sig.to_csv(bench_dir / "paired_significance.csv", index=False)
    print(f"\nBest by per-fold macro-F1: {best}")
    print(sig.to_string(index=False))

    # ---- pooled per-class report for the best model -----------------------------
    oof = np.load(bench_dir / f"oof_{best}.npz")
    cm = confusion_matrix(oof["y_true"], oof["y_pred"], labels=np.arange(n_class))
    pd.DataFrame(cm, index=le.classes_, columns=le.classes_).to_csv(
        bench_dir / f"confusion_counts_{best}.csv")
    pd.DataFrame(cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1),
                 index=le.classes_, columns=le.classes_).to_csv(
        bench_dir / f"confusion_normalized_{best}.csv")
    rep = classification_report(oof["y_true"], oof["y_pred"], labels=np.arange(n_class),
                                target_names=le.classes_, output_dict=True, zero_division=0)
    pd.DataFrame(rep).T.to_csv(bench_dir / f"per_class_{best}.csv")

    # ---- within-subject upper bound on the SAME clean features ------------------
    if not args.no_within_subject:
        ws_rows = []
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2,
                                              random_state=SEED, stratify=y)
        for model_name in ["xgboost", "hist_gradient_boosting", "random_forest", "mlp"]:
            if model_name not in registry:
                continue
            p = registry[model_name]
            p.fit(Xtr, ytr)
            m = fold_metrics(yte, p.predict(Xte), n_class)
            ws_rows.append({"model": model_name, **m})
        ws = pd.DataFrame(ws_rows).sort_values("macro_f1", ascending=False)
        ws.to_csv(bench_dir / "within_subject_upper_bound.csv", index=False)
        print("\nwithin-subject upper bound (stratified 80/20, clean features):")
        print(ws.to_string(index=False))

    # ---- inference latency (windows/sec and us/window) --------------------------
    lat_rows = []
    for model_name in summary.model:
        p = registry[model_name]
        # quick fit on a balanced mini-sample purely for the latency estimate
        idx = np.random.RandomState(0).choice(len(y), size=800, replace=False)
        p.fit(X[idx], y[idx])
        us = estimate_latency_us(p, X[:500])
        lat_rows.append({"model": model_name, "inference_us_per_window": us,
                         "windows_per_second": 1e6 / max(us, 1e-6)})
    pd.DataFrame(lat_rows).sort_values("inference_us_per_window").to_csv(
        bench_dir / "inference_latency.csv", index=False)

    json.dump({"best_model": best,
               "summary": summary.to_dict(orient="records"),
               "classes": list(le.classes_)},
              open(bench_dir / "results.json", "w"), indent=2, default=float)


if __name__ == "__main__":
    main()
