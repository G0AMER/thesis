#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v3_rebuild — Stage 3: modality ablations under the SAME subject-disjoint LOSO protocol.

Answers Reviewer 3.3 / Reviewer 4.5 ("justify that fusion actually helps"):
  * EEG-only        : 96 EEG channel features
  * physio-only     : ECG + EDA + EMG + RESP features (32)
  * fused           : 128-feature union (the full pipeline input)
  * two-tower fusion: neural model with explicit EEG tower + physio tower (fusion
                      architecture tested against plain fused-tabular models)

For every modality x model we run the full 52-subject LOSO; then:
  * fusion-benefit table   fused vs best single modality per model (delta + CI)
  * paired Wilcoxon        fused vs best single modality across the 52 folds
Outputs under .../v3_rebuild/ablation/
"""
from __future__ import annotations
import argparse, json, copy
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import f1_score, balanced_accuracy_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from models_registry import make_models

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

REPO = Path(__file__).resolve().parents[2]
OUT_ROOT = REPO / "output" / "research_outputs" / "fusion_training" / "v3_rebuild"
SEED = 42
ABLATION_MODELS = ["xgboost", "hist_gradient_boosting", "random_forest", "mlp"]
# fmt: off
MODALITY_SUBSETS = {
    "eeg":    lambda c: c.startswith("EEG_channel_"),
    "physio": lambda c: c.split("__", 1)[0] in {"ECG", "EDA", "EMG", "RESP"},
    "fused":  lambda c: True,  # placeholder, handled specially
}
# fmt: on


def load():
    df = pd.read_parquet(OUT_ROOT / "dataset_clean.parquet")
    manifest = pd.read_csv(OUT_ROOT / "feature_manifest.csv")
    feats = manifest.loc[manifest.decision == "keep", "column"].tolist()
    feats = [c for c in df.columns if c in feats]
    X = df[feats].to_numpy(dtype=float)
    y = LabelEncoder().fit_transform(df["pseudo_label"].astype(str))
    subjects = df["subject_id"].astype(int).to_numpy()
    return df, X, y, subjects, feats


def loso_sklearn(make_est, X, y, subjects, n_class):
    """Return per-fold macro-F1 array (fixed label space) + pooled y_true/y_pred."""
    folds, oof_t, oof_p = [], [], []
    for subj in sorted(np.unique(subjects)):
        tr, te = subjects != subj, subjects == subj
        est = make_est()
        est.fit(X[tr], y[tr])
        yt, yp = y[te], est.predict(X[te])
        oof_t.append(yt); oof_p.append(yp)
        folds.append(f1_score(yt, yp, labels=np.arange(n_class),
                              average="macro", zero_division=0))
    return np.array(folds), np.concatenate(oof_t), np.concatenate(oof_p)


def cols_of(feats, subset):
    if subset == "fused":
        return feats
    keep = [c for c in feats if MODALITY_SUBSETS[subset](c)]
    return keep


def indices_of(feats, Xdf, subset):
    keep = cols_of(feats, subset)
    return [feats.index(c) for c in keep]


# ----------------------------- two-tower neural model --------------------------
class TwoTower(nn.Module):
    def __init__(self, d_phys, d_eeg, n_class, hid=128, dropout=0.2):
        super().__init__()
        self.p = nn.Sequential(nn.Linear(d_phys, hid), nn.ReLU(), nn.Dropout(dropout),
                               nn.Linear(hid, hid // 2))
        self.e = nn.Sequential(nn.Linear(d_eeg, hid), nn.ReLU(), nn.Dropout(dropout),
                               nn.Linear(hid, hid // 2))
        self.head = nn.Sequential(nn.Linear(hid, hid // 2), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(hid // 2, n_class))

    def forward(self, xp, xe):
        return self.head(torch.cat([self.p(xp), self.e(xe)], dim=1))


def loso_twotower(X, y, subjects, feats, seed=SEED, max_epochs=50, patience=8):
    """Corrected subject-disjoint two-tower training.

    Fixes (compared with the first v3 run) the validation collapse that produced a
    0.07 macro-F1:
      * validation slice is *stratified* on the fold's train labels (not a random 15%),
        so the early-stopping score sees every class;
      * the early-stopping macro-F1 is computed over the FIXED label space
        (labels=np.arange(n_class), absent class -> 0);
      * the loss is class-weighted by inverse frequency (Eq. 1 of the paper),
        instead of plain cross-entropy dominated by the majority class;
      * RNG is re-seeded per fold so the 52 runs are reproducible.
    """
    idx_phys = indices_of(feats, X, "physio")
    idx_eeg = indices_of(feats, X, "eeg")
    n_class = int(y.max()) + 1
    counts = np.bincount(y, minlength=n_class).astype(float)
    w = counts.sum() / (n_class * counts + 1e-9)   # inverse-frequency weights
    w = w / w.mean()
    wt = torch.tensor(w, dtype=torch.float32)
    folds, oof_t, oof_p = [], [], []
    for subj in sorted(np.unique(subjects)):
        torch.manual_seed(seed + int(subj))
        tr, te = subjects != subj, subjects == subj

        def _t(a):
            return torch.tensor(np.ascontiguousarray(a), dtype=torch.float32)

        def _impute(tr_np, te_np):
            """Fill missing values with the training-fold column medians (train-only
            statistic). torch mean/std are not NaN-aware: without this, a single
            missing feature turns the whole fold's standardized input into NaN and
            every prediction degenerates to argmax-of-NaN == class 0."""
            med = np.nanmedian(tr_np, axis=0)
            med = np.where(np.isnan(med), 0.0, med)
            return (np.where(np.isnan(tr_np), med, tr_np),
                    np.where(np.isnan(te_np), med, te_np))

        Xp_tr_n, Xp_te_n = _impute(X[tr][:, idx_phys], X[te][:, idx_phys])
        Xe_tr_n, Xe_te_n = _impute(X[tr][:, idx_eeg], X[te][:, idx_eeg])
        Xp_tr0, Xe_tr0 = _t(Xp_tr_n), _t(Xe_tr_n)
        Xp_te, Xe_te = _t(Xp_te_n), _t(Xe_te_n)
        y_tr, y_te = y[tr], y[te]

        # standardize per-fold (train stats only)
        def std(a, b):
            mu, sd = a.mean(0), a.std(0) + 1e-6
            return (a - mu) / sd, (b - mu) / sd

        Xp_tr0, Xp_te = std(Xp_tr0, Xp_te)
        Xe_tr0, Xe_te = std(Xe_tr0, Xe_te)

        # stratified validation split inside the fold's training windows
        i_tr, i_va = train_test_split(np.arange(len(y_tr)), test_size=0.15,
                                      random_state=seed + int(subj), stratify=y_tr)
        Xp_tr, Xp_va = Xp_tr0[i_tr], Xp_tr0[i_va]
        Xe_tr, Xe_va = Xe_tr0[i_tr], Xe_tr0[i_va]
        ytr = torch.tensor(y_tr[i_tr], dtype=torch.long)
        yva = y_tr[i_va]

        model = TwoTower(len(idx_phys), len(idx_eeg), n_class)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        crit = nn.CrossEntropyLoss(weight=wt)
        best_sd, best_f1, wait = None, -1.0, 0
        dl = DataLoader(TensorDataset(Xp_tr, Xe_tr, ytr), batch_size=256, shuffle=True)
        for ep in range(max_epochs):
            model.train()
            for xp, xe, yy in dl:
                opt.zero_grad()
                loss = crit(model(xp, xe), yy)
                loss.backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                vp = model(Xp_va, Xe_va).argmax(1).numpy()
                vf = f1_score(yva, vp, labels=np.arange(n_class),
                              average="macro", zero_division=0)
            if vf > best_f1:
                best_f1, best_sd, wait = vf, copy.deepcopy(model.state_dict()), 0
            else:
                wait += 1
                if wait >= patience:
                    break
        model.load_state_dict(best_sd)
        model.eval()
        with torch.no_grad():
            pred = model(Xp_te, Xe_te).argmax(1).numpy()
        oof_t.append(y_te)
        oof_p.append(pred)
        folds.append(f1_score(y_te, pred, labels=np.arange(n_class),
                              average="macro", zero_division=0))
    return np.array(folds), np.concatenate(oof_t), np.concatenate(oof_p)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(ABLATION_MODELS))
    args = ap.parse_args()
    mods = args.models.split(",")
    out_dir = OUT_ROOT / "ablation"
    out_dir.mkdir(parents=True, exist_ok=True)

    df, X, y, subjects, feats = load()
    n_class = int(y.max()) + 1
    registry = make_models(seed=SEED)

    rows = []
    for subset in ["eeg", "physio", "fused"]:
        idx = indices_of(feats, X, subset)
        for mn in mods:
            if mn not in registry:
                print(f"skip {mn} (unavailable)"); continue
            mf, yt, yp = loso_sklearn(lambda: copy.deepcopy(registry[mn]), X[:, idx], y, subjects, n_class)
            rows.append({"modality": subset, "model": mn,
                         "macro_f1_mean": float(mf.mean()), "macro_f1_std": float(mf.std(ddof=1)),
                         "pooled_macro_f1": float(f1_score(yt, yp, average="macro", zero_division=0))})
            print(f"[{subset:>6}][{mn:>22}] macro-F1 {mf.mean():.4f}±{mf.std(ddof=1):.4f}")
            np.savez_compressed(out_dir / f"oof_{subset}_{mn}.npz", y_true=yt, y_pred=yp, folds=mf)

    # two-tower neural fusion architecture
    mf_tt, yt_tt, yp_tt = loso_twotower(X, y, subjects, feats)
    rows.append({"modality": "fused", "model": "two_tower",
                 "macro_f1_mean": float(mf_tt.mean()), "macro_f1_std": float(mf_tt.std(ddof=1)),
                 "pooled_macro_f1": float(f1_score(yt_tt, yp_tt, average="macro", zero_division=0))})
    print(f"[fused  ][two_tower             ] macro-F1 {mf_tt.mean():.4f}±{mf_tt.std(ddof=1):.4f}")
    np.savez_compressed(out_dir / "oof_fused_two_tower.npz", y_true=yt_tt, y_pred=yp_tt, folds=mf_tt)

    res = pd.DataFrame(rows)
    res.to_csv(out_dir / "modality_ablation.csv", index=False)

    # ---- fusion-benefit: fused vs best single modality, paired over folds ------
    benefit = []
    for mn in mods:
        if mn not in registry:
            continue
        f_fus = np.load(out_dir / f"oof_fused_{mn}.npz")["folds"]
        best_sub, best_f = None, -np.inf
        for sub in ["eeg", "physio"]:
            f = np.load(out_dir / f"oof_{sub}_{mn}.npz")["folds"]
            if f.mean() > best_f:
                best_f, best_sub = f.mean(), sub
        f_best = np.load(out_dir / f"oof_{best_sub}_{mn}.npz")["folds"]
        d = f_fus - f_best
        try:
            w, p = stats.wilcoxon(d, zero_method="wilcox")
        except ValueError:
            w, p = float("nan"), float("nan")
        benefit.append({"model": mn, "best_single": best_sub,
                        "fused_mf": float(f_fus.mean()), "single_mf": float(f_best.mean()),
                        "delta": float(d.mean()), "wilcoxon_p": float(p)})
    # two-tower vs fused-concat best
    f_tt = np.load(out_dir / "oof_fused_two_tower.npz")["folds"]
    f_xgb_fus = np.load(out_dir / "oof_fused_xgboost.npz")["folds"]
    benefit.append({"model": "two_tower_vs_xgboost_fused", "best_single": "xgboost_fused",
                    "fused_mf": float(f_tt.mean()), "single_mf": float(f_xgb_fus.mean()),
                    "delta": float(f_tt.mean() - f_xgb_fus.mean()),
                    "wilcoxon_p": float(stats.wilcoxon(f_tt - f_xgb_fus).pvalue)})
    pd.DataFrame(benefit).to_csv(out_dir / "fusion_benefit.csv", index=False)
    print("\n--- fusion benefit (fused vs best single modality) ---")
    print(pd.DataFrame(benefit).to_string(index=False))

    # ---- within-subject upper bound per modality for XGBoost --------------------
    if "xgboost" in registry:
        ub = []
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
        for subset in ["eeg", "physio", "fused"]:
            idx = indices_of(feats, X, subset)
            est = copy.deepcopy(registry["xgboost"]); est.fit(Xtr[:, idx], ytr)
            ub.append({"modality": subset,
                       "within_macro_f1": float(f1_score(yte, est.predict(Xte[:, idx]),
                                                         average="macro", zero_division=0))})
        pd.DataFrame(ub).to_csv(out_dir / "within_subject_ablation_xgb.csv", index=False)
        print("\nwithin-subject (upper bound) XGBoost by modality:")
        print(pd.DataFrame(ub).to_string(index=False))


if __name__ == "__main__":
    main()
