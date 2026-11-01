"""Cheap diagnostics on trained source-only models: label-shift correction and test-time transforms.

Label shift: InHARD's class mix differs strongly from InHARD-DT's (Assemble system 29% vs 11%).
Under pure label shift p(x|y) is shared and only p(y) changes, so the source posterior can be
re-weighted by pi_t / pi_s. pi_t is unknown, but can be estimated from *unlabeled* target clips
with the EM procedure of Saerens, Latinne & Decaestecker (Neural Computation, 2002). The prior is
estimated on the 12 target training subjects of each fold and applied to the 4 test subjects, so
no test clip and no target label is used. The 'true prior' row (class frequencies of the training
subjects, which uses labels) is an upper bound for this kind of correction, not a method.
"""
from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

from . import config as C
from .data import NO_AUG, SkeletonDataset
from .train import predict


def real_probs(bench, model, transform: Optional[Callable] = None):
    """Softmax outputs of `model` on every InHARD clip, in bench.real row order."""
    idx = np.arange(len(bench.real))
    ds = SkeletonDataset(bench.clips_real, bench.y_real, idx, bench.cfg.window, (NO_AUG,), bench.cfg.features,
                         bench.cfg.joint_set, transform=transform)
    out = predict(model, ds, bench.cfg, bench.device)
    order = np.argsort(out["idx"])
    return out["probs"][order], out["y_true"][order]


def em_prior(probs: np.ndarray, src_prior: np.ndarray, iters: int = 500, tol: float = 1e-8) -> np.ndarray:
    w = src_prior.copy()
    for _ in range(iters):
        adj = probs * (w / src_prior)
        adj /= adj.sum(1, keepdims=True)
        new = adj.mean(0)
        if np.abs(new - w).max() < tol:
            return new
        w = new
    return w


def adjust(probs: np.ndarray, src_prior: np.ndarray, tgt_prior: np.ndarray) -> np.ndarray:
    adj = probs * (np.maximum(tgt_prior, 1e-8) / src_prior)
    return adj / adj.sum(1, keepdims=True)


def _scores(y, p):
    present = sorted(set(y.tolist()))
    return {"accuracy": 100 * accuracy_score(y, p), "balanced_accuracy": 100 * balanced_accuracy_score(y, p),
            "macro_f1": 100 * f1_score(y, p, labels=present, average="macro", zero_division=0)}


def fold_table(bench, probs: np.ndarray, y: np.ndarray, tag: str) -> pd.DataFrame:
    """Per fold: plain, EM-prior corrected, true-prior corrected (upper bound)."""
    src_prior = np.bincount(bench.y_dt, minlength=C.NUM_CLASSES) / len(bench.y_dt)
    rows = []
    for f in range(len(bench.real_folds)):
        tr, te = bench._real_split(f)
        pi_em = em_prior(probs[tr], src_prior)
        pi_true = np.bincount(y[tr], minlength=C.NUM_CLASSES) / len(tr)
        for name, P in [("plain", probs[te]), ("em_prior", adjust(probs[te], src_prior, pi_em)),
                        ("true_prior (upper bound)", adjust(probs[te], src_prior, pi_true))]:
            rows.append({"model": tag, "correction": name, "fold": f, **_scores(y[te], P.argmax(1)),
                         "prior_L1_error": float(np.abs(pi_em - pi_true).sum()) if name == "em_prior" else np.nan})
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["model", "correction"], sort=False)
    return pd.DataFrame({
        "acc": g.accuracy.mean(), "macro_f1": g.macro_f1.mean(), "macro_f1_std": g.macro_f1.std(ddof=0),
        "bal_acc": g.balanced_accuracy.mean(), "prior_L1_err": g.prior_L1_error.mean(),
    }).round(2)


def load_model(bench, ckpt_path):
    model = bench._new_model()
    model.load_state_dict(torch.load(ckpt_path, map_location=bench.device))
    model.eval()
    return model
