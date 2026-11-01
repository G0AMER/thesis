"""Classification metrics and paired statistics."""
from typing import Dict

import numpy as np
from scipy import stats
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score

from .config import NUM_CLASSES


def classification_metrics(y_true, y_pred) -> Dict:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    labels = list(range(NUM_CLASSES))
    present = sorted(set(y_true.tolist()))
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        # macro over the classes present in the ground truth of this test set
        "macro_f1": float(f1_score(y_true, y_pred, labels=present, average="macro", zero_division=0)),
        "per_class_f1": f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0).tolist(),
        "confusion": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def per_group_scores(y_true, y_pred, groups) -> Dict[str, Dict[str, float]]:
    y_true, y_pred, groups = map(np.asarray, (y_true, y_pred, groups))
    out = {}
    for g in sorted(set(groups.tolist())):
        m = groups == g
        present = sorted(set(y_true[m].tolist()))
        out[str(g)] = {
            "n": int(m.sum()),
            "accuracy": float(accuracy_score(y_true[m], y_pred[m])),
            "macro_f1": float(f1_score(y_true[m], y_pred[m], labels=present, average="macro", zero_division=0)),
        }
    return out


def paired_bootstrap_macro_f1(y_true, pred_a, pred_b, n_boot: int = 2000, seed: int = 0) -> Dict:
    """CI of macroF1(a) - macroF1(b), resampling test clips with replacement."""
    y_true, pred_a, pred_b = map(np.asarray, (y_true, pred_a, pred_b))
    rng = np.random.default_rng(seed)
    n = len(y_true)

    def mf1(y, p):
        return f1_score(y, p, labels=sorted(set(y.tolist())), average="macro", zero_division=0)

    diffs = np.empty(n_boot)
    for b in range(n_boot):
        i = rng.integers(0, n, n)
        diffs[b] = mf1(y_true[i], pred_a[i]) - mf1(y_true[i], pred_b[i])
    return {"diff": float(mf1(y_true, pred_a) - mf1(y_true, pred_b)),
            "ci_low": float(np.percentile(diffs, 2.5)), "ci_high": float(np.percentile(diffs, 97.5)),
            "p_two_sided": float(min(1.0, 2 * min((diffs <= 0).mean(), (diffs >= 0).mean())))}


def wilcoxon_paired(a, b) -> Dict:
    a, b = np.asarray(a, float), np.asarray(b, float)
    try:
        res = stats.wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
        W, p = float(res.statistic), float(res.pvalue)
    except ValueError:  # e.g. all differences zero
        W, p = float("nan"), float("nan")
    return {"n": int(len(a)), "mean_diff": float((a - b).mean()), "W": W, "p": p}
