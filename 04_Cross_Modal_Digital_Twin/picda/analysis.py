"""Aggregation of saved runs into tables and statistics. Reads only result JSONs."""
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from .config import CLASS_NAMES, NUM_CLASSES
from .metrics import paired_bootstrap_macro_f1, per_group_scores, wilcoxon_paired

METHOD_ORDER = [
    "dt_in_domain", "reverse_zero_shot", "source_only", "dann", "coral", "mmd", "picda", "fixmatch",
    "oracle", "source_plus_target",
]


def load_results(out_dir) -> List[Dict]:
    files = sorted((Path(out_dir) / "results").rglob("*.json"))
    if not files:
        raise FileNotFoundError(f"No result files under {out_dir}/results")
    out = []
    for p in files:
        with open(p) as f:
            out.append(json.load(f))
    return out


def runs_frame(results: List[Dict]) -> pd.DataFrame:
    rows = []
    for r in results:
        m = r["metrics"]
        rows.append({"method": r["method"], "fold": str(r["fold"]), "seed": r["seed"],
                     "test_domain": r["test_domain"], "n": m["n"], "accuracy": m["accuracy"],
                     "balanced_accuracy": m["balanced_accuracy"], "macro_f1": m["macro_f1"],
                     "train_time_s": r["extra"].get("train_time_s", np.nan)})
    return pd.DataFrame(rows)


def _method_sort_key(m):
    if m in METHOD_ORDER:
        return (0, METHOD_ORDER.index(m), "")
    return (1, 0, m)


def summary_table(results: List[Dict]) -> pd.DataFrame:
    """Mean and std over (fold, seed) runs. source_only uses per-fold rows, not the 'all' row."""
    df = runs_frame(results)
    df = df[~((df.method != "reverse_zero_shot") & (df.fold == "all"))]
    g = df.groupby("method")
    tab = pd.DataFrame({
        "runs": g.size(),
        "acc_mean": g.accuracy.mean() * 100, "acc_std": g.accuracy.std(ddof=0) * 100,
        "f1_mean": g.macro_f1.mean() * 100, "f1_std": g.macro_f1.std(ddof=0) * 100,
        "bacc_mean": g.balanced_accuracy.mean() * 100,
        "minutes_per_run": g.train_time_s.mean() / 60,
    })
    return tab.loc[sorted(tab.index, key=_method_sort_key)].round(2)


def pooled_predictions(results: List[Dict], method: str, seed: int) -> pd.DataFrame:
    """All test predictions of one method/seed across folds (each target subject appears once)."""
    rows = [r for r in results if r["method"] == method and r["seed"] == seed and r["fold"] != "all"]
    if not rows:
        raise KeyError(f"No fold results for {method} seed {seed}")
    frames = [pd.DataFrame({"clip_id": r["clip_id"], "subject": r["subject"], "y_true": r["y_true"],
                            "y_pred": r["y_pred"], "fold": r["fold"]}) for r in rows]
    return pd.concat(frames, ignore_index=True).sort_values("clip_id").reset_index(drop=True)


def seeds_of(results, method) -> List[int]:
    return sorted({r["seed"] for r in results if r["method"] == method})


def per_subject_f1(results: List[Dict], method: str) -> pd.DataFrame:
    """subject x seed macro-F1 from pooled predictions."""
    cols = {}
    for s in seeds_of(results, method):
        p = pooled_predictions(results, method, s)
        sc = per_group_scores(p.y_true, p.y_pred, p.subject)
        cols[s] = {k: v["macro_f1"] for k, v in sc.items()}
    return pd.DataFrame(cols)


def compare(results: List[Dict], a: str, b: str) -> Dict:
    """Paired tests of method a vs b on the same target clips.

    Wilcoxon: over target subjects, on seed-averaged per-subject macro-F1.
    Bootstrap: over clips, on pooled predictions, one CI per common seed.
    """
    sa, sb = per_subject_f1(results, a), per_subject_f1(results, b)
    common = sa.index.intersection(sb.index)
    out = {"a": a, "b": b, "wilcoxon_subjects": wilcoxon_paired(sa.loc[common].mean(1), sb.loc[common].mean(1))}
    boots = []
    for s in sorted(set(seeds_of(results, a)) & set(seeds_of(results, b))):
        pa, pb = pooled_predictions(results, a, s), pooled_predictions(results, b, s)
        m = pa.merge(pb, on="clip_id", suffixes=("_a", "_b"))
        bt = paired_bootstrap_macro_f1(m.y_true_a, m.y_pred_a, m.y_pred_b, seed=s)
        boots.append({"seed": s, **bt})
    out["bootstrap_clips"] = boots
    return out


def per_class_f1(results: List[Dict], method: str) -> pd.Series:
    vals = [r["metrics"]["per_class_f1"] for r in results if r["method"] == method and r["fold"] != "all"]
    return pd.Series(np.mean(vals, axis=0) * 100, index=CLASS_NAMES, name=method)


def confusion(results: List[Dict], method: str, normalize: bool = True) -> np.ndarray:
    cm = np.zeros((NUM_CLASSES, NUM_CLASSES))
    for r in results:
        if r["method"] == method and r["fold"] != "all":
            cm += np.asarray(r["metrics"]["confusion"])
    if normalize:
        cm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    return cm


def latex_table(tab: pd.DataFrame, caption: str, label: str, names: Optional[Dict[str, str]] = None) -> str:
    names = names or {}
    best = tab.loc[[m for m in tab.index if m not in ("oracle", "source_plus_target", "dt_in_domain",
                                                      "reverse_zero_shot")], "f1_mean"]
    best_m = best.idxmax() if len(best) else None
    lines = [r"\begin{table}[t]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             r"\begin{tabular}{lccc}", r"\toprule",
             r"Method & Runs & Accuracy (\%) & Macro-F1 (\%) \\", r"\midrule"]
    for m, row in tab.iterrows():
        f1 = f"{row.f1_mean:.2f} $\\pm$ {row.f1_std:.2f}"
        if m == best_m:
            f1 = rf"\textbf{{{f1}}}"
        lines.append(f"{names.get(m, m).replace('_', ' ')} & {int(row.runs)} & "
                     f"{row.acc_mean:.2f} $\\pm$ {row.acc_std:.2f} & {f1} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)
