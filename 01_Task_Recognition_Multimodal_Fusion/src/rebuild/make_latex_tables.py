#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v3_rebuild — Stage 4b: emit LaTeX tables from the sealed v3_rebuild results.

Reads benchmark_summary.csv / per_class_{best}.csv / confusion_counts / ablation tables
and writes LNCS-ready, booktabs table bodies into paper/tables/*.tex.
The paper main file should simply \\input them (preamble needs \\usepackage{booktabs}).

Run AFTER the LOSO benchmark and ablations have written their CSVs.
"""
from __future__ import annotations
import ast
from pathlib import Path
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "output" / "research_outputs" / "fusion_training" / "v3_rebuild"
BENCH = OUT / "benchmark"
ABL = OUT / "ablation"
TAB = REPO / "paper" / "tables"
TAB.mkdir(parents=True, exist_ok=True)

NICE = {"xgboost": "XGBoost", "hist_gradient_boosting": "HistGB", "random_forest": "RF",
        "extra_trees": "ExtraTrees", "gradient_boosting": "GB", "logreg": "LogReg",
        "svm_rbf": "SVM-RBF", "knn": "kNN", "decision_tree": "Tree",
        "gaussian_nb": "GaussNB", "mlp": "MLP", "two_tower": "Two-Tower"}

SHORT = {"cognitive_load": "Cognitive load", "high_stress": "High stress",
         "industrial_task": "Industrial task", "low_load": "Low load",
         "other": "Other (vr-job-sim)"}


def f3(x: float) -> str:
    return f"{x:.3f}"


def parse_ci(s) -> tuple[float, float] | None:
    try:
        lo, hi = ast.literal_eval(s)
        return float(lo), float(hi)
    except Exception:
        return None


def nice(name: str) -> str:
    return NICE.get(name, name.replace("_", r"\_"))


def write(name: str, body: str) -> None:
    (TAB / name).write_text(body.lstrip("\n"))
    print(f"wrote paper/tables/{name}")


# --------------------------------------------------------------------------- 1. benchmark
def tbl_benchmark():
    p = BENCH / "benchmark_summary.csv"
    if not p.exists():
        print("skip benchmark table (missing)"); return
    s = pd.read_csv(p).sort_values("rank")
    lat = pd.read_csv(BENCH / "inference_latency.csv").set_index("model")
    best_mf = s.macro_f1_mean.iloc[0]
    best_ece = s.calib_ece.min()

    head = (r"""\begin{table}[t]
\centering
\small
\caption{Subject-disjoint leave-one-subject-out (52 operators) results on the clean
128-feature set. Macro-F1 and balanced accuracy are averaged over the 52 held-out
subjects and shown as mean~$\pm$~std; ECE is computed on pooled out-of-fold
probabilities; latency is the per-window CPU inference time of a model fitted on a
training mini-sample. Best value per column in bold.}
\label{tab:benchmark}
\begin{tabular}{lcccccc}
\toprule
Model & Macro-F1 & Balanced acc. & Accuracy & ECE & Inference \\
 & (mean$\pm$std) & (mean$\pm$std) & (mean$\pm$std) & & ($\mu$s/window) \\
\midrule
""")
    rows = []
    for _, r in s.iterrows():
        mf = f3(r.macro_f1_mean) + r" $\pm$ " + f3(r.macro_f1_std)
        ba = f3(r.balanced_acc_mean) + r" $\pm$ " + f3(r.balanced_acc_std)
        acc = f3(r.accuracy_mean) + r" $\pm$ " + f3(r.accuracy_std)
        ece = f3(r.calib_ece) if pd.notna(r.calib_ece) else "---"
        us = float(lat.loc[r.model, "inference_us_per_window"]) if r.model in lat.index else float("nan")
        lat_s = f"{us:.0f}" if np.isfinite(us) else "---"
        mm = rf"\textbf{{{mf}}}" if abs(r.macro_f1_mean - best_mf) < 1e-9 else mf
        ee = rf"\textbf{{{ece}}}" if pd.notna(r.calib_ece) and abs(r.calib_ece - best_ece) < 1e-9 else ece
        rows.append(f"{nice(r.model)} & {mm} & {ba} & {acc} & {ee} & {lat_s} \\\\")
    tail = (r"""\bottomrule
\end{tabular}
\end{table}
""")
    write("tbl_benchmark.tex", head + "\n".join(rows) + "\n" + tail)


# --------------------------------------------------------------------------- 2. per-class
def tbl_perclass():
    s = pd.read_csv(BENCH / "benchmark_summary.csv").sort_values("rank")
    best = s.iloc[0].model
    pc = pd.read_csv(BENCH / f"per_class_{best}.csv", index_col=0)
    counts = pd.read_csv(BENCH / f"confusion_counts_{best}.csv", index_col=0)
    cls = [c for c in pc.index if c in SHORT]
    tot = int(counts.sum(axis=1).sum())
    best_disp = nice(best)
    head = (f"""\\begin{{table}}[t]
\\centering
\\small
\\caption{{Pooled out-of-fold per-class metrics of the best LOSO model
({best_disp}) over all 5,640 test windows; each subject contributes only out-of-fold
predictions. Window counts are the true-label column totals of the pooled
confusion matrix.}}
\\label{{tab:perclass}}
\\begin{{tabular}}{{lcccr}}
\\toprule
True label & Precision & Recall & F1 & Windows ($n$) \\\\
\\midrule
"""
    )
    rows = []
    for c in cls:
        n = int(counts.loc[c].sum())
        rows.append(f"{SHORT[c]} & {f3(float(pc.loc[c, 'precision']))} & "
                    f"{f3(float(pc.loc[c, 'recall']))} & {f3(float(pc.loc[c, 'f1-score']))} & {n} \\\\")
    rows.append(rf"\midrule Total & & & & {tot} \\")
    tail = r"""\bottomrule
\end{tabular}
\end{table}
"""
    write("tbl_perclass.tex", head + "\n".join(rows) + "\n" + tail)


# --------------------------------------------------------------------------- 3. fusion benefit
def tbl_ablation():
    p = ABL / "fusion_benefit.csv"
    if not p.exists():
        print("skip fusion-benefit table (missing)"); return
    b = pd.read_csv(p)
    head = (r"""\begin{table}[t]
\centering
\small
\caption{Benefit of feature fusion under the identical 52-subject LOSO protocol.
Fused (128-d) is compared with the better single modality (EEG-only or physio-only)
per model; \emph{p}-values are paired Wilcoxon signed-rank tests over the 52 folds.
Two-Tower is the neural model with explicit per-modality towers, compared with fused
XGBoost.}
\label{tab:ablation}
\begin{tabular}{lccccc}
\toprule
Model & Best single & Single & Fused & $\Delta$ & Wilcoxon $p$ \\
\midrule
""")
    rows = []
    for _, r in b.iterrows():
        label = nice(r.model) if not r.model.startswith("two_tower") else "Two-Tower $vs$ fused XGB"
        bs = nice(r.best_single) if not r.best_single.endswith("_fused") else "fused XGB"
        rows.append(f"{label} & {bs} & {f3(r.single_mf)} & {f3(r.fused_mf)} & "
                    f"{'+' if r.delta >= 0 else ''}{f3(r.delta)} & {f3(r.wilcoxon_p)} \\\\")
    tail = r"""\bottomrule
\end{tabular}
\end{table}
"""
    write("tbl_ablation.tex", head + "\n".join(rows) + "\n" + tail)


# --------------------------------------------------------------------------- 4. generalization gap
def tbl_gap():
    s = pd.read_csv(BENCH / "benchmark_summary.csv").sort_values("rank")
    w = BENCH / "within_subject_upper_bound.csv"
    if not w.exists():
        print("skip gap table (missing)"); return
    wi = pd.read_csv(w).set_index("model")
    head = (r"""\begin{table}[t]
\centering
\small
\caption{Generalization gap on the identical clean features. Within-subject is a
stratified random 80/20 split (same operators appear in train and test --- an
optimistic ceiling); LOSO holds whole operators out. The gap quantifies how much
performance is lost when no data from the target operator is available at training
time.}
\label{tab:gap}
\begin{tabular}{lccc}
\toprule
Model & Within-subject & LOSO & Gap \\
\midrule
""")
    rows = []
    for _, r in s.iterrows():
        m = r.model
        if m not in wi.index:
            continue
        within = float(wi.loc[m, "macro_f1"])
        rows.append(f"{nice(m)} & {f3(within)} & {f3(r.macro_f1_mean)} & "
                    f"{within - r.macro_f1_mean:+.3f} \\\\")
    tail = r"""\bottomrule
\end{tabular}
\end{table}
"""
    write("tbl_gap.tex", head + "\n".join(rows) + "\n" + tail)


if __name__ == "__main__":
    tbl_benchmark()
    tbl_perclass()
    tbl_ablation()
    tbl_gap()
    print("done ->", TAB)
