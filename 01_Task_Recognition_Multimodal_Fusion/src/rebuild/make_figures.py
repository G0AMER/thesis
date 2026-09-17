#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v3_rebuild — Stage 4: publication figures from the sealed v3_rebuild results.

Consistent, print/CVD-safe styling (see dataviz method):
  * categorical slots for identity; single blue hue for magnitude-rankings with the
    best model emphasized; sequential blue for confusion heatmaps; diverging blue<->red
    for the pairwise significance matrix.
  * recessive gridlines, thin marks, direct labels where they add value.
Figures are saved as PDF + 300-dpi PNG into .../v3_rebuild/figures/.
"""
from __future__ import annotations
import ast
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

OUT = Path(__file__).resolve().parents[2] / "output" / "research_outputs" / "fusion_training" / "v3_rebuild"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)
BENCH = OUT / "benchmark"
ABL = OUT / "ablation"

INK, INK2 = "#0b0b0b", "#52514e"
BLUE, ORANGE, AQUA, MAGENTA, RED = "#2a78d6", "#eb6834", "#1baf7a", "#e87ba4", "#e34948"
GREY = "#9aa0a6"
SEQ_BLUES = ListedColormap(["#cde2fb", "#9ec5f4", "#3987e5", "#1c5cab", "#0d366b"])
DIV = ["#d03b3b", "#e8a09e", "#f0efec", "#9ec5f4", "#184f95"]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.edgecolor": "#b6babf", "axes.linewidth": 0.8,
    "axes.labelcolor": INK, "xtick.color": INK, "ytick.color": INK,
    "axes.titlesize": 10, "axes.titleweight": "bold",
    "figure.dpi": 120, "savefig.dpi": 300,
})

NICE = {"xgboost": "XGBoost", "hist_gradient_boosting": "HistGB", "random_forest": "RF",
        "extra_trees": "ExtraTrees", "gradient_boosting": "GB", "logreg": "LogReg",
        "svm_rbf": "SVM-RBF", "knn": "kNN", "decision_tree": "Tree",
        "gaussian_nb": "GaussNB", "mlp": "MLP", "two_tower": "Two-Tower"}


def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {name}")


def load_summary():
    s = pd.read_csv(BENCH / "benchmark_summary.csv")
    s["label"] = s.model.map(NICE)
    return s


def best_model():
    return pd.read_csv(BENCH / "benchmark_summary.csv").iloc[0].model


# ---- 1. pooled LOSO confusion matrix of the best model --------------------------
def fig_confusion():
    best = best_model()
    norm_path = BENCH / f"confusion_normalized_{best}.csv"
    cnt_path = BENCH / f"confusion_counts_{best}.csv"
    if not (norm_path.exists() and cnt_path.exists()):
        print("skip confusion (files missing)"); return
    norm = pd.read_csv(norm_path, index_col=0)
    cnt = pd.read_csv(cnt_path, index_col=0)
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    im = ax.imshow(norm.values, cmap=SEQ_BLUES, vmin=0, vmax=1)
    ax.set_xticks(range(len(norm))); ax.set_yticks(range(len(norm)))
    ax.set_xticklabels(norm.columns, rotation=35, ha="right")
    ax.set_yticklabels(norm.index)
    for i in range(len(norm)):
        for j in range(len(norm)):
            v = norm.values[i, j]
            ax.text(j, i, f"{v:.2f}\n({int(cnt.values[i, j])})",
                    ha="center", va="center", fontsize=6.6,
                    color="white" if v > 0.55 else INK)
    ax.set_xlabel("Predicted label"); ax.set_ylabel("True label")
    ax.set_title(f"Pooled LOSO confusion — {NICE.get(best, best)}\n"
                 f"(row-normalized; raw window counts in parentheses)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="row proportion")
    save(fig, "fig1_confusion_loso")


# ---- 2. per-class precision / recall / F1 (best model) --------------------------
def fig_perclass():
    best = best_model()
    p = BENCH / f"per_class_{best}.csv"
    if not p.exists():
        print("skip per-class (missing)"); return
    df = pd.read_csv(p, index_col=0)
    cls = [c for c in df.index if c not in {"accuracy", "macro avg", "weighted avg"}]
    x = np.arange(len(cls)); w = 0.26
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    for k, (col, c, h) in enumerate([("precision", BLUE, ""), ("recall", AQUA, "///"),
                                     ("f1-score", MAGENTA, "xxx")]):
        vals = df.loc[cls, col].astype(float).values
        b = ax.bar(x + (k - 1) * w, vals, w, label=col.title(), color=c,
                   edgecolor="white", linewidth=0.5, hatch=h, alpha=0.92)
        for r, v in zip(b, vals):
            ax.text(r.get_x() + r.get_width() / 2, v + 0.012, f"{v:.2f}",
                    ha="center", va="bottom", fontsize=6.4, color=INK2)
    ax.set_xticks(x); ax.set_xticklabels(cls, rotation=18, ha="right")
    ax.set_ylim(0, 1.18); ax.set_ylabel("score")
    ax.legend(ncol=3, frameon=False, loc="upper right")
    ax.set_title(f"Per-class metrics, pooled LOSO — {NICE.get(best, best)}")
    ax.grid(axis="y", alpha=0.25, lw=0.6); ax.set_axisbelow(True)
    save(fig, "fig2_perclass")


# ---- 3. per-subject macro-F1 distribution across models -------------------------
def fig_subject_box():
    s = load_summary()
    order = s.sort_values("macro_f1_mean").model.tolist()
    data, labels = [], []
    for m in order:
        f = pd.read_csv(BENCH / f"fold_metrics_{m}.csv")
        data.append(f.macro_f1.values)
        labels.append(NICE.get(m, m))
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    bp = ax.boxplot(data, labels=labels, patch_artist=True, showfliers=False,
                    medianprops=dict(color=INK, lw=1.4),
                    widths=0.62)
    best = s.iloc[0].model
    for i, (patch, m) in enumerate(zip(bp["boxes"], order)):
        if m == best:
            patch.set(facecolor=BLUE, alpha=0.95)
        else:
            patch.set(facecolor="#dbe7f8", alpha=0.7)
        patch.set(edgecolor="#7d8ea6", linewidth=0.8)
    # overlay per-subject points (thin, low alpha) for the best model
    i_best = order.index(best)
    vals = data[i_best]
    rng = np.random.RandomState(0)
    ax.plot(i_best + 1 + rng.uniform(-0.22, 0.22, size=len(vals)), vals, ".",
            color=INK, alpha=0.35, ms=4, mew=0)
    ax.set_ylabel("macro-F1 across held-out subjects"); ax.set_xticklabels(labels, rotation=40, ha="right")
    ax.set_title("Subject-disjoint LOSO — macro-F1 over the 52 held-out operators")
    ax.grid(axis="y", alpha=0.25, lw=0.6); ax.set_axisbelow(True)
    save(fig, "fig3_loso_subject_box")


# ---- 4. modality ablation (fused vs EEG-only vs physio-only) ---------------------
def fig_ablation():
    if not (ABL / "modality_ablation.csv").exists():
        print("skip ablation (missing)"); return
    a = pd.read_csv(ABL / "modality_ablation.csv")
    a = a[a.model != "two_tower"].copy()
    mods = [m for m in ["xgboost", "hist_gradient_boosting", "random_forest", "mlp"] if m in a.model.values]
    width = 0.26
    x = np.arange(len(mods))
    fig, ax = plt.subplots(figsize=(6.4, 3.9))
    for k, (mod, c, h) in enumerate([("fused", BLUE, ""), ("eeg", ORANGE, "///"),
                                     ("physio", AQUA, "xxx")]):
        sub = a[a.modality == mod].set_index("model").loc[mods]
        b = ax.bar(x + (k - 1) * width, sub.macro_f1_mean, width, yerr=sub.macro_f1_std,
                   label=mod, color=c, edgecolor="white", linewidth=0.5,
                   hatch=h, alpha=0.92, capsize=2, error_kw=dict(lw=0.8))
        for r, v in zip(b, sub.macro_f1_mean):
            ax.text(r.get_x() + r.get_width() / 2, v + 0.02, f"{v:.2f}",
                    ha="center", fontsize=6.6, color=INK2)
    ax.set_xticks(x); ax.set_xticklabels([NICE.get(m, m) for m in mods])
    ax.set_ylabel("LOSO macro-F1 (mean ± std, 52 folds)")
    ax.legend(ncol=3, frameon=False, loc="upper left")
    ax.set_title("Modality ablation under subject-disjoint LOSO")
    ax.grid(axis="y", alpha=0.25, lw=0.6); ax.set_axisbelow(True)
    ax.set_ylim(0, max(a[a.model != "two_tower"].macro_f1_mean + a[a.model != "two_tower"].macro_f1_std) * 1.35)
    save(fig, "fig4_modality_ablation")


# ---- 5. within-subject upper bound vs LOSO generalization gap --------------------
def fig_gap():
    wub = BENCH / "within_subject_upper_bound.csv"
    if not wub.exists():
        print("skip gap (missing)"); return
    w = pd.read_csv(wub)
    s = load_summary()
    rows = []
    for m in w.model:
        if m not in set(s.model):
            continue
        lo, hi = (ast.literal_eval(s.loc[s.model == m, "macro_f1_ci95"].iloc[0])
                  if isinstance(s.loc[s.model == m, "macro_f1_ci95"].iloc[0], str)
                  else (float("nan"), float("nan")))
        rows.append({"model": NICE.get(m, m), "within": w.loc[w.model == m, "macro_f1"].iloc[0],
                     "loso": s.loc[s.model == m, "macro_f1_mean"].iloc[0],
                     "loso_lo": lo,
                     "loso_hi": hi})
    d = pd.DataFrame(rows)
    x = np.arange(len(d)); wb = 0.38
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    b1 = ax.bar(x - wb / 2, d.within, wb, color="#9ec5f4", label="within-subject (stratified 80/20)",
                edgecolor="white", linewidth=0.5)
    b2 = ax.bar(x + wb / 2, d.loso, wb, color=BLUE, label="subject-disjoint LOSO",
                edgecolor="white", linewidth=0.5, yerr=[d.loso - d.loso_lo, d.loso_hi - d.loso],
                capsize=2, error_kw=dict(lw=0.8, color=INK2))
    for bars in (b1, b2):
        for r in bars:
            ax.text(r.get_x() + r.get_width() / 2, r.get_height() + 0.015,
                    f"{r.get_height():.2f}", ha="center", fontsize=7, color=INK2)
    ax.set_xticks(x); ax.set_xticklabels(d.model)
    ax.set_ylabel("macro-F1"); ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, loc="upper left", fontsize=7.5)
    ax.set_title("Generalization gap: within-subject ceiling vs across-operator")
    ax.grid(axis="y", alpha=0.25, lw=0.6); ax.set_axisbelow(True)
    save(fig, "fig5_generalization_gap")


# ---- 6. pairwise significance matrix (delta macro-F1, best anchored) -------------
def fig_significance():
    sig = BENCH / "paired_significance.csv"
    s = load_summary()
    if not sig.exists():
        print("skip significance (missing)"); return
    df = pd.read_csv(sig)
    models = s.sort_values("macro_f1_mean", ascending=False).model.tolist()
    best = models[0]
    delta = {m: 0.0 for m in models}
    star = {m: "" for m in models}
    star[best] = "—"
    for _, r in df.iterrows():
        delta[r.model] = r.delta_macro_f1
        p = r.wilcoxon_p
        star[r.model] = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
    rows = models[1:]
    y = np.arange(len(rows))[::-1]
    vals = np.array([delta[m] for m in rows])
    sign = np.sign(vals)
    # diverging: negative red, positive blue, ~0 neutral
    cmap = ListedColormap(DIV)
    norm_vals = np.clip(vals / (max(abs(vals).max(), 1e-9)), -1, 1)
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    sc = ax.scatter([0] * len(rows), y, c=norm_vals, cmap=cmap, vmin=-1, vmax=1,
                    s=430, marker="s", edgecolors="white", linewidths=1)
    for yy, m, v, st in zip(y, rows, vals, [star[m] for m in rows]):
        ax.text(0, yy, f"{v:+.3f} {st}", ha="center", va="center",
                fontsize=7, color="white" if abs(norm_vals[np.where(y == yy)[0][0]]) > 0.35 else INK)
    ax.set_yticks(y); ax.set_yticklabels([NICE.get(m, m) for m in rows])
    ax.set_xticks([]); ax.set_xlim(-1, 1)
    ax.set_title(f"Δ macro-F1 vs best ({NICE.get(best, best)}), per-fold Wilcoxon over 52 folds")
    for s_ in ax.spines.values():
        s_.set_visible(False)
    save(fig, "fig6_significance")


if __name__ == "__main__":
    fig_confusion(); fig_perclass(); fig_subject_box(); fig_ablation(); fig_gap(); fig_significance()
