#!/usr/bin/env python3
"""Create manuscript figures from the validated benchmark summary."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "output_v2/benchmark_ctrgcn/summary.csv"
OUT = ROOT / "output_v2/benchmark_ctrgcn/figures/paper_ready"
BLUE = "#0072B2"
ORANGE = "#D55E00"
GREEN = "#009E73"
PURPLE = "#9B6BB3"
SKY = "#56B4E9"
GRAY = "#525A61"
GRID = "#E4E9ED"
TEXT = "#20262E"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "axes.edgecolor": "#68717A",
    "axes.linewidth": 0.8,
    "text.color": TEXT,
    "axes.labelcolor": TEXT,
    "xtick.color": TEXT,
    "ytick.color": TEXT,
    "savefig.facecolor": "white",
    "figure.facecolor": "white",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def load_summary() -> dict[str, dict[str, float]]:
    with SUMMARY.open(newline="", encoding="utf-8") as stream:
        return {
            row["method"]: {
                name: float(row[name])
                for name in ("runs", "acc_mean", "acc_std", "f1_mean", "f1_std")
            }
            for row in csv.DictReader(stream)
        }


def save(fig: plt.Figure, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", dpi=320, bbox_inches="tight")
    plt.close(fig)


def methodology() -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ax.set(xlim=(0, 7.2), ylim=(0, 5.2))
    ax.axis("off")
    cards = [
        ("01", "DATA DOMAINS",
         "Virtual source: InHARD-DT (4,696 clips; 12 subjects)\n"
         "Physical target: InHARD (≈4,803 clips; 16 workers)",
         "#EAF4FA", BLUE),
        ("02", "SHARED PREPROCESSING",
         "Common 59-joint BVH hierarchy → body19 representation\n"
         "30 Hz · 64-frame windows · centering · torso scaling · yaw normalization",
         "#F0F4F7", GRAY),
        ("03", "TRANSFER STRATEGIES",
         "Zero-shot: source-only, MMD, CORAL, DANN, FixMatch, PI-CDA\n"
         "Few-shot: 1, 5, 10, 20, or 50 labelled target clips per class",
         "#F5EFF8", PURPLE),
        ("04", "EVALUATION",
         "One CTR-GCN backbone · 4 subject-disjoint folds · 3 seeds\n"
         "Accuracy and macro-F1 · held-out target labels used only for testing",
         "#EAF5F0", GREEN),
    ]
    x, width, height = 0.22, 6.76, 0.91
    bottoms = [4.05, 2.98, 1.91, 0.84]
    for i, (number, title, text, fill, edge) in enumerate(cards):
        bottom = bottoms[i]
        ax.add_patch(FancyBboxPatch(
            (x, bottom), width, height,
            boxstyle="round,pad=0.025,rounding_size=0.08",
            linewidth=1.0, edgecolor=edge, facecolor=fill,
        ))
        ax.text(x + .20, bottom + height / 2, number, ha="center", va="center",
                fontsize=14, weight="bold", color=edge)
        ax.plot([x + .48, x + .48], [bottom + .12, bottom + height - .12],
                color=edge, linewidth=1.1, alpha=.5)
        ax.text(x + .68, bottom + height - .16, title, ha="left", va="top",
                weight="bold", fontsize=9.4, color=edge)
        ax.text(x + .68, bottom + height - .38, text, ha="left", va="top",
                fontsize=8.5, linespacing=1.38, color=TEXT)
        if i < 3:
            ax.add_patch(FancyArrowPatch(
                (3.6, bottom - .02), (3.6, bottoms[i + 1] + height + .02),
                arrowstyle="-|>", mutation_scale=10, linewidth=1,
                color="#7B858D",
            ))
    ax.text(3.6, .34, "Same backbone and fixed evaluation protocol across methods",
            ha="center", va="center", fontsize=8.5, color="#606A73")
    fig.subplots_adjust(left=.015, right=.985, top=.99, bottom=.025)
    save(fig, "methodology")


def zero_shot(data: dict[str, dict[str, float]]) -> None:
    rows = [
        ("DT in-domain", "dt_in_domain", BLUE),
        ("Source-only", "source_only", GRAY),
        ("DANN", "dann", ORANGE),
        ("CORAL", "coral", "#E69F00"),
        ("MMD", "mmd", SKY),
        ("PI-CDA", "picda", PURPLE),
        ("FixMatch", "fixmatch", "#3A91C4"),
        ("Target oracle", "oracle", GREEN),
        ("Source + target", "source_plus_target", "#006B4F"),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    xs = np.arange(len(rows))
    for x, (label, key, color) in zip(xs, rows):
        r = data[key]
        ax.errorbar(x, r["f1_mean"], yerr=r["f1_std"], fmt="o", color=color,
                    ecolor="#343B41", elinewidth=1.2, capsize=3, markersize=6,
                    markeredgecolor="white", markeredgewidth=.8, zorder=3)
        ax.text(x, r["f1_mean"] + r["f1_std"] + 1.3,
                f'{r["f1_mean"]:.1f} ± {r["f1_std"]:.1f}',
                ha="center", va="bottom", fontsize=7.1, color=TEXT)
    ax.axhline(data["source_only"]["f1_mean"], color=GRAY,
               linestyle=(0, (4, 3)), linewidth=1)
    tick_labels = [
        "DT\nin-domain", "Source-\nonly", "DANN", "CORAL", "MMD",
        "PI-CDA", "FixMatch", "Target\noracle", "Source +\ntarget",
    ]
    ax.set_xticks(xs, tick_labels)
    ax.set_ylim(30, 88)
    ax.set_ylabel("Macro-F1 (%)")
    ax.text(.01, .96, "Point = mean; error bar = ±1 standard deviation (12 runs)",
            transform=ax.transAxes, ha="left", va="top", fontsize=7.2,
            color="#606A73")
    ax.text(8.45, data["source_only"]["f1_mean"] + .7,
            "Source-only mean", fontsize=7.0, color=GRAY,
            ha="right", va="bottom",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": .8, "pad": 1})
    ax.grid(axis="y", color=GRID, linewidth=.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", length=0, pad=5)
    ax.set_xlim(-.5, 8.5)
    fig.subplots_adjust(left=.09, right=.98, top=.92, bottom=.16)
    save(fig, "zero_shot")


def few_shot(data: dict[str, dict[str, float]]) -> None:
    budgets = [1, 5, 10, 20, 50]
    rows = [data[f"fewshot_k{k}_joint_proto_ft"] for k in budgets]
    x = np.arange(len(budgets))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6), sharex=True)
    metrics = [
        ("f1_mean", "f1_std", BLUE, "Macro-F1"),
        ("acc_mean", "acc_std", ORANGE, "Accuracy"),
    ]
    for ax, (mean, std, color, title) in zip(axes, metrics):
        ax.errorbar(
            x, [r[mean] for r in rows], yerr=[r[std] for r in rows],
            marker="o", markersize=5.2, linewidth=1.8, capsize=3,
            color=color, markeredgecolor="white", markeredgewidth=.7,
            zorder=3,
        )
        for xi, row in zip(x, rows):
            ax.annotate(
                f'{row[mean]:.1f}',
                (xi, row[mean]),
                xytext=(0, 8 if title == "Macro-F1" else -13),
                textcoords="offset points",
                ha="center", va="bottom" if title == "Macro-F1" else "top",
                fontsize=7.5, color=TEXT,
            )
        ax.set_ylabel(title + " (%)")
        ax.set_xticks(x, ["1", "5", "10", "20", "50"])
        ax.set_ylim(35, 85)
        ax.grid(axis="y", color=GRID, linewidth=.8)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(length=0, pad=4)
    axes[0].axhline(data["source_only"]["f1_mean"], color=GRAY,
                    linestyle=(0, (4, 3)), linewidth=1)
    axes[0].axhline(data["oracle"]["f1_mean"], color=GREEN,
                    linestyle=(0, (4, 3)), linewidth=1)
    axes[0].text(4.15, data["source_only"]["f1_mean"] + 1.4,
                 "Source-only", fontsize=6.8, color=GRAY,
                 ha="right", va="bottom")
    axes[0].text(4.15, data["oracle"]["f1_mean"] + 1.4,
                 "Target oracle", fontsize=6.8, color=GREEN,
                 ha="right", va="bottom")
    axes[0].set_ylabel("Score (%)")
    axes[1].set_ylabel("Score (%)")
    fig.supxlabel("Labelled target clips per class", y=.02, fontsize=9)
    fig.text(.5, .105,
             "Points show means; error bars show ±1 standard deviation across 12 runs.",
             ha="center", va="center", fontsize=7.2, color="#606A73")
    fig.subplots_adjust(left=.10, right=.98, top=.88, bottom=.20, wspace=.20)
    save(fig, "few_shot")


def initializations(data: dict[str, dict[str, float]]) -> None:
    budgets = [1, 5, 10, 20, 50]
    strategies = [
        ("Training from scratch", "scratch"),
        ("Source-only", "source_only"),
        ("PI-CDA", "picda"),
        ("Joint source–target", "joint"),
        ("Joint + FixMatch", "joint_fixmatch"),
        ("Joint + prototype", "joint_proto"),
        ("Joint prototype fine-tuning", "joint_proto_ft"),
    ]
    vals = np.array([
        [data[f"fewshot_k{k}_{key}"]["f1_mean"] for k in budgets]
        for _, key in strategies
    ])
    stds = np.array([
        [data[f"fewshot_k{k}_{key}"]["f1_std"] for k in budgets]
        for _, key in strategies
    ])
    labels = [label for label, _ in strategies]
    y = np.arange(len(strategies))

    for col, budget in enumerate(budgets):
        fig, ax = plt.subplots(figsize=(7.2, 3.5))
        winner = int(vals[:, col].argmax())
        for row in range(len(strategies)):
            color = GREEN if row == winner else GRAY
            ax.errorbar(
                vals[row, col], y[row], xerr=stds[row, col],
                fmt="o", markersize=6, capsize=3, elinewidth=1.2,
                color=color, ecolor=color, markeredgecolor="white",
                markeredgewidth=.7, zorder=3,
            )
            ax.text(vals[row, col] + stds[row, col] + 1.0, y[row],
                    f'{vals[row, col]:.1f} ± {stds[row, col]:.1f}',
                    va="center", fontsize=8.2, color=TEXT,
                    weight="bold" if row == winner else "normal")
        ax.set_xlim(0, 92)
        ax.set_xticks([0, 20, 40, 60, 80])
        ax.set_yticks(y, labels)
        ax.set_ylim(-.6, len(strategies) - .4)
        ax.invert_yaxis()
        ax.grid(axis="x", color=GRID, linewidth=.7)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(length=0, labelsize=8.5, pad=5)
        ax.set_xlabel("Macro-F1 (%)", fontsize=9, labelpad=4)
        fig.text(
            .28, .015,
            "Dots show means; whiskers and labels show ±1 standard deviation (12 runs). "
            "Green marks the highest mean only.",
            ha="left", va="bottom", fontsize=7.6, color="#606A73",
        )
        fig.subplots_adjust(left=.29, right=.98, top=.84, bottom=.22)
        save(fig, f"initializations_k{budget}")


def main() -> None:
    data = load_summary()
    required = (
        "dt_in_domain", "source_only", "dann", "coral", "mmd", "picda",
        "fixmatch", "oracle", "source_plus_target",
    )
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"Missing benchmark rows: {missing}")
    for k in (1, 5, 10, 20, 50):
        for strategy in (
            "scratch", "source_only", "picda", "joint", "joint_fixmatch",
            "joint_proto", "joint_proto_ft",
        ):
            key = f"fewshot_k{k}_{strategy}"
            if key not in data or data[key]["runs"] != 12:
                raise ValueError(f"Missing or incomplete configuration: {key}")
    methodology()
    zero_shot(data)
    few_shot(data)
    initializations(data)
    print(f"Created 8 PDF and 8 PNG figures in {OUT}")


if __name__ == "__main__":
    main()