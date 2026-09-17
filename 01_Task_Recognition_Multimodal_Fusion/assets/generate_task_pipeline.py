"""Generate a reproducible pipeline figure for the paper (task_pipeline.png).

Drawn architecture matches the models and protocol ACTUALLY described in the paper
(fixes the previous figure that advertised a sequence-encoding/BiLSTM path while the
text only describes tabular + two-tower models). Color-blind safe, print friendly.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
OUT_PATHS = [SCRIPT_DIR / "task_pipeline.png",
             SCRIPT_DIR.parent / "paper" / "task_pipeline.png"]

# scheme: data = blue, process = teal, representation = indigo, models/eval = orange,
# output = green — all chosen to stay distinguishable in grayscale via lightness.
C_DATA = "#2a78d6"
C_PROC = "#1baf7a"
C_REPR = "#4a3aa7"
C_MOD = "#eb6834"
C_OUT = "#008300"
C_EDGE = "#1a1a19"


def rbox(ax, x, y, w, h, color, text, fs=10.5, bold=True, tc="white"):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.015,rounding_size=0.06",
                         linewidth=1.4, edgecolor=C_EDGE, facecolor=color)
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", color=tc)


def arrow(ax, x1, y1, x2, y2, lw=2.0):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", lw=lw, color=C_EDGE))


def build_figure():
    fig_w, fig_h = 16.5, 8.2
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, 16.5)
    ax.set_ylim(0, 8.2)
    ax.axis("off")

    # ---- stage 1 : data ------------------------------------------------
    rbox(ax, 0.3, 3.0, 2.5, 2.4, C_DATA,
         "MultiPhysio-HRC\nEEG (8 ch)\nECG · EDA · EMG · RESP\n52 operators, 21 task blocks\n5,640 windows", fs=10)

    # ---- stage 2 : preprocessing ---------------------------------------
    rbox(ax, 3.6, 3.0, 2.9, 2.4, C_PROC,
         "Preprocessing\nsubject-level normalization\n60-s windows (50% overlap)\nstatistical features\nmedian imputation", fs=10)
    arrow(ax, 2.85, 4.2, 3.55, 4.2)

    # ---- stage 3 : modality feature groups ------------------------------
    rbox(ax, 7.2, 4.55, 2.6, 1.05, C_REPR, "EEG features\n(96-d)", fs=9.5)
    rbox(ax, 7.2, 2.55, 2.6, 1.05, C_REPR, "Peripheral physio\nECG·EDA·EMG·RESP (32-d)", fs=9.5)
    arrow(ax, 6.55, 4.85, 7.15, 5.05)   # preprocess -> EEG tower
    arrow(ax, 6.55, 3.55, 7.15, 3.05)   # preprocess -> physio tower
    # no fusion-fork: both groups flow to fusion box

    # ---- stage 4 : fused representation ----------------------------------
    rbox(ax, 10.6, 3.55, 2.2, 1.3, C_REPR, "Fused tabular\nrepresentation\n(128-d)", fs=10)
    arrow(ax, 9.85, 4.9, 10.55, 4.55)   # EEG -> fusion
    arrow(ax, 9.85, 3.0, 10.55, 3.7)    # physio -> fusion

    # ---- stage 5 : models + protocol --------------------------------------
    rbox(ax, 13.4, 3.55, 2.8, 2.4, C_MOD,
         "Detectors\nLogReg · RF · XGBoost · HistGB\nMLP · Two-Tower fusion\n\nsubject-disjoint LOSO\nper-fold preprocessing\nclass-weighted objective", fs=8.8)
    arrow(ax, 12.85, 4.2, 13.35, 4.2)

    # ---- stage 6 : output label --------------------------------------------
    rbox(ax, 13.4, 0.55, 2.8, 1.5, C_OUT,
         "Task-condition label\ncognitive load · high stress\nindustrial task · low load · other", fs=9)
    arrow(ax, 14.8, 3.5, 14.8, 2.1)

    # ---- annotation of the two-tower path (explicit modality separation) ----
    ax.text(8.5, 6.35, "modality feature groups  →  explicit EEG / physio towers optional",
            ha="center", fontsize=9.5, style="italic", color="#333333")
    ax.plot([7.2, 10.6], [6.1, 6.1], color=C_EDGE, lw=1.0, ls=(0, (2, 2)))
    ax.plot([10.6, 10.6], [5.95, 5.0], color=C_EDGE, lw=1.0, ls=(0, (2, 2)))

    # ---- title / caption -----------------------------------------------------
    ax.text(8.25, 7.9, "Task-condition recognition from multimodal physiological signals",
            ha="center", fontsize=17, fontweight="bold", color="#1a1a19")
    ax.text(8.25, 0.12,
            "One clean preprocessing path; all detectors share the same subject-disjoint LOSO evaluation protocol.",
            ha="center", fontsize=10, color="#333333")

    plt.tight_layout()
    for p in OUT_PATHS:
        fig.savefig(p, dpi=300, bbox_inches="tight", facecolor="white")
        print("Saved", p)
    plt.close(fig)


if __name__ == "__main__":
    build_figure()
