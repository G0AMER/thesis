"""Generate a reproducible pipeline figure for the paper (task_pipeline.png).

This script draws four rounded boxes with arrows, a title and footer.
Text is wrapped and padded to avoid overlapping the box borders.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


FIG_PATH = "/home/g0amer/Desktop/thesis/IEEE-conference-template-062824/task_pipeline.png"


def rounded_box(ax, x, y, w, h, color, text, fontsize=24, pad=0.25):
    # Create a rounded box and centered text with padding to avoid overlap
    box = FancyBboxPatch((x, y), w, h,
                         boxstyle="round,pad=0.02,rounding_size=0.08",
                         linewidth=2.2, edgecolor="#222222", facecolor=color)
    ax.add_patch(box)
    # Add text centered; use wrap by manual newlines provided in text
    ax.text(x + w/2, y + h/2, text, ha="center", va="center",
            fontsize=fontsize, fontweight="bold", color="white")


def arrow(ax, x1, y1, x2, y2, lw=2.2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', lw=lw, color='#222222'))


def build_figure():
    fig_w, fig_h = 16, 6
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # Colors matching original palette
    C1 = '#6FA7DB'  # blue
    C2 = '#93C47D'  # green
    C3 = '#E6A45A'  # orange
    C4 = '#9B83C9'  # purple

    box_w = 3.2
    box_h = 1.6
    y = 3.0

    # Positions
    xs = [0.6, 4.4, 8.2, 12.0]

    texts = [
        "Physiological\nSignals",
        "Preprocessing\nImputation\n + Scaling",
        "Feature /\nSequence Encoding",
        "Classifier\nXGBoost /\nBiLSTM / Fusion",
    ]

    colors = [C1, C2, C3, C4]

    # Draw boxes
    for x, col, txt in zip(xs, colors, texts):
        rounded_box(ax, x, y - box_h/2, box_w, box_h, col, txt, fontsize=20)

    # Draw arrows between boxes (center right of box i to center left of box i+1)
    for i in range(len(xs) - 1):
        x1 = xs[i] + box_w
        x2 = xs[i+1]
        y0 = y
        arrow(ax, x1 - 0.05, y0, x2 + 0.05, y0)

    # Title
    ax.text(8.0, 5.6, 'Task-type detection from physiological state', ha='center', va='center', fontsize=26, fontweight='bold')

    # Footer
    ax.text(8.0, 0.35, 'Unified pipeline for multimodal fusion and benchmarked classification', ha='center', va='center', fontsize=16, color='#333333')

    plt.tight_layout()
    fig.savefig(FIG_PATH, dpi=300, bbox_inches='tight', facecolor='white')
    print('Saved', FIG_PATH)


if __name__ == '__main__':
    build_figure()
