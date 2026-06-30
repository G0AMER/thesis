"""
Generate publication-quality PDF figures for the Cobot Safety DL Pipeline.
These are designed for direct inclusion in a LaTeX research paper (Methodology section).
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import os

# ── Global Style ──
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.titlesize': 12,
    'figure.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

OUT_DIR = "pipeline_figures"
os.makedirs(OUT_DIR, exist_ok=True)

# Color palette (academic / professional)
C = {
    'primary':   '#1a365d',   # Dark navy
    'secondary': '#2b6cb0',   # Medium blue
    'accent':    '#c53030',   # Red for danger
    'success':   '#276749',   # Green for safe
    'light':     '#ebf8ff',   # Light blue bg
    'gray':      '#4a5568',   # Text gray
    'orange':    '#c05621',   # Orange accent
    'purple':    '#553c9a',   # Purple accent
}

def draw_box(ax, x, y, w, h, text, color=C['secondary'], text_color='white', fontsize=9, bold=False):
    """Draw a rounded rectangle with centered text."""
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                         boxstyle="round,pad=0.05", facecolor=color, edgecolor='white', linewidth=1.5)
    ax.add_patch(box)
    weight = 'bold' if bold else 'normal'
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
            color=text_color, fontweight=weight, wrap=True,
            linespacing=1.3)

def draw_arrow(ax, x1, y1, x2, y2, color=C['gray']):
    """Draw an arrow between two points."""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.5))

# ═══════════════════════════════════════════════════════════
# FIGURE 1: End-to-End Pipeline Overview
# ═══════════════════════════════════════════════════════════
def fig1_pipeline_overview():
    fig, ax = plt.subplots(figsize=(12, 2.5))
    ax.set_xlim(-0.5, 8.5)
    ax.set_ylim(-1, 1.5)
    ax.axis('off')
    ax.set_title('End-to-End Pipeline Architecture', fontsize=14, fontweight='bold', pad=15)

    stages = [
        ("Raw IMU\nSensors\n(60 Subjects)", C['primary']),
        ("Data Loading\n& Labelling", C['secondary']),
        ("Windowing &\nNormalisation", C['secondary']),
        ("StratifiedKFold\n5-Fold CV", C['purple']),
        ("GPU Training\nLoop", C['accent']),
        ("Validation\n& TTA", C['purple']),
        ("Post-Processing\n(Smoothing\n+ Threshold)", C['orange']),
        ("Final Prediction\nSAFE / DANGER", C['success']),
    ]

    for i, (label, color) in enumerate(stages):
        draw_box(ax, i + 0.25, 0.3, 0.9, 1.0, label, color=color, fontsize=8)
        if i < len(stages) - 1:
            draw_arrow(ax, i + 0.75, 0.3, i + 0.8, 0.3, color=C['gray'])

    # Phase labels
    for i, phase in enumerate(["Phase 1:\nData", "Phase 2:\nTraining", "Phase 3:\nEvaluation", "Phase 4:\nDeployment"]):
        x_positions = [[0.25, 1.25, 2.25], [3.25, 4.25], [5.25], [6.25, 7.25]]
        cx = np.mean(x_positions[i])
        ax.text(cx, -0.55, phase, ha='center', va='center', fontsize=7, color=C['gray'], style='italic')

    fig.savefig(f'{OUT_DIR}/fig1_pipeline_overview.pdf')
    plt.close()
    print("  ✅ fig1_pipeline_overview.pdf")


# ═══════════════════════════════════════════════════════════
# FIGURE 2: Data Pipeline (Ingestion → Windowing → Features)
# ═══════════════════════════════════════════════════════════
def fig2_data_pipeline():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(-1, 11)
    ax.set_ylim(-1, 7)
    ax.axis('off')
    #ax.set_title('Data Preprocessing Pipeline', fontsize=14, fontweight='bold', pad=15)

    # Row 1: Data Ingestion
    draw_box(ax, 2, 6, 3.5, 0.8, "DASIG Dataset\n60 subjects × ~3 trials\n65 IMU channels @ 200Hz", C['primary'], fontsize=8)
    draw_arrow(ax, 2, 5.55, 2, 5.15)

    draw_box(ax, 2, 4.7, 3.5, 0.8, "load_all_trials()\nLabel Generation\nSAFE vs DANGER", C['secondary'], fontsize=8)
    draw_arrow(ax, 2, 4.25, 2, 3.85)

    # Row 2: Windowing
    draw_box(ax, 2, 3.4, 3.5, 0.8, "Sliding Window Segmentation\nWindow: 0.5s (100 steps)\nOverlap: 50%", C['purple'], fontsize=8)
    draw_arrow(ax, 2, 2.95, 2, 2.55)

    # Row 3: Normalisation
    draw_box(ax, 2, 2.1, 3.5, 0.8, "Per-Trial Z-Score Normalisation\nx_norm = (x − μ) / σ", C['orange'], fontsize=8)
    draw_arrow(ax, 2, 1.65, 2, 1.25)

    # Row 4: Window Label
    draw_box(ax, 2, 0.8, 3.5, 0.8, "Window-Level Labelling\ny = max(timestep labels)\nAny DANGER → Window = DANGER", C['accent'], fontsize=8)
    draw_arrow(ax, 2, 0.35, 2, -0.05)

    # Output
    draw_box(ax, 2, -0.5, 3.5, 0.8, "69,094 Windows\nX: (69094, 65, 100)\ny: (69094,) binary", C['success'], fontsize=8, bold=True)

    # Side panel: Dynamic Feature Expansion
    #draw_arrow(ax, 3.8, -0.5, 5.7, -0.5)
    draw_box(ax, 7.5, 0.8, 3.2, 0.8, "Position (Raw)\n65 channels", C['secondary'], fontsize=8)
    draw_box(ax, 7.5, -0.1, 3.2, 0.8, "Velocity (1st Derivative)\n65 channels", C['purple'], fontsize=8)
    draw_box(ax, 7.5, -1.0, 3.2, 0.5, "Acceleration (2nd Derivative)\n65 channels", C['orange'], fontsize=8)

    ax.text(7.5, 1.5, "Dynamic GPU Feature\nExpansion (On-the-fly)", ha='center', va='center',
            fontsize=9, fontweight='bold', color=C['primary'])

    # Brace-like connector (fan-out from output to feature boxes)
    for yy in [0.8, -0.1, -1.0]:
        draw_arrow(ax, 3.8, -0.5, 5.9, yy)

    # Total
    ax.text(9.5, -1.0, "→ 195 total", ha='left', va='center', fontsize=9, fontweight='bold', color=C['accent'])

    # Class distribution box
    # ax.text(7.5, 3.4, "Class Distribution", ha='center', fontsize=10, fontweight='bold', color=C['primary'])
    # ax.text(7.5, 2.8, "SAFE:   60,559  (87.6%)", ha='center', fontsize=9, color=C['success'], fontfamily='monospace')
    # ax.text(7.5, 2.3, "DANGER: 8,535   (12.4%)", ha='center', fontsize=9, color=C['accent'], fontfamily='monospace')

    # rect = FancyBboxPatch((5.5, 1.9), 4.0, 1.8, boxstyle="round,pad=0.1",
    #                       facecolor='#f7fafc', edgecolor=C['gray'], linewidth=1, linestyle='--')
    # ax.add_patch(rect)

    fig.savefig(f'{OUT_DIR}/fig2_data_pipeline.pdf')
    plt.close()
    print("  ✅ fig2_data_pipeline.pdf")


# ═══════════════════════════════════════════════════════════
# FIGURE 3: Training Loop
# ═══════════════════════════════════════════════════════════
def fig3_training_loop():
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xlim(-1, 11)
    ax.set_ylim(-1, 8)
    ax.axis('off')
    ax.set_title('Training Loop Architecture', fontsize=14, fontweight='bold', pad=15)

    # Flow
    steps = [
        (2.5, 7, "Mini-Batch (128, 65, 100)\nWeightedRandomSampler", C['primary']),
        (2.5, 5.8, "Data Augmentation\nScale Jitter (±5%) + Gaussian Noise (σ=0.01)", C['secondary']),
        (2.5, 4.6, "add_derivatives() [GPU]\n(128, 65, 100) → (128, 195, 100)", C['purple']),
        (2.5, 3.4, "Mixup Augmentation (30% probability)\nα = 0.1 Beta distribution", C['orange']),
        (2.5, 2.2, "Neural Network Forward Pass\n(TCN / ConvNeXt / MLP-Mixer / Transformer / InceptionTime)", C['accent']),
        (2.5, 1.0, "Focal Loss (γ=2.0)\nLabel Smoothing = 0.01, Inverse-Freq Class Weights", C['primary']),
        (2.5, -0.2, "AdamW (lr=1e-3, wd=1e-4)\nOneCycleLR + Gradient Clipping (1.0)", C['secondary']),
    ]

    for x, y, text, color in steps:
        draw_box(ax, x, y, 4.5, 0.85, text, color=color, fontsize=8)

    for i in range(len(steps) - 1):
        draw_arrow(ax, 2.5, steps[i][1] - 0.45, 2.5, steps[i+1][1] + 0.45)

    # Loop-back arrow
    ax.annotate('', xy=(2.5, 7.45), xytext=(-0.3, -0.2),
                arrowprops=dict(arrowstyle='->', color=C['gray'], lw=1.5,
                                connectionstyle='arc3,rad=-0.4'))
    ax.text(-0.6, 3.5, "× 30\nEpochs", ha='center', va='center', fontsize=9,
            fontweight='bold', color=C['gray'], rotation=90)

    # Hyperparameter table on the right
    params = [
        ("Epochs", "30"),
        ("Batch Size", "128"),
        ("Optimizer", "AdamW"),
        ("LR", "1e-3"),
        ("Weight Decay", "1e-4"),
        ("Scheduler", "OneCycleLR"),
        ("Loss", "Focal (γ=2)"),
        ("Grad Clip", "1.0"),
        ("Mixup α", "0.1"),
    ]

    ax.text(8, 7, "Hyperparameters", ha='center', fontsize=10, fontweight='bold', color=C['primary'])
    rect = FancyBboxPatch((6, 2.8), 4, 4.0, boxstyle="round,pad=0.15",
                          facecolor='#f7fafc', edgecolor=C['gray'], linewidth=1, linestyle='--')
    ax.add_patch(rect)

    for i, (k, v) in enumerate(params):
        y_pos = 6.4 - i * 0.42
        ax.text(6.5, y_pos, k, ha='left', fontsize=8, color=C['gray'], fontfamily='monospace')
        ax.text(9.5, y_pos, v, ha='right', fontsize=8, fontweight='bold', color=C['primary'], fontfamily='monospace')

    fig.savefig(f'{OUT_DIR}/fig3_training_loop.pdf')
    plt.close()
    print("  ✅ fig3_training_loop.pdf")


# ═══════════════════════════════════════════════════════════
# FIGURE 4: All 5 Model Architectures (Side by Side)
# ═══════════════════════════════════════════════════════════
def fig4_model_architectures():
    fig, axes = plt.subplots(1, 5, figsize=(16, 7))
    fig.suptitle('Neural Network Architectures for 1D Time-Series Classification', fontsize=14, fontweight='bold', y=0.98)

    models = [
        {
            'name': 'TCN',
            'layers': [
                ("Input\n(B, 195, 100)", C['primary']),
                ("Conv1d 1×1\n195→256\nBN + GELU", C['secondary']),
                ("TCNBlock ×6\nk=3, d=1..32\nResidual", C['purple']),
                ("AdaptiveAvgPool1d", C['orange']),
                ("Dropout(0.2)", C['gray']),
                ("Linear\n256→2", C['accent']),
                ("SAFE / DANGER", C['success']),
            ]
        },
        {
            'name': 'ConvNeXt 1D',
            'layers': [
                ("Input\n(B, 195, 100)", C['primary']),
                ("Conv1d 4×1\nstride=2\n195→128 + BN", C['secondary']),
                ("ConvNeXtBlock ×4\nDWConv k=7\nInverted Bottleneck", C['purple']),
                ("AdaptiveAvgPool1d", C['orange']),
                ("LayerNorm(128)", C['gray']),
                ("Linear\n128→2", C['accent']),
                ("SAFE / DANGER", C['success']),
            ]
        },
        {
            'name': 'MLP-Mixer 1D',
            'layers': [
                ("Input\n(B, 195, 100)", C['primary']),
                ("Conv1d k=5\nstride=5\n195→128", C['secondary']),
                ("MixerBlock ×4\nToken Mixing\nChannel Mixing", C['purple']),
                ("Mean Pooling", C['orange']),
                ("LayerNorm(128)", C['gray']),
                ("Linear\n128→2", C['accent']),
                ("SAFE / DANGER", C['success']),
            ]
        },
        {
            'name': 'Transformer 1D',
            'layers': [
                ("Input\n(B, 195, 100)", C['primary']),
                ("Linear\n195→128\n+ Pos. Encoding", C['secondary']),
                ("TransformerEncoder\n4 layers, 4 heads\nFFN=256, drop=0.2", C['purple']),
                ("Mean Pooling", C['orange']),
                ("LayerNorm(128)", C['gray']),
                ("Linear\n128→2", C['accent']),
                ("SAFE / DANGER", C['success']),
            ]
        },
        {
            'name': 'InceptionTime',
            'layers': [
                ("Input\n(B, 195, 100)", C['primary']),
                ("InceptionModule 1\nk=1,3,5 + MaxPool\n195→256", C['secondary']),
                ("InceptionModule 2\nk=1,3,5 + MaxPool\n256→256", C['purple']),
                ("AdaptiveAvgPool1d", C['orange']),
                ("", 'none'),
                ("Linear\n256→2", C['accent']),
                ("SAFE / DANGER", C['success']),
            ]
        },
    ]

    for col, (ax, model) in enumerate(zip(axes, models)):
        ax.set_xlim(-1, 3)
        ax.set_ylim(-0.5, 8)
        ax.axis('off')
        ax.set_title(model['name'], fontsize=11, fontweight='bold', pad=10)

        n = len(model['layers'])
        for i, (text, color) in enumerate(model['layers']):
            y = 7 - i * 1.0
            if color == 'none':
                continue
            box = FancyBboxPatch((0, y - 0.35), 2, 0.7,
                                 boxstyle="round,pad=0.05", facecolor=color,
                                 edgecolor='white', linewidth=1)
            ax.add_patch(box)
            ax.text(1, y, text, ha='center', va='center', fontsize=6.5,
                    color='white', fontweight='normal', linespacing=1.2)

            if i < n - 1 and model['layers'][i+1][1] != 'none':
                ax.annotate('', xy=(1, y - 0.4), xytext=(1, y - 0.6),
                            arrowprops=dict(arrowstyle='->', color=C['gray'], lw=1))
            elif i < n - 2:
                ax.annotate('', xy=(1, y - 0.4), xytext=(1, y - 1.6),
                            arrowprops=dict(arrowstyle='->', color=C['gray'], lw=1))

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(f'{OUT_DIR}/fig4_model_architectures.pdf')
    plt.close()
    print("  ✅ fig4_model_architectures.pdf")


# ═══════════════════════════════════════════════════════════
# FIGURE 5: Validation + TTA + Post-Processing
# ═══════════════════════════════════════════════════════════
def fig5_evaluation_postprocessing():
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_xlim(-0.5, 12.5)
    ax.set_ylim(-1, 5)
    ax.axis('off')
    ax.set_title('Evaluation & Post-Processing Pipeline', fontsize=14, fontweight='bold', pad=15)

    # TTA Section
    ax.text(1.5, 4.3, "Test-Time Augmentation (TTA)", ha='center', fontsize=10, fontweight='bold', color=C['primary'])
    
    draw_box(ax, 1.5, 3.2, 2.5, 0.6, "Best Model Checkpoint\n(max val Macro F1)", C['primary'], fontsize=8)
    
    tta_labels = ["Clean Input", "+ Noise #1", "+ Noise #2", "+ Noise #3"]
    for i, label in enumerate(tta_labels):
        x = 0.2 + i * 0.9
        draw_box(ax, x, 2.0, 0.8, 0.55, label, C['secondary'], fontsize=6.5)
        draw_arrow(ax, 1.5, 2.85, x, 2.3)
    
    draw_arrow(ax, 1.5, 1.7, 1.5, 1.3)
    draw_box(ax, 1.5, 0.9, 2.5, 0.6, "Average Softmax\nProbabilities", C['purple'], fontsize=8)

    # Arrow to post-processing
    draw_arrow(ax, 2.8, 0.9, 4.5, 0.9)

    # Post-Processing Section
    ax.text(7.5, 4.3, "Post-Processing (Production)", ha='center', fontsize=10, fontweight='bold', color=C['primary'])

    draw_box(ax, 5.5, 3.2, 2.2, 0.6, "Raw P(DANGER)\nProbability", C['primary'], fontsize=8)
    draw_arrow(ax, 5.5, 2.85, 5.5, 2.4)
    
    draw_box(ax, 5.5, 2.0, 2.2, 0.6, "Median Filter\n(kernel = 3)", C['purple'], fontsize=8)
    draw_arrow(ax, 5.5, 1.65, 5.5, 1.3)
    
    draw_box(ax, 5.5, 0.9, 2.2, 0.6, "Fixed Threshold\np > 0.77", C['orange'], fontsize=8)
    draw_arrow(ax, 6.65, 0.9, 7.8, 0.9)

    # Decision
    draw_box(ax, 9.5, 1.5, 2.0, 0.6, "DANGER\nEmergency Stop", C['accent'], fontsize=8, bold=True)
    draw_box(ax, 9.5, 0.3, 2.0, 0.6, "SAFE\nContinue", C['success'], fontsize=8, bold=True)
    
    draw_arrow(ax, 8.0, 1.1, 8.45, 1.5)
    draw_arrow(ax, 8.0, 0.7, 8.45, 0.3)
    
    ax.text(8.3, 1.5, "Yes", fontsize=7, color=C['accent'])
    ax.text(8.3, 0.3, "No", fontsize=7, color=C['success'])

    # Impact table at bottom
    ax.text(6, -0.5, "Impact:  FP 1,324 → 434 (−67%)  |  FN 142 → 133 (−6%)  |  Precision 83.9% → 95.1%  |  Recall 82.8% → 98.4%",
            ha='center', fontsize=8, color=C['primary'], fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#f7fafc', edgecolor=C['gray'], linewidth=0.5))

    fig.savefig(f'{OUT_DIR}/fig5_evaluation_postprocessing.pdf')
    plt.close()
    print("  ✅ fig5_evaluation_postprocessing.pdf")


# ═══════════════════════════════════════════════════════════
# FIGURE 6: Final Model Ranking Table
# ═══════════════════════════════════════════════════════════
def fig6_model_rankings():
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.axis('off')
    ax.set_title('Model Performance Comparison (Threshold=0.77, Median Smoothed)', fontsize=13, fontweight='bold', pad=15)

    columns = ['Rank', 'Model', 'Accuracy', 'Macro F1', 'Danger\nRecall', 'Danger\nPrecision']
    data = [
        ['1', 'TCN',             '99.22%', '97.88%', '98.34%', '95.09%'],
        ['2', 'ConvNeXt 1D',     '99.17%', '97.76%', '97.84%', '95.62%'],
        ['3', 'MLP-Mixer 1D',    '99.28%', '97.34%', '95.87%', '95.49%'],
        ['4', 'InceptionTime',   '98.30%', '96.22%', '97.53%', '89.62%'],
        ['5', 'Transformer 1D',  '96.47%', '92.50%', '95.85%', '79.71%'],
    ]

    table = ax.table(cellText=data, colLabels=columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.6)

    # Style header
    for j in range(len(columns)):
        cell = table[0, j]
        cell.set_facecolor(C['primary'])
        cell.set_text_props(color='white', fontweight='bold')

    # Style rows (alternating)
    for i in range(1, len(data) + 1):
        for j in range(len(columns)):
            cell = table[i, j]
            cell.set_facecolor('#f7fafc' if i % 2 == 0 else 'white')
            cell.set_edgecolor('#e2e8f0')
            if j == 0:  # Rank column
                cell.set_text_props(fontweight='bold', color=C['primary'])

    # Highlight winner row
    for j in range(len(columns)):
        table[1, j].set_facecolor('#ebf8ff')
        table[1, j].set_text_props(fontweight='bold')

    fig.savefig(f'{OUT_DIR}/fig6_model_rankings.pdf')
    plt.close()
    print("  ✅ fig6_model_rankings.pdf")


# ═══════════════════════════════════════════════════════════
# FIGURE 7: Cross-Validation Strategy
# ═══════════════════════════════════════════════════════════
def fig7_cross_validation():
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-0.5, 5)
    ax.axis('off')
    ax.set_title('5-Fold Stratified Cross-Validation Strategy', fontsize=14, fontweight='bold', pad=15)

    for fold in range(5):
        y = 4.0 - fold * 0.85
        ax.text(-0.3, y, f"Fold {fold+1}", ha='right', va='center', fontsize=9, fontweight='bold', color=C['primary'])

        for block in range(5):
            x = 0.2 + block * 1.8
            if block == fold:
                color = C['accent']
                label = "Validation"
            else:
                color = C['secondary']
                label = "Train"

            box = FancyBboxPatch((x, y - 0.3), 1.5, 0.6,
                                 boxstyle="round,pad=0.03", facecolor=color,
                                 edgecolor='white', linewidth=1)
            ax.add_patch(box)
            ax.text(x + 0.75, y, label, ha='center', va='center', fontsize=7, color='white')

    # Legend
    train_patch = mpatches.Patch(color=C['secondary'], label='Training Set (~55,275 windows)')
    val_patch = mpatches.Patch(color=C['accent'], label='Validation Set (~13,819 windows)')
    ax.legend(handles=[train_patch, val_patch], loc='lower center', ncol=2, fontsize=9, frameon=True)

    fig.savefig(f'{OUT_DIR}/fig7_cross_validation.pdf')
    plt.close()
    print("  ✅ fig7_cross_validation.pdf")


# ═══════════════════════════════════════════════════════════
# RUN ALL
# ═══════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("Generating publication-quality PDF figures...")
    print(f"Output directory: {OUT_DIR}/\n")
    
    fig1_pipeline_overview()
    fig2_data_pipeline()
    fig3_training_loop()
    fig4_model_architectures()
    fig5_evaluation_postprocessing()
    fig6_model_rankings()
    fig7_cross_validation()
    
    print(f"\n✅ All 7 figures saved to '{OUT_DIR}/' directory.")
    print("\nLaTeX usage:")
    print(r"  \includegraphics[width=\textwidth]{pipeline_figures/fig1_pipeline_overview.pdf}")
