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
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_xlim(0, 8)
    ax.set_ylim(0, 4.5)
    ax.axis('off')

    stages = [
        ("Raw Wearable\nSensors\n(60 Subjects)", C['primary']),
        ("Data Loading\n& Labelling", C['secondary']),
        ("Windowing &\nNormalisation", C['secondary']),
        ("Stratified\n5-Fold Cross-Val", C['purple']),
        ("Deep Learning\nTraining Loop", C['accent']),
        ("Validation\n& Augmentation", C['purple']),
        ("Post-Processing\n(Smoothing\n+ Threshold)", C['orange']),
        ("Final Prediction\nSAFE / DANGER", C['success']),
    ]

    coords = [
        (1.5, 3.5), (4.0, 3.5), (6.5, 3.5),
        (6.5, 2.0), (4.0, 2.0), (1.5, 2.0),
        (1.5, 0.5), (4.0, 0.5)
    ]

    # Draw boxes
    for i, (label, color) in enumerate(stages):
        x, y = coords[i]
        draw_box(ax, x, y, 2.0, 1.0, label, color=color, fontsize=9)

    # Draw arrows
    for i in range(len(coords) - 1):
        x1, y1 = coords[i]
        x2, y2 = coords[i+1]
        
        # Horizontal arrows
        if y1 == y2:
            if x1 < x2: # Right
                draw_arrow(ax, x1 + 1.05, y1, x2 - 1.05, y2, color=C['gray'])
            else: # Left
                draw_arrow(ax, x1 - 1.05, y1, x2 + 1.05, y2, color=C['gray'])
        # Vertical arrows
        elif x1 == x2:
            draw_arrow(ax, x1, y1 - 0.55, x2, y2 + 0.55, color=C['gray'])

    # Phase labels
    ax.text(1.5, 4.2, "Phase 1: Data", ha='center', va='center', fontsize=9, color=C['gray'], style='italic')
    ax.text(4.0, 2.7, "Phase 2: Training", ha='center', va='center', fontsize=9, color=C['gray'], style='italic')
    ax.text(1.5, 2.7, "Phase 3: Evaluation", ha='center', va='center', fontsize=9, color=C['gray'], style='italic')
    ax.text(2.75, 1.2, "Phase 4: Deployment", ha='center', va='center', fontsize=9, color=C['gray'], style='italic')

    fig.savefig(f'{OUT_DIR}/fig1_pipeline_overview.pdf')
    plt.close()
    print("  ✅ fig1_pipeline_overview.pdf")


# ═══════════════════════════════════════════════════════════
# FIGURE 2: Data Pipeline (Ingestion → Windowing → Features)
# ═══════════════════════════════════════════════════════════
def fig2_data_pipeline():
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 5)
    ax.axis('off')

    # Card 1: Data Ingestion & Preprocessing
    rect1 = FancyBboxPatch((0.4, 0.6), 4.8, 3.8, boxstyle="round,pad=0.1",
                           facecolor='#f7fafc', edgecolor=C['primary'], linewidth=1.5)
    ax.add_patch(rect1)
    ax.text(2.8, 4.0, "1. Preprocessing & Windowing", ha='center', va='center',
            fontsize=10, fontweight='bold', color=C['primary'])

    # Steps inside Card 1
    draw_box(ax, 2.8, 3.2, 4.2, 0.6, "DASIG Corpus: 60 Subjects × 3 Trials\n65 Sensor Channels @ 200 Hz", C['primary'], fontsize=8)
    draw_arrow(ax, 2.8, 2.85, 2.8, 2.6, color=C['gray'])

    draw_box(ax, 2.8, 2.2, 4.2, 0.6, "Sliding Window: 0.5 s (100 steps, 50% overlap)\nPer-Trial Z-Score: x_norm = (x − μ) / σ", C['secondary'], fontsize=8)
    draw_arrow(ax, 2.8, 1.85, 2.8, 1.6, color=C['gray'])

    draw_box(ax, 2.8, 1.1, 4.2, 0.6, "Window-Level Labeling: y = max(t_labels)\n69,094 Windows (87.6% SAFE / 12.4% DANGER)", C['purple'], fontsize=8)

    # Transition Arrow
    draw_arrow(ax, 5.3, 2.5, 5.8, 2.5, color=C['primary'])

    # Card 2: Dynamic Kinematic Feature Expansion
    rect2 = FancyBboxPatch((5.9, 0.6), 4.7, 3.8, boxstyle="round,pad=0.1",
                           facecolor='#ebf8ff', edgecolor=C['purple'], linewidth=1.5)
    ax.add_patch(rect2)
    ax.text(8.25, 4.0, "2. Dynamic Kinematic Feature Expansion", ha='center', va='center',
            fontsize=10, fontweight='bold', color=C['purple'])

    # Dynamic derivatives sub-boxes
    draw_box(ax, 8.25, 3.2, 3.8, 0.5, "Position (Raw): X ∈ ℝ^(65 × 100)", C['secondary'], fontsize=8)
    draw_box(ax, 8.25, 2.5, 3.8, 0.5, "Velocity (1st Deriv): V = dX/dt ∈ ℝ^(65 × 100)", C['purple'], fontsize=8)
    draw_box(ax, 8.25, 1.8, 3.8, 0.5, "Acceleration (2nd Deriv): A = d²X/dt² ∈ ℝ^(65 × 100)", C['orange'], fontsize=8)

    draw_arrow(ax, 8.25, 1.5, 8.25, 1.25, color=C['purple'])

    # Final Tensor Output
    draw_box(ax, 8.25, 0.9, 4.0, 0.55, "Expanded Kinematic Tensor: [X; V; A]\nDimensions: (B, 195, 100)", C['success'], fontsize=8, bold=True)

    fig.savefig(f'{OUT_DIR}/fig2_data_pipeline.pdf')
    plt.close()
    print("  ✅ fig2_data_pipeline.pdf")


# ═══════════════════════════════════════════════════════════
# FIGURE 3: Training Loop Architecture
# ═══════════════════════════════════════════════════════════
def fig3_training_loop():
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis('off')

    # Left Side: Circular Loop Diagram
    center_x, center_y, radius = 2.7, 2.5, 1.5

    # Draw cyclic ring background
    circle = plt.Circle((center_x, center_y), radius, color='#ebf8ff', ec=C['primary'], lw=1.5, ls='--')
    ax.add_patch(circle)
    ax.text(center_x, center_y, "30 Epochs\nTraining\nLoop", ha='center', va='center',
            fontsize=9, fontweight='bold', color=C['primary'])

    # 6 Radial Nodes
    nodes = [
        ("1. Batch Sampling\n(WeightedSampler)", 90, C['primary']),
        ("2. Scale Jitter & Noise\n(±5%, σ=0.01)", 30, C['secondary']),
        ("3. Kinematic Derivs\n(65→195)", 330, C['purple']),
        ("4. Mixup Aug.\n(p=0.3, α=0.1)", 270, C['orange']),
        ("5. Forward Pass\n(1D Backbones)", 210, C['accent']),
        ("6. Focal Loss & AdamW\n(OneCycleLR)", 150, C['success']),
    ]

    for label, angle_deg, color in nodes:
        angle_rad = np.radians(angle_deg)
        nx = center_x + (radius + 0.1) * np.cos(angle_rad)
        ny = center_y + (radius + 0.1) * np.sin(angle_rad)
        draw_box(ax, nx, ny, 1.6, 0.6, label, color=color, fontsize=6.5)

    # Right Side: Parameter Panel
    rect = FancyBboxPatch((5.8, 0.6), 3.8, 3.8, boxstyle="round,pad=0.1",
                          facecolor='#f7fafc', edgecolor=C['gray'], linewidth=1.2)
    ax.add_patch(rect)
    ax.text(7.7, 4.0, "Hyperparameters & Config", ha='center', va='center', fontsize=10, fontweight='bold', color=C['primary'])

    params = [
        ("Batch Size (B)", "128"),
        ("Optimizer", "AdamW"),
        ("Initial Learning Rate", "1e-3"),
        ("Weight Decay", "1e-4"),
        ("Scheduler", "OneCycleLR"),
        ("Loss Function", "Focal (γ=2.0)"),
        ("Label Smoothing", "0.01"),
        ("Grad Clipping", "1.0"),
    ]

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
    #ax.set_title('Evaluation & Post-Processing Pipeline', fontsize=14, fontweight='bold', pad=15)

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
    
    ax.text(8.0, 1.5, "Yes", fontsize=7, color=C['accent'])
    ax.text(8.0, 0.3, "No", fontsize=7, color=C['success'])

    # Impact table at bottom
    ax.text(6, -0.5, "Impact:  False Positives 1,324 → 166 (−87.5%)  |  Danger Precision 83.9% → 98.0%  |  Recall 97.1%",
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
