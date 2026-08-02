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
#arial
# ── Global Style ──
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'Nimbus Roman', 'Liberation Serif'],
    'mathtext.fontset': 'stix',
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

def draw_box(ax, x, y, w, h, text, color=C['secondary'], text_color='white', fontsize=9, bold=False, edgecolor='#475569', linewidth=1.2):
    """Draw a rounded rectangle with centered text."""
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                         boxstyle="round,pad=0.05", facecolor=color, edgecolor=edgecolor, linewidth=linewidth)
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
    fig, ax = plt.subplots(figsize=(12, 5.4))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5.4)
    ax.axis('off')

    # Minimal, academic color scheme
    color_card_bg = '#f8fafc'
    color_border = '#334155'
    color_box_bg = '#ffffff'
    color_unifying_bg = '#f1f5f9'
    color_unifying_border = '#64748b'
    color_final_bg = '#1e293b'
    color_arrow = '#475569'

    # Card 1: Data Ingestion & Preprocessing
    rect1 = FancyBboxPatch((0.4, 0.4), 5.3, 4.6, boxstyle="round,pad=0.08",
                           facecolor=color_card_bg, edgecolor=color_border, linewidth=1.2)
    ax.add_patch(rect1)
    ax.text(3.05, 4.65, "1. Preprocessing & Windowing", ha='center', va='center',
            fontsize=11, fontweight='bold', color='#0f172a')

    # Card 1 Boxes & Arrows
    b1_y, b1_h = 3.8, 0.75
    draw_box(ax, 3.05, b1_y, 4.7, b1_h,
             "DASIG: 60 Subjects × 3 Trials\n5 Wireless MIMUs (65 Channels) @ 200 Hz",
             color=color_box_bg, text_color='#0f172a', fontsize=11)

    b2_y, b2_h = 2.5, 0.75
    draw_box(ax, 3.05, b2_y, 4.7, b2_h,
             "Sliding Window: 0.5 s (100 steps, 50% overlap)\nPer-Trial Z-Score Standardization",
             color=color_box_bg, text_color='#0f172a', fontsize=11)

    b3_y, b3_h = 1.2, 0.75
    draw_box(ax, 3.05, b3_y, 4.7, b3_h,
             "Window-Level Labeling: y = max(t_labels)\n69,094 Windows (87.6% SAFE / 12.4% DANGER)",
             color=color_box_bg, text_color='#0f172a', fontsize=11)

    # Arrows inside Card 1
    draw_arrow(ax, 3.05, b1_y - b1_h/2, 3.05, b2_y + b2_h/2, color=color_arrow)
    draw_arrow(ax, 3.05, b2_y - b2_h/2, 3.05, b3_y + b3_h/2, color=color_arrow)

    # Transition Arrow between Card 1 and Card 2
    draw_arrow(ax, 5.7, 2.5, 6.3, 2.5, color=color_border)

    # Card 2: Dynamic Kinematic Feature Expansion
    rect2 = FancyBboxPatch((6.3, 0.4), 5.3, 4.6, boxstyle="round,pad=0.08",
                           facecolor=color_card_bg, edgecolor=color_border, linewidth=1.2)
    ax.add_patch(rect2)
    ax.text(8.95, 4.65, "2. Dynamic Kinematic Feature Expansion", ha='center', va='center',
            fontsize=11, fontweight='bold', color='#0f172a')

    # BIG UNIFYING BOX for Position, Velocity, Acceleration
    unifying_bottom = 1.85
    unifying_box_h = 2.4
    unifying_rect = FancyBboxPatch((6.5, unifying_bottom), 4.9, unifying_box_h,
                                    boxstyle="round,pad=0.05", facecolor=color_unifying_bg,
                                    edgecolor=color_unifying_border, linewidth=1.0, linestyle='--')
    ax.add_patch(unifying_rect)
    ax.text(8.95, 4.05, "Kinematic Channels (3 × 65 = 195)", ha='center', va='center',
            fontsize=9.5, fontweight='bold', color='#334155')

    # Sub-boxes inside Unifying Box
    p_y, p_h = 3.55, 0.52
    v_y, v_h = 2.85, 0.52
    a_y, a_h = 2.15, 0.52

    draw_box(ax, 8.95, p_y, 4.5, p_h, r"Position (Raw): $X \in \mathbb{R}^{65 \times 100}$",
             color=color_box_bg, text_color='#0f172a', fontsize=11)
    draw_box(ax, 8.95, v_y, 4.5, v_h, r"Velocity (1st Deriv): $V = dX/dt \in \mathbb{R}^{65 \times 100}$",
             color=color_box_bg, text_color='#0f172a', fontsize=11)
    draw_box(ax, 8.95, a_y, 4.5, a_h, r"Acceleration (2nd Deriv): $A = d^2X/dt^2 \in \mathbb{R}^{65 \times 100}$",
             color=color_box_bg, text_color='#0f172a', fontsize=11)

    # Arrows inside Unifying Box connecting sub-boxes
    draw_arrow(ax, 8.95, p_y - p_h/2, 8.95, v_y + v_h/2, color=color_arrow)
    draw_arrow(ax, 8.95, v_y - v_h/2, 8.95, a_y + a_h/2, color=color_arrow)

    # Final Tensor Output Box
    final_y, final_h = 1.05, 0.70
    draw_box(ax, 8.95, final_y, 4.7, final_h,
             "Expanded Kinematic Tensor: [X; V; A]\nDimensions: (B, 195, 100)",
             color=color_final_bg, text_color='#ffffff', fontsize=11, bold=True)

    # Arrow from Unifying Box bottom edge directly to Final Tensor top edge
    final_top = final_y + final_h/2
    draw_arrow(ax, 8.95, unifying_bottom, 8.95, final_top, color=color_border)

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
        ax.text(6.5, y_pos, k, ha='left', fontsize=8, color=C['gray'])
        ax.text(9.5, y_pos, v, ha='right', fontsize=8, fontweight='bold', color=C['primary'])

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
# FIGURE 5: Temporal Debouncing & Threshold Optimization
# ═══════════════════════════════════════════════════════════
def fig5_evaluation_postprocessing():
    fig, ax = plt.subplots(figsize=(13.5, 5.0))
    ax.set_xlim(0, 13.5)
    ax.set_ylim(0, 5.0)
    ax.axis('off')

    # Color definitions (Minimal, Academic)
    color_card_bg = '#f8fafc'
    color_border = '#334155'
    color_box_bg = '#ffffff'
    color_danger_bg = '#991b1b'
    color_safe_bg = '#166534'

    # ── Stage 1: Raw Probability Generation ──
    rect1 = FancyBboxPatch((0.4, 0.4), 3.5, 4.2, boxstyle="round,pad=0.08",
                           facecolor=color_card_bg, edgecolor=color_border, linewidth=1.2)
    ax.add_patch(rect1)
    ax.text(2.15, 4.25, "1. Raw Probabilities", ha='center', va='center',
            fontsize=12.5, fontweight='bold', color='#0f172a')

    draw_box(ax, 2.15, 2.65, 3.0, 1.1,
             r"Raw Model Output:" + "\n" + r"$p_\theta(y=1 \mid \mathbf{X}_i)$",
             color=color_box_bg, text_color='#0f172a', fontsize=11, edgecolor='#64748b')

    ax.text(2.15, 1.25, "Contains transient spikes\nfrom sensor noise",
            ha='center', va='center', fontsize=10.5, style='italic', color='#475569')

    # Arrow Stage 1 → Stage 2
    draw_arrow(ax, 3.9, 2.65, 4.5, 2.65, color=color_border)

    # ── Stage 2: Temporal Debouncing (Median Filter) ──
    rect2 = FancyBboxPatch((4.5, 0.4), 3.8, 4.2, boxstyle="round,pad=0.08",
                           facecolor=color_card_bg, edgecolor=color_border, linewidth=1.2)
    ax.add_patch(rect2)
    ax.text(6.4, 4.25, "2. Temporal Debouncing", ha='center', va='center',
            fontsize=12.5, fontweight='bold', color='#0f172a')

    draw_box(ax, 6.4, 2.65, 3.4, 1.1,
             r"3-Tap Median Filter:" + "\n" + r"$p_{\text{smooth}} = \text{Med}(p_{t-1}, p_t, p_{t+1})$",
             color=color_box_bg, text_color='#0f172a', fontsize=11, edgecolor='#64748b')

    ax.text(6.4, 1.25, "Eliminates isolated\nfalse-alarm predictions",
            ha='center', va='center', fontsize=10.5, style='italic', color='#475569')

    # Arrow Stage 2 → Stage 3
    draw_arrow(ax, 8.3, 2.65, 8.9, 2.65, color=color_border)

    # ── Stage 3: Threshold Optimization & Classification ──
    rect3 = FancyBboxPatch((8.9, 0.4), 4.2, 4.2, boxstyle="round,pad=0.08",
                           facecolor=color_card_bg, edgecolor=color_border, linewidth=1.2)
    ax.add_patch(rect3)
    ax.text(11.0, 4.25, "3. Threshold Decision", ha='center', va='center',
            fontsize=12.5, fontweight='bold', color='#0f172a')

    # Threshold rule box
    draw_box(ax, 9.85, 2.65, 1.7, 1.1,
             r"Threshold Rule:" + "\n" + r"$\tau = 0.77$",
             color=color_box_bg, text_color='#0f172a', fontsize=11, edgecolor='#64748b')

    # Decision Branching Arrows
    draw_arrow(ax, 10.7, 2.9, 11.35, 3.45, color='#991b1b')
    draw_arrow(ax, 10.7, 2.4, 11.35, 1.85, color='#166534')

    ax.text(10.85, 3.4, r"$p > \tau$", fontsize=10.5, fontweight='bold', color='#991b1b')
    ax.text(10.85, 1.7, r"$p \leq \tau$", fontsize=10.5, fontweight='bold', color='#166534')

    # Outcome boxes
    draw_box(ax, 12.2, 3.45, 1.7, 0.85,
             r"DANGER" + "\n" + r"$(\hat{y} = 1)$",
             color=color_danger_bg, text_color='#ffffff', fontsize=11, bold=True, edgecolor='#7f1d1d')

    draw_box(ax, 12.2, 1.85, 1.7, 0.85,
             r"SAFE" + "\n" + r"$(\hat{y} = 0)$",
             color=color_safe_bg, text_color='#ffffff', fontsize=11, bold=True, edgecolor='#14532d')

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
