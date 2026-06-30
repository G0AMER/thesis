"""Generate FLAIR architecture diagram for the IEEE paper."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

# Unified sizing: use the largest text size currently present for all labels.
UNIFORM_TEXT_SIZE = 20.5
BOX_W = 6
BOX_H = 1.8

fig, ax = plt.subplots(1, 1, figsize=(21, 11))
ax.set_xlim(0, 33)
ax.set_ylim(0, 12)
ax.axis('off')

# Colors
C_INPUT = '#4A90D9'       # blue
C_SHARED = '#5B9BD5'      # lighter blue for shared layers
C_FILM = '#ED7D31'        # orange for FiLM
C_HEAD_SHARED = '#70AD47' # green for shared head
C_HEAD_TASK = '#FFC000'   # gold for task head
C_OUTPUT = '#7030A0'      # purple for output
C_BUFFER = '#C00000'      # dark red for replay buffer
C_FISHER = '#00B0F0'      # cyan for Fisher/importance
C_BLEND = '#BF8F00'       # dark gold for blending

def rounded_box(x, y, w, h, color, text, fontsize=UNIFORM_TEXT_SIZE, text_color='white', alpha=0.9, lw=1.0):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                         facecolor=color, edgecolor='#333333', linewidth=lw, alpha=alpha)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center',
            fontsize=fontsize, fontweight='bold', color=text_color)

def arrow(x1, y1, x2, y2, color='#333333', style='->', lw=1.8):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                                arrowprops=dict(arrowstyle=style, color=color, lw=lw, mutation_scale=24))

def dashed_arrow(x1, y1, x2, y2, color='#666666', lw=1.5):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw, linestyle='dashed', mutation_scale=22))

def frozen_marker(x, y, r=0.26):
        marker = plt.Circle((x-0.25, y-0.25), r, facecolor='#E8F3FF', edgecolor='#1F5FA8', linewidth=1.4, alpha=1.0)
        ax.add_patch(marker)
        ax.text(x-0.25, y-0.25, 'F', ha='center', va='center', fontsize=UNIFORM_TEXT_SIZE, fontweight='bold', color='#1F5FA8')

# ============================================================
# INPUT
# ============================================================
rounded_box(1.5, 5.0, BOX_W, BOX_H, C_INPUT, 'Input\n$o \\in \\mathbb{R}^{12}$')

# ============================================================
# SHARED BACKBONE (2 layers)
# ============================================================
# Layer 1
rounded_box(8.5, 5.0, BOX_W, BOX_H, C_SHARED, 'Shared\nLayer 1')
# FiLM 1
rounded_box(8.5, 7.8, BOX_W, BOX_H, C_FILM, 'FiLM$_t^{(1)}$\n$\\gamma_t, \\beta_t$')

# Layer 2
rounded_box(15.5, 5.0, BOX_W, BOX_H, C_SHARED, 'Shared\nLayer 2')
# FiLM 2
rounded_box(15.5, 7.8, BOX_W, BOX_H, C_FILM, 'FiLM$_t^{(2)}$\n$\\gamma_t, \\beta_t$')

# Arrows: input -> layer1 -> layer2
arrow(7.5, 5.9, 8.5, 5.9)
arrow(14.5, 5.9, 15.5, 5.9)

# FiLM arrows down to layers
arrow(11.5, 7.8, 11.5, 6.8, color=C_FILM, lw=1.5)
arrow(18.5, 7.8, 18.5, 6.8, color=C_FILM, lw=1.5)

# Label: "Task-specific" above FiLM
ax.text(15.0, 9.9, 'Task-specific FiLM Parameters', ha='center', va='center',
        fontsize=UNIFORM_TEXT_SIZE, fontstyle='italic', color=C_FILM)

# ============================================================
# MULTI-HEAD OUTPUT
# ============================================================
# Shared head
rounded_box(22.5, 7.0, BOX_W, BOX_H, C_HEAD_SHARED, 'Shared\nHead')
# Task head
rounded_box(22.5, 3.8, BOX_W, BOX_H, C_HEAD_TASK, 'Task\nHead $f_t$', text_color='#333333')

# Arrows from backbone to heads
arrow(21.5, 6.3, 22.5, 7.9)
arrow(21.5, 5.5, 22.5, 4.7)

# Blending circle
circle = plt.Circle((30.5, 6.0), 0.9, facecolor=C_BLEND, edgecolor='#333333', linewidth=2.2, alpha=0.95)
ax.add_patch(circle)
ax.text(30.5, 6.0, '$\\oplus$', ha='center', va='center', fontsize=UNIFORM_TEXT_SIZE, fontweight='bold', color='#111111')

# Arrows from heads to blending
arrow(28.5, 7.9, 29.6, 6.7)
arrow(28.5, 4.7, 29.6, 5.6)

# Labels for blending weights
ax.text(28.0+1, 7.3, '$(1{-}\\alpha_h)$', fontsize=UNIFORM_TEXT_SIZE, color='#333333')
ax.text(28.0+0.6, 5.0+0.25, '$\\alpha_h$', fontsize=UNIFORM_TEXT_SIZE, color='#333333')

# Output
rounded_box(22.5, 0.8, BOX_W, BOX_H, C_OUTPUT, 'Output\n$\\hat{a} \\in \\mathbb{R}^6$')
arrow(29.8, 5.3, 25.0+3, 2.6)

# ============================================================
# REPLAY BUFFER (bottom)
# ============================================================
rounded_box(1.5-1, 0.8, BOX_W+1, BOX_H, C_BUFFER, 'Task-Aware Replay Buffer\n$(o, a, \\hat{a}, t)$')

# Arrow from buffer up to backbone (replay path)
dashed_arrow(4.5, 2.6, 8.5, 5.1, color=C_BUFFER, lw=1.3)

# Adaptive replay weight label
ax.text(2.7, 3.8, 'Adaptive\nReplay $w_r$', ha='center', va='center',
        fontsize=UNIFORM_TEXT_SIZE, fontstyle='italic', color=C_BUFFER)

# ============================================================
# IMPORTANCE REGULARIZATION (bottom right)
# ============================================================
rounded_box(8.5, 0.8, BOX_W, BOX_H, C_FISHER, 'Fisher Importance $\\Omega_i$\n+ RetroBoost')

# Arrow from Fisher up to backbone
dashed_arrow(11.5, 2.6, 15.5, 5.1, color=C_FISHER, lw=1.3)

# Label
ax.text(10.8+2, 4.0, '$\\mathcal{L}_{\\mathrm{reg}}$', ha='center', va='center',
        fontsize=UNIFORM_TEXT_SIZE, color=C_FISHER, fontweight='bold')

# ============================================================
# WARM-START (top)
# ============================================================
rounded_box(1.5, 9.4, BOX_W, BOX_H, '#808080', 'FiLM Warm-Start\n$\\cos(\\mu_t, \\mu_{t\'})$', alpha=0.7)
dashed_arrow(4.5, 9.4, 8.5, 8.5, color='#808080', lw=1.0)

# ============================================================
# LOSS COMPONENTS (bottom center)
# ============================================================
# Loss box
rounded_box(15.5, 0.8, BOX_W, BOX_H, '#333333', '$\\mathcal{L} = \\mathcal{L}_{task} + \\alpha\\mathcal{L}_{logit}$\n$+ \\beta\\mathcal{L}_{rep} + \\frac{\\lambda}{2}\\mathcal{L}_{reg}$',
            text_color='white')

# Arrows to loss box
dashed_arrow(14.5, 1.7, 15.5, 1.7, color='#333333')
dashed_arrow(22.5, 1.7, 21.5, 1.7, color='#333333')

# ============================================================
# LEGEND
# ============================================================
legend_elements = [
    mpatches.Patch(facecolor=C_SHARED, edgecolor='#333', label='Shared Weights'),
    mpatches.Patch(facecolor=C_FILM, edgecolor='#333', label='Task-Specific FiLM'),
    mpatches.Patch(facecolor=C_HEAD_SHARED, edgecolor='#333', label='Shared Head'),
    mpatches.Patch(facecolor=C_HEAD_TASK, edgecolor='#333', label='Task Head'),
    mpatches.Patch(facecolor=C_BUFFER, edgecolor='#333', label='Replay Buffer'),
    mpatches.Patch(facecolor=C_FISHER, edgecolor='#333', label='Importance Reg.'),
]
ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 1.02), fontsize=UNIFORM_TEXT_SIZE,
          framealpha=0.9, edgecolor='#999', ncol=3)

# Frozen indicator
ax.text(19.0, 4.9-1, 'F = Frozen after training', fontsize=UNIFORM_TEXT_SIZE, color='#1F5FA8',
        fontstyle='italic', ha='center')
# Frozen markers near FiLM and task head
#frozen_marker(14.1, 8.7)
frozen_marker(22.2, 4.95)

plt.tight_layout()
plt.savefig('/home/g0amer/Desktop/thesis/IEEE-conference-template-062824/flair_architecture.png',
            dpi=350, bbox_inches='tight', facecolor='white')
plt.savefig('/home/g0amer/Desktop/thesis/IEEE-conference-template-062824/flair_architecture.pdf',
            bbox_inches='tight', facecolor='white')
print("Architecture figure saved.")
