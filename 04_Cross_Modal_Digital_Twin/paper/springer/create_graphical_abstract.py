"""7:5 graphical abstract for the IJIDeM submission (TOC graphic).

Journal rules (text now added on request, which departs from the rule): no text, no specific results, no depiction of a person, original art not reused
from the manuscript, landscape 7:5, vector EPS or TIFF >= 300 dpi. Workers are abstract joint
graphs. No transparency is used, so the EPS renders identically to the PDF.

Story, read top to bottom:
  row 1  virtual workcell (many free labels) | domain shift | physical workcell (one label)
  row 2  skeleton sequences from each domain  | label-efficiency curve
  row 3  stage 1 joint training with class prototypes -> stage 2 fine-tuning -> cobot assistance
"""
from pathlib import Path

import matplotlib.pyplot as plt

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "DejaVu Sans"})
import numpy as np
from matplotlib.patches import Arc, Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle

OUT = Path(__file__).resolve().parent
VIRT, PHYS, NET, GOOD, GREY, DARK = "#3B82C4", "#E07B39", "#6B5CA5", "#2A9D8F", "#9AA0A6", "#4A4F55"
VIRT_BG, PHYS_BG, NET_BG, GOOD_BG = "#EAF2FB", "#FCEFE6", "#F1EEF8", "#E6F4F2"
VIRT_LIGHT, NET_LIGHT, VIRT_DOT, PHYS_LIGHT = "#9DC0E3", "#B5ACD3", "#6FA3D6", "#F4B183"

# body joint graph in unit body coordinates
J = {"head": (0, 1.72), "neck": (0, 1.5), "chest": (0, 1.25), "hips": (0, 0.95),
     "rsho": (0.18, 1.45), "relb": (0.42, 1.32), "rwri": (0.62, 1.38),
     "lsho": (-0.18, 1.45), "lelb": (-0.26, 1.18), "lwri": (-0.18, 0.95),
     "rhip": (0.12, 0.92), "rkne": (0.15, 0.48), "rank": (0.16, 0.02),
     "lhip": (-0.12, 0.92), "lkne": (-0.14, 0.48), "lank": (-0.17, 0.02)}
E = [("head", "neck"), ("neck", "chest"), ("chest", "hips"), ("neck", "rsho"), ("rsho", "relb"),
     ("relb", "rwri"), ("neck", "lsho"), ("lsho", "lelb"), ("lelb", "lwri"), ("hips", "rhip"),
     ("rhip", "rkne"), ("rkne", "rank"), ("hips", "lhip"), ("lhip", "lkne"), ("lkne", "lank")]
# right arm going from resting to reaching, used for the sequence strips
ARM = [((0.24, 1.18), (0.26, 0.92)), ((0.36, 1.22), (0.5, 1.05)), ((0.42, 1.3), (0.62, 1.25)),
       ((0.42, 1.36), (0.66, 1.46)), ((0.38, 1.4), (0.6, 1.58))]


def skeleton(ax, x0, y0, s, color, dashed=False, flip=False, arm=None, lw=2.2, joints=True):
    pose = dict(J)
    if arm is not None:
        pose["relb"], pose["rwri"] = arm
    p = {k: (x0 + s * (-x if flip else x), y0 + s * y) for k, (x, y) in pose.items()}
    for a, b in E:
        ax.plot(*zip(p[a], p[b]), color=color, lw=lw, ls=(0, (2, 1.2)) if dashed else "-",
                solid_capstyle="round", zorder=5)
    ax.add_patch(Circle(p["head"], s * 0.08, fc="white", ec=color, lw=lw * 0.75, zorder=6))
    if joints:
        for k, (x, y) in p.items():
            if k != "head":
                ax.add_patch(Circle((x, y), s * 0.035, fc="white", ec=color, lw=lw * 0.7, zorder=6))
    return p


def tag(ax, x, y, s, color):
    """Label tag icon: pointed card with a hole."""
    ax.add_patch(Polygon([(x, y), (x + 0.25 * s, y + 0.18 * s), (x + 0.75 * s, y + 0.18 * s),
                          (x + 0.75 * s, y - 0.18 * s), (x + 0.25 * s, y - 0.18 * s)],
                         closed=True, fc=color, ec="white", lw=0.8, zorder=7))
    ax.add_patch(Circle((x + 0.2 * s, y), 0.045 * s, fc="white", ec="none", zorder=8))


def panel(ax, x, y, w, h, fc, ec, dashed=False, lw=1.8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.14", fc=fc,
                                ec=ec, lw=lw, ls="--" if dashed else "-", zorder=1))


def arrow(ax, a, b, color, rad=0.0, lw=2.4, both=False):
    style = "<|-|>" if both else "-|>"
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle=f"{style},head_length=6,head_width=4",
                                 connectionstyle=f"arc3,rad={rad}", color=color, lw=lw, zorder=9))


def label(ax, x, y, txt, color=DARK, size=8.5, ha="left", weight="bold"):
    ax.text(x, y, txt, color=color, fontsize=size, ha=ha, va="center", fontweight=weight, zorder=10)


def dots(ax, x, y, n, color):
    """Stage badge without text: one or two filled dots."""
    for i in range(n):
        ax.add_patch(Circle((x + i * 0.13, y), 0.045, fc=color, ec="none", zorder=8))


# ---------------------------------------------------------------- row 1: the two workcells
def headset(ax, x, y, s):
    ax.add_patch(FancyBboxPatch((x - 0.4 * s, y - 0.17 * s), 0.8 * s, 0.34 * s,
                                boxstyle=f"round,pad=0,rounding_size={0.12 * s}", fc=VIRT, ec="none", zorder=7))
    for dx in (-0.19, 0.19):
        ax.add_patch(Circle((x + dx * s, y), 0.1 * s, fc="white", ec="none", zorder=8))


def virtual_scene(ax, x, y, w, h):
    panel(ax, x, y, w, h, VIRT_BG, VIRT, dashed=True)
    hx, fy = x + w * 0.45, y + 0.15
    for t in np.linspace(-1, 1, 9):
        ax.plot([hx + t * 0.3, hx + t * (w * 0.45 - 0.1)], [fy + 0.45, fy], color=VIRT_LIGHT, lw=0.7, zorder=2)
    for v in np.linspace(0, 1, 4):
        half = (w * 0.45 - 0.1) - v * (w * 0.45 - 0.1 - 0.3)
        ax.plot([hx - half, hx + half], [fy + 0.45 * v] * 2, color=VIRT_LIGHT, lw=0.7, zorder=2)
    bx, by = x + w * 0.52, y + 0.3
    ax.add_patch(Rectangle((bx, by), 0.62, 0.36, fc="none", ec=VIRT, lw=1.2, ls="--", zorder=3))
    ax.add_patch(Rectangle((bx + 0.08, by + 0.36), 0.17, 0.14, fc="none", ec=VIRT, lw=1, ls="--", zorder=3))
    ax.add_patch(Rectangle((bx + 0.36, by + 0.36), 0.15, 0.22, fc="none", ec=VIRT, lw=1, ls="--", zorder=3))
    s = 0.8
    skeleton(ax, x + w * 0.3, y + 0.14, s, VIRT, dashed=True, lw=2)
    headset(ax, x + w * 0.3 + 0.015, y + 0.14 + s * 1.73, 0.3)
    for i in range(8):  # labels come for free in the twin
        tag(ax, x + w - 0.98 + (i % 4) * 0.17, y + h - 0.22 - (i // 4) * 0.24 - (i % 4) * 0.045, 0.48, VIRT)


def physical_scene(ax, x, y, w, h):
    panel(ax, x, y, w, h, PHYS_BG, PHYS)
    ax.plot([x + 0.1, x + w - 0.1], [y + 0.15, y + 0.15], color=PHYS, lw=1.5, zorder=2)
    bx, by = x + w * 0.07, y + 0.52
    ax.add_patch(Rectangle((bx, by), 0.78, 0.07, fc=PHYS, ec="none", zorder=3))
    for lx in (bx + 0.05, bx + 0.69):
        ax.add_patch(Rectangle((lx, y + 0.15), 0.045, 0.37, fc=PHYS, ec="none", zorder=3))
    for dx, ww, hh in ((0.08, 0.2, 0.16), (0.36, 0.13, 0.27), (0.55, 0.15, 0.11)):
        ax.add_patch(Rectangle((bx + dx, by + 0.07), ww, hh, fc=PHYS_LIGHT, ec=PHYS, lw=1, zorder=3))
    # screwdriver on the bench
    ax.plot([bx + 0.2, bx + 0.42], [by + 0.11, by + 0.11], color=DARK, lw=1.2, zorder=4)
    ax.add_patch(Rectangle((bx + 0.42, by + 0.085), 0.12, 0.05, fc=DARK, ec="none", zorder=4))
    skeleton(ax, x + w * 0.68, y + 0.14, 0.8, PHYS, flip=True, lw=2)
    tag(ax, x + w - 0.55, y + h - 0.22, 0.48, PHYS)  # labels are scarce


def domain_shift(ax, x, y, w, h):
    panel(ax, x, y, w, h, "white", GREY)
    base, xs = y + 0.25, np.linspace(x + 0.12, x + w - 0.12, 200)
    for mu, col, fill, ls in ((x + w * 0.35, VIRT, VIRT_BG, (0, (3, 1.5))), (x + w * 0.65, PHYS, PHYS_BG, "-")):
        g = base + (h - 0.75) * np.exp(-0.5 * ((xs - mu) / (w * 0.13)) ** 2)
        ax.fill_between(xs, base, g, color=fill, zorder=2)
        ax.plot(xs, g, color=col, lw=2.2, ls=ls, zorder=3)
    ax.plot([x + 0.12, x + w - 0.12], [base, base], color=DARK, lw=1.2, zorder=3)
    arrow(ax, (x + w * 0.35, y + h - 0.3), (x + w * 0.65, y + h - 0.3), NET, lw=1.8, both=True)


# ---------------------------------------------------------------- row 2: sequences and the curve
def filmstrip(ax, x, y, w, h, color, dashed):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.06", fc=DARK, ec="none", zorder=2))
    for hx in np.arange(x + 0.08, x + w - 0.05, 0.14):
        for hy in (y + 0.04, y + h - 0.08):
            ax.add_patch(Rectangle((hx, hy), 0.07, 0.04, fc="white", ec="none", zorder=3))
    n = len(ARM)
    fw = (w - 0.12) / n
    for i, arm in enumerate(ARM):
        fx = x + 0.06 + i * fw
        ax.add_patch(Rectangle((fx + 0.02, y + 0.13), fw - 0.04, h - 0.26, fc="white", ec="none", zorder=3))
        skeleton(ax, fx + fw * 0.42, y + 0.15, (h - 0.32) / 1.8, color, dashed=dashed,
                 flip=color == PHYS, arm=arm, lw=1.3, joints=False)


def efficiency(ax, x, y, w, h):
    panel(ax, x, y, w, h, "white", GREY)
    ox, oy, aw, ah = x + 0.17, y + 0.2, w - 0.32, h - 0.38
    arrow(ax, (ox, oy), (ox + aw, oy), DARK, lw=1.2)
    arrow(ax, (ox, oy), (ox, oy + ah), DARK, lw=1.2)
    t = np.linspace(0, 1, 100)
    fast = lambda u: 0.08 + 0.8 * (1 - np.exp(-7 * u))
    slow = lambda u: 0.02 + 0.8 * (1 - np.exp(-1.7 * u))
    ax.plot(ox + t * aw * 0.92, oy + ah * fast(t), color=GOOD, lw=2.2, zorder=4)
    ax.plot(ox + t * aw * 0.92, oy + ah * slow(t), color=GREY, lw=1.8, ls="--", zorder=4)
    lvl = 0.55  # same accuracy reached with far fewer labels
    ug, ub = -np.log(1 - (lvl - 0.08) / 0.8) / 7, -np.log(1 - (lvl - 0.02) / 0.8) / 1.7
    ax.add_patch(FancyArrowPatch((ox + ub * aw * 0.92, oy + ah * lvl), (ox + ug * aw * 0.92, oy + ah * lvl),
                                 arrowstyle="-|>,head_length=4,head_width=2.5", color=GOOD, lw=1.4,
                                 shrinkA=2, shrinkB=2, zorder=5))


# ---------------------------------------------------------------- row 3: method and use
def network(ax, cx, cy, s=1.0):
    layers = [np.linspace(-0.4, 0.4, 3), np.linspace(-0.5, 0.5, 4), np.linspace(-0.4, 0.4, 3)]
    xs = np.linspace(cx - 0.3 * s, cx + 0.3 * s, len(layers))
    for i in range(len(layers) - 1):
        for a in layers[i]:
            for b in layers[i + 1]:
                ax.plot([xs[i], xs[i + 1]], [cy + a * s, cy + b * s], color=NET_LIGHT, lw=0.7, zorder=3)
    for x, l in zip(xs, layers):
        for yv in l:
            ax.add_patch(Circle((x, cy + yv * s), 0.065 * s, fc=NET, ec="white", lw=0.8, zorder=4))


def stage1(ax, x, y, w, h):
    panel(ax, x, y, w, h, NET_BG, NET)
    label(ax, x + 0.14, y + h - 0.16, "Stage 1: twin + few labels", NET)
    network(ax, x + 0.5, y + h * 0.48, 0.95)
    # class prototypes: physical prototypes pulled onto twin prototypes
    px, rng = x + 0.95, np.random.default_rng(3)
    ax.add_patch(FancyBboxPatch((px, y + 0.12), w - 1.05, h - 0.52, boxstyle="round,pad=0,rounding_size=0.08",
                                fc="white", ec=NET_LIGHT, lw=1, zorder=2))
    for cx, cy, dx in ((px + (w - 1.05) * 0.3, y + h * 0.58, 0.07), (px + (w - 1.05) * 0.72, y + h * 0.58, -0.07)):
        v = rng.normal(0, 0.09, (12, 2))
        ax.scatter(cx + v[:, 0], cy + v[:, 1], s=16, color=VIRT_DOT, lw=0, zorder=4)
        ox, oy = cx + dx, y + h * 0.24
        r = rng.normal(0, 0.07, (4, 2))
        ax.scatter(ox + r[:, 0], oy + r[:, 1], s=26, marker="^", color=PHYS, lw=0, zorder=4)
        ax.scatter([cx], [cy], s=200, marker="*", color=VIRT, ec="white", lw=0.6, zorder=5)
        ax.scatter([ox], [oy], s=200, marker="*", color=PHYS, ec="white", lw=0.6, zorder=5)
        ax.add_patch(FancyArrowPatch((ox, oy), (cx, cy), arrowstyle="-|>,head_length=4,head_width=2.5",
                                     color=NET, lw=1.3, shrinkA=6, shrinkB=6, zorder=5))


def stage2(ax, x, y, w, h):
    panel(ax, x, y, w, h, NET_BG, NET)
    label(ax, x + 0.14, y + h - 0.16, "Stage 2: fine-tuning", NET)
    cx, cy = x + w * 0.55, y + h * 0.48
    network(ax, cx, cy, 0.95)
    # refinement loop around the network
    ax.add_patch(Arc((cx, cy), 1.05, 1.15, theta1=20, theta2=330, color=GOOD, lw=1.8, zorder=5))
    ax.add_patch(FancyArrowPatch((cx + 0.5, cy + 0.16), (cx + 0.47, cy + 0.26), arrowstyle="-|>,head_length=5,head_width=3.5",
                                 color=GOOD, lw=1.8, zorder=6))
    for i in range(3):  # the few physical labels drive this stage
        tag(ax, x + 0.1, y + h * 0.62 - i * 0.2, 0.36, PHYS)


def icon_box(ax, x, y, s, active):
    ax.add_patch(FancyBboxPatch((x, y), s, s, boxstyle="round,pad=0,rounding_size=0.05",
                                fc=GOOD_BG if active else "white", ec=GOOD if active else GREY,
                                lw=1.8 if active else 1.0, zorder=3))


def cobot_scene(ax, x, y, w, h):
    panel(ax, x, y, w, h, GOOD_BG, GOOD)
    # robot arm: base, two links, gripper offering a part
    bx, by = x + 0.35, y + 0.22
    ax.add_patch(Rectangle((bx - 0.18, by - 0.06), 0.36, 0.12, fc=DARK, ec="none", zorder=3))
    j1, j2, j3 = (bx, by + 0.1), (bx + 0.12, by + 0.72), (bx + 0.62, by + 0.92)
    for a, b in ((j1, j2), (j2, j3)):
        ax.plot(*zip(a, b), color=GREY, lw=7, solid_capstyle="round", zorder=4)
    for j in (j1, j2, j3):
        ax.add_patch(Circle(j, 0.07, fc=DARK, ec="white", lw=1, zorder=5))
    ax.plot([j3[0], j3[0] + 0.14], [j3[1] + 0.05, j3[1] + 0.05], color=DARK, lw=2, zorder=5)
    ax.plot([j3[0], j3[0] + 0.14], [j3[1] - 0.05, j3[1] - 0.05], color=DARK, lw=2, zorder=5)
    ax.add_patch(Rectangle((j3[0] + 0.08, j3[1] - 0.035), 0.08, 0.07, fc=PHYS_LIGHT, ec=PHYS, lw=0.8, zorder=6))
    # recognised action among assembly classes: assemble, pick, consult
    s, ix, iy = 0.38, x + w - 0.52, y + h - 0.64
    for i in range(3):
        icon_box(ax, ix, iy - i * 0.47, s, active=i == 0)
    cx, cy = ix + s / 2, iy + s / 2  # wrench
    ax.plot([cx - 0.11, cx + 0.08], [cy - 0.11, cy + 0.08], color=DARK, lw=3, solid_capstyle="round", zorder=4)
    ax.add_patch(Circle((cx + 0.09, cy + 0.09), 0.06, fc=DARK, ec="none", zorder=4))
    ax.add_patch(Circle((cx + 0.12, cy + 0.12), 0.035, fc=GOOD_BG, ec="none", zorder=5))
    cx, cy = ix + s / 2, iy - 0.47 + s / 2  # part box with grasp arrow
    ax.add_patch(Rectangle((cx - 0.11, cy - 0.12), 0.22, 0.15, fc=PHYS_LIGHT, ec=PHYS, lw=1, zorder=4))
    ax.add_patch(FancyArrowPatch((cx, cy + 0.02), (cx, cy + 0.15), arrowstyle="-|>,head_length=3,head_width=2",
                                 color=DARK, lw=1.2, zorder=5))
    cx, cy = ix + s / 2, iy - 0.94 + s / 2  # instruction sheet
    ax.add_patch(Rectangle((cx - 0.09, cy - 0.13), 0.18, 0.26, fc="white", ec=DARK, lw=1, zorder=4))
    for k in range(4):
        ax.plot([cx - 0.055, cx + 0.055 - (k % 2) * 0.03], [cy + 0.08 - k * 0.05] * 2, color=GREY, lw=1, zorder=5)
    # check mark on the recognised action
    ax.plot([ix - 0.2, ix - 0.13, ix - 0.02], [iy + 0.2, iy + 0.12, iy + 0.3], color=GOOD, lw=2.2,
            solid_capstyle="round", zorder=5)
    arrow(ax, (ix - 0.06, iy + 0.05), (j3[0] + 0.25, j3[1] + 0.02), GOOD, rad=-0.3, lw=1.4)


def main():
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 7)
    ax.set_ylim(0, 5)
    ax.set_aspect("equal")
    ax.axis("off")

    # row 1
    virtual_scene(ax, 0.08, 3.1, 2.45, 1.82)
    domain_shift(ax, 2.72, 3.4, 1.56, 1.22)
    physical_scene(ax, 4.47, 3.1, 2.45, 1.82)
    arrow(ax, (2.55, 4.0), (2.95, 3.95), VIRT, rad=-0.2, lw=1.6)
    arrow(ax, (4.45, 4.0), (4.05, 3.95), PHYS, rad=0.2, lw=1.6)
    # row 2
    filmstrip(ax, 0.08, 2.15, 2.45, 0.78, VIRT, dashed=True)
    efficiency(ax, 2.82, 2.08, 1.36, 1.08)
    filmstrip(ax, 4.47, 2.15, 2.45, 0.78, PHYS, dashed=False)
    arrow(ax, (1.3, 3.08), (1.3, 2.95), VIRT, lw=2.2)
    arrow(ax, (5.7, 3.08), (5.7, 2.95), PHYS, lw=2.2)
    # row 3
    stage1(ax, 0.08, 0.08, 2.45, 1.68)
    stage2(ax, 2.72, 0.08, 1.56, 1.68)
    cobot_scene(ax, 4.47, 0.08, 2.45, 1.68)
    arrow(ax, (1.3, 2.13), (1.3, 1.78), VIRT, lw=2.2)
    arrow(ax, (5.0, 2.13), (2.25, 1.78), PHYS, lw=2.0)
    arrow(ax, (5.45, 2.13), (3.95, 1.78), PHYS, lw=2.0)
    arrow(ax, (2.55, 0.92), (2.7, 0.92), NET, lw=2.6)
    arrow(ax, (4.3, 0.92), (4.45, 0.92), NET, lw=2.6)
    arrow(ax, (3.5, 1.78), (3.5, 2.06), GOOD, lw=1.8)

    # text labels
    label(ax, 0.22, 4.78, "Digital twin (VR)", VIRT, 9.5)
    label(ax, 2.1, 4.12, "labels for free", VIRT, 7, "center", "normal")
    label(ax, 4.6, 4.78, "Physical workcell", PHYS, 9.5)
    label(ax, 6.52, 4.5, "few labels", PHYS, 7, "center", "normal")
    label(ax, 3.5, 4.76, "Virtual-to-physical gap", DARK, 7.5, "center")
    label(ax, 3.5, 3.28, "Same accuracy, fewer labels", GOOD, 6.8, "center")
    label(ax, 3.06, 3.07, "macro-F1", DARK, 6, "left", "normal")
    label(ax, 4.12, 2.17, "physical labels", DARK, 6, "right", "normal")
    label(ax, 1.42, 3.02, "skeleton clips", VIRT, 6.5, "left", "normal")
    label(ax, 5.82, 3.02, "skeleton clips", PHYS, 6.5, "left", "normal")
    label(ax, 1.75, 1.43, "class prototypes", NET, 6.5, "center", "normal")
    label(ax, 4.6, 1.6, "Collaborative robot assistance", GOOD, 8.5)
    label(ax, 6.32, 1.0, "recognised action", GOOD, 6.5, "right", "normal")
    fig.savefig(OUT / "graphical_abstract.pdf")
    fig.savefig(OUT / "graphical_abstract.eps")
    fig.savefig(OUT / "graphical_abstract.tiff", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(OUT / "graphical_abstract.png", dpi=220)


if __name__ == "__main__":
    main()
