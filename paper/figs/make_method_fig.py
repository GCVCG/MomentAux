"""Figure 1: the instrument and what it measures.

Four panels across a two-column span (7.0 in):
 (a) the controlled instrument -- frozen recipe + committed subsets, so every
     intervention sees byte-identical images;
 (b) MomentAux -- the fused knowledge source and its decaying weight;
 (c) the two measurements per cell -- end-to-end Delta and feature gain G;
 (d) the comparator ladder, ordered by declared training cost.

Revision 2026-08-07, author requests:
  * Title Case everywhere, matching the graphical abstract;
  * panel (b) redrawn to match the graphical abstract's fusion panel: FOUR
    backbone stages (not three) named by feature depth inside a labelled
    container, with the tap on the THIRD of four (ResNet layer3). The old
    three-box drawing put the tap on the final stage, which is wrong;
  * the lambda inset now says what it plots (axis labels + the consequence),
    instead of being an unlabelled curve;
  * panel (c) relaid out so no label sits under an arrow, and so every box
    that feeds another has an arrow to it;
  * "probe" replaced by "Linear Evaluation", the formal term.

No LaTeX escapes (matplotlib mathtext is not LaTeX).
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

BLUE, VERM, GREEN = "#0072B2", "#D55E00", "#009E73"
ORANGE, PURPLE, GREY = "#E69F00", "#CC79A7", "#8a8a8a"
INK, MUTED, LINE = "#1a1a1a", "#5c5c5c", "#b0b0b0"

plt.rcParams.update({
    "font.size": 6, "axes.linewidth": 0.5, "pdf.fonttype": 42,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.labelsize": 5, "ytick.labelsize": 5,
    "axes.spines.top": False, "axes.spines.right": False,
    "text.color": INK, "axes.labelcolor": INK})

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
# Two rows. The top row is the instrument and what it measures; the bottom row
# is the ANSWER -- the three fusion outcomes the paper's question asks for --
# and the scope those outcomes were measured over. R1_* keeps the top row's
# internal geometry identical to the four-panel version it grew from: the
# panels are given the same 2.06 in of height, only moved up.
FIG_W_IN, FIG_H_IN = 7.0, 2.86
# SCOPE_IN is the strip's total height, and it has to hold BOTH the separator
# rule and the line of text beneath it. At 0.16 in the rule sat ~2 pt off the
# text and read as an underline; 0.24 in puts ~5 pt of air on each side of it.
ROW_H_IN, ROW_TOP_IN, TITLE_IN, SCOPE_IN = 2.16, 0.24, 0.22, 0.24
R1_Y = SCOPE_IN / FIG_H_IN
R1_H = ROW_H_IN / FIG_H_IN
# Five columns in one row. The inter-panel gutters were the slack: trimming
# them from ~0.03 to 0.018 of the width is what makes room for (e) without
# taking any width from the four panels that were already tight.
PX = {}
_x, _gap = 0.010, 0.018
for _k, _w in (("a", 0.166), ("b", 0.222), ("c", 0.172), ("d", 0.152),
               ("e", 0.198)):
    PX[_k] = (_x, _w)
    _x += _w + _gap
fig = plt.figure(figsize=(FIG_W_IN, FIG_H_IN), dpi=300)
fig.patch.set_facecolor("white")

fig.text(0.012, 1 - 0.055, "When Does Fusing Hand-Crafted Knowledge With "
         "Learned Representations Pay?", fontsize=7.4, fontweight="bold",
         color=INK, va="center", ha="left")
fig.add_artist(plt.Line2D([0.012, 0.992], [1 - 0.093, 1 - 0.093],
                          color=LINE, lw=0.6))


def text_width_pt(s, fontsize, bold=False):
    """Conservative advance-width estimate for DejaVu Sans, in points."""
    w = 0.0
    for ch in s:
        if ch in "IJil.,:;'|! ":
            w += 0.30
        elif ch in "MW@":
            w += 0.92
        elif ch.isupper() or ch.isdigit():
            w += 0.68
        else:
            w += 0.56
    return w * fontsize * (1.06 if bold else 1.0)


def fit_fontsize(label, box_w_pt, start, floor=3.2, bold=False):
    lines = label.split("\n")
    fs = start
    while fs > floor:
        if max(text_width_pt(l, fs, bold) for l in lines) <= box_w_pt:
            break
        fs -= 0.1
    return fs


def box(ax, x, y, w, h, label, fc, unit, fs=4.9, tc="white", ec=None,
        lw=0.5, pad=0.35):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.05,rounding_size=0.18",
                                fc=fc, ec=ec or fc, lw=lw, zorder=2))
    fs = fit_fontsize(label, (w - pad) * unit, fs)
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            fontsize=fs, color=tc, zorder=3, linespacing=1.18)


def arrow(ax, x1, y1, x2, y2, color=GREY, ls="-", lw=0.65, z=1):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=4.2, lw=lw, color=color,
                                 linestyle=ls, zorder=z))


def panel(rect, title):
    ax = fig.add_axes(rect)
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    ax.text(0, 10.35, title, fontsize=5.8, fontweight="bold", color=INK,
            va="top", ha="left", linespacing=1.05)
    return ax, rect[2] * FIG_W_IN * 72.0 / 10.0


# ---------------------------------------------------------------- (a)
axa, UA = panel([PX["a"][0], R1_Y, PX["a"][1], R1_H], "(a)  Controlled Instrument")
box(axa, 0.2, 7.9, 9.4, 1.7,
    "Frozen Recipe: SGD 0.1, Cosine,\n200 Epochs, Batch 128, Crop and Flip",
    "#eef2f6", UA, tc=INK, fs=4.6, ec=BLUE, lw=0.5)
box(axa, 0.2, 5.9, 9.4, 1.5,
    "Committed Subset Indices,\nPer Dataset and Fraction",
    "#eef2f6", UA, tc=INK, fs=4.6, ec=BLUE, lw=0.5)
box(axa, 0.2, 4.2, 9.4, 1.2, "Pinned Filter Bank (Fingerprinted)",
    "#eef2f6", UA, tc=INK, fs=4.6, ec=BLUE, lw=0.5)
# NOT arrows between the three: they are simultaneous controls, and an arrow
# would assert a pipeline that does not exist. "+" says all three at once,
# and the single arrow below carries what they jointly buy.
for gy in (7.65, 5.65):
    axa.text(4.9, gy, "+", ha="center", va="center", fontsize=6.0, color=BLUE)
arrow(axa, 4.9, 4.1, 4.9, 3.2, BLUE)
axa.add_patch(Rectangle((0.2, 0.22), 9.4, 2.88, fc="#f5f5f5", ec="none"))
axa.text(4.9, 2.62, "Every Intervention Sees", ha="center", fontsize=4.7,
         color=MUTED)
axa.text(4.9, 1.92, "Byte-Identical Images", ha="center", fontsize=5.4,
         color=INK, fontweight="bold")
# two lines: at 4.4 pt this ran ~15 pt past the panel on one
axa.text(4.9, 0.92, "Deviations Quarantined,\nNever Headline", ha="center",
         va="center", fontsize=4.4, color=MUTED, linespacing=1.30)

# ---------------------------------------------------------------- (b)
def _load_assets():
    """Real content for panel (b): the deterministic STL-10 sample fig:bank
    uses, two pinned even kernels, and two calibrated magnitude maps.
    Synthetic-free: on any failure the panel keeps its plain boxes."""
    try:
        sys.path.insert(0, ROOT)
        import torch
        import data as data_mod
        from momentstem import EnergyStem
        stem = EnergyStem(feature_type="magnitude")
        stem.calibrate(data_mod.calibration_batch("stl10",
                                                  os.path.join(ROOT, "data")))
        ds = data_mod.build_dataset("stl10", os.path.join(ROOT, "data"),
                                    train=False)
        x = ds[613][0].unsqueeze(0)
        mean, std = data_mod.STATS["stl10"]
        img = (x[0] * torch.tensor(std).view(3, 1, 1)
               + torch.tensor(mean).view(3, 1, 1)).clamp(0, 1)
        img = img.permute(1, 2, 0).numpy()
        with torch.no_grad():
            resp = (stem._energy(stem._luma(x))
                    * stem.calib_scale.view(1, -1, 1, 1))[0]
        # one scale, all four orientations: the smallest set that still reads
        # as a bank, with each map produced by the kernel directly above it.
        kerns = [stem.even.squeeze(1)[i].numpy() for i in range(4)]
        maps = [resp[i].numpy() for i in range(4)]
        return img, kerns, maps
    except Exception as e:                                    # noqa: BLE001
        print(f"method-fig assets unavailable ({e}); plain boxes kept")
        return None, None, None


B_IMG, B_KERNS, B_MAPS = _load_assets()

axb, UB = panel([PX["b"][0], R1_Y, PX["b"][1], R1_H], "(b)  MomentAux: The Fused Prior")

PY, PH = 7.82, 1.35
if B_IMG is not None:
    ub_x = PX["b"][1] * FIG_W_IN / 10.0
    ub_y = R1_H * FIG_H_IN / 10.0
    iw = 1.20
    ih = iw * ub_x / ub_y
    iy0 = PY + PH / 2 - ih / 2
    IMX = 0.0
    axb.imshow(B_IMG, extent=(IMX, IMX + iw, iy0, iy0 + ih), zorder=3,
               aspect="auto")
    axb.add_patch(Rectangle((IMX, iy0), iw, ih, fc="none", ec=INK, lw=0.45,
                            zorder=4))
    axb.text(IMX + iw / 2, iy0 + ih + 0.10, "Image", ha="center", va="bottom",
             fontsize=3.9, color=MUTED)
    # the bank reads the SAME image: an explicit arrow down to it, which the
    # figure previously left the reader to infer
    arrow(axb, IMX + iw / 2, iy0 - 0.10, IMX + iw / 2, 6.20, GREEN)
else:
    box(axb, 0.05, PY, 1.25, PH, "Image", "#e6e6e6", UB, tc=INK, fs=4.4)
axb.add_patch(FancyBboxPatch((1.62, PY - 0.18), 5.50, PH + 0.36,
                             boxstyle="round,pad=0.05,rounding_size=0.18",
                             fc="none", ec=GREY, lw=0.5, ls=(0, (2, 1.4)),
                             zorder=1))
STAGE_X = [1.74, 3.10, 4.46, 5.82]
for xx, name in zip(STAGE_X, ["Early", "Mid", "Deep", "Final"]):
    box(axb, xx, PY, 1.16, PH, name, BLUE, UB, fs=4.2, pad=0.16)
# label ABOVE: below the container the tap arrow would run through it
axb.text(1.66, PY + PH + 0.52, "Backbone Feature Stages, Unchanged",
         ha="left", va="top", fontsize=4.0, color=MUTED)
box(axb, 7.46, PY, 2.45, PH, "Classifier", "#e6e6e6", UB, tc=INK, fs=4.3)
for x1, x2 in [(1.26, 1.58), (7.14, 7.44)]:
    arrow(axb, x1, PY + PH / 2, x2, PY + PH / 2)
axb.text(8.68, PY - 0.34, "Cross-Entropy", ha="center", va="top",
         fontsize=4.0, color=MUTED)

TAP_X = STAGE_X[2] + 0.58
# a gentle diagonal, so the tap lands near the aux head's middle instead of on
# its top-left corner
arrow(axb, TAP_X, PY - 0.24, TAP_X, 6.26, VERM, ls=(0, (2, 1.2)))
# left of the arrow's origin: set to its right the label ran under the arrow
# itself and out past the panel edge. "(3 of 4)" went with it -- the stage
# boxes are already named, so the count restated what the drawing shows.
axb.text(TAP_X - 0.16, 7.18, "Tap: Deep Stage", fontsize=4.1,
         color=VERM, ha="right", va="center")
# The bank box shows the bank AS a bank: one row of kernels over the row of
# magnitude maps each one produces on the image above, so the correspondence
# is visible rather than asserted. Four orientations at one scale, which is
# the smallest set that still reads as oriented.
BKX, BKY, BKW, BKH = 0.0, 4.72, 3.02, 1.44
box(axb, BKX, BKY, BKW, BKH, "", GREEN, UB, fs=4.4)
# The bank's name sits ABOVE the box, where it has the panel's full width. Set
# inside it, the label forced the tiles into a narrower box and still ran past
# its own edges at any legible size. Two lines, and clear of the arrow coming
# down from the image at x = 0.60.
axb.text(1.00, BKY + BKH + 0.12, "Fixed Gabor\nEnergy Bank", ha="left",
         va="bottom", color=GREEN, fontsize=4.0, linespacing=1.15)
if B_KERNS is not None:
    ub_x = PX["b"][1] * FIG_W_IN / 10.0
    ub_y = R1_H * FIG_H_IN / 10.0
    n = len(B_KERNS)
    gap, pad, vgap = 0.06, 0.12, 0.07
    # tiles must fit the box in BOTH directions: width sets the natural size,
    # and the two rows cap it in height. Taking the smaller of the two is what
    # stops them overflowing the box when the panel narrows.
    tw = (BKW - 2 * pad - (n - 1) * gap) / n
    th = tw * ub_x / ub_y
    th_max = (BKH - 2 * pad - vgap) / 2
    if th > th_max:
        th = th_max
        tw = th * ub_y / ub_x
    x00 = BKX + (BKW - (n * tw + (n - 1) * gap)) / 2
    y00 = BKY + (BKH + 2 * th + vgap) / 2      # centred, not hung from the top
    for row, (arrs, cm) in enumerate(((B_KERNS, "RdBu_r"), (B_MAPS, "magma"))):
        ty0 = y00 - (row + 1) * th - row * vgap
        for j_, arr in enumerate(arrs):
            x0 = x00 + j_ * (tw + gap)
            axb.imshow(arr, extent=(x0, x0 + tw, ty0, ty0 + th), cmap=cm,
                       zorder=4, aspect="auto")
# centred on the tap, so the arrow from the deep stage drops straight into it
AHW, AHH = 2.44, 1.44
AHX, AHY = TAP_X - AHW / 2, 4.72
box(axb, AHX, AHY, AHW, AHH, "Aux Head\nTraining Only", VERM, UB, fs=4.3)
# the bank supplies the regression TARGET, so the arrow runs bank -> head. It
# previously spanned 0.32 units between two touching boxes and never showed.
arrow(axb, BKX + BKW + 0.08, BKY + BKH / 2, AHX - 0.10, AHY + AHH / 2,
      GREEN, lw=0.8)
axb.text(AHX + AHW + 0.22, AHY + AHH / 2, "MSE × λ(t)", fontsize=4.3,
         color=VERM, va="center")
axb.text(5.0, 4.06, "+2% Training Compute, +0 at Inference", fontsize=4.3,
         color=MUTED, ha="center")

# --- lambda schedule: say what it plots and what follows from it
# Placed in figure coordinates, so it has to follow the row when the row moves:
# 0.143 in above the row's floor, 0.559 in tall, as in the four-panel version.
# Centred on column (b): the title is the widest element, so it is the title
# that is centred, which puts the axes box itself on the column's midline.
INS_W = 0.105
ins = fig.add_axes([PX["b"][0] + PX["b"][1] / 2 - INS_W / 2,
                    (R1_Y * FIG_H_IN + 0.143) / FIG_H_IN,
                    INS_W, 0.559 / FIG_H_IN])
t = np.linspace(0, 1, 300)
lam = 0.5 * (1 + np.cos(np.pi * t))
ins.plot(t, lam, lw=1.0, color=VERM)
ins.fill_between(t, 0, lam, color=VERM, alpha=0.13, lw=0)
ins.set_xlim(0, 1); ins.set_ylim(0, 1.26)     # headroom for a level y label
ins.set_xticks([0, 1]); ins.set_xticklabels(["Start", "End"], fontsize=4.1)
ins.set_yticks([0, 1]); ins.set_yticklabels(["0", "λ₀"], fontsize=4.1)
ins.tick_params(length=1.5, pad=1)
ins.set_xlabel("Training Progress", fontsize=4.2, labelpad=1.0)
# set as a y-label matplotlib turns this on its side, and it was then the only
# rotated element anywhere in the manuscript. Laid flat in the axes headroom.
ins.text(0.03, 0.985, "Aux Weight λ(t)", transform=ins.transAxes, ha="left",
         va="top", fontsize=4.2, color=INK)
ins.set_title("Prior Dominates Early, Then Vanishes", fontsize=4.4,
              color=INK, pad=2.0)
ins.annotate("λ = 0 Exactly:\nPure Cross-Entropy", xy=(0.975, 0.035),
             xytext=(0.70, 0.83), fontsize=3.9, color=VERM, ha="center",
             va="center", linespacing=1.25,
             arrowprops=dict(arrowstyle="-|>", lw=0.5, color=VERM,
                             shrinkA=1, shrinkB=1, mutation_scale=3.6))
for s in ins.spines.values():
    s.set_linewidth(0.4)

# ---------------------------------------------------------------- (c)
axc, UC = panel([PX["c"][0], R1_Y, PX["c"][1], R1_H], "(c)  Two Measurements")

box(axc, 0.2, 8.45, 4.4, 1.35, "Baseline Arm", "#9e9e9e", UC, fs=4.6)
box(axc, 5.4, 8.45, 4.4, 1.35, "Prior Arm", GREEN, UC, fs=4.6)
arrow(axc, 2.4, 8.35, 2.4, 7.55); arrow(axc, 7.6, 8.35, 7.6, 7.55)
box(axc, 0.2, 6.15, 4.4, 1.35, "Test Accuracy", "#eef2f6", UC, tc=INK,
    fs=4.5, ec=LINE)
box(axc, 5.4, 6.15, 4.4, 1.35, "Test Accuracy", "#eef2f6", UC, tc=INK,
    fs=4.5, ec=LINE)

# Delta is the DIFFERENCE of the two arms: draw it as a bracket joining them,
# with the label in the clear gap between the two descending arrows.
for x in (2.4, 7.6):
    axc.plot([x, x], [6.05, 5.60], lw=0.6, color=INK, zorder=1)
axc.plot([2.4, 7.6], [5.60, 5.60], lw=0.6, color=INK, zorder=1)
axc.text(5.0, 5.18, "Δ = End-to-End Gain", ha="center", va="top",
         fontsize=fit_fontsize("Δ = End-to-End Gain", 9.4 * UC, 5.0,
                               bold=True),
         color=INK, fontweight="bold")

# the frozen-feature evaluation reads the same checkpoints, not the accuracies
for x in (1.05, 8.95):
    axc.plot([x, x], [6.05, 4.05], lw=0.6, color=VERM, ls=(0, (2, 1.2)),
             zorder=1)
    arrow(axc, x, 4.15, x, 3.72, VERM, ls=(0, (2, 1.2)))
axc.text(1.30, 4.55, "Same\nCheckpoints", fontsize=3.9, color=VERM,
         ha="left", va="center", linespacing=1.2)

box(axc, 0.2, 2.30, 9.6, 1.40,
    "Linear Evaluation on Frozen Features\n(Identical Protocol, Both Arms)",
    VERM, UC, fs=4.4)
arrow(axc, 5.0, 2.22, 5.0, 1.80, VERM)
axc.text(5.0, 1.66, "G = Feature Gain", ha="center", va="top", fontsize=5.0,
         color=INK, fontweight="bold")
arrow(axc, 5.0, 1.18, 5.0, 0.86, VERM)
axc.add_patch(Rectangle((0.2, 0.0), 9.6, 0.82, fc="#fdf1e9", ec=VERM, lw=0.5))
axc.text(5.0, 0.40, "Δ = G + readout(base)", ha="center", va="center",
         fontsize=fit_fontsize("Δ = G + readout(base)", 9.2 * UC, 5.4,
                               bold=True),
         color=VERM, fontweight="bold")

# ---------------------------------------------------------------- (d)
axd, UD = panel([PX["d"][0], R1_Y, PX["d"][1], R1_H],
                "(d)  Comparators,\n       By Cost")
ladder = [("ImageNet Transfer", "External Data", PURPLE),
          ("SimCLR, SimSiam, DINO", "2× Compute", ORANGE),
          ("FitNets Learned Teacher", "2× Compute", ORANGE),
          ("DeiT Augmentation", "Recipe Change", BLUE),
          ("HOG Target", "1.02×", GREEN),
          ("MomentAux (Ours)", "1.02×", GREEN)]
y = 9.2
for name, cost, col in ladder:
    axd.add_patch(Rectangle((0.15, y - 1.05), 0.18, 1.0, fc=col, ec="none"))
    axd.text(0.65, y - 0.18, name, va="top", color=INK,
             fontsize=fit_fontsize(name, 9.1 * UD, 4.6))
    axd.text(0.65, y - 0.68, cost, fontsize=4.2, color=MUTED, va="top")
    y -= 1.42
axd.text(0.15, 0.62, "All Fused Into the Same Frozen Recipe;\n"
         "Combinations Measured Too", fontsize=4.3, color=MUTED, va="top",
         linespacing=1.3)

# ---------------------------------------------------------------- (e)
# The answer, as a fifth column: for each outcome, the two sources ALONE and
# then BOTH, as three labelled horizontal bars on one shared scale of gain over
# the same from-scratch baseline. Named rows and a printed value per bar mean
# the chart needs no key and no decoding: the "Both" bar is longer than either
# single, level with them, or dragged back across zero. Earlier drafts asked
# the reader to learn an encoding first (which height is which source; bar
# against tick), which is exactly what a one-glance panel cannot afford.
axe, UEp = panel([PX["e"][0], R1_Y, PX["e"][1], R1_H], "(e)  When Fusion Pays")
SOLO = "#d0d0d0"
blocks = [
    ("STACK", GREEN, "Food-101 5%",
     (("Prior", +5.63), ("Augmentation", +9.46), ("Both", +13.78))),
    ("SUBSTITUTE", ORANGE, "ViT-Tiny CIFAR-100 10%",
     (("Prior", +13.26), ("SimCLR Init", +13.30), ("Both", +13.46))),
    ("INTERFERE", VERM, "CIFAR-100 7%",
     (("Prior", +4.87), ("ImageNet Init", +15.39), ("Both", -1.46))),
]
VMIN, VMAX = -3.0, 16.5                    # one shared scale for all three
BX0, BX1 = 2.72, 8.50                      # the bars' own span
BLK_TOP, BLK_H = 9.55, 3.05                # clear of the panel title
ROW_P, BARH = 0.60, 0.42


def _ex(v):
    return BX0 + (v - VMIN) / (VMAX - VMIN) * (BX1 - BX0)


ZX = _ex(0.0)
for b, (verdict, col, cell, rows) in enumerate(blocks):
    top = BLK_TOP - b * BLK_H
    axe.add_patch(FancyBboxPatch((0.05, top - 0.58), 2.86, 0.56,
                                 boxstyle="round,pad=0.02,rounding_size=0.20",
                                 fc=col, ec=col, lw=0.5, zorder=3))
    axe.text(1.48, top - 0.30, verdict, ha="center", va="center", zorder=4,
             fontsize=fit_fontsize(verdict, 2.60 * UEp, 4.4, bold=True),
             color="white", fontweight="bold")
    axe.text(3.10, top - 0.30, cell, ha="left", va="center", fontsize=3.5,
             color=MUTED)
    ry0 = top - 1.06
    axe.plot([ZX, ZX], [ry0 + BARH * 0.85,
                        ry0 - 2 * ROW_P - BARH * 0.85],
             lw=0.5, color="#cfcfcf", zorder=1)
    for i, (name, v) in enumerate(rows):
        ry = ry0 - i * ROW_P
        both = name == "Both"
        axe.text(BX0 - 0.16, ry, name, ha="right", va="center", fontsize=3.5,
                 color=INK if both else MUTED,
                 fontweight="bold" if both else "normal")
        xv = _ex(v)
        axe.add_patch(Rectangle((min(ZX, xv), ry - BARH / 2), abs(xv - ZX),
                                BARH, fc=col if both else SOLO, ec="none",
                                zorder=3))
        axe.text(9.92, ry, f"{v:+.1f}".replace("-", "\u2212"), ha="right",
                 va="center", fontsize=3.5, color=col if both else MUTED,
                 fontweight="bold" if both else "normal")
axe.text(ZX, BLK_TOP - 2 * BLK_H - 1.06 - 2 * ROW_P - BARH / 2 - 0.14, "0",
         ha="center", va="top", fontsize=3.3, color=MUTED)
axe.text(5.0, 0.44, "Accuracy Gain Over From-Scratch Baseline (pts)",
         ha="center", va="top", fontsize=3.4, color=MUTED)

# ------------------------------------------------------- scope strip
fig.add_artist(plt.Line2D([0.010, 0.992], [SCOPE_IN / FIG_H_IN - 0.030,
                                           SCOPE_IN / FIG_H_IN - 0.030],
                          color=LINE, lw=0.6))
fig.text(0.010, 0.020, "Measured Over  21 Datasets  ·  9 Backbones  ·  "
         "500 to 1.28M Images  ·  32 to 224 px  ·  Classification, "
         "Segmentation and Detection", fontsize=4.6, color=MUTED,
         va="center", ha="left")

# bbox_inches="tight" matters here: without it matplotlib writes the whole
# canvas, and this figure's hand-placed boxes leave ~23pt empty at the top and
# ~23pt at the right. Included at \\linewidth that became a visible white band
# above the diagram inside the float -- 17% of the graphic height was blank.
# pad_inches keeps a hairline so strokes on the edge are not clipped.
fig.savefig(os.path.join(HERE, "method.pdf"), facecolor="white",
            bbox_inches="tight", pad_inches=0.02)
fig.savefig(os.path.join(HERE, "method.png"), facecolor="white",
            bbox_inches="tight", pad_inches=0.02)
print("method figure written")
