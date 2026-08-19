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
FIG_W_IN, FIG_H_IN = 7.0, 2.78
ROW_H_IN, ROW_TOP_IN, TITLE_IN, SCOPE_IN = 2.16, 0.24, 0.22, 0.16
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
            va="top", ha="left")
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
arrow(axa, 5.0, 4.1, 5.0, 3.2, BLUE)
axa.add_patch(Rectangle((0.2, 0.5), 9.4, 2.6, fc="#f5f5f5", ec="none"))
axa.text(4.9, 2.55, "Every Intervention Sees", ha="center", fontsize=4.7,
         color=MUTED)
axa.text(4.9, 1.75, "Byte-Identical Images", ha="center", fontsize=5.4,
         color=INK, fontweight="bold")
axa.text(4.9, 0.95, "Deviations Quarantined, Never Headline", ha="center",
         fontsize=4.4, color=MUTED)

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
    arrow(axb, IMX + iw / 2, iy0 - 0.10, IMX + iw / 2, 6.42, GREEN)
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
arrow(axb, TAP_X, PY - 0.24, TAP_X, 6.18, VERM, ls=(0, (2, 1.2)))
axb.text(TAP_X + 0.22, 7.10, "Tap: Deep Stage (3 of 4)", fontsize=4.1,
         color=VERM, ha="left", va="center")
# The bank box shows the bank AS a bank: one row of kernels over the row of
# magnitude maps each one produces on the image above, so the correspondence
# is visible rather than asserted. Four orientations at one scale, which is
# the smallest set that still reads as oriented.
BKX, BKY, BKW, BKH = 0.0, 4.62, 4.30, 1.80
box(axb, BKX, BKY, BKW, BKH, "", GREEN, UB, fs=4.4)
axb.text(BKX + BKW / 2, BKY + BKH - 0.30, "Fixed Gabor Energy Bank",
         ha="center", va="center", zorder=4, color="white",
         fontsize=fit_fontsize("Fixed Gabor Energy Bank",
                               (BKW - 0.30) * UB, 4.2))
if B_KERNS is not None:
    ub_x = PX["b"][1] * FIG_W_IN / 10.0
    ub_y = R1_H * FIG_H_IN / 10.0
    n = len(B_KERNS)
    gap, pad, hdr, vgap = 0.09, 0.16, 0.46, 0.07
    # tiles must fit the box in BOTH directions: width sets the natural size,
    # and the two rows plus the header cap it in height. Taking the smaller of
    # the two is what stops them overflowing the box when the panel narrows.
    tw = (BKW - 2 * pad - (n - 1) * gap) / n
    th = tw * ub_x / ub_y
    th_max = (BKH - hdr - vgap - 0.12) / 2
    if th > th_max:
        th = th_max
        tw = th * ub_y / ub_x
    x00 = BKX + (BKW - (n * tw + (n - 1) * gap)) / 2
    for row, (arrs, cm) in enumerate(((B_KERNS, "RdBu_r"), (B_MAPS, "magma"))):
        ty0 = BKY + BKH - hdr - (row + 1) * th - row * vgap
        for j_, arr in enumerate(arrs):
            x0 = x00 + j_ * (tw + gap)
            axb.imshow(arr, extent=(x0, x0 + tw, ty0, ty0 + th), cmap=cm,
                       zorder=4, aspect="auto")
box(axb, 4.62, 4.74, 2.62, 1.35, "Aux Head\nTraining Only", VERM, UB, fs=4.3)
arrow(axb, 4.52, 5.42, 4.20, 5.42, VERM, ls=(0, (2, 1.2)))
axb.text(7.38, 5.42, "MSE × λ(t)", fontsize=4.3, color=VERM, va="center")
axb.text(5.0, 4.06, "+2% Training Compute, +0 at Inference", fontsize=4.3,
         color=MUTED, ha="center")

# --- lambda schedule: say what it plots and what follows from it
# Placed in figure coordinates, so it has to follow the row when the row moves:
# 0.143 in above the row's floor, 0.559 in tall, as in the four-panel version.
ins = fig.add_axes([0.300, (R1_Y * FIG_H_IN + 0.143) / FIG_H_IN,
                    0.105, 0.559 / FIG_H_IN])
t = np.linspace(0, 1, 300)
lam = 0.5 * (1 + np.cos(np.pi * t))
ins.plot(t, lam, lw=1.0, color=VERM)
ins.fill_between(t, 0, lam, color=VERM, alpha=0.13, lw=0)
ins.set_xlim(0, 1); ins.set_ylim(0, 1.13)
ins.set_xticks([0, 1]); ins.set_xticklabels(["Start", "End"], fontsize=4.1)
ins.set_yticks([0, 1]); ins.set_yticklabels(["0", "λ₀"], fontsize=4.1)
ins.tick_params(length=1.5, pad=1)
ins.set_xlabel("Training Progress", fontsize=4.2, labelpad=1.0)
ins.set_ylabel("Aux Weight λ(t)", fontsize=4.2, labelpad=1.0)
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
axd, UD = panel([PX["d"][0], R1_Y, PX["d"][1], R1_H], "(d)  Comparators, By Cost")
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
# The answer, as a fifth column: three stacked blocks, one per outcome, each a
# small grouped bar of the two sources ALONE and then together, all as gain
# over the same from-scratch baseline and on ONE shared scale so the blocks
# are comparable. The outcome is then the shape of the group -- the combined
# bar clears both singles, sits level with the taller, or drops below zero.
axe, UEp = panel([PX["e"][0], R1_Y, PX["e"][1], R1_H], "(e)  When Fusion Pays")
GREY1, GREY2 = "#bdbdbd", "#7d7d7d"
blocks = [
    ("STACK", GREEN, "Prior + Augmentation", "Food-101 5%",
     +5.63, +9.46, +13.78),
    ("SUBSTITUTE", ORANGE, "Prior + SimCLR Init", "ViT-Tiny CIFAR-100 10%",
     +13.26, +13.30, +13.46),
    ("INTERFERE", VERM, "Prior + ImageNet Init", "CIFAR-100 7%",
     +4.87, +15.39, -1.46),
]
VMIN, VMAX = -2.6, 16.6                      # one shared scale for all three
BLK_H, BLK_TOP = 3.14, 9.90                  # block pitch and first block top
BAR_W, BAR_GAP = 1.62, 0.42
for b, (verdict, col, pair, cell, va_, vb_, vc_) in enumerate(blocks):
    top = BLK_TOP - b * BLK_H
    axe.add_patch(FancyBboxPatch((0.05, top - 0.72), 4.35, 0.72,
                                 boxstyle="round,pad=0.02,rounding_size=0.22",
                                 fc=col, ec=col, lw=0.5, zorder=3))
    axe.text(2.22, top - 0.36, verdict, ha="center", va="center", zorder=4,
             fontsize=fit_fontsize(verdict, 4.0 * UEp, 4.4, bold=True),
             color="white", fontweight="bold")
    axe.text(4.62, top - 0.20, pair, ha="left", va="center", fontsize=3.9,
             color=INK)
    axe.text(4.62, top - 0.58, cell, ha="left", va="center", fontsize=3.5,
             color=MUTED)
    # plot area for this block
    ay1, ay0 = top - 0.98, top - 2.62
    def _y(v):
        return ay0 + (v - VMIN) / (VMAX - VMIN) * (ay1 - ay0)
    zero = _y(0.0)
    axe.plot([0.05, 9.95], [zero, zero], lw=0.5, color=LINE, zorder=1)
    x = 0.55
    for v, c in ((va_, GREY1), (vb_, GREY2), (vc_, col)):
        h = _y(v) - zero
        axe.add_patch(Rectangle((x, zero if h >= 0 else zero + h), BAR_W,
                                abs(h), fc=c, ec="none", zorder=3))
        if h >= 0:
            axe.text(x + BAR_W / 2, _y(v) + 0.09,
                     f"{v:+.1f}", ha="center", va="bottom", fontsize=3.5,
                     color=c, fontweight="bold" if c is col else "normal")
        else:
            axe.text(x + BAR_W + 0.12, zero + h / 2,
                     f"{v:+.1f}".replace("-", "\u2212"), ha="left",
                     va="center", fontsize=3.5, color=c, fontweight="bold")
        x += BAR_W + BAR_GAP
    if b == 0:
        for k, lbl in enumerate(("source 1", "source 2", "both")):
            axe.text(0.55 + k * (BAR_W + BAR_GAP) + BAR_W / 2, ay0 - 0.04,
                     lbl, ha="center", va="top", fontsize=3.4, color=MUTED)

# ------------------------------------------------------- scope strip
fig.add_artist(plt.Line2D([0.010, 0.992], [SCOPE_IN / FIG_H_IN - 0.028,
                                           SCOPE_IN / FIG_H_IN - 0.028],
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
