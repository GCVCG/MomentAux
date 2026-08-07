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
FIG_W_IN, FIG_H_IN = 7.0, 2.60
fig = plt.figure(figsize=(FIG_W_IN, FIG_H_IN), dpi=300)
fig.patch.set_facecolor("white")


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
axa, UA = panel([0.012, 0.06, 0.215, 0.79], "(a)  Controlled Instrument")
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
axb, UB = panel([0.263, 0.06, 0.235, 0.79], "(b)  MomentAux: The Fused Prior")

PY, PH = 7.82, 1.35
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
for x1, x2 in [(1.31, 1.60), (7.14, 7.44)]:
    arrow(axb, x1, PY + PH / 2, x2, PY + PH / 2)
axb.text(8.68, PY - 0.34, "Cross-Entropy", ha="center", va="top",
         fontsize=4.0, color=MUTED)

TAP_X = STAGE_X[2] + 0.58
arrow(axb, TAP_X, PY - 0.24, TAP_X, 6.30, VERM, ls=(0, (2, 1.2)))
axb.text(TAP_X + 0.22, 7.10, "Tap: Deep Stage (3 of 4)", fontsize=4.1,
         color=VERM, ha="left", va="center")
box(axb, 3.95, 4.90, 3.15, 1.35, "Aux Head\nTraining Only", VERM, UB, fs=4.4)
arrow(axb, 3.85, 5.57, 3.20, 5.57, VERM, ls=(0, (2, 1.2)))
box(axb, 0.05, 4.90, 3.05, 1.35, "Fixed Gabor\nEnergy Bank", GREEN, UB, fs=4.4)
axb.text(7.30, 5.57, "MSE × λ(t)", fontsize=4.3, color=VERM, va="center")
axb.text(5.0, 4.20, "+2% Training Compute, +0 at Inference", fontsize=4.3,
         color=MUTED, ha="center")

# --- lambda schedule: say what it plots and what follows from it
ins = fig.add_axes([0.300, 0.115, 0.105, 0.215])
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
axc, UC = panel([0.535, 0.06, 0.215, 0.79], "(c)  Two Measurements Per Cell")

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
axd, UD = panel([0.787, 0.06, 0.205, 0.79], "(d)  Comparators, By Cost")
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

fig.savefig(os.path.join(HERE, "method.pdf"), facecolor="white")
fig.savefig(os.path.join(HERE, "method.png"), facecolor="white")
print("method figure written")
