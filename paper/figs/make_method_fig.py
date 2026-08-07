"""Figure 1: the instrument and what it measures.

Four panels across a two-column span (7.0 in):
 (a) the controlled instrument -- frozen recipe + committed subsets, so every
     intervention sees byte-identical images;
 (b) MomentAux -- the fused knowledge source and its decaying weight;
 (c) the two measurements per cell -- end-to-end Delta and probe G;
 (d) the comparator ladder, ordered by declared training cost.

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
fig = plt.figure(figsize=(7.0, 2.35), dpi=300)
fig.patch.set_facecolor("white")


def box(ax, x, y, w, h, label, fc, fs=4.9, tc="white", ec=None, lw=0.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.05,rounding_size=0.18",
                                fc=fc, ec=ec or fc, lw=lw, zorder=2))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            fontsize=fs, color=tc, zorder=3, linespacing=1.15)


def arrow(ax, x1, y1, x2, y2, color=GREY, ls="-", lw=0.65):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=4.2, lw=lw, color=color,
                                 linestyle=ls, zorder=1))


def panel(rect, title):
    ax = fig.add_axes(rect)
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    ax.text(0, 10.3, title, fontsize=5.8, fontweight="bold", color=INK,
            va="top", ha="left")
    return ax


# ---------------------------------------------------------------- (a)
axa = panel([0.012, 0.06, 0.215, 0.80], "(a)  Controlled instrument")
box(axa, 0.2, 7.9, 9.4, 1.7,
    "frozen recipe: SGD 0.1, cosine, 200 ep,\nbatch 128, crop+flip only",
    "#eef2f6", tc=INK, fs=4.6, ec=BLUE, lw=0.5)
box(axa, 0.2, 5.9, 9.4, 1.5,
    "committed subset indices (per dataset,\nper fraction) — version controlled",
    "#eef2f6", tc=INK, fs=4.6, ec=BLUE, lw=0.5)
box(axa, 0.2, 4.2, 9.4, 1.2, "pinned filter bank (fingerprinted)",
    "#eef2f6", tc=INK, fs=4.6, ec=BLUE, lw=0.5)
arrow(axa, 5.0, 4.1, 5.0, 3.2, BLUE)
axa.add_patch(Rectangle((0.2, 0.5), 9.4, 2.6, fc="#f5f5f5", ec="none"))
axa.text(4.9, 2.55, "every intervention sees", ha="center", fontsize=4.7,
         color=MUTED)
axa.text(4.9, 1.75, "byte-identical images", ha="center", fontsize=5.4,
         color=INK, fontweight="bold")
axa.text(4.9, 0.95, "deviations quarantined, never headline", ha="center",
         fontsize=4.4, color=MUTED)

# ---------------------------------------------------------------- (b)
axb = panel([0.263, 0.06, 0.235, 0.80], "(b)  MomentAux: the fused prior")
box(axb, 0.1, 7.6, 1.5, 1.5, "image", "#e6e6e6", tc=INK, fs=4.5)
for i, xx in enumerate([2.0, 3.35, 4.7]):
    box(axb, xx, 7.6, 1.1, 1.5, f"s{i+1}", BLUE, fs=4.6)
box(axb, 6.1, 7.6, 1.6, 1.5, "classifier", "#e6e6e6", tc=INK, fs=4.3)
for x1, x2 in [(1.65, 1.95), (3.15, 3.30), (4.5, 4.65), (5.85, 6.05)]:
    arrow(axb, x1, 8.35, x2, 8.35)
axb.text(7.85, 8.35, "CE", fontsize=4.6, color=INK, va="center")

arrow(axb, 5.25, 7.5, 5.25, 6.4, VERM, ls=(0, (2, 1.2)))
box(axb, 3.9, 5.0, 2.9, 1.35, "aux head\n(train only)", VERM, fs=4.5)
arrow(axb, 3.8, 5.68, 3.15, 5.68, VERM, ls=(0, (2, 1.2)))
box(axb, 0.1, 5.0, 2.9, 1.35, "fixed Gabor\nenergy bank", GREEN, fs=4.5)
axb.text(7.0, 5.68, "MSE·λ", fontsize=4.5, color=VERM, va="center")

# lambda schedule inset
ins = fig.add_axes([0.288, 0.10, 0.075, 0.20])
t = np.linspace(0, 1, 200)
ins.plot(t, 0.5 * (1 + np.cos(np.pi * t)), lw=1.0, color=VERM)
ins.set_xticks([0, 1]); ins.set_xticklabels(["0", "end"], fontsize=4.2)
ins.set_yticks([0, 1]); ins.set_yticklabels(["0", "λ₀"], fontsize=4.2)
ins.tick_params(length=1.5, pad=1)
ins.set_title("λ decays to exactly 0", fontsize=4.4, color=MUTED, pad=1.5)
for s in ins.spines.values():
    s.set_linewidth(0.4)

axb.text(5.0, 4.15, "+2% train compute · +0 at inference", fontsize=4.4,
         color=MUTED, ha="center")

# ---------------------------------------------------------------- (c)
axc = panel([0.535, 0.06, 0.215, 0.80], "(c)  Two measurements per cell")
box(axc, 0.2, 7.8, 4.4, 1.6, "baseline\narm", "#9e9e9e", fs=4.6)
box(axc, 5.4, 7.8, 4.4, 1.6, "prior\narm", GREEN, fs=4.6)
arrow(axc, 2.4, 7.7, 2.4, 6.6); arrow(axc, 7.6, 7.7, 7.6, 6.6)
box(axc, 0.2, 5.1, 4.4, 1.5, "test acc.", "#eef2f6", tc=INK, fs=4.5, ec=LINE)
box(axc, 5.4, 5.1, 4.4, 1.5, "test acc.", "#eef2f6", tc=INK, fs=4.5, ec=LINE)
axc.text(5.0, 4.35, "Δ = end-to-end gain", ha="center", fontsize=5.0,
         color=INK, fontweight="bold")

arrow(axc, 2.4, 5.0, 2.4, 3.5, VERM); arrow(axc, 7.6, 5.0, 7.6, 3.5, VERM)
box(axc, 0.2, 2.0, 9.6, 1.5, "linear probe on FROZEN features\n(identical protocol both arms)",
    VERM, fs=4.4)
axc.text(5.0, 1.25, "G = feature gain", ha="center", fontsize=5.0,
         color=INK, fontweight="bold")
axc.add_patch(Rectangle((0.2, 0.0), 9.6, 0.95, fc="#fdf1e9", ec=VERM, lw=0.5))
axc.text(5.0, 0.47, "Δ = G + readout(base)", ha="center", va="center",
         fontsize=5.4, color=VERM, fontweight="bold")

# ---------------------------------------------------------------- (d)
axd = panel([0.787, 0.06, 0.205, 0.80], "(d)  Comparators, by cost")
ladder = [("ImageNet transfer", "external data", PURPLE),
          ("SimCLR / SimSiam / DINO", "2× compute", ORANGE),
          ("FitNets learned teacher", "2× compute", ORANGE),
          ("DeiT augmentation", "recipe change", BLUE),
          ("HOG target", "1.02×", GREEN),
          ("MomentAux (ours)", "1.02×", GREEN)]
y = 9.2
for name, cost, col in ladder:
    axd.add_patch(Rectangle((0.15, y - 1.05), 0.18, 1.0, fc=col, ec="none"))
    axd.text(0.65, y - 0.18, name, fontsize=4.6, color=INK, va="top")
    axd.text(0.65, y - 0.68, cost, fontsize=4.2, color=MUTED, va="top")
    y -= 1.42
axd.text(0.15, 0.55, "all fused into the SAME frozen recipe;\n"
         "combinations measured too", fontsize=4.3, color=MUTED, va="top",
         linespacing=1.3)

fig.savefig(os.path.join(HERE, "method.pdf"), facecolor="white")
fig.savefig(os.path.join(HERE, "method.png"), facecolor="white")
print("method figure written")
