"""Graphical abstract for the Information Fusion submission.

Elsevier spec: min 531x1328 px (h x w), readable at 5x13 cm. Rendered at
13x8 cm / 300 dpi = 1535x945 px, so fonts are true print points.

Three beats, left to right: (1) what is fused and at what cost, (2) the law
that organizes the grid, (3) when fusion pays, ties, or taxes.
Palette: Okabe-Ito (CVD-safe), fixed assignment.

NOTE: no LaTeX escapes in labels -- matplotlib mathtext is not LaTeX, so
"\%" and "\," render literally. Use plain % and unicode dashes.
"""
import csv, math, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

BLUE, VERM, GREEN = "#0072B2", "#D55E00", "#009E73"
ORANGE, GREY = "#E69F00", "#8a8a8a"
INK, MUTED = "#1a1a1a", "#5c5c5c"

plt.rcParams.update({
    "font.size": 6, "axes.labelsize": 6, "xtick.labelsize": 5.5,
    "ytick.labelsize": 5.5, "axes.linewidth": 0.5,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "text.color": INK, "axes.labelcolor": INK})

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "..", "..", "results", "all_results.csv")
LO, HI = 31.8, 40.3

fig = plt.figure(figsize=(13 / 2.54, 6.4 / 2.54), dpi=300)
fig.patch.set_facecolor("white")

fig.text(0.5, 0.985, "A controlled, cost-normalized benchmark of data-efficiency "
         "interventions, and the law that organizes it", ha="center",
         va="top", fontsize=5.6, color=MUTED)

# ---------------------------------------------------------------- panel A
axA = fig.add_axes([0.025, 0.06, 0.30, 0.84])
axA.set_xlim(0, 10); axA.set_ylim(0, 10); axA.axis("off")
axA.text(0.1, 9.8, "1  The fusion", fontsize=6.4, fontweight="bold", color=INK,
         va="top")

def box(ax, x, y, w, h, label, fc, fs=5.0, tc="white"):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.06,rounding_size=0.2",
                                fc=fc, ec=fc, lw=0.5, zorder=2))
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            fontsize=fs, color=tc, zorder=3, linespacing=1.15)

def arrow(ax, x1, y1, x2, y2, color=GREY, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=4.5, lw=0.7, color=color,
                                 linestyle=ls, zorder=1))

box(axA, 0.2, 6.7, 1.5, 1.3, "image", "#e6e6e6", tc=INK, fs=4.8)
for i, xx in enumerate([2.15, 3.55, 4.95]):
    box(axA, xx, 6.7, 1.15, 1.3, f"stage\n{i+1}", BLUE, fs=4.6)
box(axA, 6.4, 6.7, 1.5, 1.3, "classifier", "#e6e6e6", tc=INK, fs=4.6)
for x1, x2 in [(1.75, 2.10), (3.35, 3.50), (4.75, 4.90), (6.15, 6.35)]:
    arrow(axA, x1, 7.35, x2, 7.35)

arrow(axA, 5.5, 6.6, 5.5, 5.15, VERM, ls=(0, (2, 1.2)))
axA.text(5.75, 5.9, "λ decays 1 to 0", fontsize=5.0, color=VERM, ha="left",
         va="center", style="italic")
box(axA, 4.15, 3.75, 2.9, 1.35, "aux head\n(training only)", VERM, fs=4.8)
arrow(axA, 4.05, 4.42, 3.35, 4.42, VERM, ls=(0, (2, 1.2)))
box(axA, 0.2, 3.75, 3.05, 1.35, "fixed Gabor\nenergy bank", GREEN, fs=4.8)

axA.add_patch(Rectangle((0.2, 0.9), 7.7, 1.9, fc="#f2f2f2", ec="none", zorder=0))
axA.text(4.05, 2.15, "+2% training compute", ha="center", fontsize=5.4, color=INK)
axA.text(4.05, 1.35, "+0 at inference · identical deployed net",
         ha="center", fontsize=5.0, color=MUTED)

# ---------------------------------------------------------------- panel B
fig.text(0.372, 0.905, "2  The law", fontsize=6.4, fontweight="bold",
         color=INK, va="top")
fig.text(0.372, 0.856, "Δ = G + readout(base) · 268 of 278 testable cells correct", fontsize=5.0, color=MUTED, va="top")
axB = fig.add_axes([0.415, 0.15, 0.225, 0.58])

pts = {"ok": [], "bad": [], "un": [], "br": []}
for r in csv.DictReader(open(CSV)):
    if not r.get("aux_target") or r.get("init_from") or r.get("pretrained"):
        continue
    if (r.get("stem") or "none") != "none":
        continue
    try:
        d, g, base = float(r["delta"]), float(r["G"]), float(r["base_acc"])
        if int(r["n_seeds"]) < 3 or int(r["n_probe_seeds"]) < 3:
            continue
        sem = math.hypot(float(r["delta_sem"] or 0), float(r["G_sem"] or 0))
    except (ValueError, KeyError):
        continue
    ro = d - g
    if LO <= base <= HI: pts["br"].append((base, ro))
    elif sem <= 0 or abs(ro) <= 2 * sem: pts["un"].append((base, ro))
    elif (base < LO and ro < 0) or (base > HI and ro > 0): pts["ok"].append((base, ro))
    else: pts["bad"].append((base, ro))

axB.axvspan(LO, HI, color="#000000", alpha=0.07, lw=0)
axB.axhline(0, color="#777777", lw=0.5)
axB.scatter(*zip(*pts["un"]), s=1.4, c="#d4d4d4", lw=0)
axB.scatter(*zip(*pts["br"]), s=1.4, c=GREY, lw=0)
axB.scatter(*zip(*pts["ok"]), s=2.2, c=BLUE, lw=0, alpha=0.85)
axB.scatter(*zip(*pts["bad"]), s=4.5, c=VERM, marker="x", lw=0.65)
axB.set_xlim(0, 100); axB.set_ylim(-7.5, 5)
axB.set_yticks([-6, -4, -2, 0, 2, 4])
axB.set_xlabel("baseline accuracy (%)", fontsize=5.4, labelpad=1.2)
axB.set_ylabel("readout = Δ − G", fontsize=5.4, labelpad=1.2)
axB.text(2, 4.4, "left flank: features gained,\naccuracy cannot cash them in",
         fontsize=4.3, color=MUTED, ha="left", va="top", linespacing=1.25)
axB.text((LO + HI) / 2, -7.35, "crossing", fontsize=4.3, color=MUTED,
         ha="center", va="bottom")

# ---------------------------------------------------------------- panel C
axC = fig.add_axes([0.685, 0.06, 0.30, 0.84])
axC.set_xlim(0, 10); axC.set_ylim(0, 10); axC.axis("off")
axC.text(0.1, 9.8, "3  When it pays", fontsize=6.4, fontweight="bold",
         color=INK, va="top")

rows = [
    ("STACK", "prior + augmentation", "different currencies", "×1.4–2.4 gain", GREEN),
    ("SUBSTITUTE", "prior vs. SimCLR (2×)", "same currency", "combo ≤ best single", ORANGE),
    ("TAX", "prior on ImageNet init", "overwrites mature features", "−17 pts, feature-side", VERM),
]
y = 8.6
for tag, what, why, num, col in rows:
    axC.add_patch(Rectangle((0.15, y - 1.55), 0.2, 1.45, fc=col, ec="none"))
    axC.text(0.65, y, tag, fontsize=5.4, fontweight="bold", color=col, va="top")
    axC.text(0.65, y - 0.55, what, fontsize=5.0, color=INK, va="top")
    axC.text(0.65, y - 1.05, why, fontsize=4.4, color=MUTED, va="top")
    axC.text(0.65, y - 1.52, num, fontsize=4.4, color=MUTED, va="top",
             style="italic")
    y -= 2.30

axC.add_patch(Rectangle((0.15, 0.35), 9.2, 1.8, fc="#eaf2f8", ec=BLUE, lw=0.5))
axC.text(4.75, 1.88, "Headline: small ViTs, modern recipe", ha="center",
         va="top", fontsize=5.2, fontweight="bold", color=BLUE)
axC.text(4.75, 1.28, "+13.0 (ViT-S), +26.0 (ViT-B) at 224 px", ha="center",
         va="top", fontsize=4.6, color=INK)
axC.text(4.75, 0.80, "+3.2 at 1.28M images, every CNN neutral",
         ha="center", va="top", fontsize=4.6, color=INK)

for ext in ("pdf", "png"):
    fig.savefig(os.path.join(HERE, f"graphical_abstract.{ext}"),
                facecolor="white")
print("graphical abstract written; law-scope points:",
      {k: len(v) for k, v in pts.items()})
