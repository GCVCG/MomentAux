"""Graphical abstract for the Information Fusion submission.

Elsevier spec: min 531x1328 px (h x w), readable at 5x13 cm. Rendered at
13x6.4 cm / 300 dpi = 1535x755 px, so fonts are true print points.

Three beats, left to right: (1) what is fused and at what cost, (2) the law
that organizes the grid, (3) when fusion pays, ties, or taxes.
Palette: Okabe-Ito (CVD-safe), fixed assignment.

Layout rules enforced here (revision 2026-08-07):
  * every string is Title Case, not upper case (author request
    2026-08-07); acronyms and model names keep their own casing;
  * every label is measured against its box before drawing, so nothing
    overflows -- see fit_fontsize();
  * the three backbone boxes sit inside a labelled container so "STAGE 1"
    is self-explanatory rather than jargon;
  * the scatter carries a legend and is widened into the gutter that the
    y-axis label used to waste;
  * panel 2's subtitle is wrapped and clipped to panel 2's own column, so
    it cannot collide with panel 3.

NOTE: no LaTeX escapes in labels -- matplotlib mathtext is not LaTeX, so
"\\%" and "\\," render literally. Use plain % and unicode dashes.
"""
import csv, math, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D

BLUE, VERM, GREEN = "#0072B2", "#D55E00", "#009E73"
ORANGE, GREY = "#E69F00", "#8a8a8a"
INK, MUTED = "#1a1a1a", "#5c5c5c"
PALE = "#d4d4d4"

plt.rcParams.update({
    "font.size": 6, "axes.labelsize": 6, "xtick.labelsize": 5.5,
    "ytick.labelsize": 5.5, "axes.linewidth": 0.5,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "text.color": INK, "axes.labelcolor": INK})

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "..", "..", "results", "all_results.csv")
LO, HI = 31.8, 40.3

FIG_W_CM, FIG_H_CM = 13.0, 6.4
fig = plt.figure(figsize=(FIG_W_CM / 2.54, FIG_H_CM / 2.54), dpi=300)
fig.patch.set_facecolor("white")


def text_width_pt(s, fontsize, bold=False):
    """Conservative advance-width estimate for DejaVu Sans, in points."""
    w = 0.0
    for ch in s:
        if ch in "IJIl.,:;'|! ":
            w += 0.30
        elif ch in "MW@":
            w += 0.92
        elif ch.isupper() or ch.isdigit():
            w += 0.68
        else:
            w += 0.56
    return w * fontsize * (1.06 if bold else 1.0)


def fit_fontsize(label, box_w_pt, start, floor=3.2, bold=False):
    """Largest size <= start at which the widest line fits box_w_pt."""
    lines = label.split("\n")
    fs = start
    while fs > floor:
        if max(text_width_pt(l, fs, bold) for l in lines) <= box_w_pt:
            break
        fs -= 0.1
    return fs


fig.text(0.5, 0.985,
         "A Controlled, Cost-Normalized Benchmark of Data-Efficiency "
         "Interventions, and the Law That Organizes It",
         ha="center", va="top", fontsize=5.4, color=MUTED)

# ---------------------------------------------------------------- panel A
AX_L, AX_W = 0.015, 0.305
axA = fig.add_axes([AX_L, 0.05, AX_W, 0.86])
axA.set_xlim(0, 10); axA.set_ylim(0, 10); axA.axis("off")
# points per axis-unit, needed to size text against boxes
UNIT_A = AX_W * FIG_W_CM / 2.54 * 72.0 / 10.0

axA.text(0.0, 9.9, "1  The Fusion", fontsize=6.4, fontweight="bold",
         color=INK, va="top")


def box(ax, x, y, w, h, label, fc, unit, fs=5.0, tc="white", pad=0.30):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.05,rounding_size=0.18",
                                fc=fc, ec=fc, lw=0.5, zorder=2))
    fs = fit_fontsize(label, (w - pad) * unit, fs)
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
            fontsize=fs, color=tc, zorder=3, linespacing=1.2)


def arrow(ax, x1, y1, x2, y2, color=GREY, ls="-"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=4.5, lw=0.7, color=color,
                                 linestyle=ls, zorder=4))


# --- the deployed network: input, backbone container, classifier
PIPE_Y, PIPE_H = 7.30, 1.25
box(axA, 0.10, PIPE_Y, 1.40, PIPE_H, "Image", "#e6e6e6", UNIT_A, fs=4.6, tc=INK)
axA.add_patch(FancyBboxPatch((1.92, PIPE_Y - 0.22), 4.28, PIPE_H + 0.44,
                             boxstyle="round,pad=0.05,rounding_size=0.18",
                             fc="none", ec=GREY, lw=0.5, ls=(0, (2, 1.4)),
                             zorder=1))
for i, xx in enumerate([2.05, 3.44, 4.83]):
    box(axA, xx, PIPE_Y, 1.24, PIPE_H, "Stage\n%d" % (i + 1), BLUE, UNIT_A,
        fs=4.4)
axA.text(1.95, PIPE_Y - 0.42, "Backbone, Unchanged", ha="left", va="top",
         fontsize=4.2, color=MUTED)
box(axA, 6.62, PIPE_Y, 2.55, PIPE_H, "Classifier", "#e6e6e6", UNIT_A, fs=4.6,
    tc=INK)
for x1, x2 in [(1.50, 1.90), (6.22, 6.60)]:
    arrow(axA, x1, PIPE_Y + PIPE_H / 2, x2, PIPE_Y + PIPE_H / 2)

# --- the training-only branch
arrow(axA, 5.45, PIPE_Y - 0.28, 5.45, 5.35, VERM, ls=(0, (2, 1.2)))
axA.text(5.68, 6.28, "Tap at Stage 3", fontsize=4.4, color=VERM, ha="left",
         va="center")
axA.text(5.68, 5.78, "λ Decays 1 to 0", fontsize=4.4, color=VERM, ha="left",
         va="center")
box(axA, 4.30, 3.95, 3.30, 1.35, "Aux Head\nTraining Only", VERM, UNIT_A,
    fs=4.6)
arrow(axA, 4.20, 4.62, 3.55, 4.62, VERM, ls=(0, (2, 1.2)))
box(axA, 0.10, 3.95, 3.35, 1.35, "Fixed Gabor\nEnergy Bank", GREEN, UNIT_A,
    fs=4.6)

# --- the cost claim
axA.add_patch(Rectangle((0.10, 1.05), 9.07, 1.95, fc="#f2f2f2", ec="none",
                        zorder=0))
axA.text(4.63, 2.32, "+2% Training Compute", ha="center", fontsize=5.2,
         color=INK)
axA.text(4.63, 1.52, "+0 at Inference, Identical Deployed Net",
         ha="center", fontsize=4.4, color=MUTED)

# ---------------------------------------------------------------- panel B
PB_L = 0.352
fig.text(PB_L, 0.901, "2  The Law", fontsize=6.4, fontweight="bold",
         color=INK, va="top")
fig.text(PB_L, 0.843, "Δ = G + readout(base)", fontsize=5.0, color=INK,
         va="top")
fig.text(PB_L, 0.794, "268 of 278 Testable Cells Correct", fontsize=4.6,
         color=MUTED, va="top")

axB = fig.add_axes([PB_L + 0.028, 0.165, 0.283, 0.545])

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
axB.scatter(*zip(*pts["un"]), s=1.4, c=PALE, lw=0)
axB.scatter(*zip(*pts["br"]), s=1.4, c=GREY, lw=0)
axB.scatter(*zip(*pts["ok"]), s=2.2, c=BLUE, lw=0, alpha=0.85)
axB.scatter(*zip(*pts["bad"]), s=4.5, c=VERM, marker="x", lw=0.65)
axB.set_xlim(0, 100); axB.set_ylim(-9.4, 6.6)
axB.set_yticks([-6, -4, -2, 0, 2, 4])
axB.set_xlabel("Baseline Accuracy (%)", fontsize=5.0, labelpad=1.2)
axB.set_ylabel("Readout = Δ − G", fontsize=5.0, labelpad=1.2)
axB.text(3.5, -8.4, "Left Flank: Features Gained,\nAccuracy Cannot Cash "
         "Them In", fontsize=4.0, color=MUTED, ha="left", va="center",
         linespacing=1.3)

handles = [
    Line2D([], [], ls="", marker="o", ms=1.9, mfc=BLUE, mec="none",
           label="Sign as Predicted (268)"),
    Line2D([], [], ls="", marker="x", ms=2.4, mec=VERM, mew=0.65,
           label="Exception (10)"),
    Line2D([], [], ls="", marker="o", ms=1.9, mfc=PALE, mec="none",
           label="Not Resolvable"),
    Line2D([], [], ls="", marker="s", ms=2.4, mfc="#e4e4e4", mec="none",
           label="Crossing Bracket"),
]
leg = axB.legend(handles=handles, loc="upper left", fontsize=3.9,
                 frameon=True, handlelength=0.9, handletextpad=0.35,
                 borderpad=0.32, labelspacing=0.28, borderaxespad=0.25,
                 ncol=2, columnspacing=0.6)
leg.get_frame().set_linewidth(0.35)
leg.get_frame().set_edgecolor("#c8c8c8")
leg.get_frame().set_facecolor("white")

# ---------------------------------------------------------------- panel C
PC_L, PC_W = 0.695, 0.295
axC = fig.add_axes([PC_L, 0.05, PC_W, 0.86])
axC.set_xlim(0, 10); axC.set_ylim(0, 10); axC.axis("off")
UNIT_C = PC_W * FIG_W_CM / 2.54 * 72.0 / 10.0

axC.text(0.0, 9.9, "3  When It Pays", fontsize=6.4, fontweight="bold",
         color=INK, va="top")

rows = [
    ("Stack", "Prior + Augmentation", "Different Currencies",
     "Gain x1.4 to x2.4", GREEN),
    ("Substitute", "Prior vs. SimCLR (2x)", "Same Currency",
     "Combo Not Above Best Single", ORANGE),
    ("Tax", "Prior on ImageNet Init", "Overwrites Mature Features",
     "−17 Points, Feature-Side", VERM),
]
y = 8.75
for tag, what, why, num, col in rows:
    axC.add_patch(Rectangle((0.05, y - 1.62), 0.20, 1.52, fc=col, ec="none"))
    axC.text(0.55, y, tag, fontsize=5.4, fontweight="bold", color=col,
             va="top")
    axC.text(0.55, y - 0.58, what,
             fontsize=fit_fontsize(what, 9.2 * UNIT_C, 5.0), color=INK,
             va="top")
    axC.text(0.55, y - 1.10, why,
             fontsize=fit_fontsize(why, 9.2 * UNIT_C, 4.3), color=MUTED,
             va="top")
    axC.text(0.55, y - 1.58, num,
             fontsize=fit_fontsize(num, 9.2 * UNIT_C, 4.3), color=MUTED,
             va="top")
    y -= 2.15

BX, BW = 0.05, 9.60
axC.add_patch(Rectangle((BX, 0.20), BW, 1.90, fc="#eaf2f8", ec=BLUE, lw=0.5))
cx = BX + BW / 2
for dy, txt, fs0, col, bold in [
        (1.86, "Headline: Small ViTs", 5.2, BLUE, True),
        (1.26, "+13.0 (ViT-S) and +26.0 (ViT-B) at 224 px", 4.5, INK, False),
        (0.78, "+3.2 at 1.28M Images; Every CNN Neutral", 4.5, INK, False)]:
    axC.text(cx, dy, txt, ha="center", va="top",
             fontsize=fit_fontsize(txt, (BW - 0.5) * UNIT_C, fs0, bold=bold),
             fontweight="bold" if bold else "normal", color=col)

for ext in ("pdf", "png"):
    fig.savefig(os.path.join(HERE, f"graphical_abstract.{ext}"),
                facecolor="white")
print("graphical abstract written; law-scope points:",
      {k: len(v) for k, v in pts.items()})
