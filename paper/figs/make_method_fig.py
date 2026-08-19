"""Figure 1: how the prior is fused, and what fusing two sources can do.

Redesign 2026-08-18, referee request: the earlier box diagram did not show
what is inside the boxes, carried no formula symbols, and never showed the
fusion outcomes. Two panels across the two-column span:

 (a) the training-time architecture, drawn with REAL content: the actual
     sample image, the backbone as feature-map slabs of decreasing spatial
     size, the tap marked on the THIRD of four stages, the task head with its
     cross-entropy loss, and the auxiliary 1x1-conv head regressed onto the
     fixed Gabor-magnitude target m(x) -- computed from the SAME input by the
     study's own pinned bank (the thumbnails ARE the bank's kernels and its
     calibrated response, imported from momentstem, not drawings of them).
     The loss is written with its real symbols, the cosine lambda(t) schedule
     is an inset with "lambda reaches exactly 0" annotated, and a deployment
     strip shows the aux head dropped: backbone + task head only, identical
     FLOPs to the baseline.

 (b) the three fusion outcomes as schematic bar groups -- stack, substitute,
     interfere -- each with a real measured example underneath. The bars are
     schematic (axis says so); the examples are the measurements.

The decomposition Delta = G + readout sits under panel (b) with a one-phrase
gloss of each term.

No LaTeX escapes (matplotlib mathtext is not LaTeX). Falls back to synthetic
thumbnails if the data or momentstem import is unavailable, so the script
always writes a figure.
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import (FancyBboxPatch, FancyArrowPatch, Rectangle,
                                Polygon)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import pubstyle as PS  # noqa: E402

BLUE, VERM, GREEN = PS.OI["blue"], PS.OI["verm"], PS.OI["green"]
ORANGE, GREY = PS.OI["orange"], "#8a8a8a"
INK, MUTED, LINE = PS.INK, "#5c5c5c", "#b0b0b0"

plt.rcParams.update({
    "font.size": 6, "axes.linewidth": 0.5, "pdf.fonttype": 42,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.labelsize": 5, "ytick.labelsize": 5,
    "axes.spines.top": False, "axes.spines.right": False,
    "text.color": INK, "axes.labelcolor": INK,
    "mathtext.fontset": "dejavusans"})

FIG_W_IN, FIG_H_IN = 7.0, 3.05
fig = plt.figure(figsize=(FIG_W_IN, FIG_H_IN), dpi=300)
fig.patch.set_facecolor("white")


# ----------------------------------------------------------------- assets
def _load_assets():
    """The real knowledge source: one STL-10 test image (the deterministic
    sample fig:bank uses), two of the pinned bank's even kernels, and the
    channel-mean of the CALIBRATED magnitude response m(x). Synthetic
    fallback keeps the script runnable without the dataset."""
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
        x = ds[613][0].unsqueeze(0)          # the fig:bank sample (deer)
        mean, std = data_mod.STATS["stl10"]
        img = (x[0] * torch.tensor(std).view(3, 1, 1)
               + torch.tensor(mean).view(3, 1, 1)).clamp(0, 1)
        img = img.permute(1, 2, 0).numpy()
        with torch.no_grad():
            m = (stem._energy(stem._luma(x))
                 * stem.calib_scale.view(1, -1, 1, 1))[0].mean(0).numpy()
        kerns = [stem.even.squeeze(1)[i].numpy() for i in (1, 6)]
        return img, m, kerns
    except Exception as e:                                    # noqa: BLE001
        print(f"method-fig assets fallback ({e})")
        yy, xx = np.mgrid[0:96, 0:96] / 96.0
        img = np.stack([0.4 + 0.4 * xx, 0.5 + 0.3 * yy, 0.45 + 0.2 * xx], -1)
        m = np.abs(np.sin(14 * xx) * np.cos(9 * yy))
        k = np.outer(np.hanning(11), np.hanning(11)) * np.cos(
            np.linspace(-6, 6, 11))[None, :]
        return img, m, [k, k.T]


IMG, MMAP, KERNS = _load_assets()


# ----------------------------------------------------------------- helpers
def text_width_pt(s, fontsize, bold=False):
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
        if max(text_width_pt(ln, fs, bold) for ln in lines) <= box_w_pt:
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
            fontsize=fs, color=tc, zorder=4, linespacing=1.18)


def arrow(ax, x1, y1, x2, y2, color=GREY, ls="-", lw=0.65, z=1):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=4.2, lw=lw, color=color,
                                 linestyle=ls, zorder=z))


def panel(rect, title):
    ax = fig.add_axes(rect)
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    ax.text(0, 10.30, title, fontsize=5.8, fontweight="bold", color=INK,
            va="top", ha="left")
    return ax, rect[2] * FIG_W_IN * 72.0 / 10.0


def shade(color, f):
    """Blend color toward white (f>0) or black (f<0)."""
    c = np.array(matplotlib.colors.to_rgb(color))
    return tuple(c + (1 - c) * f if f >= 0 else c * (1 + f))


def slab(ax, x0, yc, a_in, t_in, ux, uy, label=None, tap=False):
    """A feature-map slab: front face a_in x a_in inches (spatial extent),
    thickness t_in inches (channels), drawn as a shallow cuboid. ux/uy are
    inches-per-axes-unit so the front face is square on the page."""
    w, h = a_in / ux, a_in / uy
    dx, dy = 0.55 * t_in / ux, 0.55 * t_in / uy
    y0 = yc - h / 2
    ax.add_patch(Polygon([(x0 + w, y0), (x0 + w + dx, y0 + dy),
                          (x0 + w + dx, y0 + h + dy), (x0 + w, y0 + h)],
                         fc=shade(BLUE, -0.25), ec="none", zorder=3))
    ax.add_patch(Polygon([(x0, y0 + h), (x0 + dx, y0 + h + dy),
                          (x0 + w + dx, y0 + h + dy), (x0 + w, y0 + h)],
                         fc=shade(BLUE, 0.45), ec="none", zorder=3))
    ax.add_patch(Rectangle((x0, y0), w, h, fc=shade(BLUE, 0.15),
                           ec=shade(BLUE, -0.35), lw=0.4, zorder=3))
    for f in (1 / 3, 2 / 3):               # faint grid: it is a feature MAP
        ax.plot([x0 + f * w] * 2, [y0, y0 + h], color="white", lw=0.3,
                alpha=0.7, zorder=3)
        ax.plot([x0, x0 + w], [y0 + f * h] * 2, color="white", lw=0.3,
                alpha=0.7, zorder=3)
    if label:
        ax.text(x0 + w / 2, y0 - 0.32, label, ha="center", va="top",
                fontsize=4.6, color=VERM if tap else MUTED, zorder=4)
    return x0 + w + dx, y0                  # right edge, bottom


# ================================================================ panel (a)
axa, UA = panel([0.008, 0.045, 0.610, 0.875],
                "(a)  MomentAux Training: One Input, Two Losses")
ux, uy = 0.610 * FIG_W_IN / 10.0, 0.875 * FIG_H_IN / 10.0

Y = 7.55                                    # training-path centerline
# --- input image (the real sample)
IMG_W = 1.32
IMG_H = IMG_W * ux / uy                      # square on the page
axa.imshow(IMG, extent=(0.10, 0.10 + IMG_W, Y - IMG_H / 2, Y + IMG_H / 2),
           zorder=3, aspect="auto")
axa.add_patch(Rectangle((0.10, Y - IMG_H / 2), IMG_W, IMG_H, fc="none",
                        ec=INK, lw=0.4, zorder=4))
axa.text(0.76, Y + IMG_H / 2 + 0.24, "Input $x$", ha="center", va="bottom",
         fontsize=4.8, color=INK)

# --- backbone: four slabs, spatial size falling, channels growing
sizes = [(0.470, 0.080), (0.375, 0.115), (0.290, 0.160), (0.220, 0.210)]
xs, gap = 1.80, 0.34
tap_xy = None
axa.text(2.00, 9.72, "Backbone: Four Feature Stages, Unchanged",
         ha="left", va="center", fontsize=4.4, color=MUTED)
right_edges = []
for i, (a_in, t_in) in enumerate(sizes):
    lab = f"$f_{i+1}(x)$"
    r, _ = slab(axa, xs, Y, a_in, t_in, ux, uy, label=lab, tap=(i == 2))
    if i == 2:
        tap_xy = (xs + a_in / ux / 2, Y - a_in / uy / 2)
    right_edges.append(r)
    if i < 3:
        arrow(axa, r + 0.06, Y, r + gap - 0.06, Y)
    xs = r + gap
arrow(axa, 1.48, Y, 1.75, Y)
arrow(axa, right_edges[-1] + 0.06, Y, 7.06, Y)

# --- task head + CE
box(axa, 7.12, Y - 0.62, 1.62, 1.24, "Task Head\n(Classifier)", "#e6e6e6",
    UA, tc=INK, fs=4.4, ec=LINE)
arrow(axa, 8.80, Y, 9.20, Y)
axa.text(9.62, Y, r"$\mathcal{L}_{\mathrm{CE}}$", ha="center", va="center",
         fontsize=6.0, color=INK)

# --- tap branch: 1x1 conv aux head (training only)
arrow(axa, tap_xy[0], tap_xy[1] - 0.55, tap_xy[0], 5.62, VERM,
      ls=(0, (2, 1.2)), lw=0.8, z=2)
axa.plot([tap_xy[0]], [tap_xy[1]], marker="o", ms=3.2, color=VERM, zorder=5)
axa.text(tap_xy[0] + 0.16, 6.10, "Tap", fontsize=4.4, color=VERM,
         ha="left", va="center")
box(axa, tap_xy[0] - 1.18, 4.42, 2.36, 1.18,
    "Aux Head: $1{\\times}1$ Conv $W$\n(Training Only)", VERM, UA, fs=4.2)

# --- bank path: same input -> pinned bank -> m(x)
arrow(axa, 0.76, Y - IMG_H / 2 - 0.10, 0.76, 5.75, GREEN,
      ls=(0, (2, 1.2)), lw=0.8)
box(axa, 0.10, 4.42, 2.10, 1.18, "", "#ffffff", UA, ec=GREEN, lw=0.6)
KW = 0.60
KH = KW * ux / uy
for j, k in enumerate(KERNS):
    x0 = 0.28 + j * 0.74
    axa.imshow(k, extent=(x0, x0 + KW, 5.01 - KH / 2, 5.01 + KH / 2),
               cmap="RdBu_r", zorder=3, aspect="auto")
axa.text(1.90, 5.01, "$g_{\\sigma,\\theta}$", ha="center", va="center",
         fontsize=5.0, color=GREEN)
axa.text(1.15, 4.20, "Pinned Gabor Bank\n(Fixed, Fingerprinted)",
         ha="center", va="top", fontsize=4.2, color=GREEN, linespacing=1.25)
arrow(axa, 2.28, 5.01, 2.72, 5.01, GREEN)
MW = 0.86
MH = MW * ux / uy
axa.imshow(MMAP, extent=(2.80, 2.80 + MW, 5.01 - MH / 2, 5.01 + MH / 2),
           cmap="magma", zorder=3, aspect="auto")
axa.add_patch(Rectangle((2.80, 5.01 - MH / 2), MW, MH, fc="none",
                        ec=shade(GREEN, -0.2), lw=0.4, zorder=4))
axa.text(3.23, 4.20, "Target\n$m(x)=|x*g_{\\sigma,\\theta}|$",
         ha="center", va="top", fontsize=4.2, color=GREEN, linespacing=1.25)

# --- the loss, with its real symbols, fed by all three branches
LOSS = ("$\\mathcal{L} \\,=\\, \\mathcal{L}_{\\mathrm{CE}} \\;+\\; "
        "\\lambda(t)\\,\\Vert\\, W f_3(x) - m(x) \\,\\Vert_2^2$")
axa.add_patch(FancyBboxPatch((0.30, 1.92), 6.30, 1.05,
                             boxstyle="round,pad=0.05,rounding_size=0.18",
                             fc="#fdf1e9", ec=VERM, lw=0.6, zorder=2))
axa.text(3.45, 2.44, LOSS, ha="center", va="center", fontsize=5.6,
         color=INK, zorder=4)
arrow(axa, 3.23, 3.62, 3.23, 3.10, GREEN)            # m(x) -> loss
arrow(axa, tap_xy[0], 4.36, tap_xy[0], 3.10, VERM)   # aux head -> loss
axa.plot([9.72, 9.72], [Y - 0.50, 2.44], color=GREY, lw=0.65, zorder=1)
arrow(axa, 9.72, 2.44, 6.78, 2.44, GREY)             # CE -> loss, around inset

# --- lambda(t) inset: cosine 1 -> exactly 0
ins = fig.add_axes([0.480, 0.335, 0.100, 0.205])
t = np.linspace(0, 1, 300)
lam = 0.5 * (1 + np.cos(np.pi * t))
ins.plot(t, lam, lw=1.0, color=VERM)
ins.fill_between(t, 0, lam, color=VERM, alpha=0.13, lw=0)
ins.set_xlim(0, 1); ins.set_ylim(0, 1.13)
ins.set_xticks([0, 1]); ins.set_xticklabels(["0", "$T$"], fontsize=4.1)
ins.set_yticks([0, 1]); ins.set_yticklabels(["0", "$\\lambda_0$"],
                                            fontsize=4.1)
ins.tick_params(length=1.5, pad=1)
ins.set_xlabel("Epoch $t$", fontsize=4.2, labelpad=0.6)
ins.set_title("Cosine Schedule $\\lambda(t)$", fontsize=4.4, color=INK,
              pad=2.0)
ins.annotate("$\\lambda$ Reaches Exactly 0:\nLate Training Is Pure CE",
             xy=(0.955, 0.075), xytext=(0.70, 0.47), fontsize=3.9,
             color=VERM, ha="center", va="center", linespacing=1.25,
             arrowprops=dict(arrowstyle="-|>", lw=0.5, color=VERM,
                             shrinkA=1, shrinkB=1, mutation_scale=3.6))
for s in ins.spines.values():
    s.set_linewidth(0.4)

# --- deployment strip: aux head dropped, identical FLOPs
axa.add_patch(Rectangle((0.10, 0.02), 9.80, 1.42, fc="#f5f5f5", ec=LINE,
                        lw=0.4, zorder=1))
axa.text(0.30, 1.16, "Deployment", fontsize=4.6, color=INK,
         fontweight="bold", va="center")
dxs = 1.90
for a_in in (0.16, 0.13, 0.105, 0.085):
    h = a_in / uy * 1.6
    axa.add_patch(Rectangle((dxs, 0.72 - h / 2), 0.22, h,
                            fc=shade(BLUE, 0.15), ec="none", zorder=2))
    dxs += 0.36
arrow(axa, dxs, 0.72, dxs + 0.30, 0.72, lw=0.5)
box(axa, dxs + 0.36, 0.42, 1.30, 0.60, "Task Head", "#e6e6e6", UA, tc=INK,
    fs=3.9, ec=LINE)
axa.text(5.30, 0.72, "$W$ and the Bank Are Dropped:  Backbone + Task Head"
         " Only,\nFLOPs and Parameters Identical to the Baseline",
         fontsize=4.2, color=MUTED, ha="left", va="center", linespacing=1.3)

# ================================================================ panel (b)
axb, UB = panel([0.665, 0.045, 0.330, 0.875],
                "(b)  Fusion Outcomes: What Two Sources Can Do")

groups = [
    ("Stack", ORANGE, "Aug.", 1.00, 0.55, 1.55,
     "Prior + DeiT Aug.:\n$+21.0$ vs $+13.3$ Alone\n(ViT, C100@10%)"),
    ("Substitute", BLUE, "SimCLR", 1.00, 0.95, 1.00,
     "Prior + SimCLR\n$\\approx$ SimCLR Alone"),
    ("Interfere", GREEN, "Transfer", 1.00, 0.80, 0.38,
     "Prior on ImageNet Init:\n$-16$ at $\\lambda_0{=}1.0$,\n"
     "${\\approx}0$ at $\\lambda_0{\\leq}0.3$"),
]
BASE_Y, TOP = 3.30, 8.45                    # bars live between these
SCALE = (TOP - BASE_Y - 0.85) / 1.55        # 1.55 = tallest schematic bar
gw, bw = 3.05, 0.72                         # group width, bar width
for gi, (name, bcol, bname, va, vb, vc, example) in enumerate(groups):
    gx = 0.35 + gi * (gw + 0.22)
    xa, xb, xc = gx, gx + bw + 0.22, gx + 2 * (bw + 0.22)
    for x, v, fc, hatch, ec in (
            (xa, va, VERM, None, VERM),
            (xb, vb, bcol, None, bcol),
            (xc, vc, "#ffffff", "/////", None)):
        if hatch:                            # the fused bar: both colours
            h_in = v * SCALE
            axb.add_patch(Rectangle((x, BASE_Y), bw, h_in, fc=bcol,
                                    ec="none", zorder=2))
            axb.add_patch(Rectangle((x, BASE_Y), bw, h_in, fc="none",
                                    hatch=hatch, ec=VERM, lw=0.0, zorder=3))
            axb.add_patch(Rectangle((x, BASE_Y), bw, h_in, fc="none",
                                    ec=INK, lw=0.4, zorder=4))
        else:
            axb.add_patch(Rectangle((x, BASE_Y), bw, v * SCALE, fc=fc,
                                    ec="none", zorder=2))
    # dashed reference at max of the singles, extended over the fused bar
    ymax = BASE_Y + max(va, vb) * SCALE
    axb.plot([xa - 0.10, xc + bw + 0.10], [ymax, ymax], color=INK, lw=0.5,
             ls=(0, (2, 1.2)), zorder=5)
    rel = {"Stack": "$>$", "Substitute": "$\\approx$",
           "Interfere": "$<$"}[name]
    axb.text(xc + bw / 2, BASE_Y + vc * SCALE + 0.22,
             rel, ha="center", va="bottom", fontsize=5.4, color=INK)
    # bar labels + group title + measured example
    for x, lab in ((xa, "Prior"), (xb, bname), (xc, "Both")):
        axb.text(x + bw / 2, BASE_Y - 0.14, lab, ha="center", va="top",
                 fontsize=4.0, color=MUTED)
    axb.text(gx + (2 * (bw + 0.22) + bw) / 2, 9.12, name,
             ha="center", va="center", fontsize=5.2, fontweight="bold",
             color=INK)
    axb.text(gx + (2 * (bw + 0.22) + bw) / 2, 2.35, example, ha="center",
             va="top", fontsize=3.9, color=MUTED, linespacing=1.35)

axb.plot([0.15, 9.85], [BASE_Y, BASE_Y], color=INK, lw=0.6, zorder=5)
axb.text(0.02, (BASE_Y + TOP) / 2, "Gain Over Baseline (Schematic)",
         rotation=90, ha="center", va="center", fontsize=4.2, color=MUTED)
axb.text(5.0, 9.72, "Bars Schematic; Dashed Line: Better Single Source",
         ha="center", va="center", fontsize=4.0, color=MUTED)

# --- the decomposition, with a one-phrase gloss of each term
axb.add_patch(FancyBboxPatch((0.15, 0.10), 9.70, 1.26,
                             boxstyle="round,pad=0.05,rounding_size=0.18",
                             fc="#fdf1e9", ec=VERM, lw=0.5, zorder=2))
axb.text(5.0, 1.04, "$\\Delta \\,=\\, G \\,+\\, \\mathrm{readout}$",
         ha="center", va="center", fontsize=5.4, color=VERM,
         fontweight="bold", zorder=4)
axb.text(5.0, 0.50, "$G$: Frozen-Feature Gain (Linear Evaluation);"
         "  readout: What the\nCell's Own Classifier Realizes",
         ha="center", va="center", fontsize=3.9, color=INK, zorder=4,
         linespacing=1.3)

# bbox_inches="tight" trims the hand-layout margins; pad keeps a hairline so
# strokes on the edge are not clipped.
fig.savefig(os.path.join(HERE, "method.pdf"), facecolor="white",
            bbox_inches="tight", pad_inches=0.02)
fig.savefig(os.path.join(HERE, "method.png"), facecolor="white",
            bbox_inches="tight", pad_inches=0.02)
print("method figure written")
