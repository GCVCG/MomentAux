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

FIG_W_IN, FIG_H_IN = 7.0, 3.95
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
            resp = (stem._energy(stem._luma(x))
                    * stem.calib_scale.view(1, -1, 1, 1))[0]
        m = resp.mean(0).numpy()
        maps = [resp[i].numpy() for i in (0, 1, 2, 3)]   # sigma=2, 4 thetas
        kerns = [stem.even.squeeze(1)[i].numpy() for i in (0, 1, 2, 3)]
        return img, m, kerns, maps
    except Exception as e:                                    # noqa: BLE001
        print(f"method-fig assets fallback ({e})")
        yy, xx = np.mgrid[0:96, 0:96] / 96.0
        img = np.stack([0.4 + 0.4 * xx, 0.5 + 0.3 * yy, 0.45 + 0.2 * xx], -1)
        m = np.abs(np.sin(14 * xx) * np.cos(9 * yy))
        k = np.outer(np.hanning(11), np.hanning(11)) * np.cos(
            np.linspace(-6, 6, 11))[None, :]
        return img, m, [k, k.T, k, k.T], [m, m.T, m, m.T]


IMG, MMAP, KERNS, MAPS = _load_assets()


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
    # channel stack: two offset back sheets, then the cuboid faces, so the
    # stage reads as a STACK of feature maps rather than a flat box
    for k in (2, 1):
        off_x, off_y = k * dx / 2.6, k * dy / 2.6
        ax.add_patch(Rectangle((x0 + off_x, y0 + off_y), w, h,
                               fc=shade(BLUE, 0.42 + 0.13 * k),
                               ec=shade(BLUE, -0.05), lw=0.3, zorder=3 - 0.1 * k))
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
                "(a)  MomentAux in Three Phases: Fixed Source, Fused Training, Plain Deployment")

ux = 0.610 * FIG_W_IN / 10.0     # inches per x-unit
uy = 0.875 * FIG_H_IN / 10.0     # inches per y-unit


def band(y0, y1, label, color):
    axa.add_patch(Rectangle((0.02, y0), 9.96, y1 - y0, fc=shade(color, 0.94),
                            ec=shade(color, 0.55), lw=0.5, zorder=0))
    axa.text(0.14, y1 - 0.14, label, fontsize=4.7, fontweight="bold",
             color=shade(color, -0.25), ha="left", va="top", zorder=4)


# ---- Phase 1: the knowledge source, fixed before any training -----------
band(8.30, 10.10, "Phase 1  \u00b7  The Knowledge Source (Fixed and Fingerprinted Before Any Training)", GREEN)
ky = 8.98
KW = 0.42
KH = KW * ux / uy
for j, k in enumerate(KERNS):
    x0 = 1.66 + j * 0.52
    axa.imshow(k, extent=(x0, x0 + KW, ky - KH / 2, ky + KH / 2),
               cmap="RdBu_r", zorder=3, aspect="auto")
axa.text(1.54, ky, "$g_{\\sigma,\\theta}$:", ha="right", va="center",
         fontsize=5.0, color=shade(GREEN, -0.25))
axa.text(2.70, 8.48, "Pinned Gabor Bank, 8 Quadrature Pairs\n"
         "(2 Scales $\\times$ 4 Orientations), Calibrated Once",
         ha="center", va="center", fontsize=3.9, color=MUTED, linespacing=1.3)
arrow(axa, 3.82, ky, 4.16, ky, GREEN)
MW = 0.52
MH = MW * ux / uy
for j, mm in enumerate(MAPS):
    x0 = 4.26 + j * 0.62
    axa.imshow(mm, extent=(x0, x0 + MW, ky - MH / 2, ky + MH / 2),
               cmap="magma", zorder=3, aspect="auto")
    axa.add_patch(Rectangle((x0, ky - MH / 2), MW, MH, fc="none",
                            ec=shade(GREEN, -0.2), lw=0.35, zorder=4))
    axa.text(x0 + MW / 2, ky - MH / 2 - 0.10,
             ["$0^\\circ$", "$45^\\circ$", "$90^\\circ$",
              "$135^\\circ$"][j],
             ha="center", va="top", fontsize=3.6, color=shade(GREEN, -0.2))
axa.text(6.90, ky + 0.24, "Target $m(x)=|x*g_{\\sigma,\\theta}|$",
         ha="left", va="center", fontsize=4.7, color=shade(GREEN, -0.25))
axa.text(6.90, ky - 0.28, "Phase-Invariant Oriented Energy,\nIdentical for "
         "Every Dataset, Backbone, Task", ha="left", va="center",
         fontsize=3.8, color=MUTED, linespacing=1.25)

# ---- Phase 2: fused training --------------------------------------------
band(1.80, 8.16, "Phase 2  \u00b7  Fused Training: One Input, Two Losses", VERM)
Y = 6.30                                     # backbone row centreline
IMG_W = 1.05
IMG_H = IMG_W * ux / uy
IY = 6.60
axa.imshow(IMG, extent=(0.10, 0.10 + IMG_W, IY - IMG_H / 2, IY + IMG_H / 2),
           zorder=3, aspect="auto")
axa.add_patch(Rectangle((0.10, IY - IMG_H / 2), IMG_W, IMG_H, fc="none",
                        ec=INK, lw=0.5, zorder=4))
axa.text(0.62, IY - IMG_H / 2 - 0.18, "Input $x$", ha="center", va="top",
         fontsize=4.8, color=INK)
# input feeds Phase 1's bank too
arrow(axa, 0.62, IY + IMG_H / 2 + 0.06, 0.62, ky, GREEN, lw=0.7)
arrow(axa, 0.62, ky, 1.10, ky, GREEN, lw=0.7)

axa.text(4.62, 7.62, "Backbone: Four Stages, Unchanged "
         "(Spatial $\\downarrow$, Channels $\\uparrow$)",
         ha="center", va="center", fontsize=4.1, color=MUTED)
stage_names = ("Early", "Mid", "Deep", "Final")
xs, gap = 1.80, 1.02
right_edges = []
for i, (a_in, t_in) in enumerate(
        ((0.32, 0.055), (0.26, 0.085), (0.215, 0.13), (0.17, 0.185))):
    if i == 2:
        axa.text(xs - 0.36, Y, "$\\cdots$", ha="center", va="center",
                 fontsize=9, color=INK)
    r, ybot = slab(axa, xs, Y, a_in, t_in, ux, uy, label=None, tap=(i == 2))
    h = a_in / uy
    axa.text(xs + a_in / ux / 2, Y - h / 2 - 0.14, stage_names[i],
             ha="center", va="top", fontsize=4.2,
             color=VERM if i == 2 else MUTED,
             fontweight="bold" if i == 2 else "normal")
    if i == 2:
        tap_xy = (xs + a_in / ux / 2, Y - h / 2)
    right_edges.append(r)
    if i and i != 2:
        arrow(axa, right_edges[-2] + 0.05, Y, xs - 0.05, Y)
    xs += gap
arrow(axa, 0.10 + IMG_W + 0.05, IY - 0.15, 1.74, Y)
arrow(axa, right_edges[-1] + 0.05, Y, 6.55, Y)

box(axa, 6.60, Y - 0.52, 1.48, 1.04, "Task Head", "#e6e6e6",
    UA, tc=INK, fs=4.4, ec=LINE)
arrow(axa, 8.14, Y, 8.50, Y)
axa.text(8.92, Y, "$\\mathcal{L}_{\\mathrm{CE}}$", ha="center",
         va="center", fontsize=5.6, color=INK)

# tap -> aux head -> loss
arrow(axa, tap_xy[0], tap_xy[1] - 0.44, tap_xy[0], 4.66, VERM,
      ls=(0, (2, 1.2)), lw=0.8)
axa.plot([tap_xy[0]], [tap_xy[1]], marker="o", ms=3.2, color=VERM, zorder=5)
axa.text(tap_xy[0] + 0.15, 5.06, "Tap", fontsize=4.2, color=VERM,
         ha="left", va="center")
box(axa, tap_xy[0] - 1.16, 3.52, 2.32, 1.04,
    "Aux Head: $1{\\times}1$ Conv $W$\n(Training Only)", VERM, UA, fs=4.1)

LOSS = ("$\\mathcal{L} \\,=\\, \\mathcal{L}_{\\mathrm{CE}} "
        "\\,+\\, \\lambda(t)\\,\\Vert\\, W f_{\\mathrm{deep}}(x)"
        " - m(x) \\,\\Vert_2^2$")
axa.add_patch(FancyBboxPatch((1.55, 2.02), 5.10, 0.98,
                             boxstyle="round,pad=0.05,rounding_size=0.18",
                             fc="#fdf1e9", ec=VERM, lw=0.6, zorder=2))
axa.text(4.10, 2.52, LOSS, ha="center", va="center", fontsize=5.2,
         color=INK, zorder=4)
arrow(axa, tap_xy[0], 3.46, tap_xy[0], 3.10, VERM)          # aux head -> loss
arrow(axa, 6.10, 8.28, 6.10, 3.10, GREEN, ls=(0, (2, 1.2)), lw=0.85)
axa.text(6.24, 4.90, "$m(x)$", fontsize=4.4, color=shade(GREEN, -0.2),
         ha="left", va="center")
axa.plot([8.92, 8.92], [Y - 0.42, 2.52], color=GREY, lw=0.65, zorder=1)
arrow(axa, 8.92, 2.52, 6.72, 2.52, GREY)                    # CE -> loss

# lambda(t) inset
ins = fig.add_axes([0.441, 0.335, 0.092, 0.155])
t = np.linspace(0, 1, 300)
lam = 0.5 * (1 + np.cos(np.pi * t))
ins.plot(t, lam, lw=1.0, color=VERM)
ins.fill_between(t, 0, lam, color=VERM, alpha=0.13, lw=0)
ins.set_xlim(0, 1); ins.set_ylim(0, 1.13)
ins.set_xticks([0, 1]); ins.set_xticklabels(["0", "$T$"], fontsize=4.0)
ins.set_yticks([0, 1]); ins.set_yticklabels(["0", "$\\lambda_0$"],
                                            fontsize=4.0)
ins.tick_params(length=1.5, pad=1)
ins.set_title("Cosine $\\lambda(t)$: Reaches Exactly 0,\n"
              "Late Training Is Pure CE", fontsize=3.9, color=VERM, pad=2.0,
              linespacing=1.2)
for sp in ins.spines.values():
    sp.set_linewidth(0.4)
ins.set_zorder(6)
ins.patch.set_alpha(0.0)

# ---- Phase 3: deployment, one mechanism, three tasks --------------------
band(0.02, 1.66, "Phase 3  \u00b7  Deployment: $W$ and the Bank Are Dropped", BLUE)
dxs = 0.55
for a_in in (0.14, 0.115, 0.095, 0.075):
    r, _ = slab(axa, dxs, 0.62, a_in, a_in * 0.5, ux, uy)
    dxs = r + 0.16
arrow(axa, dxs, 0.62, dxs + 0.24, 0.62, lw=0.5)
chip_x = dxs + 0.30
chips = [("Classification", 1.05, "cls"), ("Segmentation", 0.62, "seg"),
         ("Detection", 0.19, "det")]
for lab, hy, icon in chips:
    axa.add_patch(FancyBboxPatch((chip_x, hy - 0.145), 2.05, 0.29,
                                 boxstyle="round,pad=0.03,rounding_size=0.12",
                                 fc="white", ec=shade(BLUE, -0.15), lw=0.5,
                                 zorder=3))
    ix = chip_x + 0.14
    if icon == "cls":
        for bi, bl in enumerate((0.26, 0.16, 0.09)):
            axa.add_patch(Rectangle((ix, hy + 0.065 - bi * 0.075), bl, 0.05,
                                    fc=shade(BLUE, -0.1), ec="none", zorder=4))
    elif icon == "seg":
        for gi in range(4):
            gx, gy = ix + (gi % 2) * 0.13, hy - 0.09 + (gi // 2) * 0.11
            axa.add_patch(Rectangle((gx, gy), 0.115, 0.095,
                                    fc=[shade(BLUE, 0.1), shade(GREEN, 0.2),
                                        shade(ORANGE, 0.3),
                                        shade(VERM, 0.35)][gi],
                                    ec="none", zorder=4))
    else:
        axa.add_patch(Rectangle((ix, hy - 0.095), 0.27, 0.19, fc="none",
                                ec=shade(VERM, -0.05), lw=0.7, zorder=4))
        axa.plot([ix + 0.135], [hy], marker=".", ms=1.6,
                 color=shade(BLUE, -0.2), zorder=4)
    axa.text(chip_x + 0.52, hy, lab, ha="left", va="center", fontsize=3.9,
             color=INK, zorder=4)
    axa.plot([dxs + 0.26, chip_x], [0.62, hy], color=GREY, lw=0.45, zorder=1)
axa.text(6.15, 0.62, "One Task Head, Trained for the Task at Hand: the Same\n"
         "Bank, Tap and Schedule Serve All Three; FLOPs and\n"
         "Parameters Identical to the Baseline",
         fontsize=4.1, color=MUTED, ha="left", va="center", linespacing=1.3)

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
