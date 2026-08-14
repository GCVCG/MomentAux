"""Generate the paper's two main-body figures from study data.

Fig 1 (law_scatter): readout = Delta - G vs baseline accuracy for every
law-scope cell with >=3 seeds on both arms, with the measured crossing
bracket shaded. Point classes follow the audit: resolvable-correct,
resolvable-wrong, unresolved, in-bracket (no prediction).

Fig 2 (envelopes): champion-prior gain vs data fraction; panel (a) three
conv populations, panel (b) ViT-tiny under plain vs DeiT recipe.

Style, palette and the two placement widths come from pubstyle, so these
figures cannot drift from the rest of the paper's.
"""
import csv, math, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pubstyle as PS   # noqa: E402

OI = PS.OI
PS.use()
CSV = os.path.join(HERE, "..", "..", "results", "all_results.csv")
LO, HI = 31.8, 40.3

# The readout term's uncertainty MUST be formed per seed pair: Delta and G are
# computed from the same checkpoints, so their errors are correlated and the
# independent form hypot(delta_sem, G_sem) overstates the SEM by a median
# factor of ~1.8, admitting far too few cells. This figure used the
# independent form and therefore drew the retracted 96% audit under a caption
# that had already been corrected to 79%. analysis/audit_law_paired.py is the
# single source of truth for the audit, so the figure now calls it rather than
# re-deriving the classification.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "..", "analysis"))
import audit_law_paired as ALP   # noqa: E402

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
pts = {"correct": [], "wrong": [], "unres": [], "bracket": []}
for _r in ALP.load(os.path.join(REPO, "runs"), CSV):
    base, ro, sem = _r["base"], _r["ro"], _r["sem"]
    if LO <= base <= HI:
        pts["bracket"].append((base, ro))
    elif sem <= 0 or abs(ro) <= 2 * sem:
        pts["unres"].append((base, ro))
    elif (base < LO and ro < 0) or (base > HI and ro > 0):
        pts["correct"].append((base, ro))
    else:
        pts["wrong"].append((base, ro))

fig, ax = plt.subplots(figsize=(PS.COL, 2.95))
ax.axvspan(LO, HI, color="#000000", alpha=0.07, lw=0, zorder=0)
ax.axhline(0, color=PS.RULE, lw=0.7, zorder=1)
# plotted in the order the audit reads them: the two classes that TEST the
# law first, then the two that cannot.
ax.scatter(*zip(*pts["unres"]), s=6, c="#c9c9c9", lw=0, alpha=0.75, zorder=2,
           label=f"unresolved ({len(pts['unres'])})")
ax.scatter(*zip(*pts["bracket"]), s=7, c=OI["grey"], marker="D", lw=0,
           alpha=0.8, zorder=2,
           label=f"in bracket ({len(pts['bracket'])})")
ax.scatter(*zip(*pts["correct"]), s=9, c=OI["blue"], lw=0, alpha=0.85,
           zorder=3, label=f"sign as predicted ({len(pts['correct'])})")
# The wrong class is a fifth the size of the correct one, so it is drawn at
# comparable ink weight rather than the heavier stroke it had when there were
# only ten of them: a marker sized for 10 points reads as a crowd at 107.
ax.scatter(*zip(*pts["wrong"]), s=12, c=OI["verm"], marker="x", lw=0.85,
           alpha=0.9, zorder=4,
           label=f"sign wrong ({len(pts['wrong'])})")
ax.set_xlabel("baseline accuracy (%)")
ax.set_ylabel(r"readout $=\Delta-G$ (points)")
ax.set_xlim(0, 100)
ax.set_ylim(-8, 6)

# state what the law actually predicts in each region, so the point classes
# are readable without consulting the caption
ax.annotate("crossing bracket:\nno sign predicted", xy=((LO + HI) / 2, 5.85),
            ha="center", va="top", fontsize=PS.SMALL, color=PS.MUTED,
            linespacing=1.2)
ax.annotate("law predicts\nreadout $<0$", xy=(17, -7.15), ha="center",
            va="center", fontsize=PS.SMALL, color=PS.MUTED, linespacing=1.2)
ax.annotate("law predicts\nreadout $>0$", xy=(99, 5.6), ha="right",
            va="top", fontsize=PS.SMALL, color=PS.MUTED, linespacing=1.2)

# legend BELOW the axes: every in-axes position overlapped the point cloud
handles, labels = ax.get_legend_handles_labels()
order = [2, 3, 0, 1]
leg = ax.legend([handles[i] for i in order], [labels[i] for i in order],
                loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2,
                frameon=False, handletextpad=0.35, columnspacing=1.0,
                borderaxespad=0.0, markerscale=1.7, fontsize=PS.LEGEND)
fig.tight_layout(pad=0.4)
# pad_inches=0.02 rather than matplotlib's default 0.1in: the default
# leaves 7.2pt of blank on every side, which at these figure widths is
# 5-8% of the graphic's height and reads as a white band inside the float.
PS.save(fig, os.path.join(HERE, "law_scatter.pdf"))
print("law_scatter:", {k: len(v) for k, v in pts.items()})

# ---------------------------------------------------------------- fig 2
fr = [1, 2, 3, 5, 7, 10, 15, 25, 100]
c100 = [1.42, 2.50, 3.68, 5.15, 4.87, 3.75, 2.55, 0.25, 0.08]
c10 = [6.37, 6.66, 5.38, 4.41, 2.21, 1.09, -0.66, -0.83, -0.26]
tin_f = [1, 2, 5, 10, 25, 100]
tin = [1.49, 1.81, 2.12, 1.65, 0.10, -0.42]
vit_p = [1.35, 3.25, 6.21, 9.35, 10.99, 13.26, 14.44, 13.67, 9.88]
vit_d = [3.20, 6.01, 9.48, 16.34, 18.69, 21.00, 24.59, 25.05, 13.86]

# One column, two rows: authored at the width it is placed at, so the
# panel contents are drawn at the size the reader sees.
# 3.6in rather than 4.25: at the taller size the two panels sat far apart and
# the float ran to 306pt in a ~690pt column, which reads as air above and below
# the data. The panels share an x axis, so the height buys nothing.
fig, axes = plt.subplots(2, 1, figsize=(PS.COL, 3.6), sharex=True)
for ax in axes:
    ax.axhline(0, color=PS.RULE, lw=0.7)
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 5, 10, 25, 100])
    ax.set_xticklabels(["1", "2", "5", "10", "25", "100"])
axes[1].set_xlabel("data fraction (%)")

a = axes[0]
a.plot(fr, c100, "-o", ms=2.8, c=OI["blue"], label="CIFAR-100")
a.plot(fr, c10, "-s", ms=2.8, c=OI["orange"], label="CIFAR-10")
a.plot(tin_f, tin, "-^", ms=3.0, c=OI["green"], label="Tiny-ImageNet")
a.set_ylabel(r"prior gain $\Delta$ (points)")
a.set_title("(a) ResNet-18, frozen recipe", loc="left")
a.legend(frameon=False, loc="upper right", handlelength=1.6)

b = axes[1]
b.plot(fr, vit_p, "-o", ms=2.8, c=OI["purple"], label="plain recipe")
b.plot(fr, vit_d, "-s", ms=2.8, c=OI["verm"], label="DeiT augmentation")
b.set_ylabel(r"prior gain $\Delta$ (points)")
b.set_title("(b) ViT-tiny, CIFAR-100", loc="left")
b.legend(frameon=False, loc="upper left", handlelength=1.6)

fig.tight_layout(pad=0.3, h_pad=0.6)
PS.save(fig, os.path.join(HERE, "envelopes.pdf"))
print("envelopes written")
