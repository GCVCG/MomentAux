"""Generate the paper's two main-body figures from study data.

Fig 1 (law_scatter): readout = Delta - G vs baseline accuracy for every
law-scope cell with >=3 seeds on both arms, with the measured crossing
bracket shaded. Point classes follow the audit: resolvable-correct,
resolvable-wrong, unresolved, in-bracket (no prediction).

Fig 2 (envelopes): champion-prior gain vs data fraction; panel (a) three
conv populations, panel (b) ViT-tiny under plain vs DeiT recipe.

Colors: Okabe-Ito (CVD-safe), fixed assignment.
"""
import csv, math, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OI = {"blue": "#0072B2", "verm": "#D55E00", "green": "#009E73",
      "orange": "#E69F00", "purple": "#CC79A7", "grey": "#999999",
      "sky": "#56B4E9"}
plt.rcParams.update({
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "legend.fontsize": 7, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "pdf.fonttype": 42})

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "..", "..", "results", "all_results.csv")
LO, HI = 31.8, 40.3

# ---------------------------------------------------------------- fig 1
pts = {"correct": [], "wrong": [], "unres": [], "bracket": []}
for r in csv.DictReader(open(CSV)):
    if not r.get("aux_target") or r.get("init_from") or r.get("pretrained"):
        continue
    if (r.get("stem") or "none") != "none":
        continue
    try:
        d, g, base = float(r["delta"]), float(r["G"]), float(r["base_acc"])
        ns, nps = int(r["n_seeds"]), int(r["n_probe_seeds"])
        sem = math.hypot(float(r["delta_sem"] or 0), float(r["G_sem"] or 0))
    except (ValueError, KeyError):
        continue
    if ns < 3 or nps < 3:
        continue
    ro = d - g
    if LO <= base <= HI:
        pts["bracket"].append((base, ro))
    elif sem <= 0 or abs(ro) <= 2 * sem:
        pts["unres"].append((base, ro))
    elif (base < LO and ro < 0) or (base > HI and ro > 0):
        pts["correct"].append((base, ro))
    else:
        pts["wrong"].append((base, ro))

fig, ax = plt.subplots(figsize=(3.45, 2.95))
ax.axvspan(LO, HI, color="#000000", alpha=0.07, lw=0, zorder=0)
ax.axhline(0, color="#666666", lw=0.7, zorder=1)
# plotted in the order the audit reads them: the two classes that TEST the
# law first, then the two that cannot.
ax.scatter(*zip(*pts["unres"]), s=6, c="#c9c9c9", lw=0, alpha=0.75, zorder=2,
           label=f"cannot test it: unresolved ({len(pts['unres'])})")
ax.scatter(*zip(*pts["bracket"]), s=7, c=OI["grey"], marker="D", lw=0,
           alpha=0.8, zorder=2,
           label=f"cannot test it: in bracket ({len(pts['bracket'])})")
ax.scatter(*zip(*pts["correct"]), s=9, c=OI["blue"], lw=0, alpha=0.85,
           zorder=3, label=f"tests it: sign as predicted ({len(pts['correct'])})")
ax.scatter(*zip(*pts["wrong"]), s=18, c=OI["verm"], marker="x", lw=1.1,
           zorder=4, label=f"tests it: sign wrong ({len(pts['wrong'])})")
ax.set_xlabel("baseline accuracy (%)")
ax.set_ylabel(r"readout $=\Delta-G$ (points)")
ax.set_xlim(0, 100)
ax.set_ylim(-8, 6)

# state what the law actually predicts in each region, so the point classes
# are readable without consulting the caption
ax.annotate("crossing bracket:\nno sign predicted", xy=((LO + HI) / 2, 5.85),
            ha="center", va="top", fontsize=6.0, color="#555555",
            linespacing=1.2)
ax.annotate("law predicts\nreadout $<0$", xy=(17, -7.15), ha="center",
            va="center", fontsize=6.2, color="#555555", linespacing=1.2)
ax.annotate("law predicts\nreadout $>0$", xy=(99, 5.6), ha="right",
            va="top", fontsize=6.2, color="#555555", linespacing=1.2)

# legend BELOW the axes: every in-axes position overlapped the point cloud
handles, labels = ax.get_legend_handles_labels()
order = [2, 3, 0, 1]
leg = ax.legend([handles[i] for i in order], [labels[i] for i in order],
                loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=2,
                frameon=False, handletextpad=0.35, columnspacing=1.0,
                borderaxespad=0.0, markerscale=1.7, fontsize=6.4)
fig.tight_layout(pad=0.4)
fig.savefig(os.path.join(HERE, "law_scatter.pdf"))
print("law_scatter:", {k: len(v) for k, v in pts.items()})

# ---------------------------------------------------------------- fig 2
fr = [1, 2, 3, 5, 7, 10, 15, 25, 100]
c100 = [1.42, 2.50, 3.68, 5.15, 4.87, 3.75, 2.55, 0.25, 0.08]
c10 = [6.37, 6.66, 5.38, 4.41, 2.21, 1.09, -0.66, -0.83, -0.26]
tin_f = [1, 2, 5, 10, 25, 100]
tin = [1.49, 1.81, 2.12, 1.65, 0.10, -0.42]
vit_p = [1.35, 3.25, 6.21, 9.35, 10.99, 13.26, 14.44, 13.67, 9.88]
vit_d = [3.20, 6.01, 9.48, 16.34, 18.69, 21.00, 24.59, 25.05, 13.86]

fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.35), sharex=True)
for ax in axes:
    ax.axhline(0, color="#666666", lw=0.7)
    ax.set_xscale("log")
    ax.set_xticks([1, 2, 5, 10, 25, 100])
    ax.set_xticklabels(["1", "2", "5", "10", "25", "100"])
    ax.set_xlabel("data fraction (%)")

a = axes[0]
a.plot(fr, c100, "-o", ms=2.6, lw=1.3, c=OI["blue"], label="CIFAR-100")
a.plot(fr, c10, "-s", ms=2.6, lw=1.3, c=OI["orange"], label="CIFAR-10")
a.plot(tin_f, tin, "-^", ms=2.8, lw=1.3, c=OI["green"], label="Tiny-ImageNet")
a.set_ylabel(r"prior gain $\Delta$ (points)")
a.set_title("(a) ResNet-18, frozen recipe", loc="left")
a.legend(frameon=False, loc="upper right", handlelength=1.6)

b = axes[1]
b.plot(fr, vit_p, "-o", ms=2.6, lw=1.3, c=OI["purple"], label="plain recipe")
b.plot(fr, vit_d, "-s", ms=2.6, lw=1.3, c=OI["verm"], label="DeiT augmentation")
b.set_title("(b) ViT-tiny, CIFAR-100", loc="left")
b.legend(frameon=False, loc="upper right", handlelength=1.6)

fig.tight_layout(pad=0.4, w_pad=1.2)
fig.savefig(os.path.join(HERE, "envelopes.pdf"))
print("envelopes written")
