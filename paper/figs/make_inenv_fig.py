"""The ImageNet-scale envelope, and why its shape is a backbone property.

The point of this figure is a comparison the classification grid cannot make:
dataset, recipe, fixed subsets and label space are all held constant, and only
the backbone varies. Under those conditions the envelope has three different
shapes, which the readout account explains as one thing -- each backbone's gain
decays as its own baseline enters the crossing bracket, and the three do so at
very different fractions. ResNet-18 is inside the band by 7-15% and is neutral
there; MobileNetV3 only reaches it at 100%; ViT-tiny passes above it at 100%,
which is exactly where its gain finally falls.

Panel (b) is what turns that from an assertion into something the reader can
check: it plots the same three backbones' BASELINE trajectories against the
same x axis with the bracket shaded, so "which backbone is in the band at
which fraction" and "where its gain decays" can be read off one figure.

Numbers come from results/all_results.csv (3 seeds/cell), not from prose.
"""
import csv
import os
import sys

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pubstyle as ps

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
CSV = os.path.join(ROOT, "results", "all_results.csv")

# The crossing bracket, measured on the classification grid and quoted
# throughout the paper. Shaded, not fitted here.
LO, HI = 31.8, 40.3

BACKBONES = [
    ("resnet18",              "ResNet-18",   ps.OI["blue"],  "o"),
    ("vit_tiny",              "ViT-tiny",    ps.OI["verm"],  "^"),
    ("mobilenetv3_small_100", "MobileNetV3", ps.OI["green"], "s"),
]


def load():
    """{backbone: [(pct, base, delta), ...]} for ImageNet64, sorted by pct."""
    out = {b: [] for b, _, _, _ in BACKBONES}
    for r in csv.DictReader(open(CSV)):
        if r["dataset"] != "imagenet64" or not r["delta"]:
            continue
        if r["backbone"] in out:
            out[r["backbone"]].append(
                (float(r["subset_pct"]), float(r["base_acc"]), float(r["delta"])))
    for b in out:
        out[b].sort()
    return out


def main():
    ps.use()
    data = load()
    # Two rows in ONE column rather than two panels across the page: the
    # panels share an x axis, so stacking lets them share tick labels and
    # keeps the figure out of the scarce full-width float class.
    fig, (a, b) = plt.subplots(2, 1, figsize=(ps.COL, 3.55), sharex=True)

    # ---- (a) the envelopes themselves ------------------------------------
    for key, label, colour, mark in BACKBONES:
        pts = data[key]
        if not pts:
            continue
        a.plot([p[0] for p in pts], [p[2] for p in pts],
               color=colour, marker=mark, label=label, clip_on=False)
    a.axhline(0.0, color=ps.RULE, lw=0.6, ls=(0, (3, 2)), zorder=0)
    a.set_xscale("log")
    a.set_xticks([1, 2, 5, 10, 25, 100])
    a.set_xticklabels(["1", "2", "5", "10", "25", "100"])
    a.set_ylabel(r"$\Delta$ (points)")
    ps.panel(a, "(a) one dataset, three envelope shapes")
    a.legend(frameon=False, loc="upper left", handlelength=1.6)

    # ---- (b) the baselines, against the crossing bracket ------------------
    b.axhspan(LO, HI, color=ps.OI["yellow"], alpha=0.45, lw=0, zorder=0)
    b.text(1.05, (LO + HI) / 2.0, "crossing bracket", fontsize=ps.SMALL,
           color=ps.MUTED, va="center", ha="left", zorder=3)
    for key, label, colour, mark in BACKBONES:
        pts = data[key]
        if not pts:
            continue
        b.plot([p[0] for p in pts], [p[1] for p in pts],
               color=colour, marker=mark, label=label, clip_on=False)
    b.set_xscale("log")
    b.set_xticks([1, 2, 5, 10, 25, 100])
    b.set_xticklabels(["1", "2", "5", "10", "25", "100"])
    b.set_xlabel("ImageNet64 training fraction (%)")
    b.set_ylabel("baseline accuracy (%)")
    ps.panel(b, "(b) each gain decays as its baseline enters it")

    fig.tight_layout(pad=0.3, h_pad=0.9)
    ps.save(fig, os.path.join(HERE, "inenv.pdf"))
    print("inenv figure written")


if __name__ == "__main__":
    main()
