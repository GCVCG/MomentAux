"""The imprint dissociation: alignment gap against feature gain.

WHY A FIGURE AND NOT ONLY THE TABLE. The claim is that two quantities are
INDEPENDENT -- the oriented-energy imprint tracks whether the moment target was
present, while G tracks what the initialization supplied. A table asks the
reader to hold five rows in mind and do that cross-comparison themselves. Plotted
against each other the dissociation is immediate: the vertical position is set by
the target, and it does not follow the horizontal axis at all.

The two control points carry most of the weight and are labelled in the axes
rather than left to the caption, since a reader scanning the figure should not
have to reconstruct which point is which:

  * moment prior tapped two stages EARLIER -- identical target, imprint absent
    when read at layer 3, so the effect is localised to where the target is
    imposed;
  * the same self-supervised initialization as the negative-gap cells, PLUS the
    moment target -- the largest gap measured.

Reads results/imprint_specificity.json, the record the table is built from.
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pubstyle as PS  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "..", "results", "imprint_specificity.json")

# Marker encodes the factor that matters: filled = the moment target was
# present during training, open = it was not.
STYLE = {
    "prior_l3":  ("Moment prior (tap L3)",   PS.OI["verm"],  "o", True),
    "prior_tap": ("Moment prior, tap L1/L2", PS.OI["orange"], "s", True),
    "ssl":       ("SimCLR init, no target",  PS.OI["blue"],  "^", False),
    "combo":     ("SimCLR init $+$ target",  PS.OI["green"], "D", True),
    "rand":      ("Random fixed target",     PS.OI["purple"], "v", False),
}


def bucket(r):
    if "simclraux" in r["cell"]:
        return "combo"
    if r["family"] == "SimCLR init":
        return "ssl"
    if r["family"] == "random target":
        return "rand"
    return "prior_tap" if "tap" in r["cell"] else "prior_l3"


def main():
    PS.use()
    if not os.path.exists(SRC):
        raise SystemExit("missing %s -- run analysis/imprint_specificity.py" % SRC)
    d = json.load(open(SRC))
    fig, ax = plt.subplots(figsize=(PS.COL, 2.45))
    ax.axhline(0, color=PS.RULE, lw=0.7, zorder=0)

    seen = set()
    for r in d:
        k = bucket(r)
        label, col, mk, filled = STYLE[k]
        ax.plot(r["G"], r["align_gap"], marker=mk, ms=4.4, ls="none",
                color=col, mfc=col if filled else "none", mew=1.0,
                label=None if k in seen else label, zorder=3)
        seen.add(k)

    # the two single-cell controls, named in the axes
    for k, txt, dx, dy, ha in (
            ("prior_tap", "same target,\nimposed earlier", -0.7, -0.085, "right"),
            ("combo", "same init,\nplus the target", -0.3, 0.045, "right")):
        pts = [r for r in d if bucket(r) == k]
        x = sum(p["G"] for p in pts) / len(pts)
        y = sum(p["align_gap"] for p in pts) / len(pts)
        ax.annotate(txt, xy=(x, y), xytext=(x + dx, y + dy), ha=ha,
                    fontsize=PS.SMALL - 0.8, color=PS.MUTED, linespacing=1.15,
                    arrowprops=dict(arrowstyle="-", lw=0.6, color=PS.MUTED,
                                    shrinkA=1, shrinkB=3))

    ax.set_xlabel("feature gain $G$ (points)", fontsize=PS.LABEL)
    ax.set_ylabel("oriented-energy alignment gap", fontsize=PS.LABEL)
    # The two clusters leave a wide empty band around zero; the legend goes
    # there rather than over the points it explains.
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo - 0.02 * (hi - lo), hi + 0.10 * (hi - lo))
    ax.legend(fontsize=PS.LEGEND - 1.0, frameon=False, ncol=2,
              loc="center", bbox_to_anchor=(0.5, 0.42),
              handlelength=1.2, borderpad=0.1, labelspacing=0.25,
              columnspacing=0.8, handletextpad=0.35)
    PS.panel(ax, "Alignment tracks the target, not $G$")

    fig.tight_layout(pad=0.35)
    out = os.path.join(HERE, "imprint_dissociation.pdf")
    # pad_inches=0.02 rather than matplotlib's default 0.1in: the default
    # leaves 7.2pt of blank on every side, which at these figure widths is
    # 5-8% of the graphic's height and reads as a white band inside the float.
    PS.save(fig, out)
    print("imprint_dissociation written; %d cells, %d families" % (len(d), len(seen)))


if __name__ == "__main__":
    main()
