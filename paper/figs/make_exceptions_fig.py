"""Where the account fails: better features, worse accuracy at sufficiency.

Every point is a law-scope cell whose readout is RESOLVABLE against its own
seed-paired uncertainty, plotted as end-to-end gain against frozen-feature
gain. On the diagonal, accuracy realizes exactly the feature gain.

The exceptions are the cells in the shaded quadrant: G clearly positive while
Delta is negative -- the representation improved and accuracy did not follow --
drawn larger because the paper's claim is that they CONCENTRATE rather than
scatter. Reproduced from the released per-cell table and the committed audit,
so this figure and Table 9 cannot disagree.
"""
import csv
import os
import sys

import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pubstyle as ps

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "analysis"))
import audit_law_paired as A  # noqa: E402

# What counts as an instance of the phenomenon, and why the filter has three
# clauses rather than one:
#   G >= G_MATERIAL  -- cells with G ~ +0.05 carry the SIGN but not the effect.
#   Delta <= 0       -- accuracy did not follow the features.
#   base > CROSS_HI  -- and this is the clause that matters. ABOVE the crossing
#                       the account predicts readout ~ 0, so a large negative
#                       one is genuinely anomalous. BELOW it a negative readout
#                       is what the account predicts, so the deep-left-flank
#                       CUB and CIFAR-100 cells that also satisfy the first two
#                       clauses are the account WORKING, not failing, and
#                       highlighting them would misreport the paper's claim.
G_MATERIAL = 0.5
CROSS_HI = 40.3


def main():
    ps.use()
    rows = A.load(os.path.join(ROOT, "runs"),
                  os.path.join(ROOT, "results", "all_results.csv"))
    meta = {r["cell"]: r for r in
            csv.DictReader(open(os.path.join(ROOT, "results", "all_results.csv")))}
    _, _, res = A.audit(rows, 2.0)

    ord_, exc = [], []
    for x in res:
        m = meta[x["cell"]]
        if not m["delta"] or not m["G"]:
            continue
        d, g = float(m["delta"]), float(m["G"])
        above = float(m["base_acc"]) > CROSS_HI
        (exc if (g >= G_MATERIAL and d <= 0.0 and above) else ord_).append((g, d, m))

    fig, ax = plt.subplots(figsize=(ps.COL, 2.55))
    lim = (-4.0, 26.0)
    ax.axhspan(lim[0], 0.0, color=ps.OI["yellow"], alpha=0.30, lw=0, zorder=0)
    ax.plot(lim, lim, color=ps.RULE, lw=0.7, ls=(0, (3, 2)), zorder=1)
    ax.axhline(0.0, color=ps.RULE, lw=0.5, zorder=1)

    ax.scatter([p[0] for p in ord_], [p[1] for p in ord_], s=7,
               color=ps.OI["blue"], alpha=0.45, lw=0, zorder=2,
               label="resolvable cells (%d)" % len(ord_))
    ax.scatter([p[0] for p in exc], [p[1] for p in exc], s=26,
               color=ps.OI["verm"], marker="X", lw=0, zorder=3,
               label="better features, worse accuracy (%d)" % len(exc))

    ax.text(25.0, 23.5, r"$\Delta=G$", fontsize=ps.SMALL, color=ps.MUTED,
            ha="right", va="top")
    ax.text(0.6, -3.5, "accuracy falls while features improve",
            fontsize=ps.SMALL, color=ps.MUTED, ha="left", va="bottom")

    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_xlabel("frozen-feature gain $G$ (points)")
    ax.set_ylabel(r"end-to-end gain $\Delta$ (points)")
    ax.legend(frameon=False, loc="upper left", handlelength=1.0,
              borderpad=0.1, labelspacing=0.25)

    fig.tight_layout(pad=0.3)
    ps.save(fig, os.path.join(HERE, "exceptions.pdf"))
    print("exceptions: %d ordinary, %d in the quadrant" % (len(ord_), len(exc)))
    for g, d, m in sorted(exc, key=lambda z: -z[0]):
        print("   %-10s %-14s %4s%%  D=%+.2f  G=%+.2f"
              % (m["dataset"], m["backbone"][:14], m["subset_pct"], d, g))


if __name__ == "__main__":
    main()
