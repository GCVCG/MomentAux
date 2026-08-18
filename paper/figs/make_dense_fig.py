"""The dense-prediction figure for Section 10.

Two panels, and the choice of what to plot is itself the finding:

  (a) RELATIVE delta, not absolute mIoU. The six populations' baselines differ
      eightfold (VOC 51.5 at full data, Pascal-Context 6.4), so absolute mIoU
      deltas are not comparable across them -- reading them as if they were is
      what produced a "13x label-space collapse" that was really ~1.6x. Plotted
      relatively, the envelopes are unimodal and land in the same 5-20% band
      classification occupies, which is the actual cross-task result.

  (b) READOUT against PIXEL ACCURACY, not against mIoU. The crossing bracket is
      an accuracy bracket; scoring the law on mIoU put every dense cell on the
      wrong flank. This panel shows why the dense grid can only half-test the
      law: every resolvable cell sits ABOVE the bracket, on the decayed
      positive branch, so segmentation confirms the positive branch and tests
      the negative branch not at all.

Reads the released CSVs rather than the run tree, so the figure cannot disagree
with the tables in the paper.
"""
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pubstyle as PS  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "..", "results")
BRACKET = (31.8, 40.3)

# Order by label-space width, which is the axis the same-pixels control probes.
POPS = [
    ("voc", "VOC (21 cls)", PS.OI["verm"], "o"),
    ("cityscapes", "Cityscapes (19)", PS.OI["orange"], "s"),
    ("foodseg103", "FoodSeg103 (104)", PS.OI["green"], "^"),
    ("ade20k", "ADE20K (150)", PS.OI["sky"], "v"),
    ("pascalcontext", "Pascal-Ctx (254)", PS.OI["purple"], "D"),
    ("diagswin_voc", "VOC, Swin-T (21)", PS.OI["blue"], "*"),
]


def load(name):
    p = os.path.join(RES, name)
    if not os.path.exists(p):
        raise SystemExit("missing %s -- run analysis/aggregate_dense.py first" % p)
    with open(p) as f:
        return list(csv.DictReader(f))


def main():
    PS.use()
    res, law = load("dense_results.csv"), load("dense_law.csv")
    # Two rows in one column: the panels answer different questions and do
    # not share an axis, but at COL width they still read top-to-bottom and
    # the figure stops competing for full-width float slots.
    fig, (axA, axB) = plt.subplots(2, 1, figsize=(PS.COL, 4.15))

    # ---- (a) relative envelope ------------------------------------------
    for pop, label, col, mk in POPS:
        rows = sorted((r for r in res if r["population"] == pop),
                      key=lambda r: int(r["pct"]))
        xs = [int(r["pct"]) for r in rows if r["delta_rel_pct"] != ""]
        ys = [float(r["delta_rel_pct"]) for r in rows if r["delta_rel_pct"] != ""]
        if not xs:
            continue
        axA.plot(xs, ys, marker=mk, color=col, label=label, lw=1.2, ms=3.4,
                 ls="--" if pop == "diagswin_voc" else "-")
    axA.axhline(0, color=PS.RULE, lw=0.7, zorder=0)
    axA.set_xscale("log")
    axA.set_xticks([1, 2, 5, 10, 25, 100])
    axA.set_xticklabels(["1", "2", "5", "10", "25", "100"])
    axA.set_xlabel("training data (% of split)", fontsize=PS.LABEL)
    axA.set_ylabel(r"$\Delta$ mIoU, % of baseline", fontsize=PS.LABEL)
    # headroom so the legend cannot sit on the Swin peak at 2%
    _lo, _hi = axA.get_ylim()
    axA.set_ylim(_lo, _hi + 0.42 * (_hi - _lo))
    axA.legend(fontsize=PS.LEGEND - 0.6, frameon=False, ncol=2,
               loc="upper left", handlelength=1.6, columnspacing=0.9)
    PS.panel(axA, "(a) unimodal, neutral at sufficiency")

    # ---- (b) readout vs the head's own classification scale --------------
    axB.axvspan(BRACKET[0], BRACKET[1], color=PS.OI["yellow"], alpha=0.30, lw=0,
                zorder=0)
    axB.axhline(0, color=PS.RULE, lw=0.7, zorder=0)
    for pop, label, col, mk in POPS:
        rows = [r for r in law if r["population"] == pop]
        for r in rows:
            resolvable = r["resolvable"] == "True"
            axB.plot(float(r["baseline_pixel_acc"]), float(r["readout"]),
                     marker=mk, color=col, ms=4.2 if resolvable else 3.0,
                     mfc=col if resolvable else "none", mew=0.8, ls="none",
                     zorder=3 if resolvable else 2)
    axB.set_xlabel("baseline pixel accuracy (%)", fontsize=PS.LABEL)
    axB.set_ylabel("readout $= \\Delta - G$ (mIoU)", fontsize=PS.LABEL)
    # EVERY encoding in this panel appears in a legend: the fill/size pair
    # (resolvable vs unresolved) gets its own proxy entries -- the earlier
    # one-line annotation named only the filled half -- and the yellow band
    # is entered too, alongside its in-place label. Marker shape and colour
    # are the population, so those handles stay in panel (a)'s legend and a
    # footnote here says where to read them.
    proxies = [
        Line2D([], [], marker="o", color=PS.INK, mfc=PS.INK, ls="none",
               ms=4.2, mew=0.8,
               label=r"filled, larger: resolvable ($|$readout$|>2\,$SEM)"),
        Line2D([], [], marker="o", color=PS.INK, mfc="none", ls="none",
               ms=3.0, mew=0.8, label="open, smaller: unresolved"),
        Patch(fc=PS.OI["yellow"], alpha=0.30, ec="none",
              label="crossing bracket [31.8, 40.3]"),
    ]
    proxies.append(
        Line2D([], [], ls="none", marker="", color="none",
               label="shape and colour: population, as in (a)"))
    axB.legend(handles=proxies, fontsize=PS.SMALL - 0.5, frameon=False,
               loc="upper left", handlelength=1.0, handletextpad=0.5,
               labelspacing=0.25, borderaxespad=0.2)
    PS.panel(axB, "(b) every resolvable cell is above the crossing")

    fig.tight_layout(pad=0.3, h_pad=1.0)
    out = os.path.join(HERE, "dense_envelope.pdf")
    # pad_inches=0.02 rather than matplotlib's default 0.1in: the default
    # leaves 7.2pt of blank on every side, which at these figure widths is
    # 5-8% of the graphic's height and reads as a white band inside the float.
    PS.save(fig, out)
    n_res = sum(1 for r in law if r["resolvable"] == "True" and r["predicted_sign"])
    print("dense_envelope written; %d law cells, %d resolvable, "
          "pixAcc range %.0f-%.0f"
          % (len(law), n_res,
             min(float(r["baseline_pixel_acc"]) for r in law),
             max(float(r["baseline_pixel_acc"]) for r in law)))


if __name__ == "__main__":
    main()
