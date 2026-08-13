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
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(PS.WIDE, 2.5))

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
    PS.panel(axA, "(a) the envelope is unimodal, and neutral at sufficiency")

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
    axB.annotate("crossing\nbracket", xy=(36, axB.get_ylim()[1] * 0.62),
                 ha="center", fontsize=PS.SMALL - 0.5, color=PS.MUTED)
    axB.annotate(r"filled = resolvable ($|$readout$|>2\,$SEM)",
                 xy=(0.03, 0.94), xycoords="axes fraction", ha="left", va="top",
                 fontsize=PS.SMALL - 0.5, color=PS.MUTED)
    PS.panel(axB, "(b) every resolvable cell sits above the crossing")

    fig.tight_layout(pad=0.4, w_pad=1.2)
    out = os.path.join(HERE, "dense_envelope.pdf")
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    n_res = sum(1 for r in law if r["resolvable"] == "True" and r["predicted_sign"])
    print("dense_envelope written; %d law cells, %d resolvable, "
          "pixAcc range %.0f-%.0f"
          % (len(law), n_res,
             min(float(r["baseline_pixel_acc"]) for r in law),
             max(float(r["baseline_pixel_acc"]) for r in law)))


if __name__ == "__main__":
    main()
