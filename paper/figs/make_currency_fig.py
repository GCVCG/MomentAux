"""Figure: does 'same currency' show up in WHAT gets fixed and WHAT is learned?

Reads results/currency_vit_c100_10pct.json (analysis/currency_evidence.py) and
draws two panels:

  (a) the four agreement measures for the three intervention pairs, each
      normalized by the same intervention's own seed-to-seed agreement, so
      1.0 = "these two sources differ no more than two seeds of one of them";
  (b) how many test images each intervention fixes, and how many of those the
      other member of the pair fixes too.

Deliberately plots ALL FOUR measures rather than the most favourable one: the
Jaccard gap is partly a set-size effect, and the size-robust overlap
coefficient separates the pairs much less. Showing one measure would overstate
the result.
"""
import json
import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SRC = os.path.join(ROOT, "results", "currency_vit_c100_10pct.json")

sys.path.insert(0, HERE)
import pubstyle as PS   # noqa: E402

PAIR_LABEL = {
    "prior|simclr": "prior + SimCLR\n(same currency)",
    "prior|augmentation": "prior + aug.\n(different)",
    "simclr|augmentation": "SimCLR + aug.\n(different)",
}
MEASURES = [
    ("fixed_similarity", "Jaccard"),
    ("overlap_similarity", "overlap coef."),
    ("lift_similarity", "lift"),
    ("cka_similarity", "CKA"),
]


def main():
    with open(SRC) as f:
        r = json.load(f)
    pairs = list(PAIR_LABEL)

    # Column width (3.33in), panels STACKED: authored at the width it is
    # placed at, so the font sizes below are what the reader sees.
    PS.use()
    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(PS.COL, 3.5), gridspec_kw={"height_ratios": [1.45, 1.0]})

    # ---- (a) normalized agreement, all four measures ----------------------
    x = np.arange(len(pairs))
    w = 0.2
    # one ordered ramp around the shared blue: these are four measures of
    # ONE quantity, not four arms, so they read as a family rather than
    # borrowing the arm colours.
    colors = ["#00548A", PS.OI["blue"], "#4C9FD4", PS.OI["sky"]]
    for k, (key, name) in enumerate(MEASURES):
        vals = [r["pairwise"][p][key] for p in pairs]
        ax.bar(x + (k - 1.5) * w, vals, w, label=name, color=colors[k],
               edgecolor="white", linewidth=0.4)
    ax.axhline(1.0, color=PS.ARM["prior"], lw=1.0, ls="--", zorder=0)
    ax.text(len(pairs) - 0.45, 1.03, "same-seed level", color=PS.ARM["prior"],
            fontsize=PS.SMALL, va="bottom", ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels([PAIR_LABEL[p] for p in pairs], fontsize=PS.SMALL)
    ax.set_ylabel("agreement / own seed-to-seed\nagreement", fontsize=PS.LABEL)
    ax.set_ylim(0, 1.42)
    ax.tick_params(labelsize=PS.TICK)
    ax.legend(fontsize=PS.SMALL, ncol=2, frameon=False, loc="upper right")
    PS.panel(ax, "(a) do they change the same things?")

    # ---- (b) how many images each fixes, and the shared part --------------
    arms = ["prior", "simclr", "augmentation"]
    names = {"prior": "prior", "simclr": "SimCLR", "augmentation": "aug."}
    n_fixed = [r["arms"][a]["n_fixed_mean"] for a in arms]
    deltas = [r["arms"][a]["delta"] for a in arms]
    y = np.arange(len(arms))
    ax2.barh(y, n_fixed, 0.55, color=[PS.ARM[a] for a in arms],
         edgecolor="white", linewidth=0.4)
    for i, (n, d) in enumerate(zip(n_fixed, deltas)):
        ax2.text(n + 45, i, f"{n:.0f} ({d:+.1f})", va="center", fontsize=PS.SMALL)
    ax2.set_yticks(y)
    ax2.set_yticklabels([names[a] for a in arms], fontsize=PS.SMALL)
    ax2.set_xlabel("test images fixed vs. the shared baseline", fontsize=PS.LABEL)
    ax2.set_xlim(0, max(n_fixed) * 1.55)
    ax2.tick_params(labelsize=PS.TICK)
    ax2.invert_yaxis()
    PS.panel(ax2, "(b) how much they change")

    fig.tight_layout()
    out = os.path.join(HERE, "currency_evidence.pdf")
    fig.savefig(out, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
