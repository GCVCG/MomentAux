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
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SRC = os.path.join(ROOT, "results", "currency_vit_c100_10pct.json")

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

    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(7.0, 2.5), gridspec_kw={"width_ratios": [1.55, 1.0]})

    # ---- (a) normalized agreement, all four measures ----------------------
    x = np.arange(len(pairs))
    w = 0.2
    colors = ["#2a6f97", "#468faf", "#89c2d9", "#a9d6e5"]
    for k, (key, name) in enumerate(MEASURES):
        vals = [r["pairwise"][p][key] for p in pairs]
        ax.bar(x + (k - 1.5) * w, vals, w, label=name, color=colors[k],
               edgecolor="white", linewidth=0.4)
    ax.axhline(1.0, color="#b0413e", lw=1.0, ls="--", zorder=0)
    ax.text(len(pairs) - 0.42, 1.02, "same-seed level", color="#b0413e",
            fontsize=6.5, va="bottom", ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels([PAIR_LABEL[p] for p in pairs], fontsize=7)
    ax.set_ylabel("agreement / own seed-to-seed\nagreement", fontsize=7.5)
    ax.set_ylim(0, 1.35)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6.2, ncol=2, frameon=False, loc="upper right")
    ax.set_title("(a) do they change the same things?", fontsize=8, loc="left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # ---- (b) how many images each fixes, and the shared part --------------
    arms = ["prior", "simclr", "augmentation"]
    names = {"prior": "prior", "simclr": "SimCLR", "augmentation": "aug."}
    n_fixed = [r["arms"][a]["n_fixed_mean"] for a in arms]
    deltas = [r["arms"][a]["delta"] for a in arms]
    y = np.arange(len(arms))
    ax2.barh(y, n_fixed, 0.55, color="#468faf", edgecolor="white", linewidth=0.4)
    for i, (n, d) in enumerate(zip(n_fixed, deltas)):
        ax2.text(n + 40, i, f"{n:.0f}  ({d:+.1f} pts)", va="center", fontsize=6.8)
    ax2.set_yticks(y)
    ax2.set_yticklabels([names[a] for a in arms], fontsize=7)
    ax2.set_xlabel("test images fixed vs. the shared baseline", fontsize=7.5)
    ax2.set_xlim(0, max(n_fixed) * 1.55)
    ax2.tick_params(labelsize=7)
    ax2.invert_yaxis()
    ax2.set_title("(b) how much they change", fontsize=8, loc="left")
    for s in ("top", "right"):
        ax2.spines[s].set_visible(False)

    fig.tight_layout()
    out = os.path.join(HERE, "currency_evidence.pdf")
    fig.savefig(out, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
