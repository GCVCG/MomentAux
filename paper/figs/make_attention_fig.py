"""Figure: mean attention distance per block, with and without the prior.

Reads results/attention_locality_vit.json (analysis/attention_locality.py).
The reference line is the distance a UNIFORM attention map would give on this
token grid (4.136 patch units for 8x8), computed here rather than asserted:
a curve sitting on that line means the head attends everywhere equally.
"""
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
SRC = os.path.join(ROOT, "results", "attention_locality_vit.json")

STYLE = {
    "baseline": ("#7a7a7a", "o", "ViT-tiny baseline"),
    "simclr":   ("#468faf", "s", "+ SimCLR init ($2\\times$)"),
    "prior":    ("#b0413e", "^", "+ prior ($1.02\\times$)"),
}
TAP = 8  # the auxiliary target is regressed from blocks.8


def uniform_distance(grid=8):
    pos = np.array([(r, c) for r in range(grid) for c in range(grid)], float)
    d = np.linalg.norm(pos[:, None] - pos[None], axis=-1)
    return float(d.mean())


def main():
    with open(SRC) as f:
        r = json.load(f)
    ref = uniform_distance()

    fig, axes = plt.subplots(2, 1, figsize=(3.33, 3.9), sharex=True, sharey=True)
    for ax, key, title in (
        (axes[0], "block_mean", "(a) mean over all heads"),
        (axes[1], "block_min_head", "(b) most local head per block"),
    ):
        ax.axhline(ref, color="#333", lw=0.9, ls=":", zorder=0)
        ax.text(0.15, ref + 0.03, "uniform attention", fontsize=6, color="#333")
        ax.axvline(TAP, color="#b0413e", lw=0.8, ls="--", alpha=0.45, zorder=0)
        if key == "block_min_head":      # label it once; the line is in both
            ax.text(TAP - 0.18, 2.62, "tap", fontsize=6, color="#b0413e",
                    ha="right", rotation=90, va="bottom")
        for label in ("baseline", "simclr", "prior"):
            if label not in r["arms"]:
                continue
            colour, marker, name = STYLE[label]
            arr = np.array(r["arms"][label]["per_seed_block_head"])  # seeds,blocks,heads
            y = arr.mean(2) if key == "block_mean" else arr.min(2)
            m, sd = y.mean(0), y.std(0)
            x = np.arange(1, m.size + 1)
            ax.plot(x, m, marker=marker, ms=3.2, lw=1.3, color=colour, label=name)
            ax.fill_between(x, m - sd, m + sd, color=colour, alpha=0.15, lw=0)
        if key == "block_min_head":
            ax.set_xlabel("transformer block", fontsize=7)
        ax.set_title(title, fontsize=7.5, loc="left")
        ax.tick_params(labelsize=6.5)
        ax.set_xticks(range(1, 13, 2))
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    for a_ in axes:
        a_.set_ylabel("attn. distance", fontsize=7)
    axes[0].set_ylim(2.55, 4.45)
    axes[0].legend(fontsize=6, frameon=False, loc="lower right")

    fig.tight_layout()
    out = os.path.join(HERE, "attention_locality.pdf")
    fig.savefig(out, bbox_inches="tight")
    print("wrote", out, " uniform ref =", round(ref, 3))


if __name__ == "__main__":
    main()
