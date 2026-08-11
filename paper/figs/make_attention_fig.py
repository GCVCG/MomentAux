"""Figure: mean attention distance per block, with and without the prior.

Reads results/attention_locality_vit.json (analysis/attention_locality.py).
The reference line is the distance a UNIFORM attention map would give on this
token grid (4.136 patch units for 8x8), computed here rather than asserted:
a curve sitting on that line means the head attends everywhere equally.
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
SRC = os.path.join(ROOT, "results", "attention_locality_vit.json")

sys.path.insert(0, HERE)
import pubstyle as PS   # noqa: E402

# One column, two rows, authored at the width it is placed at: the panel
# contents keep their full labels and type sizes rather than being shrunk to
# fit a layout they were not drawn for.
NAME = {
    "baseline": "ViT-tiny baseline",
    "simclr":   "+ SimCLR init ($2\\times$)",
    "prior":    "+ prior ($1.02\\times$)",
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

    PS.use()
    fig, axes = plt.subplots(2, 1, figsize=(PS.COL, 4.25), sharex=True,
                             sharey=True)
    for ax, key, title in (
        (axes[0], "block_mean", "(a) mean over all heads"),
        (axes[1], "block_min_head", "(b) the most local head in each block"),
    ):
        ax.axhline(ref, color="#333333", lw=0.9, ls=":", zorder=0)
        ax.text(0.15, ref + 0.03, "uniform attention", fontsize=PS.SMALL,
                color="#333333")
        ax.axvline(TAP, color=PS.ARM["prior"], lw=0.8, ls="--", alpha=0.45,
                   zorder=0)
        if key == "block_min_head":   # the line is in both panels; the label
            ax.text(TAP - 0.16, 2.60, "tap", fontsize=PS.SMALL,   # would sit
                    color=PS.ARM["prior"], ha="right", rotation=90,  # under
                    va="bottom")                            # (a)'s legend
        for label in ("baseline", "simclr", "prior"):
            if label not in r["arms"]:
                continue
            arr = np.array(r["arms"][label]["per_seed_block_head"])  # seeds,blocks,heads
            y = arr.mean(2) if key == "block_mean" else arr.min(2)
            m, sd = y.mean(0), y.std(0)
            x = np.arange(1, m.size + 1)
            ax.plot(x, m, marker=PS.MARK[label], color=PS.ARM[label],
                    label=NAME[label])
            ax.fill_between(x, m - sd, m + sd, color=PS.ARM[label],
                            alpha=0.15, lw=0)
        ax.set_xticks(range(1, 13, 2))
        ax.set_ylabel("mean attention distance\n(patch units)")
        PS.panel(ax, title)
    axes[1].set_xlabel("transformer block")
    axes[0].set_ylim(2.55, 4.45)
    axes[0].legend(frameon=False, loc="lower right")

    fig.tight_layout(h_pad=1.0)
    out = os.path.join(HERE, "attention_locality.pdf")
    fig.savefig(out, bbox_inches="tight")
    print("wrote", out, " uniform ref =", round(ref, 3))


if __name__ == "__main__":
    main()
