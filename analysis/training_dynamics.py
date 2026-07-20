"""Training-dynamics observability from the metrics.csv every run already
writes: HOW does the aux prior change the course of training, not just its
endpoint?

One figure per baseline/aux pair, all seeds:

  panel 1  test accuracy vs epoch      -- per-seed traces + mean band. Shows
           WHERE in training the gap opens (early prior-dominated phase vs
           late pure-CE phase).
  panel 2  train loss vs epoch (log)   -- fitting speed; aux runs carry the
           extra λ·MSE term early, so their total loss starts higher.
  panel 3  λ(t) schedule + lr(t)       -- reconstructed from the config
           (cosine λ0 → λ_final over the epoch budget) next to the lr
           schedule: the "prior dominates early, data takes over" picture.
  panel 4  newer runs only: ce_loss vs aux_loss components and the tapped
           feature std (collapse diagnostic), when those columns exist in
           metrics.csv (logged by train.py since 2026-07-20). Older runs
           show what they have.

    python analysis/training_dynamics.py --pair tin_none_1pct tin_aux_1pct
"""

import argparse
import csv
import math
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from per_class_delta import find_config


def read_metrics(cell):
    """{seed: {column: np.array}} for every seed dir with a metrics.csv."""
    out = {}
    for sd in sorted(d for d in os.listdir(f"runs/{cell}")
                     if d.startswith("seed")):
        path = f"runs/{cell}/{sd}/metrics.csv"
        if not os.path.exists(path):
            continue
        with open(path) as f:
            rows = list(csv.DictReader(f))
        cols = {}
        for k in rows[0]:
            vals = [r[k] for r in rows]
            try:
                cols[k] = np.array([float(v) if v != "" else np.nan
                                    for v in vals])
            except ValueError:
                continue
        out[sd] = cols
    return out


def band(ax, runs, col, color, label, scale=1.0, log=False):
    series = [m[col] * scale for m in runs.values() if col in m]
    if not series:
        return False
    L = min(len(s) for s in series)
    arr = np.stack([s[:L] for s in series])
    x = np.arange(L)
    for s in arr:
        ax.plot(x, s, color=color, alpha=0.18, lw=0.7)
    ax.plot(x, np.nanmean(arr, 0), color=color, lw=1.8, label=label)
    if log:
        ax.set_yscale("log")
    return True


def lam_schedule(cfg, epochs):
    aux = cfg.get("moment_aux") or {}
    lam0 = aux.get("weight", 0.0)
    lamF = aux.get("weight_final", lam0)
    if aux.get("weight_schedule") == "cosine":
        t = np.arange(epochs) / max(epochs - 1, 1)
        return lamF + (lam0 - lamF) * (1 + np.cos(math.pi * t)) / 2
    return np.full(epochs, lam0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", nargs=2, metavar=("NONE_CELL", "AUX_CELL"),
                    required=True)
    ap.add_argument("--out", default="docs/viz")
    args = ap.parse_args()

    none_cell, aux_cell = args.pair
    cfg_aux = yaml.safe_load(open(find_config(aux_cell)))
    runs = {c: read_metrics(c) for c in args.pair}
    n_ep = int(cfg_aux.get("epochs", 200))

    fig, axes = plt.subplots(1, 4, figsize=(17, 3.8))

    ax = axes[0]
    band(ax, runs[none_cell], "test_acc", "#D55E00", f"baseline ({len(runs[none_cell])} seeds)", 100)
    band(ax, runs[aux_cell], "test_acc", "#0072B2", f"aux ({len(runs[aux_cell])} seeds)", 100)
    ax.set_xlabel("epoch"); ax.set_ylabel("test top-1 (%)")
    ax.set_title("test accuracy"); ax.legend(fontsize=8)

    ax = axes[1]
    band(ax, runs[none_cell], "train_loss", "#D55E00", "baseline", log=True)
    band(ax, runs[aux_cell], "train_loss", "#0072B2", "aux (CE + λ·MSE)", log=True)
    ax.set_xlabel("epoch"); ax.set_ylabel("train loss (log)")
    ax.set_title("total train loss"); ax.legend(fontsize=8)

    ax = axes[2]
    lam = lam_schedule(cfg_aux, n_ep)
    ax.plot(lam, color="#0072B2", lw=1.8, label="λ(t) (from config)")
    m0 = next(iter(runs[aux_cell].values()), {})
    if "lr" in m0:
        ax2 = ax.twinx()
        ax2.plot(m0["lr"], color="#999999", lw=1.2, ls="--", label="lr(t)")
        ax2.set_ylabel("lr", color="#666666"); ax2.tick_params(labelsize=7)
    ax.set_xlabel("epoch"); ax.set_ylabel("aux weight λ")
    ax.set_title(f"schedules: λ {cfg_aux.get('moment_aux', {}).get('weight', 0)}"
                 f" → {cfg_aux.get('moment_aux', {}).get('weight_final', 0)}"
                 " (cosine), lr (dashed)")
    ax.legend(fontsize=8, loc="upper right")

    ax = axes[3]
    have = False
    have |= band(ax, runs[aux_cell], "ce_loss", "#009E73", "aux run: CE part", log=True)
    have |= band(ax, runs[aux_cell], "aux_loss", "#CC79A7", "aux run: raw MSE part", log=True)
    if have:
        ax.set_title("loss components (aux run)")
        ax.set_xlabel("epoch"); ax.legend(fontsize=8)
        m1 = next(iter(runs[aux_cell].values()), {})
        if "tap_std" in m1:
            ax3 = ax.twinx()
            band(ax3, runs[aux_cell], "tap_std", "#E69F00", "tap feature std")
            ax3.set_ylabel("tap std", color="#B87700")
    else:
        ax.text(0.5, 0.5, "loss components not logged for this run\n"
                "(train.py logs ce_loss/aux_loss/tap_std since 2026-07-20)",
                ha="center", va="center", fontsize=8, transform=ax.transAxes)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title("loss components (n/a)")

    fig.suptitle(f"{aux_cell} vs {none_cell}: training dynamics", fontsize=11)
    fig.tight_layout()
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"dynamics_{aux_cell}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"DYNAMICS {aux_cell} -> {path}", flush=True)


if __name__ == "__main__":
    main()
