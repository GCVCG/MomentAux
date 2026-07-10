"""Learning-curve comparison across cells.

    python analysis/curves.py --cells abl_none abl_cat_calib ... [--out results/curves.png]

Reads runs/<cell>/seed*/metrics.csv, plots per-epoch test top-1 (mean over
seeds, band = min/max), and prints a compact table of curve milestones so
trajectories can be compared without the figure.
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd


def load_cell(runs_root, cell):
    curves = []
    for path in sorted(glob.glob(os.path.join(runs_root, cell, "seed*", "metrics.csv"))):
        df = pd.read_csv(path)
        curves.append(df["test_acc"].to_numpy())
    if not curves:
        return None
    n = min(len(c) for c in curves)
    return np.stack([c[:n] for c in curves])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="+", required=True)
    ap.add_argument("--runs-root", default="runs")
    ap.add_argument("--out", default=None, help="write a PNG here (optional)")
    args = ap.parse_args()

    data = {}
    for cell in args.cells:
        arr = load_cell(args.runs_root, cell)
        if arr is None:
            print(f"warning: no metrics for {cell}")
            continue
        data[cell] = arr

    if not data:
        raise SystemExit("no curves found")

    name_w = max(len(c) for c in data) + 2
    marks = [10, 25, 50, 100, 150, 200]
    header = f"{'cell':{name_w}}" + "".join(f"ep{m:>4} " for m in marks) + "  final"
    print(header)
    for cell, arr in data.items():
        mean = arr.mean(axis=0)
        cols = ""
        for m in marks:
            cols += f"{mean[m - 1] * 100:6.2f} " if len(mean) >= m else "     - "
        print(f"{cell:{name_w}}{cols}  {mean[-1] * 100:5.2f}")

    if args.out:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9, 5.5))
        for cell, arr in data.items():
            mean = arr.mean(axis=0) * 100
            x = np.arange(1, len(mean) + 1)
            ax.plot(x, mean, label=cell, linewidth=1.6)
            if arr.shape[0] > 1:
                ax.fill_between(x, arr.min(axis=0) * 100, arr.max(axis=0) * 100, alpha=0.15)
        ax.set_xlabel("epoch")
        ax.set_ylabel("test top-1 (%)")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        fig.savefig(args.out, dpi=150)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
