"""Collect all runs into the paper tables (markdown + LaTeX).

    python analysis/aggregate.py [--runs-root runs] [--out results]

Reads every runs/<cell>/seed<N>/final.json (+ robustness.json when present),
groups by cell, and emits mean+/-std top-1 over seeds with param/FLOP columns,
plus a CIFAR-C table with mean corruption error and mCE (Hendrycks &
Dietterich, ICLR 2019) normalised by the stem=none run of the same
(dataset, backbone, subset) cell. Regenerates every reported number from the
raw run artifacts -- nothing is hand-entered.
"""

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np


def load_runs(runs_root):
    runs = []
    for path in sorted(glob.glob(os.path.join(runs_root, "*", "seed*", "final.json"))):
        with open(path) as f:
            final = json.load(f)
        rob_path = os.path.join(os.path.dirname(path), "robustness.json")
        final["robustness"] = None
        if os.path.exists(rob_path):
            with open(rob_path) as f:
                final["robustness"] = json.load(f)
        runs.append(final)
    return runs


def cell_key(run):
    cfg = run["config"]
    return (cfg["dataset"], cfg.get("subset_pct") or 100, cfg["backbone"], cfg["stem"])


def mean_std(values):
    a = np.asarray(values, dtype=float)
    return a.mean(), a.std(ddof=1) if len(a) > 1 else 0.0


def fmt_pm(mean, std, scale=100.0, digits=2):
    return f"{mean * scale:.{digits}f}+/-{std * scale:.{digits}f}"


def aggregate_accuracy(runs):
    cells = defaultdict(list)
    for r in runs:
        cells[cell_key(r)].append(r)
    rows = []
    for key in sorted(cells):
        group = cells[key]
        fm, fs = mean_std([r["final_test_acc"] for r in group])
        bm, bs = mean_std([r["best_test_acc"] for r in group])
        acc = group[0]["accounting"]
        rows.append({
            "dataset": key[0], "subset_pct": key[1], "backbone": key[2],
            "stem": key[3], "seeds": len(group),
            "final_top1": fmt_pm(fm, fs), "best_top1": fmt_pm(bm, bs),
            "params_M": f"{acc['params_trainable'] / 1e6:.2f}",
            "fixed_filter_params": acc["params_stem_fixed_filters"],
            "flops_M": f"{acc['flops_total'] / 1e6:.1f}",
        })
    return rows


def corruption_errors(run):
    """Mean error per corruption (averaged over severities 1..5)."""
    errs = run["robustness"]["errors"]
    return {c: np.mean([sev[s] for s in sorted(sev)]) for c, sev in errs.items()}


def aggregate_robustness(runs):
    """mCE per Hendrycks & Dietterich: CE_c = sum_s E_c,s / sum_s E^base_c,s,
    mCE = mean_c CE_c. Baseline = stem 'none', same (dataset, subset,
    backbone). Errors are seed-averaged before normalisation."""
    cells = defaultdict(list)
    for r in runs:
        if r["robustness"] is not None:
            cells[cell_key(r)].append(r)
    per_cell = {}
    for key, group in cells.items():
        per_corr = defaultdict(list)
        for r in group:
            for c, e in corruption_errors(r).items():
                per_corr[c].append(e)
        per_cell[key] = {c: float(np.mean(v)) for c, v in per_corr.items()}

    rows = []
    for key in sorted(per_cell):
        dataset, subset, backbone, stem = key
        base_key = (dataset, subset, backbone, "none")
        errs = per_cell[key]
        mean_err = float(np.mean(list(errs.values())))
        mce = None
        if base_key in per_cell:
            base = per_cell[base_key]
            ces = [errs[c] / base[c] for c in errs if base.get(c)]
            mce = float(np.mean(ces)) if ces else None
        clean = np.mean(
            [r["robustness"]["clean_error"] for r in cells[key]]
        )
        rows.append({
            "dataset": dataset, "subset_pct": subset, "backbone": backbone,
            "stem": stem, "seeds": len(cells[key]),
            "clean_err": f"{clean * 100:.2f}",
            "mean_corruption_err": f"{mean_err * 100:.2f}",
            "mCE_vs_none": f"{mce * 100:.1f}" if mce is not None else "n/a",
        })
    return rows


def to_markdown(rows, title):
    if not rows:
        return f"## {title}\n\n(no runs found)\n"
    cols = list(rows[0].keys())
    lines = [f"## {title}", "", "| " + " | ".join(cols) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    lines += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    return "\n".join(lines) + "\n"


def to_latex(rows, title):
    if not rows:
        return f"% {title}: no runs found\n"
    cols = list(rows[0].keys())
    header = " & ".join(c.replace("_", r"\_") for c in cols)
    body = " \\\\\n".join(
        " & ".join(str(r[c]).replace("+/-", r" $\pm$ ") for c in cols) for r in rows
    )
    return (
        f"% {title}\n\\begin{{tabular}}{{{'l' * len(cols)}}}\n\\toprule\n"
        f"{header} \\\\\n\\midrule\n{body} \\\\\n\\bottomrule\n\\end{{tabular}}\n"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", default="runs")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    runs = load_runs(args.runs_root)
    print(f"loaded {len(runs)} runs from {args.runs_root}")
    acc_rows = aggregate_accuracy(runs)
    rob_rows = aggregate_robustness(runs)

    os.makedirs(args.out, exist_ok=True)
    md = (to_markdown(acc_rows, "Top-1 accuracy (mean+/-std over seeds)")
          + "\n" + to_markdown(rob_rows, "CIFAR-C robustness"))
    tex = (to_latex(acc_rows, "Top-1 accuracy") + "\n"
           + to_latex(rob_rows, "CIFAR-C robustness"))
    with open(os.path.join(args.out, "summary.md"), "w") as f:
        f.write(md)
    with open(os.path.join(args.out, "summary.tex"), "w") as f:
        f.write(tex)
    print(md)
    print(f"wrote {args.out}/summary.md and {args.out}/summary.tex")


if __name__ == "__main__":
    main()
