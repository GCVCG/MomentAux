#!/usr/bin/env python3
"""Regenerate the detection tables from runs_det/.

The third sibling of aggregate.py and aggregate_dense.py, and it exists for the
same reason they do: a number that reaches the paper through an ad-hoc query is
a number nobody can regenerate.

ONE TABLE, four measures per cell, because detection's headline metric fails at
the low fractions and the paper has to say so with the alternatives beside it:

  AP50    the task metric, and the headline. A cell where BOTH arms land below
          1.0 is marked UNINTERPRETABLE by the floor rule declared before any
          cell ran -- a difference between two near-zero detectors is not a
          measurement. Reported, never silently dropped.
  fg_acc  20-way accuracy at the locations ground truth assigns to an object.
          Pre-registered as the readout scale before any cell ran.
  fg_iou  mean IoU of the predicted against the GT box at those same locations.
          The only measure in the study that reads the coordinate regressor.
  G       the same quantities under a frozen trunk with a fresh 1x1 head fitted
          on the full split, which is what separates "the prior buys detection
          nothing" from "the head cannot cash it".

WHY AP50 COLLAPSES AND THE OTHER TWO DO NOT: AP50 is a ranked-precision
integral, so when precision is poor everywhere the integral is ~0 however much
better one arm's boxes are. fg_acc and fg_iou are conditioned on GROUND-TRUTH
foreground -- through the same assign_targets() the training loss uses, never
through predictions -- so a detector that predicts nothing still has a
well-defined value. Same compressive failure mIoU has on dense prediction,
where a +1.62 point gain in pixel accuracy was reported as "+0.09 mIoU".

COLLAPSED RUNS ARE EXCLUDED FROM DELTA AND REPORTED SEPARATELY. One baseline
seed (vocdet_none_5pct/seed2) has a dead regression branch: fg_iou exactly
0.0000 with fg_acc in line with its siblings, and a reg loss pinned at exactly
1.0000 from epoch 40 onward. Averaging it into a cell mean turned a -0.08 delta
into +0.84. The seed-level accuracy check used elsewhere in the study would not
have caught it, because the cell still trains -- it is a PARTIAL collapse of one
branch of a two-headed model.

NO LAW COLUMN, and that is a result rather than an omission. Delta = G +
readout needs G on the SAME metric as Delta, and the probe's AP50 floors at
exactly the fractions whose baseline sits below the readout crossing. See
det_summary.md for the per-cell accounting.
"""
import argparse
import csv
import glob
import json
import os
import re
import statistics as st

FLOOR_AP50 = 1.0          # pre-declared: both arms below this => uninterpretable
BRACKET = (31.8, 40.3)    # the accuracy crossing bracket, read on fg_acc


def pct_of(cell):
    m = re.search(r"(\d+)pct", cell)
    return int(m.group(1)) if m else None


def load(runs):
    """finals, probes and decomposition, keyed by cell -> list of per-seed dicts."""
    fin, prb = {}, {}
    for f in sorted(glob.glob(os.path.join(runs, "*", "seed*", "final.json"))):
        fin.setdefault(f.split(os.sep)[-3], []).append(json.load(open(f)))
    for f in sorted(glob.glob(os.path.join(runs, "*", "seed*", "det_probe.json"))):
        prb.setdefault(f.split(os.sep)[-3], []).append(json.load(open(f)))
    return fin, prb


def load_decomp(path):
    if not os.path.exists(path):
        return {}
    out = {}
    for r in json.load(open(path)):
        out.setdefault(r["cell"], []).append(r)
    return out


def collapsed(rec):
    """A run whose regression branch died: boxes of zero area at every
    foreground location. Detected on fg_iou, which is exactly 0 there, rather
    than on AP50, which is near zero for healthy low-data cells too."""
    return rec.get("fg_iou") is not None and rec["fg_iou"] < 0.05


def agg(records, key):
    v = [r[key] for r in records if r.get(key) is not None]
    if not v:
        return None, None, 0
    return (st.mean(v),
            (st.stdev(v) / len(v) ** 0.5 if len(v) > 1 else 0.0),
            len(v))


def build(runs, decomp_path):
    fin, prb = load(runs)
    dec = load_decomp(decomp_path)
    rows, notes = [], []
    for p in sorted({pct_of(c) for c in fin if pct_of(c)}):
        none, aux = f"vocdet_none_{p}pct", f"vocdet_aux_{p}pct"
        # the decomposition carries fg_iou; drop collapsed runs from both arms
        dn = [r for r in dec.get(none, []) if not collapsed(r)]
        da = [r for r in dec.get(aux, []) if not collapsed(r)]
        ndrop = len(dec.get(none, [])) - len(dn) + len(dec.get(aux, [])) - len(da)
        if ndrop:
            for r in dec.get(none, []) + dec.get(aux, []):
                if collapsed(r):
                    notes.append(f"{r['cell']}/{r['seed']}: regression branch dead "
                                 f"(fg_iou {r['fg_iou']:.4f}, fg_acc {r['fg_acc']:.2f}) "
                                 f"-- excluded from Delta")
        row = {"pct": p, "n_dropped": ndrop}
        for tag, recs in (("none", dn), ("aux", da)):
            for k in ("ap50", "fg_acc", "fg_iou"):
                m, s, n = agg(recs, k)
                row[f"{tag}_{k}"] = "" if m is None else f"{m:.4f}"
                row[f"{tag}_{k}_sem"] = "" if s is None else f"{s:.4f}"
                row[f"{tag}_n"] = n
        for k in ("ap50", "fg_acc", "fg_iou"):
            mn, sn, _ = agg(dn, k); ma, sa, _ = agg(da, k)
            if mn is None or ma is None:
                row[f"delta_{k}"] = row[f"delta_{k}_sem"] = ""
                continue
            row[f"delta_{k}"] = f"{ma - mn:.4f}"
            row[f"delta_{k}_sem"] = f"{(sn ** 2 + sa ** 2) ** 0.5:.4f}"
        # floor rule, applied to AP50 only
        both_low = (mn is not None and
                    float(row["none_ap50"] or 9) < FLOOR_AP50 and
                    float(row["aux_ap50"] or 9) < FLOOR_AP50)
        row["ap50_interpretable"] = not both_low
        # G, from the frozen-trunk probes
        for k in ("ap50", "fg_acc", "fg_iou"):
            pk = "probe_" + k
            mn, sn, _ = agg(prb.get(none, []), pk)
            ma, sa, _ = agg(prb.get(aux, []), pk)
            if mn is None or ma is None:
                row[f"G_{k}"] = row[f"G_{k}_sem"] = ""
                continue
            row[f"G_{k}"] = f"{ma - mn:.4f}"
            row[f"G_{k}_sem"] = f"{(sn ** 2 + sa ** 2) ** 0.5:.4f}"
            row[f"probe_{k}_none"] = f"{mn:.4f}"
            row[f"probe_{k}_aux"] = f"{ma:.4f}"
        # which branch of the sign law this cell would test, read on fg_acc
        b = float(row["none_fg_acc"]) if row["none_fg_acc"] else None
        row["branch"] = ("" if b is None else
                         "inside-bracket" if BRACKET[0] <= b <= BRACKET[1] else
                         "below" if b < BRACKET[0] else "above")
        rows.append(row)
    return rows, notes


def summarise(rows, notes):
    L = ["# Detection (PASCAL VOC, ResNet-18 + single-level FCOS at stride 8)", ""]
    L.append("Delta is aux minus baseline. AP50 is the headline; cells where BOTH")
    L.append("arms fall below %.1f AP50 are UNINTERPRETABLE by the floor rule and are" % FLOOR_AP50)
    L.append("marked. fg_acc and fg_iou are conditioned on ground-truth foreground and")
    L.append("do not floor. G is the same measure under a frozen trunk.")
    L.append("")
    L.append("| pct | D AP50 | D fg_acc | D fg_iou | G fg_acc | G fg_iou | base fg_acc | branch | floor |")
    L.append("|---:|---:|---:|---:|---:|---:|---:|:--|:--|")
    for r in rows:
        def f(k, p=2):
            return "--" if not r.get(k) else f"{float(r[k]):+.{p}f}"
        L.append("| {}% | {} | {} | {} | {} | {} | {} | {} | {} |".format(
            r["pct"], f("delta_ap50"), f("delta_fg_acc"), f("delta_fg_iou", 4),
            f("G_fg_acc"), f("G_fg_iou", 4),
            ("--" if not r["none_fg_acc"] else f"{float(r['none_fg_acc']):.2f}"),
            r["branch"], "" if r["ap50_interpretable"] else "UNINTERPRETABLE"))
    if notes:
        L += ["", "## Excluded runs", ""] + [f"- {n}" for n in notes]
    L += ["", "## Why there is no law column", "",
          "Delta = G + readout requires G on the same metric as Delta. The probe's",
          "AP50 floors at exactly the fractions whose baseline fg_acc sits below the",
          "crossing bracket, so no cell below the crossing is resolvable, and the",
          "cells above it have Delta and G both near zero. Detection therefore",
          "contributes no resolvable law cell -- the same structural outcome as",
          "Pascal-Context on the dense side, and for the same reason."]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs_det")
    ap.add_argument("--decomp", default="results/det_decompose.json")
    ap.add_argument("--out-dir", default="results")
    a = ap.parse_args()

    rows, notes = build(a.runs, a.decomp)
    if not rows:
        raise SystemExit("no detection cells found under %s" % a.runs)
    os.makedirs(a.out_dir, exist_ok=True)
    p = os.path.join(a.out_dir, "det_results.csv")
    keys = sorted({k for r in rows for k in r})
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["pct"] + [k for k in keys if k != "pct"])
        w.writeheader(); w.writerows(rows)
    print("wrote", p, f"({len(rows)} fractions)")
    q = os.path.join(a.out_dir, "det_summary.md")
    open(q, "w").write(summarise(rows, notes))
    print("wrote", q)
    print()
    print(summarise(rows, notes))


if __name__ == "__main__":
    main()
