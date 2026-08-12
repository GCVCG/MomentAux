#!/usr/bin/env python3
"""Regenerate every dense-prediction table from runs_dense/.

The dense sibling of aggregate.py, and it exists for the same reason: until
now every dense number in the ledger came from an ad-hoc query, which is
exactly the fragility the "tables regenerate from run records" rule prevents.

TWO TABLES, because the dense study needs both and they are not the same thing:

  dense_results.csv -- one row per (population, fraction): both arms, Delta,
    its standard error, and Delta as a PERCENTAGE of the baseline. The relative
    column is not decoration. Absolute mIoU deltas are not comparable across
    populations whose baselines differ 8x (voc@100% 51.5 vs pascalcontext@100%
    6.4), and reading them as if they were produced a "13x label-space
    collapse" that was really ~1.6x. Report both, always.

  dense_law.csv -- the Delta = G + readout decomposition, 1-25% only. The 100%
    cells are EXCLUDED by the probe-ceiling rule: there the probe's labels ARE
    the cell's labels, so no G/readout split is interpretable (the cub@100%
    precedent). They still appear in dense_results.csv, where only Delta is
    claimed.

READOUT IS REPORTED AGAINST PIXEL ACCURACY, not mIoU, and that is a finding
rather than a formatting choice. The sign law's crossing bracket [31.8, 40.3]
is an ACCURACY bracket; mIoU is a far harsher metric, so a 4.65 mIoU cell is
not a low-task-performance cell -- its head may be classifying 72% of pixels
correctly. Scoring the law on mIoU put every dense cell on the wrong flank and
produced a prediction that was wrong for a units reason.

Grouping is by config NAME, matching aggregate.py: variants that share a
population differ in ways the name records and a parsed field would not.
"""
import argparse
import csv
import glob
import json
import os
import re
import statistics as st

POPULATIONS = ["voc", "cityscapes", "foodseg103", "ade20k", "pascalcontext",
               "diagswin_voc"]
PCTS = [1, 2, 5, 10, 25, 100]
LAW_PCTS = [1, 2, 5, 10, 25]          # 100% excluded: probe-ceiling rule
CROSSING = (31.8, 40.3)                # accuracy-scale bracket, from the r18 grid


def _cell(runs, name):
    """(miou list, pixel-accuracy list) over a cell's seeds."""
    miou, pacc = [], []
    for f in sorted(glob.glob(os.path.join(runs, name, "seed*", "final.json"))):
        try:
            d = json.load(open(f))
        except Exception:
            continue                    # a torn record is not a measurement
        miou.append(d["final_miou"])
        pacc.append(d["final_pixel_acc"])
    return miou, pacc


def _probe(runs, name):
    p = os.path.join(runs, name, "dense_probe.json")
    if not os.path.exists(p):
        return None, None, 0
    d = json.load(open(p))
    n = d.get("n_seeds", 0) or 0
    sem = d["probe_miou_std"] / n ** 0.5 if n > 1 else float("nan")
    return d["probe_miou_mean"], sem, n


def _mean_sem(v):
    if not v:
        return float("nan"), float("nan")
    if len(v) == 1:
        return v[0], float("nan")
    return st.mean(v), st.stdev(v) / len(v) ** 0.5


def build(runs):
    results, law = [], []
    for pop in POPULATIONS:
        for pct in PCTS:
            nv, npa = _cell(runs, "%s_none_%dpct" % (pop, pct))
            av, apa = _cell(runs, "%s_aux_%dpct" % (pop, pct))
            if len(nv) < 2 or len(av) < 2:
                continue
            bn, sn = _mean_sem(nv)
            ba, sa = _mean_sem(av)
            delta = ba - bn
            sd = (sn ** 2 + sa ** 2) ** 0.5
            pixacc = st.mean(npa)
            results.append(dict(
                population=pop, pct=pct, n_seeds_none=len(nv), n_seeds_aux=len(av),
                baseline_miou=round(bn, 3), baseline_sem=round(sn, 3),
                aux_miou=round(ba, 3), aux_sem=round(sa, 3),
                delta_miou=round(delta, 3), delta_sem=round(sd, 3),
                delta_rel_pct=round(100 * delta / bn, 2) if bn > 0 else "",
                baseline_pixel_acc=round(pixacc, 2)))

            if pct not in LAW_PCTS:
                continue
            pb, pbs, nb = _probe(runs, "%s_none_%dpct" % (pop, pct))
            pa, pas, na = _probe(runs, "%s_aux_%dpct" % (pop, pct))
            if pb is None or pa is None:
                continue
            G = pa - pb
            gsem = (pbs ** 2 + pas ** 2) ** 0.5
            readout = delta - G
            # The law makes NO sign call inside the bracket, so say so rather
            # than scoring a cell the law declines to predict (the mnet
            # precedent, whose baseline sat inside it).
            if pixacc < CROSSING[0]:
                branch, predicted = "below", "negative"
            elif pixacc > CROSSING[1]:
                branch, predicted = "above", "positive"
            else:
                branch, predicted = "inside", ""
            # Resolvable only against its own uncertainty: on the right flank
            # Delta and G are both ~0, so readout is a small difference of small
            # numbers and its SIGN is noise. Counting those signs tests nothing.
            rsem = (sd ** 2 + gsem ** 2) ** 0.5
            resolvable = (abs(readout) > 2 * rsem) if rsem == rsem else False
            law.append(dict(
                population=pop, pct=pct,
                baseline_miou=round(bn, 3), delta_miou=round(delta, 3),
                probe_none=round(pb, 3), probe_aux=round(pa, 3),
                G=round(G, 3), G_sem=round(gsem, 3) if gsem == gsem else "",
                readout=round(readout, 3),
                readout_sem=round(rsem, 3) if rsem == rsem else "",
                baseline_pixel_acc=round(pixacc, 2), branch=branch,
                predicted_sign=predicted, resolvable=resolvable,
                sign_as_predicted=("" if not predicted or not resolvable else
                                   (readout < 0) == (predicted == "negative"))))
    return results, law


def write_csv(path, rows):
    if not rows:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote %s (%d rows)" % (path, len(rows)))


def summarise(results, law):
    out = ["# Dense-prediction results\n",
           "Regenerated by `analysis/aggregate_dense.py` from `runs_dense/`.",
           "Recipe: 200 epochs at every fraction, matching the frozen",
           "classification budget.\n",
           "## Envelope (Delta mIoU, relative % of baseline in brackets)\n"]
    hdr = "| population | " + " | ".join("%d%%" % p for p in PCTS) + " |"
    out += [hdr, "|" + "---|" * (len(PCTS) + 1)]
    for pop in POPULATIONS:
        cells = []
        for p in PCTS:
            r = next((r for r in results if r["population"] == pop and r["pct"] == p), None)
            cells.append("%+.2f (%s%%)" % (r["delta_miou"], r["delta_rel_pct"]) if r else "--")
        out.append("| %s | %s |" % (pop, " | ".join(cells)))

    res = [r for r in law if r["resolvable"] and r["predicted_sign"]]
    ok = [r for r in res if r["sign_as_predicted"]]
    out += ["\n## Law: Delta = G + readout\n",
            "%d cells with both Delta and G (1-25%%; 100%% excluded by the "
            "probe-ceiling rule)." % len(law),
            "Readout is evaluated against PIXEL ACCURACY, not mIoU -- the "
            "crossing bracket [%.1f, %.1f] is an accuracy bracket." % CROSSING,
            "",
            "- inside the bracket (no prediction): %d" %
            len([r for r in law if r["branch"] == "inside"]),
            "- unresolvable (|readout| <= 2 SEM): %d" %
            len([r for r in law if not r["resolvable"]]),
            "- resolvable and testing the law: %d" % len(res),
            "  - sign as predicted: %d" % len(ok),
            "  - wrong side: %d" % (len(res) - len(ok))]
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs_dense")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    results, law = build(a.runs)
    if not results:
        raise SystemExit("no dense cells found under %s -- refusing to write "
                         "empty tables (an empty result and a broken path must "
                         "not look alike)" % a.runs)
    write_csv(os.path.join(a.out, "dense_results.csv"), results)
    write_csv(os.path.join(a.out, "dense_law.csv"), law)
    p = os.path.join(a.out, "dense_summary.md")
    open(p, "w").write(summarise(results, law))
    print("wrote %s" % p)


if __name__ == "__main__":
    main()
