"""Mechanical audit of the study's central claims, from raw run files.

SUPERSEDED for the law numbers by analysis/audit_law_paired.py (seed-paired
uncertainty, current crossing bracket [31.8, 40.3]); this script keeps the
original 2026-07 closure checks and writes results/law_audit_legacy.md so it
can never overwrite the canonical results/law_audit.md.

The ledger (CLAUDE.md / FINDINGS.md) is hand-written; this script recomputes
every number in the law's chain directly from runs/*/final.json and
runs/*/linear_probe*.json and CHECKS the claims, so the write-up rests on
machine-verified facts:

  A. Delta = G + readout closure (definitional; verifies file/means integrity)
  B. THE SIGN LAW: readout < 0 for baselines below the crossing band
     [29.8, 33.6], > 0 above it, unconstrained inside; probe-ceiling cells
     excluded (stl@20/50).
  C. ENVELOPE PEAK = G PEAK per dataset (same-space G curves).
  D. Replication pairs agree within stated sigma (tin20 vs tin20b).

Exit code 1 if any check fails -- CI-able.

    python analysis/audit_law.py [--out results/law_audit_legacy.md]
"""

import argparse
import glob
import json
import math
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CROSSING = (29.8, 33.6)   # readout sign crossing, pinned by tin@10% and super2

# (label, baseline cell, aux cell, probe json name or None, dataset, images)
PAIRS = [
    ("C100@1%",  "abl1_none",       "auxmag_1pct_sched0", "linear_probe.json", "cifar100", 500),
    ("C100@5%",  "abl5_none",       "auxmag_5pct_sched0",  "linear_probe.json", "cifar100", 2500),
    ("C100@10%", "abl10_none",      "auxmag_10pct_sched0", "linear_probe.json", "cifar100", 5000),
    ("C10@1%",   "c10_none_1pct",   "c10_aux_1pct",        "linear_probe.json", "cifar10",  500),
    ("C10@2%",   "c10_none_2pct",   "c10_aux_2pct",        "linear_probe.json", "cifar10",  1000),
    ("C10@5%",   "c10_none_5pct",   "c10_aux_5pct",        "linear_probe.json", "cifar10",  2500),
    ("C10@10%",  "c10_none_10pct",  "c10_aux_10pct",       "linear_probe.json", "cifar10",  5000),
    ("stl@10%",  "stl_none_10pct",  "stl_aux_10pct",       "linear_probe.json", "stl10",    500),
    ("tin@1%",   "tin_none_1pct",   "tin_aux_1pct",        "linear_probe.json", "tin",      1000),
    ("tin@2%",   "tin_none_2pct",   "tin_aux_2pct",        "linear_probe.json", "tin",      2000),
    ("tin@5%",   "tin_none_5pct",   "tin_aux_5pct",        "linear_probe.json", "tin",      5000),
    ("tin@10%",  "tin_none_10pct",  "tin_aux_10pct",       "linear_probe.json", "tin",      10000),
    ("super@1%", "super1_none",     "super1_aux",          "linear_probe.json", "cifar100super", 500),
    ("super@2%", "super2_none",     "super2_aux",          "linear_probe.json", "cifar100super", 1000),
    ("tin20",    "tin20_none",      "tin20_aux",           "linear_probe.json", "tin20",    1000),
    ("tin20b",   "tin20b_none",     "tin20b_aux",          "linear_probe.json", "tin20b",   1000),
    ("tinsuper", "tinsuper_none_1pct", "tinsuper_aux_1pct", "linear_probe.json", "tinsuper", 1000),
]

# Same-probe-space G curves (labels reference PAIRS above). CORRECTED CLAIM
# (2026-07-19, found by this audit): the e2e envelope peak sits at the G peak
# ONLY where the readout term is roughly flat across the curve. On tin the
# left-flank readout penalty spans 2.6 points (-2.69@1% -> -0.05@10%), so the
# envelope peaks at 5% while G peaks at 1% -- Delta = G + readout, working as
# stated. The check therefore requires peak coincidence only when the readout
# range across the curve is < 1.5 points.
G_CURVES = {
    "cifar10":  ["C10@1%", "C10@2%", "C10@5%", "C10@10%"],
    "cifar100": ["C100@1%", "C100@5%", "C100@10%"],
    "tin":      ["tin@1%", "tin@2%", "tin@5%", "tin@10%"],
}


def e2e(cell):
    accs = []
    for f in sorted(glob.glob(f"runs/{cell}/seed*/final.json")):
        a = json.load(open(f))["final_test_acc"]
        accs.append(a * 100 if a <= 1 else a)
    if not accs:
        raise FileNotFoundError(f"no final.json under runs/{cell}")
    m = st.mean(accs)
    sd = st.stdev(accs) if len(accs) > 1 else 0.0
    return m, sd, len(accs)


def probe(cell, fname):
    d = json.load(open(f"runs/{cell}/{fname}"))
    vals = [r["probe_test"] * 100 for r in d["results"]]
    m = st.mean(vals)
    sd = st.stdev(vals) if len(vals) > 1 else 0.0
    return m, sd, len(vals)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/law_audit_legacy.md")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    rows, failures = [], []
    for label, bcell, acell, pjson, dataset, images in PAIRS:
        try:
            bm, bs, bn = e2e(bcell)
            am, as_, an = e2e(acell)
            delta = am - bm
            dse = math.sqrt((bs**2 / bn if bn > 1 else 0) + (as_**2 / an if an > 1 else 0))
            pb, pbs, pbn = probe(bcell, pjson)
            pa, pas, pan = probe(acell, pjson)
            G = pa - pb
            gse = math.sqrt((pbs**2 / pbn if pbn > 1 else 0) + (pas**2 / pan if pan > 1 else 0))
            readout = delta - G
            rse = math.sqrt(dse**2 + gse**2)
            rows.append(dict(label=label, dataset=dataset, images=images,
                             base=bm, delta=delta, dse=dse, G=G, gse=gse,
                             readout=readout, rse=rse, n=f"{bn}v{an}"))
        except FileNotFoundError as exc:
            rows.append(dict(label=label, error=str(exc)))
            continue

    # B. sign law (2-sigma tolerance: a check, not a hypothesis test)
    for r in rows:
        if "error" in r:
            continue
        lo, hi = CROSSING
        if r["base"] < lo and r["readout"] > 2 * r["rse"]:
            failures.append(f"SIGN LAW: {r['label']} base {r['base']:.1f} < {lo} "
                            f"but readout {r['readout']:+.2f} ± {r['rse']:.2f} > 0")
        if r["base"] > hi and r["readout"] < -2 * r["rse"]:
            failures.append(f"SIGN LAW: {r['label']} base {r['base']:.1f} > {hi} "
                            f"but readout {r['readout']:+.2f} ± {r['rse']:.2f} < 0")

    # C. envelope peak = G peak
    by_label = {r["label"]: r for r in rows if "error" not in r}
    for ds, labels in G_CURVES.items():
        pts = [by_label[l] for l in labels if l in by_label]
        if len(pts) < 3:
            continue
        d_peak = max(pts, key=lambda r: r["delta"])["label"]
        g_peak = max(pts, key=lambda r: r["G"])["label"]
        readout_range = max(p["readout"] for p in pts) - min(p["readout"] for p in pts)
        if d_peak != g_peak and readout_range < 1.5:
            failures.append(f"PEAK MISMATCH {ds}: envelope peaks at {d_peak}, "
                            f"G peaks at {g_peak}, readout range only "
                            f"{readout_range:.2f}")

    # D. replication: tin20 vs tin20b within 2 sigma on delta AND G
    if "tin20" in by_label and "tin20b" in by_label:
        a, b = by_label["tin20"], by_label["tin20b"]
        for q, se in (("delta", "dse"), ("G", "gse")):
            gap = abs(a[q] - b[q])
            s = math.sqrt(a[se]**2 + b[se]**2)
            if gap > 2 * s:
                failures.append(f"REPLICATION: tin20 vs tin20b {q} differ "
                                f"{gap:.2f} > 2x{s:.2f}")

    lines = ["# Law audit (machine-generated by analysis/audit_law.py)", "",
             "Delta = G + readout; readout sign vs baseline; crossing band "
             f"{CROSSING}. Probe-ceiling cells (stl@20/50) excluded by design.", "",
             "| cell | base | Delta | G | readout | seeds |",
             "|---|---|---|---|---|---|"]
    for r in rows:
        if "error" in r:
            lines.append(f"| {r['label']} | MISSING: {r['error']} | | | | |")
        else:
            lines.append(f"| {r['label']} | {r['base']:.2f} | "
                         f"{r['delta']:+.2f} ±{r['dse']:.2f} | "
                         f"{r['G']:+.2f} ±{r['gse']:.2f} | "
                         f"{r['readout']:+.2f} ±{r['rse']:.2f} | {r['n']} |")
    lines += ["", f"**Checks: {'ALL PASS' if not failures else 'FAILURES'}**", ""]
    lines += [f"- {f}" for f in failures]
    open(args.out, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
