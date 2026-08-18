#!/usr/bin/env python3
"""Verify deltas typed into LaTeX tables against the exported run records.

WHY THIS EXISTS. Recurring figures in the prose stopped drifting when they moved
into generated macros (paper/tables/numbers.tex). Table BODIES were never given
the same treatment: a table of forty deltas is typed by hand, and nothing checked
it. A referee found one inconsistency in the budget table by subtracting two
printed columns and noticing the result did not match a third table. Auditing the
whole table against results/all_results.csv turned that one flag into SIX wrong
entries, every one of them in the same column and every one off by exactly 0.02 --
the signature of a column written before a baseline was deepened to ten seeds and
not rewritten afterwards. The appendix table the referee suspected was correct.

WHAT IT CHECKS. Each TABLES entry declares where a row lives, the cell-name
pattern per data fraction, and which quantity is printed. The script reads the
delta the exporter computed (seed-paired, per Section 3.5) and compares it to the
number in the .tex. It does NOT re-derive deltas by subtracting means: doing that
is what produced the wrong values in the first place, because a paired delta over
matched seeds is not the difference of two unmatched means.

    python scripts/check_table_numbers.py            # report
    python scripts/check_table_numbers.py --strict   # non-zero exit on mismatch

Add a table here whenever one is typed rather than generated.
"""
import argparse
import csv
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results", "all_results.csv")

PCTS = ["1", "2", "3", "5", "7", "10", "15", "25"]

# file, row label as printed, cell pattern ({p} = fraction), fractions, quantity.
# "delta" is the exporter's seed-paired gain over the row's own baseline;
# "diff:A|B" is the difference of two families' deltas at the same fraction.
TABLES = [
    ("sections/s3b_budget.tex", r"\textbf{Prior (this work)}",
     "auxmag_{p}pct_sched0", PCTS, "delta"),
    ("sections/s3b_budget.tex", "SimSiam, 200 ep",
     "diaggrid_simsiam_c100_{p}pct", PCTS, "delta"),
    ("sections/s3b_budget.tex", "SimCLR, 200 ep",
     "diagssl_simclr_{p}pct", PCTS, "delta"),
    ("sections/s3b_budget.tex", "SimSiam, 800 ep",
     "diagsslbudget_simsiam800_c100_{p}pct", PCTS, "delta"),
    ("sections/s3b_budget.tex", "SimCLR, 800 ep",
     "diagsslbudget_simclr800_c100_{p}pct", PCTS, "delta"),
    ("sections/s3b_budget.tex", r"Prior $-$ SimSiam-800", None, PCTS,
     "diff:auxmag_{p}pct_sched0|diagsslbudget_simsiam800_c100_{p}pct"),
    ("sections/s3b_budget.tex", r"Prior $-$ SimCLR-800", None, PCTS,
     "diff:auxmag_{p}pct_sched0|diagsslbudget_simclr800_c100_{p}pct"),
]


def load():
    out = {}
    with open(RESULTS) as f:
        for r in csv.DictReader(f):
            out[r["cell"]] = r
    return out


def delta(rows, cell):
    r = rows.get(cell)
    if r is None or not r.get("delta"):
        return None
    return float(r["delta"])


def printed(path, label):
    """Every numeric field of every row whose first cell is `label`."""
    vals = []
    for line in open(path):
        head = line.split("&")[0].strip()
        if head != label.strip():
            continue
        for cellstr in line.split("&")[1:]:
            cellstr = cellstr.replace(r"\\", "")
            m = re.search(r"[-+−]?\d+\.\d+", cellstr.replace(r"\times", ""))
            if m and "times" not in cellstr:
                vals.append(float(m.group(0).replace("−", "-")))
    return vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()

    rows = load()
    bad = 0
    checked = 0
    for rel, label, pat, pcts, what in TABLES:
        path = os.path.join(ROOT, "paper", rel)
        vals = printed(path, label)
        want = []
        for p in pcts:
            if what == "delta":
                want.append(delta(rows, pat.format(p=p)))
            else:
                a, b = what.split(":", 1)[1].split("|")
                da, db = delta(rows, a.format(p=p)), delta(rows, b.format(p=p))
                want.append(None if da is None or db is None else da - db)
        if len(vals) != len(want):
            print(f"  SKIP  {label}: printed {len(vals)} values, expected {len(want)}")
            continue
        for p, got, exp in zip(pcts, vals, want):
            if exp is None:
                print(f"  ?     {label} @{p}%: no record for that cell")
                continue
            checked += 1
            if abs(got - exp) > 0.005:
                bad += 1
                print(f"  FAIL  {label} @{p}%: table {got:+.2f}, records {exp:+.2f}")
    print(f"\n{checked} typed table values checked against {os.path.relpath(RESULTS, ROOT)}"
          f", {bad} mismatched")
    if bad and args.strict:
        sys.exit(1)


if __name__ == "__main__":
    main()
