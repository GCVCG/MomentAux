"""Scope-wide sign-law audit over every paired cell that has G and readout.

This is the paper's central evidence, so it is a COMMITTED script rather than
an ad-hoc query (2026-08-05: the first pass at this was ad-hoc and twice
produced a wrong headline number before being caught -- see below).

    python analysis/audit_sign_law.py            # law scope only
    python analysis/audit_sign_law.py --scope all

THE LAW: Delta = G + readout, and readout's SIGN is a function of BASELINE
HEIGHT, crossing zero somewhere in the bracket [31.8, 40.3] -- negative below,
positive above.

TWO METHODOLOGICAL POINTS, both learned the hard way, both enforced here:

(1) COUNT ONLY RESOLVABLE CELLS. A naive sign count over all cells gives ~52%
    above the crossing and looks like a coin flip. That is not a failure of the
    law, it is a failure of the count: above the crossing the law PREDICTS
    readout ~ +0.4 decaying to ~0, and at high fractions Delta and G are both
    ~0, so readout is a small difference of small numbers whose sign is pure
    noise. A cell only tests the law if |readout| > 2*SEM, with SEM propagated
    from BOTH e2e arms and BOTH probe arms.

(2) RESPECT THE DERIVED SCOPE. The law was derived on aux-from-scratch cells.
    `pretrained: true` (the transfer tax) is explicitly placed outside it by
    the 2026-07-29 entry, and SSL-init cells are non-aux interventions scored
    separately. Mixing them in drops the figure from 96% to 86%. Excluding
    them is applying the definition, not cherry-picking -- so --scope all
    prints the wider number too, and both belong in any writeup.

Cells INSIDE the crossing bracket make no sign prediction and are reported
separately rather than silently counted as passes.
"""

import argparse
import csv
import math
import os
import sys

CROSS_LO, CROSS_HI = 31.8, 40.3
CSV = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results", "all_results.csv")


def fnum(row, key):
    v = row.get(key, "")
    if v in ("", None):
        return None
    try:
        return float(v)
    except ValueError:
        return None


def in_law_scope(r):
    """aux-from-scratch: an aux target, no pretrained/SSL init, plain stem."""
    if not r.get("aux_target"):
        return False
    if r.get("init_from"):            # SSL init: SimCLR / SimSiam / DINO
        return False
    if r.get("pretrained"):           # ImageNet-transfer TAX cells; the
        return False                  # 2026-07-29 entry puts these outside
                                      # the law's derived scope explicitly.
    if (r.get("stem") or "none") != "none":
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", choices=("law", "all"), default="law")
    ap.add_argument("--min-seeds", type=int, default=3)
    ap.add_argument("--csv", default=CSV)
    args = ap.parse_args()

    if not os.path.exists(args.csv):
        sys.exit(f"missing {args.csv} -- run analysis/export_results_csv.py first")

    cells = []
    with open(args.csv) as fh:
        for r in csv.DictReader(fh):
            d, g = fnum(r, "delta"), fnum(r, "G")
            ds, gs = fnum(r, "delta_sem"), fnum(r, "G_sem")
            base = fnum(r, "base_acc")
            ns, nps = fnum(r, "n_seeds"), fnum(r, "n_probe_seeds")
            if None in (d, g, base) or ns is None or nps is None:
                continue
            if ns < args.min_seeds or nps < args.min_seeds:
                continue
            if args.scope == "law" and not in_law_scope(r):
                continue
            sem = math.hypot(ds or 0.0, gs or 0.0)
            cells.append({
                "cell": r["cell"], "ds": r["dataset"], "bb": r["backbone"],
                "base": base, "delta": d, "G": g, "readout": d - g, "sem": sem,
            })

    if not cells:
        sys.exit("no cells matched")

    resolvable, right, wrong, inband, unresolved = [], 0, [], 0, 0
    for c in cells:
        if CROSS_LO <= c["base"] <= CROSS_HI:
            inband += 1
            continue
        if c["sem"] <= 0 or abs(c["readout"]) <= 2 * c["sem"]:
            unresolved += 1
            continue
        want_neg = c["base"] < CROSS_LO
        good = (c["readout"] < 0) if want_neg else (c["readout"] > 0)
        resolvable.append(c)
        if good:
            right += 1
        else:
            wrong.append(c)

    bbs, dss = {}, {}
    for c in cells:
        bbs[c["bb"]] = bbs.get(c["bb"], 0) + 1
        dss[c["ds"]] = dss.get(c["ds"], 0) + 1

    print(f"# Sign-law audit (scope={args.scope}, >={args.min_seeds} seeds/arm)")
    print(f"crossing bracket [{CROSS_LO}, {CROSS_HI}]; resolvable = |readout| > 2 SEM\n")
    print(f"cells with Delta+G:            {len(cells)}")
    print(f"  backbones ({len(bbs)}): " +
          ", ".join(f"{k} {v}" for k, v in sorted(bbs.items(), key=lambda x: -x[1])))
    print(f"  datasets  ({len(dss)}): " +
          ", ".join(f"{k} {v}" for k, v in sorted(dss.items(), key=lambda x: -x[1])))
    print(f"\ninside crossing bracket (no prediction): {inband}")
    print(f"unresolved (|readout| <= 2 SEM):         {unresolved}")
    print(f"RESOLVABLE (these test the law):         {len(resolvable)}")
    if resolvable:
        print(f"  sign as predicted: {right}  ({100.0*right/len(resolvable):.0f}%)")
        print(f"  wrong side:        {len(wrong)}")
    if wrong:
        print("\n## exceptions")
        print("| cell | dataset | backbone | base | Delta | G | readout | SEM |")
        print("|---|---|---|---|---|---|---|---|")
        for c in sorted(wrong, key=lambda x: -abs(x["readout"])):
            print(f"| {c['cell']} | {c['ds']} | {c['bb']} | {c['base']:.1f} | "
                  f"{c['delta']:+.2f} | {c['G']:+.2f} | {c['readout']:+.2f} | "
                  f"{c['sem']:.2f} |")


if __name__ == "__main__":
    main()
