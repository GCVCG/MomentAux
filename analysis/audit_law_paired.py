"""Canonical sign-law audit with seed-paired uncertainty.

Supersedes the independent-SEM audit. Every number the paper reports about
the law is printed by this one command.

WHY PAIRED: readout = Delta - G, and Delta and G are computed from the SAME
checkpoints (the linear evaluation probes the networks whose accuracy gives
Delta). Propagating SEM(readout) as sqrt(SEM(D)^2 + SEM(G)^2) assumes
Cov(D, G) = 0, which is false and positive, so that formula OVERSTATES the
uncertainty (median factor 1.8 here). Forming the readout per seed removes
the assumption:

    readout_s = (aux_acc_s - aux_eval_s) - (base_acc_s - base_eval_s)

Usage:  python analysis/audit_law_paired.py [--runs runs] [--k 2.0]
"""
import argparse, csv, json, math, os, statistics, collections

LO, HI = 31.8, 40.3


def _accs(runs, cell):
    d, out = os.path.join(runs, cell), {}
    if not os.path.isdir(d):
        return out
    for sd in sorted(os.listdir(d)):
        f = os.path.join(d, sd, "final.json")
        if os.path.isfile(f):
            try:
                out[sd] = 100.0 * json.load(open(f))["final_test_acc"]
            except Exception:
                pass
    return out


def _evals(runs, cell):
    f = os.path.join(runs, cell, "linear_probe.json")
    if not os.path.isfile(f):
        return {}
    try:
        p = json.load(open(f))
    except Exception:
        return {}
    return {r["seed"]: 100.0 * r["probe_test"]
            for r in p.get("results", []) if "probe_test" in r}


def in_scope(r):
    """aux-from-scratch: an aux target, no pretrained/SSL init, plain stem.

    THE `pretrained` TEST WAS A BUG AND IT COST 6.6 POINTS OF THE HEADLINE.
    It read `str(...).lower() not in ("true", "1")`, but export_results_csv.py
    writes this column as the STRING "yes" (184 rows) or "". "yes" is in
    neither reject set, so all 91 resolvable-scope ImageNet-TRANSFER TAX cells
    passed the filter and entered the audit -- the exact cells the 2026-07-29
    entry places OUTSIDE the law's derived scope, and which the paper's own
    methods section declares excluded. They enter with large negative Delta
    and G and only 9 of 50 land on the predicted side, so the leak DEPRESSED
    the reported rate: 511 resolvable / 79.1% with the leak, 461 / 85.7%
    without. Every downstream figure moved the same way (below-crossing
    87.7 -> 94.2%, held-out 80.1 -> 88.0%, residual SD 2.15 -> 1.95).
    Test the column truthily, exactly as the `init_from` test beside it
    already did -- that one was correct only because it never compared
    against a literal. Sibling script analysis/audit_sign_law.py always had
    this right, which is why the two disagreed.
    """
    return (r.get("aux_target") and not r.get("init_from")
            and not r.get("pretrained")
            and (r.get("stem") or "none") == "none")


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def load(runs, csv_path, min_seeds=3):
    ca, ce, rows = {}, {}, []
    for r in csv.DictReader(open(csv_path)):
        if not in_scope(r):
            continue
        b = r.get("baseline_cell")
        if not b:
            continue
        try:
            base = float(r["base_acc"])
        except Exception:
            continue
        for c in (r["cell"], b):
            ca.setdefault(c, _accs(runs, c))
            ce.setdefault(c, _evals(runs, c))
        aa, ba, ae, be = ca[r["cell"]], ca[b], ce[r["cell"]], ce[b]
        common = sorted(set(aa) & set(ba) & set(ae) & set(be))
        if len(common) < min_seeds:
            continue
        per = [(aa[s] - ae[s]) - (ba[s] - be[s]) for s in common]
        try:
            ind = math.hypot(float(r["delta_sem"] or 0), float(r["G_sem"] or 0))
        except Exception:
            ind = 0.0
        rows.append(dict(cell=r["cell"], ds=r["dataset"], bb=r["backbone"],
                         pct=r["subset_pct"], base=base,
                         ro=statistics.fmean(per),
                         sem=statistics.stdev(per) / math.sqrt(len(per)),
                         ind_sem=ind, n=len(per)))
    return rows


def audit(rows, k=2.0, lo=LO, hi=HI):
    res = [x for x in rows if x["sem"] > 0 and abs(x["ro"]) > k * x["sem"]
           and not (lo <= x["base"] <= hi)]
    ok = sum(1 for x in res if (x["base"] < lo and x["ro"] < 0)
             or (x["base"] > hi and x["ro"] > 0))
    return len(res), ok, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--csv", default="results/all_results.csv")
    ap.add_argument("--k", type=float, default=2.0)
    a = ap.parse_args()
    rows = load(a.runs, a.csv)

    ratio = sorted(x["sem"] / x["ind_sem"] for x in rows
                   if x["sem"] > 0 and x["ind_sem"] > 0)
    print("=" * 62)
    print(f"law-scope cells with >=3 seed-matched arms : {len(rows)}")
    print(f"SEM(paired)/SEM(independent) median        : "
          f"{ratio[len(ratio)//2]:.3f}  "
          f"(independent overstates in {sum(1 for x in ratio if x<1)/len(ratio):.0%})")
    inb = sum(1 for x in rows if LO <= x["base"] <= HI)
    n, ok, _ = audit(rows, a.k)
    unres = len(rows) - inb - n
    lo_, hi_ = wilson(ok, n)
    print(f"inside crossing bracket (no prediction)    : {inb}")
    print(f"unresolved (|readout| <= {a.k} SEM)            : {unres}")
    print(f"RESOLVABLE (these test the law)            : {n}")
    print(f"  sign as predicted                        : {ok} ({ok/n:.1%})")
    print(f"  wrong side                               : {n-ok}")
    print(f"  Wilson 95% CI                            : "
          f"[{100*lo_:.1f}, {100*hi_:.1f}]")
    print("=" * 62)

    print("\nTHRESHOLD SENSITIVITY")
    for k in (1.0, 1.5, 2.0, 2.5, 3.0):
        n2, ok2, _ = audit(rows, k)
        l, h = wilson(ok2, n2)
        print(f"  >{k:>3.1f} SEM : {ok2:>4}/{n2:<4} = {ok2/n2:5.1%}  "
              f"[{100*l:.1f}, {100*h:.1f}]")

    print("\nBY FLANK")
    for lab, sel in (("below crossing", lambda x: x["base"] < LO),
                     ("above crossing", lambda x: x["base"] > HI)):
        n2, ok2, _ = audit([x for x in rows if sel(x)], a.k)
        l, h = wilson(ok2, n2)
        print(f"  {lab:<15}: {ok2:>4}/{n2:<4} = {ok2/n2:5.1%}  "
              f"[{100*l:.1f}, {100*h:.1f}]")

    _, _, res = audit(rows, a.k)
    cl = collections.defaultdict(list)
    for x in res:
        cl[(x["ds"], x["bb"], x["pct"])].append(
            (x["base"] < LO and x["ro"] < 0) or (x["base"] > HI and x["ro"] > 0))
    maj = [sum(v) / len(v) > 0.5 for v in cl.values()]
    l, h = wilson(sum(maj), len(maj))
    print(f"\nCLUSTERED (one vote per dataset,backbone,fraction)")
    print(f"  {sum(maj)}/{len(maj)} = {sum(maj)/len(maj):.1%}  "
          f"[{100*l:.1f}, {100*h:.1f}]   "
          f"({len(set(x['cell'] for x in res))} cells, {len(cl)} groups)")

    print("\nLEAVE-ONE-DATASET-OUT (bracket re-estimated without that dataset)")
    tot = okt = 0
    for d in sorted(set(x["ds"] for x in rows)):
        fit = [x for x in rows if x["ds"] != d and x["sem"] > 0
               and abs(x["ro"]) > a.k * x["sem"]]
        neg = [x["base"] for x in fit if x["ro"] < 0]
        pos = [x["base"] for x in fit if x["ro"] > 0]
        if len(neg) < 10 or len(pos) < 10:
            continue
        lo2 = statistics.quantiles(neg, n=100)[89]
        hi2 = statistics.quantiles(pos, n=100)[9]
        if lo2 > hi2:
            lo2, hi2 = hi2, lo2
        n2, ok2, _ = audit([x for x in rows if x["ds"] == d], a.k, lo2, hi2)
        if n2 < 5:
            continue
        tot += n2; okt += ok2
        print(f"  {d:<12} [{lo2:5.1f},{hi2:5.1f}]  {ok2:>3}/{n2:<3} = {ok2/n2:5.1%}")
    l, h = wilson(okt, tot)
    print(f"  POOLED HELD-OUT: {okt}/{tot} = {okt/tot:.1%}  "
          f"[{100*l:.1f}, {100*h:.1f}]")

    print("\nWHAT THE READOUT DEPENDS ON (variance explained)")
    ys = [x["ro"] for x in rows]
    mu = statistics.fmean(ys)
    tv = sum((y - mu) ** 2 for y in ys)
    bins = collections.defaultdict(list)
    for x in rows:
        bins[int(x["base"] // 5)].append(x["ro"])
    pred = {b: statistics.fmean(v) if len(v) > 2 else mu for b, v in bins.items()}
    rss = sum((x["ro"] - pred[int(x["base"] // 5)]) ** 2 for x in rows)
    print(f"  baseline accuracy alone (5-point bins) : R^2 = {1 - rss/tv:.3f}")
    resid = {id(x): x["ro"] - pred[int(x["base"] // 5)] for x in rows}
    rm = statistics.fmean(resid.values())
    rt = sum((v - rm) ** 2 for v in resid.values())
    for lab, key in (("dataset", "ds"), ("backbone", "bb"), ("fraction", "pct")):
        g = collections.defaultdict(list)
        for x in rows:
            g[x[key]].append(resid[id(x)])
        bet = sum(len(v) * (statistics.fmean(v) - rm) ** 2 for v in g.values())
        print(f"  + {lab:<9} on the residual            : "
              f"R^2 = {bet/rt:.3f}  ({len(g)} levels)")
    print(f"  residual SD at fixed baseline          : "
          f"{statistics.pstdev(resid.values()):.2f} points")


if __name__ == "__main__":
    main()
