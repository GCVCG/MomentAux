"""Seed-paired sign-law audit.

Referee point (2026-08-07): Eq. 4 propagates SEM(readout) as
sqrt(SEM(D)^2 + SEM(G)^2), which assumes Cov(D, G) = 0. It is not: the
linear evaluation probes the SAME checkpoints whose end-to-end accuracy
gives D, so across seeds the two move together. The independent formula
therefore OVERSTATES the uncertainty, the 2-SEM bar sits too high, and the
surviving cells are enriched for large |readout|.

This script removes the assumption entirely by forming the readout per
seed and taking the standard error of that quantity directly:

    readout_s = (aux_acc_s - base_acc_s) - (aux_probe_s - base_probe_s)
              = (aux_acc_s - aux_probe_s) - (base_acc_s - base_probe_s)

Seeds are matched by index across all four arms; only seeds present in all
four contribute. Everything else (scope filter, crossing bracket, sign
rule) is identical to analysis/audit_sign_law.py.

Usage:  python analysis/audit_paired_readout.py [--runs runs] [--k 2.0]
"""
import argparse, csv, json, math, os, statistics, collections

LO, HI = 31.8, 40.3


def seed_accs(runs, cell):
    out = {}
    d = os.path.join(runs, cell)
    if not os.path.isdir(d):
        return out
    for sd in sorted(os.listdir(d)):
        f = os.path.join(d, sd, "final.json")
        if os.path.isfile(f):
            try:
                out[sd] = 100.0 * json.load(open(f))["final_test_acc"]
            except (KeyError, ValueError, json.JSONDecodeError):
                pass
    return out


def seed_probes(runs, cell):
    f = os.path.join(runs, cell, "linear_probe.json")
    if not os.path.isfile(f):
        return {}
    try:
        payload = json.load(open(f))
    except json.JSONDecodeError:
        return {}
    return {r["seed"]: 100.0 * r["probe_test"]
            for r in payload.get("results", []) if "probe_test" in r}


def in_scope(r):
    if not r.get("aux_target"):
        return False
    if r.get("init_from"):
        return False
    if str(r.get("pretrained", "")).lower() in ("true", "1"):
        return False
    return (r.get("stem") or "none") == "none"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--csv", default="results/all_results.csv")
    ap.add_argument("--k", type=float, default=2.0)
    ap.add_argument("--min-seeds", type=int, default=3)
    a = ap.parse_args()

    cells = [r for r in csv.DictReader(open(a.csv)) if in_scope(r)]
    cache_a, cache_p = {}, {}

    def acc(c):
        if c not in cache_a:
            cache_a[c] = seed_accs(a.runs, c)
        return cache_a[c]

    def prb(c):
        if c not in cache_p:
            cache_p[c] = seed_probes(a.runs, c)
        return cache_p[c]

    rows, skipped = [], collections.Counter()
    for r in cells:
        base = r.get("baseline_cell")
        if not base:
            skipped["no baseline"] += 1
            continue
        aa, ba, ap_, bp = acc(r["cell"]), acc(base), prb(r["cell"]), prb(base)
        common = sorted(set(aa) & set(ba) & set(ap_) & set(bp))
        if len(common) < a.min_seeds:
            skipped["<%d paired seeds" % a.min_seeds] += 1
            continue
        per = [(aa[s] - ap_[s]) - (ba[s] - bp[s]) for s in common]
        ro = statistics.fmean(per)
        sem = statistics.stdev(per) / math.sqrt(len(per)) if len(per) > 1 else 0.0
        try:
            b = float(r["base_acc"])
        except (ValueError, KeyError):
            skipped["no base_acc"] += 1
            continue
        try:
            ind = math.hypot(float(r["delta_sem"] or 0), float(r["G_sem"] or 0))
        except ValueError:
            ind = float("nan")
        rows.append(dict(cell=r["cell"], ds=r["dataset"], bb=r["backbone"],
                         pct=r["subset_pct"], base=b, ro=ro, sem=sem,
                         ind_sem=ind, n=len(per)))

    print(f"cells in scope with >={a.min_seeds} seed-matched arms: {len(rows)}")
    for k, v in skipped.items():
        print(f"  skipped, {k}: {v}")

    paired = [x["sem"] for x in rows if x["sem"] > 0]
    indep = [x["ind_sem"] for x in rows
             if x["sem"] > 0 and x["ind_sem"] == x["ind_sem"] and x["ind_sem"] > 0]
    ratio = [x["sem"] / x["ind_sem"] for x in rows
             if x["sem"] > 0 and x["ind_sem"] == x["ind_sem"] and x["ind_sem"] > 0]
    if ratio:
        ratio.sort()
        print(f"\nSEM(paired) / SEM(independent):  median {ratio[len(ratio)//2]:.3f}, "
              f"mean {statistics.fmean(ratio):.3f}, "
              f"fraction < 1 (independent overstates): "
              f"{sum(1 for x in ratio if x < 1)/len(ratio):.1%}")

    def audit(k, sel=None):
        R = [x for x in rows if sel is None or sel(x)]
        res = [x for x in R if x["sem"] > 0 and abs(x["ro"]) > k * x["sem"]
               and not (LO <= x["base"] <= HI)]
        ok = sum(1 for x in res
                 if (x["base"] < LO and x["ro"] < 0) or (x["base"] > HI and x["ro"] > 0))
        return len(res), ok

    print(f"\n{'k*SEM':>6} {'resolvable':>11} {'correct':>8} {'rate':>8}")
    for k in (1.0, 1.5, 2.0, 2.5, 3.0):
        n, ok = audit(k)
        print(f"{k:>6.1f} {n:>11} {ok:>8} {ok/n:>7.1%}" if n else f"{k:>6.1f} {0:>11}")

    n, ok = audit(a.k)
    below = [x for x in rows if x["sem"] > 0 and abs(x["ro"]) > a.k * x["sem"]
             and x["base"] < LO]
    above = [x for x in rows if x["sem"] > 0 and abs(x["ro"]) > a.k * x["sem"]
             and x["base"] > HI]
    print(f"\nat k={a.k}:  {ok}/{n} = {ok/n:.1%}")
    print(f"  below crossing: {sum(1 for x in below if x['ro']<0)}/{len(below)}")
    print(f"  above crossing: {sum(1 for x in above if x['ro']>0)}/{len(above)}")


if __name__ == "__main__":
    main()
