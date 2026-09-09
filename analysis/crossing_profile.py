"""Is the readout crossing a THRESHOLD or a TRANSITION?

The audit reports one number (86.4% of resolvable cells) and one split (below
94.9% / above 67.4%). Neither says where the law has CONTENT. If the crossing
were a sharp threshold the below-crossing error rate would be flat in baseline
height; if it is a smooth transition the rate should rise, and the mean
readout decay toward zero, as cells approach the bracket -- in which case the
"errors" near it are cells whose true readout is near zero rather than law
failures.

Run:  python analysis/crossing_profile.py
"""
import sys, argparse, collections, statistics as st
sys.path.insert(0, "analysis")
import audit_law_paired as A

# THIS SCRIPT HAD NO ARGPARSE, so a --csv/--probe-file passed on the command
# line was SILENTLY IGNORED and it printed the released best.pt numbers over a
# matched-epoch corpus with no error (2026-08-31). Same family as the audit
# bypass: a repair applied at one layer that a second layer ignores, except
# worse -- argparse at least rejects an unrecognized flag.
_ap = argparse.ArgumentParser()
_ap.add_argument("--runs", default="runs")
_ap.add_argument("--csv", default="results/all_results.csv")
_ap.add_argument("--probe-file", default="linear_probe_last.json",
                 help="probe json per cell; linear_probe_last.json for the "
                      "matched-epoch corpus (mirrors the audit and exporter)")
_a = _ap.parse_args()
print(f"# corpus: {_a.csv}   probe: {_a.probe_file}")
rows = A.load(_a.runs, _a.csv, probe_file=_a.probe_file)
n, ok, res = A.audit(rows, 2.0)
lo = [x for x in res if x["base"] < A.LO]
bins = [(0, 5), (5, 10), (10, 15), (15, 20), (20, 25), (25, 31.8)]
print(f"{'baseline band':>16s} {'n':>4s} {'positive (WRONG)':>18s}  mean readout")
for a, b in bins:
    v = [x for x in lo if a <= x["base"] < b]
    if not v: continue
    w = sum(x["ro"] > 0 for x in v)
    print(f"{f'[{a}, {b})':>16s} {len(v):>4d} {w:>8d} = {100*w/len(v):5.1f}%    "
          f"{st.fmean(x['ro'] for x in v):+6.2f}")
hi = [x for x in res if x["base"] > A.HI]
print()
bins2 = [(40.3, 50), (50, 60), (60, 75), (75, 90), (90, 101)]
print(f"{'baseline band':>16s} {'n':>4s} {'negative (WRONG)':>18s}  mean readout")
for a, b in bins2:
    v = [x for x in hi if a <= x["base"] < b]
    if not v: continue
    w = sum(x["ro"] < 0 for x in v)
    print(f"{f'[{a}, {b})':>16s} {len(v):>4d} {w:>8d} = {100*w/len(v):5.1f}%    "
          f"{st.fmean(x['ro'] for x in v):+6.2f}")

print("\nTHE LAW STATED WHERE IT HAS CONTENT")
def wil(k, m):
    import math
    if m == 0: return (0, 0)
    z = 1.96; p = k / m; d = 1 + z * z / m
    c = (p + z * z / (2 * m)) / d
    h = z * math.sqrt(p * (1 - p) / m + z * z / (4 * m * m)) / d
    return (100 * max(0, c - h), 100 * min(1, c + h))
groups = [("far below  (base < 20)", [x for x in lo if x["base"] < 20], -1),
          ("near below (20-31.8)  ", [x for x in lo if x["base"] >= 20], -1),
          ("above the bracket     ", hi, +1)]
for lab, v, sgn in groups:
    okk = sum((x["ro"] < 0) if sgn < 0 else (x["ro"] > 0) for x in v)
    a, b = wil(okk, len(v))
    print(f"  {lab} {okk:>3d}/{len(v):<3d} = {100*okk/len(v):5.1f}%  "
          f"[{a:.1f}, {b:.1f}]   mean |readout| "
          f"{st.fmean(abs(x['ro']) for x in v):.2f}")
