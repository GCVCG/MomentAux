"""Sign-law robustness to the two 2026-08-25/26 measurement defects.

Both defects damage the FEATURE side, which is exactly where the law's G term
comes from, so "does the law survive them?" is the first question a referee
will ask once they are disclosed. This answers it by re-running the canonical
audit under each filter and both together:

  identity   drop every cell whose own arm or whose baseline has a checkpoint
             that fails to reproduce its own recorded accuracy
  epoch      drop every cell whose two arms differ by more than 0.25 points in
             best-minus-final, so the Delta/G epoch mismatch cannot bias it

Usage:  python analysis/law_robustness.py [--verify-dir DIR]
The verify directory holds the identity sweep's best_shard*.json reports.
"""
import sys, os, csv, json, glob
sys.path.insert(0, "analysis")
import audit_law_paired as A

bad = set()
VDIR = (sys.argv[sys.argv.index("--verify-dir") + 1]
        if "--verify-dir" in sys.argv else "verify")
for f in glob.glob(os.path.join(VDIR, "best_shard*.json")):
    for r in json.load(open(f))["results"]:
        st = r.get("status") or ("ok" if abs(r["diff"]) <= 0.5 else "FAIL")
        if st in ("FAIL", "CORRUPT"):
            bad.add(r["cell"])
base_of = {r["cell"]: r.get("baseline_cell")
           for r in csv.DictReader(open("results/all_results.csv"))}
rows = A.load("runs", "results/all_results.csv")
suspect = lambda x: x["cell"] in bad or base_of.get(x["cell"]) in bad

for lab, sel in (("ALL cells", rows),
                 ("excluding identity-suspect", [x for x in rows if not suspect(x)])):
    n, ok, res = A.audit(sel, 2.0)
    lo = [x for x in res if x["base"] < A.LO]
    hi = [x for x in res if x["base"] > A.HI]
    okl = sum(x["ro"] < 0 for x in lo); okh = sum(x["ro"] > 0 for x in hi)
    print(f"{lab:<28s} resolvable {n:4d}  correct {ok:4d} = {ok/n:5.1%}   "
          f"below {okl}/{len(lo)} = {okl/max(len(lo),1):5.1%}   "
          f"above {okh}/{len(hi)} = {okh/max(len(hi),1):5.1%}")

strict = [x for x in rows if not suspect(x) and x["egap"] <= 0.25]
n, ok, res = A.audit(strict, 2.0)
lo = [x for x in res if x["base"] < A.LO]; hi = [x for x in res if x["base"] > A.HI]
okl = sum(x["ro"] < 0 for x in lo); okh = sum(x["ro"] > 0 for x in hi)
print(f"{'BOTH filters (strictest)':<28s} resolvable {n:4d}  correct {ok:4d} = "
      f"{ok/n:5.1%}   below {okl}/{len(lo)} = {okl/max(len(lo),1):5.1%}   "
      f"above {okh}/{len(hi)} = {okh/max(len(hi),1):5.1%}")
print(f"  cells kept: {len(strict)} of {len(rows)}")
