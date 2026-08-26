#!/usr/bin/env python
"""Score the cluster-wide checkpoint IDENTITY sweep and derive its repair.

A checkpoint can load perfectly and be the WRONG NETWORK: the duplicate-
execution incidents of 2026-08-02/03/06 put a second trainer into seed dirs
whose original had already finished, overwriting best.pt with mid-run weights
while final.json -- computed in memory at train time -- survived intact.  The
sweep evaluates every probed cell's best.pt and last.pt against its own
recorded accuracy; this script turns the shard reports into

  * the block-I score (I1-I4, F-I1/F-I3, pre-registered 2026-08-25),
  * a REPAIR worklist -- cells whose best.pt fails but whose last.pt passes
    are re-probed from last.pt at zero training cost,
  * the BLAST RADIUS -- every released row whose G was measured against a
    failing arm, which is how three unrelated interventions came to report
    the same spurious +5 on food101.

Usage:
  python analysis/aggregate_identity.py --verify-dir verify [--csv results/all_results.csv]
"""
import argparse, collections, csv, glob, json, os, sys

TOL = 0.5


def load_shards(d, tag):
    rows, files = [], sorted(glob.glob(os.path.join(d, f"{tag}_shard*.json")))
    for f in files:
        try:
            rows += json.load(open(f))["results"]
        except Exception as e:                              # noqa: BLE001
            print(f"  WARNING unreadable shard {f}: {e}", file=sys.stderr)
    return rows, files


def classify(r, tol=TOL):
    """ok | FAIL (wrong network) | CORRUPT | KEYS | ERR (not verifiable)."""
    if "status" in r:
        return r["status"]
    return "ok" if abs(r["diff"]) <= tol else "FAIL"


def by_cell(rows, tol=TOL):
    """cell -> {seed: status}. A cell is bad if ANY seed is."""
    out = collections.defaultdict(dict)
    for r in rows:
        out[r["cell"]][r["seed"]] = classify(r, tol)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify-dir", default="verify")
    ap.add_argument("--csv", default="results/all_results.csv")
    ap.add_argument("--tol", type=float, default=TOL)
    ap.add_argument("--repair-out", default=None,
                    help="write the list of cells to re-probe from last.pt")
    a = ap.parse_args()

    best, bf = load_shards(a.verify_dir, "best")
    last, lf = load_shards(a.verify_dir, "last")
    print(f"shard reports: {len(bf)} best, {len(lf)} last")
    if not best:
        sys.exit("no best.pt results -- nothing to score")

    B, L = by_cell(best, a.tol), by_cell(last, a.tol)
    n_cells = len(B)
    bad_best = {c: s for c, s in B.items()
                if any(v in ("FAIL", "CORRUPT") for v in s.values())}
    err_best = {c for c, s in B.items() if all(v == "ERR" for v in s.values())}
    verifiable = n_cells - len(err_best)

    # ---- I1: the failure rate -------------------------------------------
    rate = 100.0 * len(bad_best) / max(verifiable, 1)
    print("\n" + "=" * 62)
    print(f"cells with a best.pt result        : {n_cells}")
    print(f"  not verifiable (all seeds ERR)   : {len(err_best)}")
    print(f"  VERIFIABLE                       : {verifiable}")
    print(f"  at least one bad best.pt seed    : {len(bad_best)}  ({rate:.2f}%)")
    print("=" * 62)
    print(f"I1  band 2-10%   -> {'HIT' if 2 <= rate <= 10 else 'MISS'}")
    print(f"F-I1 (>25% => corpus unreliable)   -> "
          f"{'FIRED' if rate > 25 else 'dead'}")
    print(f"F-I3 (<0.5% => isolated accidents) -> "
          f"{'FIRED' if rate < 0.5 else 'dead'}")

    # ---- how BIG are the failures? --------------------------------------
    # A wrong-epoch checkpoint misses its record by many points; JPEG decode
    # drift across library versions costs a few tenths (the recorded tin
    # re-evaluation offset was -0.06..-0.38 on six independent cells, and
    # food101/cub/stl are JPEG populations too). Reporting one count for both
    # would let decode noise inflate the damage rate, so split the magnitudes
    # and let the reader see where the tolerance bites.
    diffs = sorted(abs(r["diff"]) for r in best
                   if "status" not in r and abs(r["diff"]) > a.tol)
    if diffs:
        band = collections.Counter()
        for d in diffs:
            band["0.5-1" if d < 1 else "1-2" if d < 2 else
                 "2-5" if d < 5 else "5-15" if d < 15 else ">15"] += 1
        print("\nFAILURE MAGNITUDES (|recorded - evaluated|, accuracy points)")
        for k in ("0.5-1", "1-2", "2-5", "5-15", ">15"):
            if band[k]:
                print(f"  {band[k]:5d}  {k}")
        big = sum(1 for d in diffs if d >= 2.0)
        print(f"  -> {big}/{len(diffs)} seed-level failures are >= 2 points, "
              f"i.e. beyond any plausible decode drift")

    # ---- I2: is the damage concentrated in the grid lane? ---------------
    grid = sum(1 for c in bad_best if c.startswith("grid_")
               or c.startswith("diaggrid_"))
    if bad_best:
        print(f"\nI2  grid-lane share of failures    : {grid}/{len(bad_best)} "
              f"({100*grid/len(bad_best):.0f}%)   band >=60% -> "
              f"{'HIT' if 100*grid/len(bad_best) >= 60 else 'MISS'}")

    # ---- I3: does last.pt survive where best.pt does not? ---------------
    both = [c for c in bad_best if c in L]
    saved = [c for c in both
             if all(v == "ok" for v in L[c].values()) and L[c]]
    if both:
        print(f"I3  last.pt intact where best fails: {len(saved)}/{len(both)} "
              f"({100*len(saved)/len(both):.0f}%)   band >=70% -> "
              f"{'HIT' if 100*len(saved)/len(both) >= 70 else 'MISS'}")

    # ---- the blast radius -----------------------------------------------
    if os.path.exists(a.csv):
        hit = []
        for r in csv.DictReader(open(a.csv)):
            c, b = r.get("cell"), r.get("baseline_cell")
            why = [n for n in (c, b) if n in bad_best]
            if why:
                hit.append((c, b, ",".join(why)))
        print(f"\nBLAST RADIUS: {len(hit)} released rows pair against a "
              f"failing arm")
        seen = collections.Counter(w for _, _, w in hit)
        for w, n in seen.most_common(15):
            print(f"  {n:4d} rows <- {w}")

    # ---- the repair list -------------------------------------------------
    if a.repair_out:
        with open(a.repair_out, "w") as f:
            for c in sorted(saved):
                f.write(c + "\n")
        print(f"\nwrote {len(saved)} repairable cells to {a.repair_out}")
    unrepairable = sorted(set(bad_best) - set(saved))
    if unrepairable:
        print(f"\n{len(unrepairable)} cells fail on BOTH checkpoints "
              f"(retraining, not re-probing):")
        for c in unrepairable[:25]:
            print(f"  {c}")


if __name__ == "__main__":
    main()
