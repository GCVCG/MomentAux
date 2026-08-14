#!/usr/bin/env python3
"""Emit one train_det.py command per detection (config, seed) with no final.json.

Same design as scripts/dense_missing.py, and for the same reason: derived from
the CONFIG DIRECTORY and the filesystem rather than from a stored worklist. A
stored list is what goes stale, and every worklist incident in this campaign
traces back to trusting one. Exits non-zero on error rather than printing an
empty list, so "generator failed" and "nothing missing" cannot look alike.
"""
import glob
import os
import re
import sys

MS = os.environ.get("MS_ROOT", "/gpfs/scratch/ub234/momentstem")
CFG = os.path.join(MS, "repo", "configs", "det")
RUNS = os.path.join(MS, "runs_det")
SEEDS = (0, 1, 2)


def main():
    cfgs = sorted(os.path.basename(p)[:-5] for p in glob.glob(os.path.join(CFG, "*.yaml")))
    if not cfgs:
        print("no detection configs at %s" % CFG, file=sys.stderr)
        return 1
    missing = [(c, s) for c in cfgs for s in SEEDS
               if not os.path.exists(os.path.join(RUNS, c, "seed%d" % s, "final.json"))]
    # Longest-job-first: with more slots than tasks the expensive cells must
    # start first or they become a tail nobody can fill around. Both arms of a
    # pair stay adjacent so a partial drain never leaves a one-armed cell.
    pct = lambda n: int(re.search(r"_(\d+)pct$", n).group(1))
    missing.sort(key=lambda t: (-pct(t[0]), re.sub(r"_(none|aux)_", "_", t[0]), t[0], t[1]))
    for c, s in missing:
        print('python train_det.py --config configs/det/%s.yaml --seed %d '
              '--data-root "$DR" --out-root "$OUT"' % (c, s))
    return 0


if __name__ == "__main__":
    sys.exit(main())
