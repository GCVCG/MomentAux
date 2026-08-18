#!/usr/bin/env python3
"""Emit one train_dense.py command per dense (config, seed) that has no
final.json -- i.e. the dense lane's reconcile.

Deliberately derived from the CONFIG DIRECTORY rather than from a stored
worklist: a stored list is exactly what goes stale, and every worklist incident
in this campaign (2026-08-02 counter rewind, 2026-08-03 swap-under-workers,
2026-08-06 phantom 7,479) traces back to trusting one. The filesystem is the
only thing that knows what actually finished.

diag200e_* are EXCLUDED: they are content-identical to their canonical
counterparts and exist only as the already-scored L1/L2 cells, so re-running
them would burn nodes reproducing numbers we have.

Prints nothing and exits 0 when the grid is complete, which the keeper reads as
DENSE_COMPLETE. It exits NON-ZERO on any error rather than printing an empty
list, because "generator failed" and "nothing missing" must not look alike --
that conflation is what silently stopped a lane on 2026-08-01.
"""
import glob
import os
import re
import sys

MS = os.environ.get("MS_ROOT", "${CLUSTER_SCRATCH}/momentstem")
CFG = os.path.join(MS, "repo", "configs", "dense")
RUNS = os.path.join(MS, "runs_dense")
SEEDS = (0, 1, 2)


def main():
    cfgs = sorted(
        os.path.basename(p)[:-5]
        for p in glob.glob(os.path.join(CFG, "*.yaml"))
        if not os.path.basename(p).startswith("diag200e_")
    )
    if not cfgs:
        print("no dense configs found at %s" % CFG, file=sys.stderr)
        return 1

    missing = []
    for c in cfgs:
        for s in SEEDS:
            if not os.path.exists(os.path.join(RUNS, c, "seed%d" % s, "final.json")):
                missing.append((c, s))

    # Longest-job-first. With many slots and few tasks the expensive cells must
    # start first or they become a tail nobody can fill around; the cheap ones
    # slot into the gaps. Both arms of a pair stay adjacent so a partial drain
    # never leaves a one-armed cell.
    def pct(name):
        m = re.search(r"_(\d+)pct$", name)
        return int(m.group(1)) if m else 0

    missing.sort(key=lambda t: (-pct(t[0]), re.sub(r"_(none|aux)_", "_", t[0]), t[0], t[1]))
    for c, s in missing:
        print('python train_dense.py --config configs/dense/%s.yaml --seed %d '
              '--data-root "$DR" --out-root "$OUT"' % (c, s))
    return 0


if __name__ == "__main__":
    sys.exit(main())
