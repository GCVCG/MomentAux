"""Re-derive the work queue from cells that have NO final.json yet.

    OUT=runs python scripts/make_missing_worklist.py [--split turing|bsc|all] > worklist.missing

The counter-queue in bsc_worker.sbatch advances when a task is CLAIMED, with no
retry: if a 5h45 worker claims an expensive cell and hits walltime mid-run, that
(config, seed) is skipped forever. This regenerates the worklist from what is
actually MISSING on disk, so a fresh pass re-attempts exactly the gaps -- the
reconciliation the "no missing cells for any configuration" goal needs.

It reuses make_worklist.py's task generation verbatim (same commands, same
cost-ordering, same flock SSL-pretrain wrapping) and then keeps only the tasks
whose $OUT/<name>/seed<seed>/final.json does not exist. Idempotent: run it, feed
the output, repeat until it emits zero lines -- that is the terminal state.
"""

import argparse
import glob
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_worklist import RATE, SEEDS  # noqa: E402  (reuse the pinned rates/seeds)

# The same dead/live split bsc_worker vs turing_grid use, so --split matches the
# two clusters' worklists exactly. Imported lazily to avoid a hard dependency.
from split_worklist import is_dead  # noqa: E402

TURING_PORTIONS = (50, 100)


def build_tasks(extra_configs=()):
    """Yield (pct, cost, cfg, name, seed, cmd) for every (config, seed).

    `extra_configs` adds individual config paths OUTSIDE configs/grid/ -- used
    for the configs/diagnostics/ cells that were deliberately queued on BSC
    (the vitenv tin ViT/DeiT envelope). Without this the reconcile silently
    DROPPED them every time it regenerated the worklist, because it only
    globbed configs/grid/ (happened twice: 2026-08-02 and 2026-08-03, both
    times taking the two open falsifier blocks out of the queue with it).
    Only explicitly-listed diagnostics configs are included -- configs/
    diagnostics/ holds 368 cells that belong to dedicated waves, not the grid
    queue, so it must never be globbed wholesale.
    """
    paths = sorted(glob.glob("configs/grid/*.yaml")) + [
        p for p in extra_configs if os.path.exists(p)
    ]
    for path in paths:
        cfg = yaml.safe_load(open(path))
        ds, bb = cfg["dataset"], cfg["backbone"]
        pct = cfg.get("subset_pct") or 100
        cost = RATE.get((ds, bb), 80) * pct
        # keep the config's REAL directory (configs/grid or configs/diagnostics);
        # hardcoding configs/grid here would emit unrunnable paths for the
        # explicitly-queued diagnostics cells
        rel = path if path.startswith("configs/") else os.path.join(
            "configs/grid", os.path.basename(path))
        name = cfg["name"]
        init = cfg.get("init_from")
        for seed in SEEDS:
            cmd = ""
            c = cost
            if init:
                pre = init.format(seed=seed)
                ep = " --epochs 50" if "simclr_pre50" in pre else ""
                # the pretrain FAMILY is encoded in the checkpoint path
                script = ("simsiam_pretrain" if "simsiam_pre" in pre
                          else "dino_pretrain" if "dino_pre" in pre
                          else "simclr_pretrain")
                cmd += (
                    'mkdir -p "$(dirname %s)" && '
                    'flock -w 7200 "%s.lock" -c '
                    "'[ -f %s ] || python scripts/%s.py "
                    '--config %s --seed %d --out %s%s --data-root "$DR"\' && '
                    % (pre, pre, pre, script, rel, seed, pre, ep)
                )
                c *= 2
            cmd += ('python train.py --config %s --seed %d '
                    '--data-root "$DR" --out-root "$OUT"' % (rel, seed))
            yield pct, c, cfg, name, seed, cmd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["turing", "bsc", "all"], default="all",
                    help="turing = live cells at 50/100%%; bsc = everything "
                         "else; all = the full grid (default)")
    ap.add_argument("--out", default=os.environ.get("OUT", "runs"),
                    help="run root to check for final.json (default $OUT or runs)")
    ap.add_argument("--extra-configs", default=None,
                    help="file listing config paths outside configs/grid/ (one "
                         "per line) to include -- the diagnostics cells that "
                         "were deliberately queued on BSC. Blank lines and "
                         "'#' comments ignored.")
    args = ap.parse_args()

    extra = []
    if args.extra_configs and os.path.exists(args.extra_configs):
        extra = [ln.strip() for ln in open(args.extra_configs)
                 if ln.strip() and not ln.startswith("#")]

    missing, done = [], 0
    for pct, cost, cfg, name, seed, cmd in build_tasks(extra):
        if args.split == "turing" and not (not is_dead(cfg) and pct in TURING_PORTIONS):
            continue
        if args.split == "bsc" and (not is_dead(cfg) and pct in TURING_PORTIONS):
            continue
        final = os.path.join(args.out, name, "seed%d" % seed, "final.json")
        if os.path.exists(final):
            done += 1
            continue
        missing.append((pct, cost, cmd))

    missing.sort(key=lambda t: (t[0], t[1]))       # cheapest first, as before
    for _, _, cmd in missing:
        print(cmd)
    print("%d missing (%d already done) [split=%s]"
          % (len(missing), done, args.split), file=sys.stderr)


if __name__ == "__main__":
    main()
