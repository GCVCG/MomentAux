"""Build the last.pt re-probe worklist from the identity sweep's own report.

Only cells whose last.pt VERIFIED on every seed are probed. Probing a
checkpoint before verifying it is exactly how the food101 exception cluster
came to exist, so the filter is the point of this script, not a detail.
"""
import json, glob, os, sys
MS = os.environ.get("MS") or sys.exit("set MS to the scratch root")
rep = sorted(glob.glob(os.path.join(MS, "verify", "last_shard*.json")))
if not rep:
    sys.exit("no verify report found -- run the identity sweep first")
ok, bad = {}, {}
for p in rep:
    for r in json.load(open(p))["results"]:
        c = r["cell"]
        good = ("status" not in r) and abs(r["diff"]) <= 0.5
        (ok if good else bad).setdefault(c, []).append(r["seed"])
clean = sorted(c for c in ok if c not in bad)
skipped = sorted(bad)
CFG_DIRS = ["configs/grid", "configs/diagnostics", "configs/ablations_full",
            "configs/sensorfusion", "configs"]
def cfg_of(cell):
    for d in CFG_DIRS:
        p = os.path.join(MS, "repo", d, cell + ".yaml")
        if os.path.exists(p):
            return os.path.join(d, cell + ".yaml")
    return None
lines, nocfg = [], []
for c in clean:
    cf = cfg_of(c)
    if not cf:
        nocfg.append(c); continue
    # $DR and $OUT are set by the lane, so tin cells read the /dev/shm
    # staging the worker built instead of hammering GPFS with 110k JPEGs.
    lines.append('python analysis/linear_probe.py --run "$OUT/%s" --config %s '
                 '--data-root "$DR" --ckpt last.pt --out-suffix _last' % (c, cf))
with open(os.path.join(MS, "worklist.reprobe"), "w") as f:
    f.write("\n".join(lines) + "\n")
print("clean cells: %d   skipped (last.pt failed identity): %d   no config: %d"
      % (len(clean), len(skipped), len(nocfg)))
print("worklist.reprobe lines:", len(lines))
if skipped:
    print("SKIPPED:", ", ".join(skipped[:20]), "..." if len(skipped) > 20 else "")
