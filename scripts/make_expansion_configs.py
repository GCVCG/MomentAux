"""Expand every NEW test family across ALL datasets and ALL valid portions.

    python scripts/make_expansion_configs.py            # writes configs + task lists
    python scripts/make_expansion_configs.py --dry-run

2026-07-23, user: "make sure the test include all datasets and their portions".
The modern-architecture and comparator families were initially seeded at
decisive fractions on 1-2 datasets; this makes them full-matrix, same as the
champion grid. The reconcile loop owns everything emitted here (configs land
in configs/grid/).

FAMILIES x the 9 core populations (cifar100, cifar10, stl10, tin, cub +
eurosat, dtd, pathmnist, food101):
  vit_pair, swin_pair, deit_pair (ViT + DeiT aug), mnet_pair   [architectures]
  simclr, simsiam, dino (ViT), transfer_pair                   [comparators]
  mag3, mag6o                                                  [re-pin arms]
The derived-control datasets (cifar100super, tin20/b, tinsuper, tinsem) are
EXCLUDED deliberately: they exist to isolate label-space effects on fixed
pixels for r18 mechanism questions; re-running them under every architecture
re-measures the same pixels for no new claim.

Validity: portion must yield >= 1 batch of 128 (drop_last floor). Existing
configs/cells (any name, matched by config_key) are skipped. Tasks whose cost
estimate exceeds BIG_H hours go to the big lane.

All three SSL pretrain scripts accept the SSL cell's OWN config as --config
(they read dataset/subset/backbone/recipe and ignore init_from), so no
parent-config lookup is needed.
"""

import argparse
import glob
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "analysis"))

from export_results_csv import config_key, load_cells

DATASETS = ("cifar100", "cifar10", "stl10", "tin", "cub",
            "eurosat", "dtd", "pathmnist", "food101")
SHORT = {"cifar100": "c100", "cifar10": "c10", "stl10": "stl", "tin": "tin",
         "cub": "cub", "eurosat": "esat", "dtd": "dtd", "pathmnist": "path",
         "food101": "food"}
TRAIN_SIZE = {"cifar100": 50000, "cifar10": 50000, "stl10": 5000,
              "tin": 100000, "cub": 5994, "eurosat": 21600, "dtd": 3760,
              "pathmnist": 89996, "food101": 75750}
PCTS = (1, 2, 3, 5, 7, 10, 15, 20, 25, 50, 100)
BATCH = 128
SEEDS = (0, 1, 2)
# rough sec per 1%% of data for an r18 SGD cell (H100), from make_worklist
RATE = {"cifar100": 80, "cifar10": 80, "stl10": 142, "tin": 115, "cub": 50,
        "eurosat": 35, "dtd": 6, "pathmnist": 145, "food101": 122}
BIG_H = 4.5   # tasks estimated above this many hours go to the big lane

SGD = """name: {name}
dataset: {ds}
backbone: {bb}
epochs: 200
batch_size: 128
lr: 0.1
weight_decay: 0.0005
momentum: 0.9
num_workers: 2
small_input: true
pretrained: {pre}
subset_pct: {pct}
stem: none
stem_calibrate: true
"""
ADAMW = """name: {name}
dataset: {ds}
backbone: {bb}
optimizer: adamw
epochs: 200
batch_size: 128
lr: 0.001
weight_decay: 0.05
momentum: 0.9
num_workers: 2
small_input: true
pretrained: false
subset_pct: {pct}
stem: none
stem_calibrate: true
"""
AUX = """moment_aux:
  stem: energy-magnitude
  tap: {tap}
  weight: 1.0
  weight_final: 0.0
  weight_schedule: cosine
  head_norm: true
"""
MAG = """moment_aux:
  stem: energy-magnitude{suffix}
  tap: layer3
  weight: 1.0
  weight_final: 0.0
  weight_schedule: cosine
  head_norm: true
"""


def family_cells(ds, pct):
    """Yield (name, yaml_body, arch_mult, pretrain_script_or_None)."""
    s = SHORT[ds]
    # --- architecture pairs ---
    for kind in ("none", "aux"):
        a = AUX.format(tap="blocks.8") if kind == "aux" else ""
        yield (f"diaggrid_vit_{s}_{kind}_{pct}pct",
               ADAMW.format(name=f"diaggrid_vit_{s}_{kind}_{pct}pct", ds=ds,
                            bb="vit_tiny", pct=pct) + a, 1.0, None)
        yield (f"diaggrid_deit_{s}_{kind}_{pct}pct",
               ADAMW.format(name=f"diaggrid_deit_{s}_{kind}_{pct}pct", ds=ds,
                            bb="vit_tiny", pct=pct) + "augment: deit\n" + a,
               1.2, None)
        aw = AUX.format(tap="layers.2") if kind == "aux" else ""
        yield (f"diaggrid_swin_{s}_{kind}_{pct}pct",
               ADAMW.format(name=f"diaggrid_swin_{s}_{kind}_{pct}pct", ds=ds,
                            bb="swin_tiny", pct=pct) + aw, 1.3, None)
        am = AUX.format(tap="blocks.3") if kind == "aux" else ""
        yield (f"grid_mnet_{s}_{kind}_{pct}pct",
               SGD.format(name=f"grid_mnet_{s}_{kind}_{pct}pct", ds=ds,
                          bb="mobilenetv3_small_100", pct=pct, pre="false") + am,
               0.5, None)
        # transfer pair (r18, ImageNet init -- diag: outside images)
        at = AUX.format(tap="layer3") if kind == "aux" else ""
        yield (f"diagtransfer2_{s}_{kind}_{pct}pct",
               SGD.format(name=f"diagtransfer2_{s}_{kind}_{pct}pct", ds=ds,
                          bb="resnet18", pct=pct, pre="true").replace(
                   "name:", "# diag: ImageNet init breaks the data contract\nname:")
               + at, 1.0, None)
        # re-pin arms (aux only makes sense)
        if kind == "aux":
            for suffix, tag, mult in (("3", "mag3", 1.0), ("6o", "mag6o", 1.0)):
                body = SGD.format(name=f"grid_{tag}_{s}_{pct}pct", ds=ds,
                                  bb="resnet18", pct=pct, pre="false")
                if suffix == "3":
                    body += MAG.format(suffix="3").replace(
                        "tap: layer3", "kernel_size: 17\n  tap: layer3")
                else:
                    body += MAG.format(suffix="6o")
                yield (f"grid_{tag}_{s}_{pct}pct", body, mult, None)
    # --- SSL families (cell config doubles as pretrain parent) ---
    for fam, script, bb, tmpl, mult in (
            ("simclr", "simclr_pretrain", "resnet18", SGD, 2.0),
            ("simsiam", "simsiam_pretrain", "resnet18", SGD, 2.0),
            ("dino", "dino_pretrain", "vit_tiny", ADAMW, 3.0)):
        name = f"diaggrid_{fam}_{s}_{pct}pct"
        if tmpl is SGD:
            body = tmpl.format(name=name, ds=ds, bb=bb, pct=pct, pre="false")
        else:
            body = tmpl.format(name=name, ds=ds, bb=bb, pct=pct)
        body += f"init_from: runs/{fam}_pre_{s}_{pct}pct/seed{{seed}}/pretrain.pt\n"
        yield (name, body, mult, script)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out-normal", default="expansion.tasks")
    ap.add_argument("--out-big", default="expansion.big")
    args = ap.parse_args()

    # config_key EXCLUDES subset_pct by design (it is the grid family key),
    # so coverage must be keyed on (family, pct) or one covered portion would
    # mark the whole family covered.
    def ckey(c):
        return (config_key(c), c.get("subset_pct") or 100)
    cells = load_cells(["runs", "runs_turing"])
    covered = set()
    for rec in cells.values():
        covered.add(ckey(rec["cfg"]))
    for f in glob.glob("configs/**/*.yaml", recursive=True):
        try:
            c = yaml.safe_load(open(f))
        except Exception:
            continue
        if isinstance(c, dict) and "dataset" in c and "backbone" in c:
            covered.add(ckey(c))

    written, skipped, normal, big = 0, 0, [], []
    for ds in DATASETS:
        for pct in PCTS:
            if int(round(TRAIN_SIZE[ds] * pct / 100.0)) < BATCH:
                continue
            for name, body, mult, script in family_cells(ds, pct):
                cfg = yaml.safe_load(body)
                if ckey(cfg) in covered:
                    skipped += 1
                    continue
                covered.add(ckey(cfg))
                path = f"configs/grid/{name}.yaml"
                if not args.dry_run:
                    open(path, "w").write(body)
                written += 1
                cost_h = RATE[ds] * pct * mult * (2 if script else 1) / 3600.0
                rel = "configs/grid/" + name + ".yaml"
                for seed in SEEDS:
                    if script:
                        pre = cfg["init_from"].format(seed=seed)
                        cmd = ('mkdir -p "$(dirname %s)" && flock -w 43200 '
                               '"%s.lock" -c \'[ -f %s ] || python scripts/%s.py '
                               '--config %s --seed %d --out %s --data-root "$DR"\''
                               ' && ' % (pre, pre, pre, script, rel, seed, pre))
                    else:
                        cmd = ""
                    cmd += ('python train.py --config %s --seed %d '
                            '--data-root "$DR" --out-root "$OUT"' % (rel, seed))
                    (big if cost_h > BIG_H else normal).append((pct, cost_h, cmd))
    normal.sort(key=lambda t: (t[0], t[1]))
    big.sort(key=lambda t: (t[0], t[1]))
    if not args.dry_run:
        open(args.out_normal, "w").write("\n".join(c for _, _, c in normal) + "\n")
        open(args.out_big, "w").write("\n".join(c for _, _, c in big) + "\n")
    print(f"configs written: {written} (skipped {skipped} already covered)")
    print(f"tasks: normal {len(normal)}, big {len(big)} "
          f"(est. big-lane hours: {sum(h for _, h, _ in big):.0f})")


if __name__ == "__main__":
    main()
