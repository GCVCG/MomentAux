"""Emit the BSC work queue: every missing (config, seed) as one shell task,
ordered CHEAPEST FIRST so the results table fills left-to-right.

    python scripts/make_worklist.py > worklist.txt

Each line is a self-contained command run by a GPU worker (see
slurm/bsc_worker.sbatch). Workers pull tasks atomically from this list, so the
huge cost spread between a 1% cell and a 100% cell never idles a GPU.

SSL cells guard their SimCLR pretrain with flock: whichever worker reaches it
first builds the checkpoint and the others wait, instead of each duplicating
the (2x cost) pretrain.
"""

import glob
import os
import sys

import yaml

SEEDS = (0, 1, 2)
# Seconds per 1% of data, measured. BSC is ~0.8x turing per GPU; only the
# relative ORDER matters here, since this just sorts the queue.
RATE = {
    ("cifar100", "resnet18"): 80, ("cifar100", "resnet34"): 86,
    ("cifar100", "resnet50"): 125, ("cifar100", "convnext_tiny"): 595,
    ("cifar100", "vit_tiny"): 53, ("cifar100super", "resnet18"): 80,
    ("cifar10", "resnet18"): 80, ("stl10", "resnet18"): 142,
    ("tin", "resnet18"): 115, ("tin", "vit_tiny"): 79,
    ("tinsuper", "resnet18"): 115, ("tinsem", "resnet18"): 115,
    ("tin20", "resnet18"): 25, ("tin20b", "resnet18"): 49,
    ("cub", "resnet18"): 50,
}


def main():
    tasks = []
    for path in sorted(glob.glob("configs/grid/*.yaml")):
        cfg = yaml.safe_load(open(path))
        ds, bb = cfg["dataset"], cfg["backbone"]
        pct = cfg.get("subset_pct") or 100
        cost = RATE.get((ds, bb), 80) * pct
        rel = "configs/grid/" + os.path.basename(path)
        init = cfg.get("init_from")
        for seed in SEEDS:
            cmd = ""
            if init:
                pre = init.format(seed=seed)
                # the simclr_pre50 variant IS a 50-epoch pretrain: that is
                # the only thing distinguishing it from simclr_pre.
                ep = " --epochs 50" if "simclr_pre50" in pre else ""
                cmd += (
                    'flock -w 7200 "%s.lock" -c '
                    "'[ -f %s ] || python scripts/simclr_pretrain.py "
                    '--config %s --seed %d --out %s%s --data-root "$DR"\' && '
                    % (pre, pre, rel, seed, pre, ep)
                )
                cost *= 2
            cmd += ('python train.py --config %s --seed %d '
                    '--data-root "$DR" --out-root "$OUT"' % (rel, seed))
            tasks.append((pct, cost, cmd))

    tasks.sort(key=lambda t: (t[0], t[1]))     # portion first, then cost
    for _, _, cmd in tasks:
        print(cmd)
    print("%d tasks" % len(tasks), file=sys.stderr)


if __name__ == "__main__":
    main()
