"""Submit the experiment grid to the turing cluster, chained two jobs at a
time (the `gpu` QOS allows max 2 running jobs): jobs are split into two
chains, each chained with --dependency=afterany:<previous>.

GPU jobs on turing MUST use --partition=gpu --qos=gpu (the default `ladon`
QOS has GrpTRES gres/gpu=0, i.e. zero GPUs -- discovered the hard way).

    python slurm/submit_grid.py --dry-run                     # show the plan
    python slurm/submit_grid.py                               # full grid
    python slurm/submit_grid.py --filter cifar100_resnet18    # cell substring
    python slurm/submit_grid.py --seeds 0 1 2 --gpu h100 \
        --python /mnt/beegfs/amughrabi/envs/momentstem/bin/python
"""

import argparse
import glob
import os
import re
import shlex
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def submit(sbatch_args, dry_run):
    cmd = ["sbatch"] + sbatch_args
    if dry_run:
        print("DRY:", " ".join(shlex.quote(c) for c in cmd))
        return f"dry{submit.counter}"
    out = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO, check=True).stdout
    m = re.search(r"Submitted batch job (\d+)", out)
    if not m:
        sys.exit(f"could not parse job id from: {out!r}")
    print(f"{out.strip()}  <- {' '.join(sbatch_args[-4:])}")
    return m.group(1)


submit.counter = 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--filter", default="", help="substring filter on config name")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--gpu", default="h100", choices=["h100", "h200"])
    ap.add_argument("--python", default="python", help="interpreter (venv) to run train.py with")
    ap.add_argument("--partition", default="gpu")
    ap.add_argument("--qos", default="gpu")
    ap.add_argument("--time", default="48:00:00")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    configs = sorted(glob.glob(os.path.join(REPO, "configs", "*.yaml")))
    configs = [c for c in configs
               if args.filter in os.path.basename(c) and "smoke" not in os.path.basename(c)]
    if not configs:
        sys.exit("no configs matched")
    os.makedirs(os.path.join(REPO, "slurm", "logs"), exist_ok=True)
    print(f"{len(configs)} configs x seeds {args.seeds} -> {len(configs)} jobs in 2 chains")

    tails = [None, None]  # last job id of each chain
    for i, config in enumerate(configs):
        chain = i % 2
        rel = os.path.relpath(config, REPO)
        name = os.path.splitext(os.path.basename(config))[0]
        train_cmds = " && ".join(
            f"{args.python} train.py --config {rel} --seed {s}" for s in args.seeds
        )
        sbatch_args = [
            f"--job-name={name}",
            f"--partition={args.partition}",
            f"--qos={args.qos}",
            f"--gres=gpu:{args.gpu}:1",
            "--cpus-per-task=8",
            "--mem=32G",
            f"--time={args.time}",
            "--output=slurm/logs/%x_%j.out",
        ]
        if tails[chain] is not None:
            sbatch_args.append(f"--dependency=afterany:{tails[chain]}")
        sbatch_args.append(f"--wrap={train_cmds}")
        submit.counter += 1
        tails[chain] = submit(sbatch_args, args.dry_run)
    print(f"chain tails: {tails}")


if __name__ == "__main__":
    main()
