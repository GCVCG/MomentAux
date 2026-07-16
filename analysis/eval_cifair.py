"""Re-evaluate trained CIFAR-100 runs on the ciFAIR-100 test set.

Why: CIFAR-100's test set contains ~9% near-duplicates of TRAINING images
(927/10000). Every headline cell in this study lives at 1-5% data, so
"are your low-data gains partly memorised duplicates?" is a question that will
be asked. ciFAIR (Barz & Denzler 2020) replaces exactly those images, keeping
labels and order, so it is a drop-in swap needing NO retraining.

What to expect, stated before looking: the contamination is against the FULL
train set, so at 1% (500 imgs) most duplicate sources are not in the training
subset at all -- the absolute drop should be SMALL at low data and larger at
100%. And the aux-vs-baseline DELTA should barely move, since both cells eat
the identical contamination. If the delta DOES move materially, that is a real
problem and worth knowing now rather than at review.

    python analysis/eval_cifair.py --run runs/auxmag_5pct_sched0 \
        --config configs/diagnostics/auxmag_5pct_sched0.yaml
"""

import argparse
import json
import os
import sys

import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import data as data_mod
from data import CiFAIRTest
from momentstem import build_model


@torch.no_grad()
def accuracy(model, loader, device):
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast("cuda", enabled=device.type == "cuda"):
            pred = model(x).argmax(1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return 100.0 * correct / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="runs/<cell> (all seeds)")
    ap.add_argument("--config", required=True)
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--ckpt", default="best.pt")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    assert cfg["dataset"] == "cifar100", "ciFAIR-100 pairs with cifar100 runs only"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tf = data_mod.build_transforms("cifar100", train=False)
    cifar = data_mod.build_dataset("cifar100", args.data_root, train=False)
    cifair = CiFAIRTest(args.data_root, transform=tf)
    l_cifar = DataLoader(cifar, batch_size=512, num_workers=4, shuffle=False)
    l_cifair = DataLoader(cifair, batch_size=512, num_workers=4, shuffle=False)

    out = []
    for seed_dir in sorted(d for d in os.listdir(args.run) if d.startswith("seed")):
        ckpt = os.path.join(args.run, seed_dir, args.ckpt)
        if not os.path.exists(ckpt):
            continue
        model = build_model(
            cfg["backbone"], cfg.get("stem", "none"), num_classes=100,
            small_input=cfg.get("small_input", True),
            stem_kernel_size=cfg.get("stem_kernel_size", 11),
            stem_kwargs=cfg.get("stem_kwargs"),
            moment_aux=cfg.get("moment_aux"),
        ).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.eval()
        a_cifar, a_cifair = accuracy(model, l_cifar, device), accuracy(model, l_cifair, device)
        out.append({"seed": seed_dir, "cifar": a_cifar, "cifair": a_cifair,
                    "drop": a_cifar - a_cifair})
        print(f"  {seed_dir}: CIFAR {a_cifar:.2f}  ciFAIR {a_cifair:.2f}  "
              f"drop {a_cifar - a_cifair:+.2f}", flush=True)

    path = os.path.join(args.run, "cifair.json")
    with open(path, "w") as f:
        json.dump({"config": args.config, "ckpt": args.ckpt, "results": out}, f, indent=2)
    if out:
        def ms(k):
            v = [r[k] for r in out]
            m = sum(v) / len(v)
            s = (sum((x - m) ** 2 for x in v) / max(len(v) - 1, 1)) ** 0.5
            return m, s
        mc, sc = ms("cifar")
        mf, sf = ms("cifair")
        md, _ = ms("drop")
        print(f"CIFAIR_RESULT {os.path.basename(args.run)} "
              f"CIFAR {mc:.2f}+/-{sc:.2f}  ciFAIR {mf:.2f}+/-{sf:.2f}  "
              f"drop {md:+.2f}  ({len(out)} seeds) -> {path}", flush=True)


if __name__ == "__main__":
    main()
