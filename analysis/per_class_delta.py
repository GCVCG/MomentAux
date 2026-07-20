"""Per-class accuracy delta: WHICH classes does the prior help, and by how
much? Evaluates every seed of a baseline/aux cell pair on the test split and
reports per-class test accuracy for both, the per-class Δ (aux − baseline,
averaged over seed pairs), and human class names throughout.

Output: runs/<aux_cell>/per_class_delta.json (full table) and
<out>/perclass_<aux_cell>.png — a sorted bar chart of the K most-helped and
K most-hurt classes, with a seed-spread whisker per bar. Diagnostic only.

    python analysis/per_class_delta.py --pair tin_none_1pct tin_aux_1pct

Reading it: the prior is a spatial-frequency/structure prior, so a natural
hypothesis is that structure-rich classes gain most. This tool is how that
kind of hypothesis gets LOOKED AT before anyone claims it.
"""

import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import data as data_mod
from momentstem import build_model
from visualize_features import class_names


def find_config(cell):
    for d in ("", "diagnostics/", "ablations_full/"):
        path = f"configs/{d}{cell}.yaml"
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"no config found for cell {cell!r}")


@torch.no_grad()
def per_class_acc(model, loader, num_classes, device):
    correct = torch.zeros(num_classes)
    total = torch.zeros(num_classes)
    for x, y in loader:
        x = x.to(device)
        pred = model(x).argmax(1).cpu()
        for c, ok in zip(y, pred == y):
            correct[c] += bool(ok)
            total[c] += 1
    return (correct / total.clamp_min(1)).numpy()


def seed_accs(cell, cfg, loader, num_classes, device):
    out = {}
    for sd in sorted(d for d in os.listdir(f"runs/{cell}") if d.startswith("seed")):
        ckpt = f"runs/{cell}/{sd}/best.pt"
        if not os.path.exists(ckpt):
            continue
        model = build_model(
            cfg["backbone"], cfg.get("stem", "none"),
            num_classes=num_classes,
            small_input=cfg.get("small_input", True),
            stem_kernel_size=cfg.get("stem_kernel_size", 11),
            stem_kwargs=cfg.get("stem_kwargs"),
            head=cfg.get("head"),
            moment_aux=cfg.get("moment_aux"),
        ).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.eval()
        out[sd] = per_class_acc(model, loader, num_classes, device)
        print(f"{cell} {sd}: mean {out[sd].mean()*100:.2f}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", nargs=2, metavar=("NONE_CELL", "AUX_CELL"),
                    required=True)
    ap.add_argument("--out", default="docs/viz")
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--top-k", type=int, default=15)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    none_cell, aux_cell = args.pair
    cfgs = {c: yaml.safe_load(open(find_config(c))) for c in args.pair}
    dataset = cfgs[aux_cell]["dataset"]
    num_classes = data_mod.NUM_CLASSES[dataset]

    test_ds = data_mod.build_dataset(dataset, args.data_root, train=False)
    loader = DataLoader(test_ds, batch_size=512, num_workers=4, shuffle=False)
    names = class_names(dataset, test_ds, args.data_root) \
        or [f"class {c}" for c in range(num_classes)]

    accs = {c: seed_accs(c, cfgs[c], loader, num_classes, device)
            for c in args.pair}
    base = np.stack(list(accs[none_cell].values()))   # (S, C)
    aux = np.stack(list(accs[aux_cell].values()))
    n = min(len(base), len(aux))
    delta_seeds = (aux[:n] - base[:n]) * 100          # paired by seed index
    delta = delta_seeds.mean(0)
    spread = delta_seeds.std(0) if n > 1 else np.zeros_like(delta)

    order = np.argsort(delta)
    rows = [{"class": int(c), "name": names[c],
             "base_acc": float(base[:, c].mean() * 100),
             "aux_acc": float(aux[:, c].mean() * 100),
             "delta": float(delta[c]), "delta_seed_std": float(spread[c])}
            for c in order[::-1]]
    payload = {"pair": args.pair, "dataset": dataset, "seeds_paired": n,
               "mean_delta": float(delta.mean()), "per_class": rows}
    jpath = f"runs/{aux_cell}/per_class_delta.json"
    with open(jpath, "w") as f:
        json.dump(payload, f, indent=1)

    k = min(args.top_k, num_classes // 2)
    show = np.concatenate([order[-k:][::-1], order[:k][::-1]])
    colors = ["#0072B2"] * k + ["#D55E00"] * k
    fig, ax = plt.subplots(figsize=(9, 0.32 * 2 * k + 1.6))
    ypos = np.arange(len(show))[::-1]
    ax.barh(ypos, delta[show], xerr=spread[show], color=colors,
            error_kw={"lw": 0.8, "alpha": 0.6})
    ax.set_yticks(ypos)
    ax.set_yticklabels([names[c] for c in show], fontsize=7)
    ax.axvline(0, color="k", lw=0.8)
    ax.axvline(delta.mean(), color="gray", lw=0.8, ls="--",
               label=f"mean Δ = {delta.mean():+.2f}")
    ax.set_xlabel("Δ test accuracy per class, aux − baseline (pts, "
                  f"mean of {n} seed pairs; whisker = seed std)")
    ax.set_title(f"{aux_cell}: {k} most-helped (blue) and {k} most-hurt "
                 "(orange) classes")
    ax.legend(fontsize=8)
    fig.tight_layout()
    os.makedirs(args.out, exist_ok=True)
    fpath = os.path.join(args.out, f"perclass_{aux_cell}.png")
    fig.savefig(fpath, dpi=150)
    plt.close(fig)

    helped = (delta > 0).sum()
    print(f"PERCLASS {aux_cell}: mean Δ {delta.mean():+.2f}, "
          f"{helped}/{num_classes} classes helped, "
          f"best {names[order[-1]]} {delta[order[-1]]:+.1f}, "
          f"worst {names[order[0]]} {delta[order[0]]:+.1f} -> {jpath}, {fpath}",
          flush=True)


if __name__ == "__main__":
    main()
