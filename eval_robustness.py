"""CIFAR-C robustness evaluation of trained checkpoints. NO retraining.

    python eval_robustness.py --run-dir runs/<cell>/seed<N> --cifar-c-root <dir>

Writes robustness.json next to the checkpoint with per-corruption,
per-severity top-1 error. mCE is computed in analysis/aggregate.py following
Hendrycks & Dietterich (ICLR 2019): CE_c = sum_s E_{c,s}(model) /
sum_s E_{c,s}(baseline), averaged over corruptions, with the vanilla
(stem=none) run of the same backbone/dataset cell as the baseline.
"""

import argparse
import json
import os

import torch
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassAccuracy

import data as data_mod
from data import CIFAR_C_CORRUPTIONS, CIFARCorrupted
from momentstem import build_model


@torch.no_grad()
def top1_error(model, loader, device, num_classes):
    metric = MulticlassAccuracy(num_classes=num_classes, average="micro").to(device)
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        metric.update(model(x), y)
    return 1.0 - metric.compute().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, help="runs/<cell>/seed<N>")
    ap.add_argument("--cifar-c-root", required=True,
                    help="dir with <corruption>.npy + labels.npy (Zenodo layout)")
    ap.add_argument("--dataset", default=None,
                    help="cifar10|cifar100 for the -C set; default: from config")
    ap.add_argument("--checkpoint", default="last.pt")
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    with open(os.path.join(args.run_dir, "final.json")) as f:
        final = json.load(f)
    cfg = final["config"]
    dataset = args.dataset or cfg["dataset"]
    assert dataset in ("cifar10", "cifar100"), "CIFAR-C evaluation only"
    num_classes = data_mod.NUM_CLASSES[dataset]

    device = torch.device(args.device)
    model = build_model(
        cfg["backbone"], cfg["stem"], num_classes=num_classes,
        small_input=cfg.get("small_input", True), pretrained=False,
        stem_kernel_size=cfg.get("stem_kernel_size", 11), stem_seed=final["seed"],
        stem_kwargs=cfg.get("stem_kwargs"),
    )
    # calibrated filter scales, if any, are restored from the checkpoint
    state = torch.load(os.path.join(args.run_dir, args.checkpoint),
                       map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()

    results = {"dataset": dataset, "checkpoint": args.checkpoint, "errors": {}}
    for corruption in CIFAR_C_CORRUPTIONS:
        results["errors"][corruption] = {}
        for severity in range(1, 6):
            ds = CIFARCorrupted(args.cifar_c_root, corruption, severity, dataset=dataset)
            loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, pin_memory=True)
            err = top1_error(model, loader, device, num_classes)
            results["errors"][corruption][str(severity)] = err
            print(f"{corruption} s{severity}: err {err:.4f}")

    clean = data_mod.build_dataset(dataset, "./data", train=False)
    loader = DataLoader(clean, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)
    results["clean_error"] = top1_error(model, loader, device, num_classes)

    out = os.path.join(args.run_dir, "robustness.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"clean err {results['clean_error']:.4f} -> {out}")


if __name__ == "__main__":
    main()
