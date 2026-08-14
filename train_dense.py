"""Training entry point for DENSE-PREDICTION cells (semantic segmentation).

    python train_dense.py --config configs/dense/<cell>.yaml --seed N

A SEPARATE ENTRY POINT, deliberately. train.py is consumed by ~9,000 finished
classification runs whose comparability rests on one frozen recipe; adding a
second task to it would put every one of those numbers behind a conditional.
Nothing here is imported by train.py and nothing in train.py is changed.

THE DENSE RECIPE, frozen in the same sense as the classification one -- no
cell tunes it, and any cell that must deviate says so in its name:
    SGD momentum 0.9, lr 0.01, weight decay 1e-4, cosine schedule,
    200 epochs, batch 16, crop 512 with random scale 0.5-2.0 and horizontal
    flip, cross-entropy ignoring index 255.

RE-PINNED 50 -> 200 EPOCHS (2026-08-11). The first dense envelope ran at 50
epochs and the pre-registered L1 control measured what that cost: the
voc@10% baseline went 7.23 -> 18.42 mIoU, a +155% rise, and the 50-epoch
curves were still climbing at ~0.7 mIoU/epoch when the cosine annealed the
step size away -- so their apparent convergence was a SCHEDULE ARTIFACT and
the small absolute Deltas reported a weak recipe as much as a weak prior.
200 is not a tuned number. It is the SAME budget the frozen classification
recipe uses at every fraction, so steps scale with data on both sides of the
study and a dense-vs-classification comparison at matched images-per-class is
no longer confounded by schedule. Picking the budget from convergence curves
was considered and REJECTED: that tunes the recipe to the task, which is
exactly what a frozen recipe exists to prevent.
The learning rate and batch size are the standard VOC segmentation settings
rather than the classification recipe's 0.1/128, because 0.1 at batch 16 on
512px crops does not train. Cosine rather than the segmentation literature's
poly schedule, to match the rest of the study and, more importantly, because
the AUXILIARY weight must reach exactly zero by the final epoch -- that is
what makes high-data neutrality structural rather than tuned, and it is the
one property of the method that a new task family must not quietly drop.

WHAT IS MEASURED. Two numbers per cell, exactly as in the classification
study: end-to-end mIoU, and (by analysis/dense_probe.py) the mIoU a frozen
backbone supports under a 1x1 convolutional head, which is the dense analog
of G. Their difference is the readout term, on a metric the law has never
been tested against.
"""
import argparse
import csv
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

import data_dense as dd
from momentstem.segmentation import build_seg_model

RECIPE = {
    "epochs": 200, "batch_size": 16, "lr": 0.01,
    "weight_decay": 1e-4, "momentum": 0.9, "crop": 512,
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    s = torch.initial_seed() % 2 ** 32
    np.random.seed(s)
    random.seed(s)


def atomic_save(obj, path):
    """tmp + replace: a torn checkpoint cannot exist even if we are killed."""
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


@torch.no_grad()
def evaluate(model, loader, device, n_classes):
    model.eval()
    cm = dd.ConfusionMatrix(n_classes)
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast("cuda", enabled=device.type == "cuda"):
            logits = model(x)
        cm.update(logits.argmax(1).flatten(), y.flatten())
    miou, per_class = cm.miou()
    return miou, cm.pixel_acc(), per_class


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--out-root", default="./runs_dense")
    ap.add_argument("--epochs", type=int, default=None,
                    help="override (smoke runs only)")
    ap.add_argument("--batch-size", type=int, default=None,
                    help="SMOKE-TEST OVERRIDE ONLY. The frozen dense recipe "
                         "is batch 16 on every population; this exists so a "
                         "code path can be exercised on a GPU that is busy "
                         "with something else. It is recorded in final.json "
                         "under batch_size, so a cell run with it can never "
                         "be mistaken for a recipe cell.")
    ap.add_argument("--eval-every", type=int, default=None,
                    help="epochs between validations; the final epoch is "
                         "always evaluated (default 5)")
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    out_dir = os.path.join(args.out_root, cfg["name"], f"seed{args.seed}")
    os.makedirs(out_dir, exist_ok=True)

    # Same run-dir guards as train.py, for the same reasons: a completed cell
    # is never silently re-run, and two trainers never race one seed dir.
    if (os.path.exists(os.path.join(out_dir, "final.json"))
            and not os.environ.get("MS_FORCE_RERUN")):
        print(f"SKIP: {out_dir}/final.json exists -- this cell is complete.")
        return
    import fcntl
    lock_fd = open(os.path.join(out_dir, ".runlock"), "w")
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        raise SystemExit(f"ABORT: {out_dir} is locked by another trainer.")
    main._run_lock = lock_fd

    set_seed(args.seed)
    device = torch.device(args.device)
    epochs = args.epochs or cfg.get("epochs", RECIPE["epochs"])
    bs = args.batch_size or cfg.get("batch_size", RECIPE["batch_size"])
    crop = cfg.get("crop", RECIPE["crop"])

    # The dataset key drives split names, class count and file layout; every
    # other line of this trainer is population-agnostic, which is the point.
    ds = cfg.get("dataset", "voc_seg")
    n_classes = dd.NUM_CLASSES[ds]
    train_ds = dd.SegmentationDataset(
        args.data_root, split=cfg.get("split", dd.TRAIN_SPLIT[ds]),
        train=True, crop=crop,
        subset_pct=cfg.get("subset_pct"), ds=ds)
    val_ds = dd.SegmentationDataset(args.data_root, split=dd.VAL_SPLIT[ds],
                                    train=False, ds=ds)
    gen = torch.Generator().manual_seed(args.seed)
    nw = cfg.get("num_workers", 4)
    train_ld = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw,
                          pin_memory=True, drop_last=True, generator=gen,
                          worker_init_fn=seed_worker)
    # Native-resolution evaluation means variable image sizes, so batch 1.
    val_ld = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=nw)

    model = build_seg_model(
        cfg.get("backbone", "resnet18"),
        n_classes=n_classes,
        output_stride=cfg.get("output_stride", 8),
        pretrained=cfg.get("pretrained", False),
        moment_aux=cfg.get("moment_aux"),
        image_size=crop,
    ).to(device)
    has_aux = cfg.get("moment_aux") is not None
    if has_aux and cfg.get("stem_calibrate", True):
        # Calibrate the fixed bank on a deterministic batch of TRAINING images,
        # exactly as the classification cells do (image statistics only, no
        # labels, so nothing leaks into the low-label protocol).
        cal = torch.stack([train_ds[i][0] for i in range(min(32, len(train_ds)))])
        model.calibrate(cal.to(device))

    # The frozen dense recipe is SGD. AdamW exists ONLY for the attention
    # cells, and its use is why those carry a diag prefix: the classification
    # study found swin-none seed-bistable or at chance under the frozen SGD
    # recipe on most datasets, so running a dense Swin pair under SGD would
    # measure a training failure rather than a feature deficit. Any cell that
    # sets this is a diagnostic and is never mixed into a headline table.
    # Both arms of a pair always share it, so Delta stays valid.
    optim_name = cfg.get("optimizer", "sgd").lower()
    if optim_name == "adamw":
        if not cfg["name"].startswith("diag"):
            raise ValueError(
                f"{cfg['name']}: optimizer adamw deviates from the frozen "
                "dense recipe, so the cell name must carry a diag prefix")
        opt = torch.optim.AdamW(model.parameters(),
                                lr=cfg.get("lr", 1e-4),
                                weight_decay=cfg.get("weight_decay", 0.05))
    elif optim_name == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=cfg.get("lr", RECIPE["lr"]),
                              momentum=RECIPE["momentum"],
                              weight_decay=cfg.get("weight_decay",
                                                   RECIPE["weight_decay"]))
    else:
        raise ValueError(f"unknown optimizer {optim_name!r}")
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    lam0 = (cfg["moment_aux"].get("weight", 1.0) if has_aux else 0.0)
    lam_final = (cfg["moment_aux"].get("weight_final", 0.0) if has_aux else 0.0)

    # Validation is 1,449 images at native resolution, which at the small
    # fractions costs as much wall-clock as the training it is measuring.
    # Both arms use the identical interval, so no comparison is affected;
    # the final epoch is always evaluated, so final_miou is never stale.
    eval_every = args.eval_every or cfg.get("eval_every", 5)

    rows, best = [], -1.0
    t0 = time.time()
    for ep in range(epochs):
        # lambda(t): cosine from lam0 to lam_final, reaching it EXACTLY at the
        # last epoch. Identical schedule to the classification cells.
        lam = lam_final + 0.5 * (lam0 - lam_final) * (1 + math.cos(math.pi * ep / max(epochs - 1, 1)))
        if has_aux:
            model.aux_weight = lam
        model.train()
        tot, n_seen = 0.0, 0
        for x, y in train_ld:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(x)
                loss = F.cross_entropy(logits, y, ignore_index=dd.IGNORE_INDEX)
                if has_aux and model.last_aux is not None:
                    loss = loss + lam * model.last_aux
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            if has_aux:
                model.project_heads()
            tot += float(loss) * x.size(0)
            n_seen += x.size(0)
        sched.step()
        due = (ep % eval_every == 0) or (ep == epochs - 1)
        miou, pacc = (float("nan"), float("nan"))
        if due:
            miou, pacc, _ = evaluate(model, val_ld, device, n_classes)
        rows.append({"epoch": ep, "loss": tot / max(n_seen, 1), "miou": miou,
                     "pixel_acc": pacc, "lr": opt.param_groups[0]["lr"],
                     "aux_lambda": lam})
        print(f"ep {ep:3d}  loss {rows[-1]['loss']:.4f}  mIoU "
              f"{miou:6.2f}  pixAcc {pacc:6.2f}  lam {lam:.3f}", flush=True)
        if due and miou > best:
            best = miou
            atomic_save(model.state_dict(), os.path.join(out_dir, "best.pt"))
    atomic_save(model.state_dict(), os.path.join(out_dir, "last.pt"))

    miou, pacc, per_class = evaluate(model, val_ld, device, n_classes)
    with open(os.path.join(out_dir, "metrics.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(out_dir, "final.json"), "w") as f:
        json.dump({
            "config": cfg, "seed": args.seed, "epochs": epochs,
            "optimizer": optim_name,
            "batch_size": bs,
            "final_miou": miou, "best_miou": best, "final_pixel_acc": pacc,
            "per_class_iou": per_class,
            "n_train": len(train_ds), "n_val": len(val_ds),
            "wall_seconds": time.time() - t0,
            "gpu_name": (torch.cuda.get_device_name(0)
                         if device.type == "cuda" else "cpu"),
            "torch": torch.__version__,
        }, f, indent=1)
    print(f"FINAL mIoU {miou:.2f}  best {best:.2f}  "
          f"({time.time() - t0:.0f}s, {len(train_ds)} train images)")


if __name__ == "__main__":
    main()
