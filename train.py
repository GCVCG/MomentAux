"""Single training entry point for ALL experimental cells.

    python train.py --config configs/<cell>.yaml --seed N

One fixed recipe (any deviation is a bug): SGD momentum 0.9, cosine schedule,
200 epochs, batch 128, crop+flip augmentation only. The config selects the
cell (dataset, subset, backbone, stem); it does NOT change the recipe.

Outputs, per run, under runs/<cell-name>/seed<N>/:
  metrics.csv  -- one row per epoch (loss, top-1, lr, time)
  final.json   -- config, seed, accuracies, param/FLOP accounting, env info
  last.pt      -- final-epoch weights (used for CIFAR-C evaluation)
  best.pt      -- best test-top-1 weights
No external services required to reproduce anything.
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
import yaml
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassAccuracy

import data as data_mod
from momentstem import MomentStem, build_model, count_params_flops

RECIPE = {
    "epochs": 200,
    "batch_size": 128,
    "lr": 0.1,
    "weight_decay": 5e-4,
    "momentum": 0.9,
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


def build_loaders(cfg, seed, data_root):
    train_ds = data_mod.build_dataset(
        cfg["dataset"], data_root, train=True, subset_pct=cfg.get("subset_pct")
    )
    test_ds = data_mod.build_dataset(cfg["dataset"], data_root, train=False)
    gen = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg.get("num_workers", 8),
        pin_memory=True,
        drop_last=True,
        generator=gen,
        worker_init_fn=seed_worker,
        persistent_workers=cfg.get("num_workers", 8) > 0,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=512,
        shuffle=False,
        num_workers=cfg.get("num_workers", 8),
        pin_memory=True,
    )
    return train_loader, test_loader


@torch.no_grad()
def conv1_group_norms(model):
    """Visibility into whether the backbone is USING the stem: L2 norm of
    conv1's weights per input-channel group (identity / gabor / zernike).
    This metric caught the v1 scale bug -- conv1 shrank its gabor weights
    below init while growing identity 3x. Empty for non-moment stems."""
    stem = model.stem
    if not isinstance(stem, MomentStem) or stem.mode != "concat":
        return {}
    conv1 = getattr(model.net, "conv1", None)
    if conv1 is None:
        return {}
    per_in = conv1.weight.detach().permute(1, 0, 2, 3).flatten(1).norm(dim=1)
    groups, lo = {}, 0
    for name, width in (
        ("identity", stem.in_channels if stem.include_identity else 0),
        ("gabor", stem.n_gabor),
        ("zernike", stem.n_zernike),
    ):
        if width:
            groups[name] = per_in[lo:lo + width].mean().item()
            lo += width
    return groups


@torch.no_grad()
def evaluate(model, loader, device, num_classes):
    metric = MulticlassAccuracy(num_classes=num_classes, average="micro").to(device)
    model.eval()
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        metric.update(model(x), y)
    return metric.compute().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--out-root", default="./runs")
    ap.add_argument("--epochs", type=int, default=None, help="override (smoke runs only)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-amp", action="store_true")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    for k, v in RECIPE.items():
        cfg.setdefault(k, v)
    if args.epochs is not None:
        cfg["epochs"] = args.epochs
        cfg["epochs_overridden"] = True

    name = cfg.get("name") or os.path.splitext(os.path.basename(args.config))[0]
    out_dir = os.path.join(args.out_root, name, f"seed{args.seed}")
    os.makedirs(out_dir, exist_ok=True)

    set_seed(args.seed)
    device = torch.device(args.device)
    if device.type != "cuda":
        print("WARNING: training on CPU (fine for smoke tests only)")

    num_classes = data_mod.NUM_CLASSES[cfg["dataset"]]
    image_size = data_mod.IMAGE_SIZE[cfg["dataset"]]
    model = build_model(
        cfg["backbone"],
        cfg["stem"],
        num_classes=num_classes,
        small_input=cfg.get("small_input", True),
        pretrained=cfg.get("pretrained", False),
        stem_kernel_size=cfg.get("stem_kernel_size", 11),
        stem_seed=args.seed,
        stem_kwargs=cfg.get("stem_kwargs"),
        head_pool=cfg.get("head_pool"),
        moment_aux=cfg.get("moment_aux"),
    ).to(device)
    if cfg.get("stem_calibrate", False):
        # Deterministic calibration batch: first N train images in index
        # order, eval transform (no augmentation) -- identical for every
        # stem, subset, and seed of a dataset. stem_calibrate: true -> per-
        # channel std; "zca" -> fixed whitening (calibration v2). For a
        # moment_aux model the target moments are calibrated instead.
        calib = data_mod.calibration_batch(cfg["dataset"], args.data_root).to(device)
        if cfg.get("moment_aux") and hasattr(model, "calibrate"):
            model.calibrate(calib)
            print(f"moment-aux target calibrated on {calib.shape[0]} images")
        elif hasattr(model.stem, "calibrate"):
            if cfg["stem_calibrate"] == "zca":
                model.stem.calibrate_zca(calib)
            else:
                model.stem.calibrate(calib)
            print(f"stem calibrated ({cfg['stem_calibrate']}) on {calib.shape[0]} images")
    accounting = count_params_flops(model, image_size=image_size)
    print(f"[{name} seed{args.seed}] {json.dumps(accounting)}")

    # stem_unfreeze_epoch: train with the stem FROZEN (prior-as-warmup)
    # until this epoch, then let its filters train. Requires a trainable
    # stem (stem: gabor-learn). The frozen params sit in the optimizer from
    # the start with requires_grad=False (SGD skips grad-less params), so
    # the cosine schedule stays intact when they wake up.
    unfreeze_at = cfg.get("stem_unfreeze_epoch")
    if unfreeze_at is not None:
        stem_params = list(model.stem.parameters())
        if not stem_params:
            raise ValueError("stem_unfreeze_epoch needs a trainable stem (gabor-learn)")
        for p in stem_params:
            p.requires_grad = False
        print(f"stem frozen until epoch {unfreeze_at}")

    train_loader, test_loader = build_loaders(cfg, args.seed, args.data_root)
    params = (list(model.parameters()) if unfreeze_at is not None
              else [p for p in model.parameters() if p.requires_grad])
    # THE RECIPE IS SGD. `optimizer: adamw` exists ONLY because the frozen recipe
    # is implicitly a RESNET recipe and does not transfer off that family:
    # ConvNeXt under SGD lr=0.1 never fits the train set (19% train acc after 200
    # epochs, loss 6.38 -> 3.31, vs ResNet-18 at 100% by epoch 99), so a Δ
    # measured there says nothing about the method. A cell using this MUST be
    # named diag* and MUST NEVER enter a headline table (see CLAUDE.md).
    opt_name = cfg.get("optimizer", "sgd").lower()
    if opt_name == "sgd":
        optimizer = torch.optim.SGD(
            params, lr=cfg["lr"], momentum=cfg["momentum"],
            weight_decay=cfg["weight_decay"],
        )
    elif opt_name == "adamw":
        if not cfg["name"].startswith("diag"):
            raise ValueError(
                f"optimizer: adamw deviates from the frozen recipe, so the cell "
                f"name must start with 'diag' (got {cfg['name']!r}). See CLAUDE.md."
            )
        optimizer = torch.optim.AdamW(
            params, lr=cfg["lr"], weight_decay=cfg["weight_decay"],
        )
    else:
        raise ValueError(f"optimizer must be 'sgd' or 'adamw', got {opt_name!r}")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])
    criterion = torch.nn.CrossEntropyLoss()
    use_amp = device.type == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    csv_path = os.path.join(out_dir, "metrics.csv")
    csv_file = open(csv_path, "w", newline="")
    writer = csv.writer(csv_file)
    norm_cols = ["conv1_identity", "conv1_gabor", "conv1_zernike"]
    writer.writerow(
        ["epoch", "train_loss", "train_acc", "test_acc", "lr", "epoch_seconds"]
        + norm_cols
    )

    # Optional moment-aux lambda SCHEDULE: start strong (prior dominates when
    # the net can't estimate features) and decay (let cross-entropy take over) --
    # the "washes out with data" mechanism inside one run. weight -> weight_final
    # over epochs; default const (weight_final = weight) so fixed-lambda cells
    # are unchanged.
    aux_cfg = cfg.get("moment_aux") or {}
    aux_w0 = aux_cfg.get("weight", 0.1)
    aux_wT = aux_cfg.get("weight_final", aux_w0)
    aux_sched = aux_cfg.get("weight_schedule", "const")

    def aux_lambda(epoch):
        if aux_sched == "const":
            return aux_w0
        frac = epoch / max(cfg["epochs"] - 1, 1)
        if aux_sched == "linear":
            return aux_w0 + (aux_wT - aux_w0) * frac
        if aux_sched == "cosine":
            return aux_wT + 0.5 * (aux_w0 - aux_wT) * (1 + math.cos(math.pi * frac))
        raise ValueError(f"unknown weight_schedule {aux_sched!r}")

    best_acc, t_start = 0.0, time.time()
    train_metric = MulticlassAccuracy(num_classes=num_classes, average="micro").to(device)
    for epoch in range(cfg["epochs"]):
        if cfg.get("moment_aux") and hasattr(model, "aux_weight"):
            model.aux_weight = aux_lambda(epoch)
        if unfreeze_at is not None and epoch == unfreeze_at:
            for p in model.stem.parameters():
                p.requires_grad = True
            print(f"stem unfrozen at epoch {epoch}")
        model.train()
        train_metric.reset()
        loss_sum, n_batches, t0 = 0.0, 0, time.time()
        lr_now = optimizer.param_groups[0]["lr"]
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(x)
                loss = criterion(logits, y)
                # moment-aux prior: soft MSE regression of an intermediate
                # feature onto the fixed moment maps (deployed path unchanged).
                aux = getattr(model, "last_aux", None)
                if aux is not None:
                    loss = loss + model.aux_weight * aux
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            # moment-aux: block the scale degeneracy (see aux.py) by restoring
            # the aux head's weight norm after each step.
            if hasattr(model, "project_heads"):
                model.project_heads()
            loss_sum += loss.item()
            n_batches += 1
            train_metric.update(logits.detach(), y)
        scheduler.step()

        test_acc = evaluate(model, test_loader, device, num_classes)
        train_acc = train_metric.compute().item()
        norms = conv1_group_norms(model)
        writer.writerow(
            [epoch, f"{loss_sum / max(n_batches, 1):.6f}", f"{train_acc:.6f}",
             f"{test_acc:.6f}", f"{lr_now:.6f}", f"{time.time() - t0:.1f}"]
            + [f"{norms[g]:.4f}" if g in norms else ""
               for g in ("identity", "gabor", "zernike")]
        )
        csv_file.flush()
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), os.path.join(out_dir, "best.pt"))
        print(
            f"epoch {epoch + 1}/{cfg['epochs']} loss {loss_sum / max(n_batches, 1):.4f} "
            f"train {train_acc:.4f} test {test_acc:.4f}"
        )
    csv_file.close()
    torch.save(model.state_dict(), os.path.join(out_dir, "last.pt"))

    final = {
        "name": name,
        "seed": args.seed,
        "config": cfg,
        "final_test_acc": test_acc,
        "best_test_acc": best_acc,
        "accounting": accounting,
        "wall_seconds": time.time() - t_start,
        "torch_version": torch.__version__,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "amp": use_amp,
    }
    with open(os.path.join(out_dir, "final.json"), "w") as f:
        json.dump(final, f, indent=2)
    print(f"done: final {test_acc:.4f} best {best_acc:.4f} -> {out_dir}")


if __name__ == "__main__":
    main()
