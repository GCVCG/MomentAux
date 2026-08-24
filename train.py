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

RESUME (`resume_every: N`, diag-only, OFF by default; block F of the
limitations campaign, 2026-08-23, for ViT-L/16 cells that cannot finish
inside the cluster's 24h MaxWall). Every N epochs the full training state
is written ATOMICALLY to <seed_dir>/resume.pt -- model, optimizer,
scheduler, AMP scaler, the epoch counter, best_acc, the aux-head init norms,
the elapsed wall clock, the number of metrics.csv rows, and the RNG states
(python, numpy, torch CPU, torch CUDA, and the sampler generator `gen`). On
start, if resume.pt exists and final.json does not, training resumes from
it: metrics.csv is truncated back to the saved row count and re-opened in
append mode, best.pt is rolled back to the copy that matched the saved
best_acc (best.pt.resume, a hard link taken at save time), and the epoch
loop continues. final.json then carries `resumed_from_epoch` (the last
resume point) and `resume_history` (every resume point). resume.pt is
deleted once final.json is written.
WHAT IS AND IS NOT RESTORED, stated honestly. RESTORED EXACTLY: every
parameter and optimizer moment, the LR schedule, the loss scaler, the
main-process RNGs (so Mixup/CutMix draws continue the same stream), and the
DATA ORDER: the sampler generator's state is restored, and with persistent
workers the DataLoader's one-off worker-base-seed draw is redirected to a
throwaway generator so the restored `gen` yields exactly the permutations the
uninterrupted run would have drawn. NOT RESTORED: the dataloader WORKERS'
own RNG streams (RandomCrop/Flip/RandAugment/RandomErasing draw from them).
Workers die with the job, and PyTorch seeds fresh workers from the base seed
only, so they restart at their epoch-0 state rather than at the state they
had after K epochs of draws. The augmentation stream after a resume is
therefore an UNBIASED re-draw from the same distribution, not the
uninterrupted run's draw -- the same status as a num_workers change (see
CLAUDE.md): Delta stays valid, byte-identity with an uninterrupted run does
not hold. With num_workers=0 the generator is not swapped (the single-process
iterator draws its base seed from `gen` every epoch, so leaving it in place
is what reproduces the uninterrupted sequence) and there are no worker RNGs
to lose, so a resumed CPU/num_workers=0 run IS byte-identical.

WARMUP / CLIPPING (`warmup_epochs: N`, `clip_grad: X`; diag-only, both OFF by
default and both leaving the default code path byte-unchanged). The frozen
recipe has no LR warmup, and the diagnostic `optimizer: adamw` path runs
lr 1e-3 at batch 128 -- eight times DeiT's batch-scaled lr. ViT-S/16 and
ViT-B/16 survive that; ViT-L/16 (304M params) does not, and every ViT-L cell
of block F ended at NaN (2026-08-24). `warmup_epochs` linearly ramps the lr
over the first N epochs' worth of OPTIMIZER STEPS (per-step, because the
divergence happens inside the first epoch) on top of the same cosine decay;
`clip_grad` clips the global grad norm after unscaling. Both are RECIPE
DEVIATIONS: a cell using either MUST be named diag*, and they must be applied
IDENTICALLY to both arms of a pair or the Delta is meaningless.
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
        cfg["dataset"], data_root, train=True, subset_pct=cfg.get("subset_pct"),
        augment=cfg.get("augment"),
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


def _save_resume_state(path, model, optimizer, scheduler, scaler, train_loader,
                       epoch_done, best_acc, wall_seconds, resume_history, out_dir):
    """Atomic full-state checkpoint for `resume_every` (module docstring).
    `epoch_done` is the number of completed epochs == the next epoch index
    == the number of metrics.csv data rows written so far."""
    state = {
        "epoch": epoch_done,
        "best_acc": best_acc,
        "wall_seconds": wall_seconds,
        "resume_history": list(resume_history),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "head_norm0": getattr(model, "_head_norm0", None),
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            # the SAMPLER's generator (RandomSampler holds `gen` by reference);
            # loader.generator may have been swapped on a previous resume.
            "sampler_gen": train_loader.sampler.generator.get_state(),
        },
    }
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)
    # best.pt at this moment is the checkpoint that scored best_acc; keep a
    # hard link so a later (post-save, pre-crash) best.pt can be rolled back
    # on resume and best.pt stays identical to the recorded best_test_acc.
    best = os.path.join(out_dir, "best.pt")
    if os.path.exists(best):
        ltmp = os.path.join(out_dir, "best.pt.resume.tmp")
        if os.path.exists(ltmp):
            os.remove(ltmp)
        os.link(best, ltmp)
        os.replace(ltmp, os.path.join(out_dir, "best.pt.resume"))
    print(f"resume state saved at epoch {epoch_done} -> {path}")


def _restore_resume_state(state, model, optimizer, scheduler, scaler, train_loader,
                          seed, out_dir, csv_path, device):
    """Inverse of _save_resume_state; returns (start_epoch, best_acc,
    wall_before). See the module docstring for what is and is not restored."""
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    scaler.load_state_dict(state["scaler"])
    if state.get("head_norm0") is not None and hasattr(model, "_head_norm0"):
        model._head_norm0 = dict(state["head_norm0"])
    rng = state["rng"]
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch"])
    if rng["cuda"] is not None and torch.cuda.is_available():
        if len(rng["cuda"]) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all(rng["cuda"])
        else:
            torch.cuda.set_rng_state(rng["cuda"][0], device)
    gen = train_loader.sampler.generator
    gen.set_state(rng["sampler_gen"])
    if train_loader.num_workers > 0 and train_loader.persistent_workers:
        # The first iter() will create the persistent workers and draw ONE
        # int64 base seed from loader.generator. In the uninterrupted run
        # that draw came from the freshly-seeded generator at epoch 0, and
        # `gen` has since advanced by K permutations. Redirect the draw to a
        # throwaway generator seeded exactly as the original was, so (a) the
        # workers get the SAME base seed the original workers got and (b)
        # `gen` is left at the restored state for the sampler, whose
        # RandomSampler holds its own reference to `gen` -- the data ORDER
        # then continues exactly where the uninterrupted run would have.
        train_loader.generator = torch.Generator().manual_seed(seed)
    start_epoch = int(state["epoch"])
    # metrics.csv: keep header + the rows the saved state accounts for; any
    # rows after it belong to the discarded post-save trajectory.
    with open(csv_path) as f:
        lines = f.readlines()
    keep = lines[: 1 + start_epoch]
    if len(keep) < 1 + start_epoch:
        raise RuntimeError(
            f"{csv_path} has {len(lines) - 1} rows but resume.pt says {start_epoch} "
            f"epochs completed -- refusing to resume from an inconsistent record")
    with open(csv_path, "w", newline="") as f:
        f.writelines(keep)
    # roll best.pt back to the copy that matched best_acc at save time
    linked = os.path.join(out_dir, "best.pt.resume")
    if os.path.exists(linked):
        os.replace(linked, os.path.join(out_dir, "best.pt"))
    print(f"RESUME: restored epoch {start_epoch}, best {state['best_acc']:.4f}, "
          f"{state['wall_seconds']:.0f}s elapsed before this segment")
    return start_epoch, float(state["best_acc"]), float(state["wall_seconds"])


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

    # --- RUN-DIR GUARDS (2026-08-07, after the duplicate-race incident that
    # left 21 wrong-epoch best.pt files: two lanes trained the same seed dir
    # concurrently, and the loser's mid-run "best so far" overwrote the
    # winner's finished weights).
    # (1) Completed-run guard: final.json is the record of a finished cell.
    #     Re-running would overwrite scored, committed numbers with a fresh
    #     draw -- every stale-worklist incident (#2..#7) reduces to this.
    #     MS_FORCE_RERUN=1 overrides for a deliberate re-measurement.
    if (os.path.exists(os.path.join(out_dir, "final.json"))
            and not os.environ.get("MS_FORCE_RERUN")):
        print(f"SKIP: {out_dir}/final.json exists -- this cell is complete. "
              f"Set MS_FORCE_RERUN=1 to retrain deliberately.")
        return
    # (2) Exclusive run lock: a second trainer on the same seed dir aborts
    #     LOUDLY instead of silently racing checkpoint writes. The fd must
    #     outlive this scope, so it is parked on the function.
    import fcntl
    lock_fd = open(os.path.join(out_dir, ".runlock"), "w")
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        raise SystemExit(
            f"ABORT: {out_dir} is locked -- another training process is "
            f"live on this exact (config, seed). Refusing to race it.")
    main._run_lock = lock_fd  # keep the lock for the process lifetime

    # resume_every (see module docstring): diag-only, off by default.
    resume_every = cfg.get("resume_every")
    if resume_every:
        resume_every = int(resume_every)
        if resume_every <= 0:
            raise ValueError("resume_every must be a positive epoch count")
        if not cfg["name"].startswith("diag"):
            raise ValueError("resume_every requires a diag* config name")
    resume_path = os.path.join(out_dir, "resume.pt")
    resume_state = None
    if resume_every and os.path.exists(resume_path):
        resume_state = torch.load(resume_path, map_location="cpu", weights_only=False)
        print(f"RESUME: found {resume_path} at epoch {resume_state['epoch']} "
              f"(best so far {resume_state['best_acc']:.4f}); continuing")

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
        head=cfg.get("head"),
        moment_aux=cfg.get("moment_aux"),
        image_size=image_size,
        # >3 only for the multispectral sensor-fusion cells (10 or 13
        # Sentinel-2 bands); defaults to 3 so every other cell is unchanged.
        in_channels=data_mod.INPUT_CHANNELS.get(cfg["dataset"], 3),
    ).to(device)
    # A non-linear classifier head is an ARCHITECTURE deviation from the
    # frozen recipe's implicit plain-linear readout: diag-only, same rule
    # as adamw below.
    if cfg.get("head") and not cfg["name"].startswith("diag"):
        raise ValueError("head: requires a diag* config name (never headline)")
    # pretrained: ImageNet weights bring OUTSIDE images into the run -- the
    # loudest possible break of the committed-subset data contract. Allowed
    # only as an explicitly diagnostic positioning comparator.
    if cfg.get("pretrained") and not cfg["name"].startswith("diag"):
        raise ValueError("pretrained: requires a diag* config name (never headline)")
    # init_from: load a pretrained state_dict (e.g. scripts/simclr_pretrain)
    # before training. "{seed}" in the path is filled with this run's seed so
    # pretrain and supervised stay paired. strict=False: the pretrain ckpt
    # omits the classifier (and any aux heads) by construction. Changing the
    # INIT deviates from the frozen recipe -> diag-only.
    # augment: 'deit' adds RandAugment/RandomErasing (data.py) plus the
    # batch-level Mixup/CutMix/label-smoothing below, all at DeiT's published
    # values. A different augmentation stack is a recipe deviation -> diag-only.
    augment = cfg.get("augment")
    if augment and not cfg["name"].startswith("diag"):
        raise ValueError("augment: requires a diag* config name (never headline)")
    mixup_fn = None
    if augment == "deit":
        from timm.data import Mixup
        from timm.loss import SoftTargetCrossEntropy

        mixup_fn = Mixup(mixup_alpha=0.8, cutmix_alpha=1.0, label_smoothing=0.1,
                         num_classes=num_classes)
        print("deit augmentation: randaug m9 + erasing .25 + mixup .8/cutmix 1.0 "
              "+ label smoothing .1")

    init_from = cfg.get("init_from")
    if init_from:
        if not cfg["name"].startswith("diag"):
            raise ValueError("init_from requires a diag* config name")
        path = init_from.format(seed=args.seed)
        sd = torch.load(path, map_location=device)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if unexpected:
            raise ValueError(f"init_from {path}: unexpected keys {unexpected[:5]}")
        print(f"init_from {path}: {len(sd)} tensors, "
              f"{len(missing)} left at fresh init")
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
    # LR WARMUP + GRADIENT CLIPPING (2026-08-24, block F). Both default OFF and
    # the default code path is BYTE-UNCHANGED: `warmup_epochs` absent/0 keeps
    # the original CosineAnnealingLR and never enters the per-step branch;
    # `clip_grad` absent keeps the original unclipped scaler.step().
    # WHY THEY EXIST: the frozen recipe has no warmup, and `optimizer: adamw`
    # runs lr 1e-3 at batch 128 -- 8x DeiT's batch-scaled lr (5e-4 at 512).
    # ViT-S/16 and ViT-B/16 tolerate that; ViT-L/16 (304M) does not: its
    # residual stream blows up inside the first 2 epochs and every run ends at
    # NaN. These are RECIPE DEVIATIONS, so they are diag-gated exactly like
    # adamw/augment/head, and must be applied to BOTH arms of a pair.
    warmup_epochs = int(cfg.get("warmup_epochs", 0) or 0)
    clip_grad = cfg.get("clip_grad")
    clip_grad = float(clip_grad) if clip_grad else None
    if (warmup_epochs or clip_grad) and not cfg["name"].startswith("diag"):
        raise ValueError(
            "warmup_epochs/clip_grad deviate from the frozen recipe, so the "
            f"cell name must start with 'diag' (got {cfg['name']!r})."
        )
    if warmup_epochs:
        # Closed form of CosineAnnealingLR(T_max=epochs) -- identical values at
        # every epoch -- but as a LambdaLR, which recomputes lr from base_lrs on
        # each step() and is therefore safe to override WITHIN an epoch (the
        # recursive CosineAnnealingLR is not).
        _E = max(cfg["epochs"], 1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda e: 0.5 * (1.0 + math.cos(math.pi * min(e, _E) / _E))
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])
    criterion = torch.nn.CrossEntropyLoss()
    # Mixup/CutMix produce SOFT targets, which plain CrossEntropyLoss cannot
    # take; timm's SoftTargetCrossEntropy is the loss DeiT itself uses.
    soft_criterion = None
    if mixup_fn is not None:
        from timm.loss import SoftTargetCrossEntropy
        soft_criterion = SoftTargetCrossEntropy()
    use_amp = device.type == "cuda" and not args.no_amp
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    csv_path = os.path.join(out_dir, "metrics.csv")
    norm_cols = ["conv1_identity", "conv1_gabor", "conv1_zernike"]
    start_epoch, best_acc, wall_before = 0, 0.0, 0.0
    resume_history = []
    if resume_state is None:
        csv_file = open(csv_path, "w", newline="")
        writer = csv.writer(csv_file)
        # ce_loss/aux_loss/lambda/tap_std (2026-07-20): loss decomposition and the
        # tapped-feature std — the scale-collapse diagnostic from the R50 trace,
        # now logged on every aux run. Empty for non-aux cells; older runs simply
        # lack the columns (analysis/training_dynamics.py handles both).
        writer.writerow(
            ["epoch", "train_loss", "train_acc", "test_acc", "lr", "epoch_seconds"]
            + norm_cols + ["ce_loss", "aux_loss", "lambda", "tap_std"]
        )
    else:
        start_epoch, best_acc, wall_before = _restore_resume_state(
            resume_state, model, optimizer, scheduler, scaler, train_loader,
            args.seed, out_dir, csv_path, device)
        resume_history = list(resume_state.get("resume_history", [])) + [start_epoch]
        csv_file = open(csv_path, "a", newline="")
        writer = csv.writer(csv_file)

    # Optional moment-aux lambda SCHEDULE: start strong (prior dominates when
    # the net can't estimate features) and decay (let cross-entropy take over) --
    # the "washes out with data" mechanism inside one run. weight -> weight_final
    # over epochs; default const (weight_final = weight) so fixed-lambda cells
    # are unchanged.
    aux_cfg = cfg.get("moment_aux") or {}
    aux_w0 = aux_cfg.get("weight", 0.1)
    aux_wT = aux_cfg.get("weight_final", aux_w0)
    aux_sched = aux_cfg.get("weight_schedule", "const")

    # weight_delay: hold the auxiliary weight at ZERO for the first N epochs,
    # then run the declared schedule over what remains. Added for the referee
    # question the tax evidence could not answer: every transfer cell in the
    # grid applies the prior at full strength from step zero, so "a shaping
    # prior taxes a mature initialization" is only established for that timing.
    # A delayed onset lets the pretrained features settle first. Default 0
    # reproduces the previous behaviour EXACTLY (span becomes epochs-1 and frac
    # becomes epoch/(epochs-1), which is the original expression), so no
    # existing cell moves.
    aux_delay = int(aux_cfg.get("weight_delay", 0))

    def aux_lambda(epoch):
        if epoch < aux_delay:
            return 0.0
        if aux_sched == "const":
            return aux_w0
        frac = (epoch - aux_delay) / max(cfg["epochs"] - 1 - aux_delay, 1)
        if aux_sched == "linear":
            return aux_w0 + (aux_wT - aux_w0) * frac
        if aux_sched == "cosine":
            return aux_wT + 0.5 * (aux_w0 - aux_wT) * (1 + math.cos(math.pi * frac))
        raise ValueError(f"unknown weight_schedule {aux_sched!r}")

    t_start = time.time()
    # warmup is counted in OPTIMIZER STEPS; a resumed run is already past it
    # (block F resumes at epoch >= 10 while warmup spans <= 10 epochs), so
    # global_step is seeded from the restored epoch and the branch is inert.
    warmup_steps = warmup_epochs * len(train_loader)
    global_step = start_epoch * len(train_loader)
    train_metric = MulticlassAccuracy(num_classes=num_classes, average="micro").to(device)
    for epoch in range(start_epoch, cfg["epochs"]):
        if cfg.get("moment_aux") and hasattr(model, "aux_weight"):
            model.aux_weight = aux_lambda(epoch)
        if unfreeze_at is not None and epoch == unfreeze_at:
            for p in model.stem.parameters():
                p.requires_grad = True
            print(f"stem unfrozen at epoch {epoch}")
        model.train()
        train_metric.reset()
        loss_sum, n_batches, t0 = 0.0, 0, time.time()
        ce_sum, aux_sum, tapstd_sum = 0.0, 0.0, 0.0
        lr_now = optimizer.param_groups[0]["lr"]
        if warmup_steps and global_step < warmup_steps:
            # log the lr actually USED at this epoch's first step, not the
            # un-warmed cosine value the scheduler happens to hold
            lr_now *= (global_step + 1) / warmup_steps
        # per-STEP linear warmup: the divergence this guards against happens
        # inside the first epoch, so epoch-granular warmup is too coarse. The
        # cosine factor for THIS epoch is whatever LambdaLR just set; warmup
        # scales it, and the next scheduler.step() restores it from base_lrs.
        epoch_lrs = [g["lr"] for g in optimizer.param_groups] if warmup_steps else None
        for x, y in train_loader:
            if warmup_steps and global_step < warmup_steps:
                warm = (global_step + 1) / warmup_steps
                for g, base in zip(optimizer.param_groups, epoch_lrs):
                    g["lr"] = base * warm
            elif warmup_steps and global_step == warmup_steps:
                for g, base in zip(optimizer.param_groups, epoch_lrs):
                    g["lr"] = base
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            y_hard = y
            if mixup_fn is not None:
                # Mixup/CutMix blend the INPUTS, so the aux target is computed
                # from the mixed image the network actually sees (consistent);
                # targets become soft, so train_acc is scored against the
                # pre-mixup labels and is only indicative.
                x, y = mixup_fn(x, y)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(x)
                loss = (soft_criterion(logits, y) if mixup_fn is not None
                        else criterion(logits, y))
                ce_sum += loss.item()
                # moment-aux prior: soft MSE regression of an intermediate
                # feature onto the fixed moment maps (deployed path unchanged).
                aux = getattr(model, "last_aux", None)
                if aux is not None:
                    loss = loss + model.aux_weight * aux
                    aux_sum += aux.item()
                    with torch.no_grad():
                        tapstd_sum += float(
                            model._feats[model.taps[0]].detach().float().std()
                        )
            scaler.scale(loss).backward()
            if clip_grad is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, clip_grad)
            scaler.step(optimizer)
            scaler.update()
            global_step += 1
            # moment-aux: block the scale degeneracy (see aux.py) by restoring
            # the aux head's weight norm after each step.
            if hasattr(model, "project_heads"):
                model.project_heads()
            loss_sum += loss.item()
            n_batches += 1
            train_metric.update(logits.detach(), y_hard)
        scheduler.step()

        test_acc = evaluate(model, test_loader, device, num_classes)
        train_acc = train_metric.compute().item()
        norms = conv1_group_norms(model)
        is_aux = cfg.get("moment_aux") and hasattr(model, "aux_weight")
        writer.writerow(
            [epoch, f"{loss_sum / max(n_batches, 1):.6f}", f"{train_acc:.6f}",
             f"{test_acc:.6f}", f"{lr_now:.6f}", f"{time.time() - t0:.1f}"]
            + [f"{norms[g]:.4f}" if g in norms else ""
               for g in ("identity", "gabor", "zernike")]
            + ([f"{ce_sum / max(n_batches, 1):.6f}",
                f"{aux_sum / max(n_batches, 1):.6f}",
                f"{model.aux_weight:.6f}",
                f"{tapstd_sum / max(n_batches, 1):.4f}"] if is_aux
               else ["", "", "", ""])
        )
        csv_file.flush()
        if test_acc > best_acc:
            best_acc = test_acc
            # atomic: write-then-rename, so a reader (or a crash) can never
            # see a half-written checkpoint (the corrupt-cnx signature).
            _tmp = os.path.join(out_dir, "best.pt.tmp")
            torch.save(model.state_dict(), _tmp)
            os.replace(_tmp, os.path.join(out_dir, "best.pt"))
        # TRAJECTORY CHECKPOINTS (2026-08-23, block A of the limitations
        # campaign): `save_every: N` writes ckpt_epXXX.pt at every N-th
        # epoch (1-based, so N=20 over 200 epochs gives ep020..ep200). Off by
        # default -- no existing cell changes behaviour. These are weights
        # only (like best/last), intended for linear_probe.py --ckpt, and
        # they are NOT resume states.
        _se = int(cfg.get("save_every", 0) or 0)
        if _se > 0 and (epoch + 1) % _se == 0:
            _tmp = os.path.join(out_dir, f"ckpt_ep{epoch + 1:03d}.pt.tmp")
            torch.save(model.state_dict(), _tmp)
            os.replace(_tmp, os.path.join(out_dir, f"ckpt_ep{epoch + 1:03d}.pt"))
        print(
            f"epoch {epoch + 1}/{cfg['epochs']} loss {loss_sum / max(n_batches, 1):.4f} "
            f"train {train_acc:.4f} test {test_acc:.4f}"
        )
        if (resume_every and (epoch + 1) % resume_every == 0
                and epoch + 1 < cfg["epochs"]):
            _save_resume_state(
                resume_path, model, optimizer, scheduler, scaler, train_loader,
                epoch + 1, best_acc, wall_before + time.time() - t_start,
                resume_history, out_dir)
    csv_file.close()
    _tmp = os.path.join(out_dir, "last.pt.tmp")
    torch.save(model.state_dict(), _tmp)
    os.replace(_tmp, os.path.join(out_dir, "last.pt"))

    final = {
        "name": name,
        "seed": args.seed,
        "config": cfg,
        "final_test_acc": test_acc,
        "best_test_acc": best_acc,
        "accounting": accounting,
        "wall_seconds": wall_before + time.time() - t_start,
        "torch_version": torch.__version__,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "amp": use_amp,
    }
    if resume_history:
        final["resumed_from_epoch"] = resume_history[-1]
        final["resume_history"] = resume_history
    with open(os.path.join(out_dir, "final.json"), "w") as f:
        json.dump(final, f, indent=2)
    if resume_every:
        for fn in (resume_path, resume_path + ".tmp", os.path.join(out_dir, "best.pt.resume")):
            if os.path.exists(fn):
                os.remove(fn)
    print(f"done: final {test_acc:.4f} best {best_acc:.4f} -> {out_dir}")


if __name__ == "__main__":
    main()
