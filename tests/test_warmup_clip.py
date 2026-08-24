"""`warmup_epochs` / `clip_grad` (train.py): the block-F stabilizers.

Added 2026-08-24 after every ViT-L/16 cell of block F ended at NaN. Both
options are RECIPE DEVIATIONS (diag-gated) and both must leave the default
code path byte-unchanged. These tests pin:
  * the LambdaLR used when warmup is on reproduces CosineAnnealingLR EXACTLY
    at every epoch, so turning warmup on does not silently change the decay;
  * the per-step warmup factor is linear over warmup_epochs * steps_per_epoch
    optimizer steps and lands exactly on the cosine value afterwards;
  * clip_grad actually bounds the global grad norm;
  * the defaults are inert: the warmup branch is guarded by `if warmup_steps`
    and the clip branch by `if clip_grad is not None`, and an absent key gives
    0 / None;
  * both are refused on a non-diag config name.
CPU only, a fraction of a second."""

import math
import re

import torch
from torch import nn

SRC = open("train.py").read()


def test_lambdalr_matches_cosine_annealing_exactly():
    """The warmup path swaps CosineAnnealingLR for a LambdaLR closed form.
    If those differ, every warmup cell would silently run a different decay."""
    for epochs in (10, 100, 200):
        p = nn.Parameter(torch.zeros(1))
        a = torch.optim.SGD([p], lr=0.001)
        b = torch.optim.SGD([p], lr=0.001)
        cos = torch.optim.lr_scheduler.CosineAnnealingLR(a, T_max=epochs)
        E = max(epochs, 1)
        lam = torch.optim.lr_scheduler.LambdaLR(
            b, lambda e: 0.5 * (1.0 + math.cos(math.pi * min(e, E) / E)))
        for _ in range(epochs):
            assert abs(a.param_groups[0]["lr"] - b.param_groups[0]["lr"]) < 1e-12
            cos.step()
            lam.step()


def test_per_step_warmup_is_linear_then_exactly_the_cosine():
    """Replicates train.py's in-loop override on a real optimizer."""
    epochs, steps_per_epoch, warmup_epochs, base = 6, 5, 2, 0.001
    p = nn.Parameter(torch.zeros(1))
    opt = torch.optim.SGD([p], lr=base)
    E = epochs
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda e: 0.5 * (1.0 + math.cos(math.pi * min(e, E) / E)))
    warmup_steps = warmup_epochs * steps_per_epoch
    gstep, seen = 0, []
    for ep in range(epochs):
        epoch_lrs = [g["lr"] for g in opt.param_groups]
        for _ in range(steps_per_epoch):
            if warmup_steps and gstep < warmup_steps:
                warm = (gstep + 1) / warmup_steps
                for g, b in zip(opt.param_groups, epoch_lrs):
                    g["lr"] = b * warm
            elif warmup_steps and gstep == warmup_steps:
                for g, b in zip(opt.param_groups, epoch_lrs):
                    g["lr"] = b
            seen.append((gstep, opt.param_groups[0]["lr"], epoch_lrs[0]))
            gstep += 1
        sched.step()

    # inside warmup: exactly the epoch's cosine lr times (step+1)/warmup_steps,
    # strictly increasing from base/warmup_steps
    for gs, lr, cos_lr in seen[:warmup_steps]:
        assert abs(lr - cos_lr * (gs + 1) / warmup_steps) < 1e-15
    warm_lrs = [lr for _, lr, _ in seen[:warmup_steps]]
    assert warm_lrs == sorted(warm_lrs) and warm_lrs[0] < warm_lrs[-1]
    assert abs(warm_lrs[0] - base * 0.5 * (1 + math.cos(0.0)) / warmup_steps) < 1e-15
    # after warmup: exactly the cosine value, no residue of the override
    for _, lr, cos_lr in seen[warmup_steps:]:
        assert abs(lr - cos_lr) < 1e-15


def test_clip_grad_bounds_the_global_norm():
    lin = nn.Linear(4, 4)
    lin.weight.grad = torch.full_like(lin.weight, 10.0)
    lin.bias.grad = torch.full_like(lin.bias, 10.0)
    before = torch.nn.utils.clip_grad_norm_(list(lin.parameters()), 1.0)
    after = torch.cat([p.grad.flatten() for p in lin.parameters()]).norm()
    assert before > 1.0
    assert abs(after.item() - 1.0) < 1e-5


def test_defaults_are_inert_and_guarded():
    cfg = {}
    warmup_epochs = int(cfg.get("warmup_epochs", 0) or 0)
    clip = cfg.get("clip_grad")
    clip = float(clip) if clip else None
    assert warmup_epochs == 0 and clip is None
    # the default path must not enter either branch, and must keep the
    # original scheduler
    assert "if warmup_steps and global_step < warmup_steps:" in SRC
    assert "if clip_grad is not None:" in SRC
    assert re.search(
        r"else:\n\s+scheduler = torch\.optim\.lr_scheduler\.CosineAnnealingLR\(",
        SRC)
    # warmup_steps is zero when the key is absent, so `if warmup_steps` is False
    assert "warmup_steps = warmup_epochs * len(train_loader)" in SRC


def test_warmup_and_clip_are_diag_only():
    assert re.search(
        r'if \(warmup_epochs or clip_grad\) and not cfg\["name"\]\.startswith\("diag"\)',
        SRC)
