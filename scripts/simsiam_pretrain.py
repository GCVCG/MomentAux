"""SimSiam pretraining on a committed subset's IMAGES (no labels) -- the
SECOND SSL comparator (2026-07-23, user: importance over cost).

WHY A SECOND SSL METHOD: every SSL-vs-prior number in the study is SimCLR,
which is (a) 2020-era and (b) negative-pair-based, so a reviewer can claim the
prior only beats a strawman. SimSiam (Chen & He 2021) is the canonical
negative-FREE method -- stop-gradient + predictor instead of negatives -- and
is reported MORE robust at small batch/data. If the prior's positioning
survives SimSiam too, "SSL comparator was weak" dies as an objection.

Same contract as simclr_pretrain.py: pretrain on the SAME committed subset
images the supervised cell sees, same epochs, then the checkpoint initializes
a standard supervised run (train.py init_from, diag-only). Same CLI so the
worklist wrappers only swap the script name.

Recipe: the paper's CIFAR appendix values -- SGD, base lr 0.03 x bs/256,
cosine, wd 5e-4, momentum 0.9, projection MLP 3-layer (hidden = feat dim,
out 2048, BN on every layer, no ReLU on output), predictor 2-layer
(2048 -> 512 -> 2048, BN+ReLU on hidden). Loss = symmetrized negative cosine
with stop-gradient. Views = the SimCLR view generator (shared code).
"""

import argparse
import math
import os
import sys

import torch
import torch.nn.functional as F
import yaml
from torch import nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import data as data_mod
from momentstem import build_model
from simclr_pretrain import TwoCrop, simclr_transform
from train import set_seed


def projection_mlp(feat_dim, out_dim=2048):
    return nn.Sequential(
        nn.Linear(feat_dim, feat_dim, bias=False), nn.BatchNorm1d(feat_dim),
        nn.ReLU(inplace=True),
        nn.Linear(feat_dim, feat_dim, bias=False), nn.BatchNorm1d(feat_dim),
        nn.ReLU(inplace=True),
        nn.Linear(feat_dim, out_dim, bias=False),
        nn.BatchNorm1d(out_dim, affine=False),   # paper: no affine on output BN
    )


def prediction_mlp(dim=2048, hidden=512):
    return nn.Sequential(
        nn.Linear(dim, hidden, bias=False), nn.BatchNorm1d(hidden),
        nn.ReLU(inplace=True),
        nn.Linear(hidden, dim),
    )


def neg_cosine(p, z):
    return -F.cosine_similarity(p, z.detach(), dim=-1).mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--base-lr", type=float, default=0.03, help="x bs/256 (SGD)")
    ap.add_argument("--adamw-lr", type=float, default=1e-3)
    ap.add_argument("--proj-dim", type=int, default=2048)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    SUPPORTED = ("cifar100", "cifar10", "stl10", "tin", "eurosat", "dtd", "pathmnist", "food101", "cub")
    if cfg["dataset"] not in SUPPORTED:
        raise ValueError(f"simsiam_pretrain supports {SUPPORTED}")
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = data_mod.build_dataset(cfg["dataset"], args.data_root, train=True,
                                subset_pct=cfg.get("subset_pct"))
    base = ds.dataset if hasattr(ds, "dataset") else ds
    base.transform = TwoCrop(simclr_transform(cfg["dataset"]))
    bs = cfg.get("batch_size", 128)
    loader = DataLoader(ds, batch_size=bs, shuffle=True,
                        num_workers=cfg.get("num_workers", 4), drop_last=True,
                        generator=torch.Generator().manual_seed(args.seed))

    model = build_model(cfg["backbone"], "none",
                        num_classes=data_mod.NUM_CLASSES[cfg["dataset"]],
                        small_input=cfg.get("small_input", True),
                        image_size=data_mod.IMAGE_SIZE[cfg["dataset"]]).to(device)
    if hasattr(model.net, "fc"):
        clf_attr, feat_dim = "fc", model.net.fc.in_features
    elif hasattr(model.net, "head"):
        clf_attr, feat_dim = "head", model.net.head.in_features
    else:
        raise ValueError(f"no classifier found on {cfg['backbone']}")
    setattr(model.net, clf_attr, projection_mlp(feat_dim, args.proj_dim).to(device))
    predictor = prediction_mlp(args.proj_dim).to(device)

    params = list(model.parameters()) + list(predictor.parameters())
    if cfg.get("optimizer", "sgd").lower() == "adamw":
        opt = torch.optim.AdamW(params, lr=args.adamw_lr,
                                weight_decay=cfg.get("weight_decay", 0.05))
    else:
        lr = args.base_lr * bs / 256.0
        opt = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler()

    steps = 0
    for epoch in range(args.epochs):
        model.train(); predictor.train()
        tot = 0.0
        for (x1, x2), _ in loader:
            x1, x2 = x1.to(device, non_blocking=True), x2.to(device, non_blocking=True)
            with torch.cuda.amp.autocast():
                z1, z2 = model(x1), model(x2)
                p1, p2 = predictor(z1), predictor(z2)
                loss = neg_cosine(p1, z2) / 2 + neg_cosine(p2, z1) / 2
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            tot += loss.item(); steps += 1
        sched.step()
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"simsiam epoch {epoch+1}/{args.epochs} loss {tot/max(1,len(loader)):.4f}")

    # Save the BACKBONE only (drop projection + predictor): init_from loads
    # with strict=False and a fresh classifier, same contract as simclr.
    sd = {k: v for k, v in model.state_dict().items()
          if not k.startswith(f"net.{clf_attr}")}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(sd, args.out)
    print(f"saved {len(sd)} tensors ({steps} steps) -> {args.out}")


if __name__ == "__main__":
    main()
