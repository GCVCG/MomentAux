"""SimCLR pretraining on a committed subset's IMAGES (no labels) -- the
external-positioning control: is MomentAux competitive with SSL pretraining
at matched (actually SSL-favoring) compute?

Design (recorded 2026-07-20): pretrain ResNet-18 with NT-Xent on the SAME
committed subset images the supervised cell sees (the honest data contract:
no outside images), for the SAME 200 epochs / step count as the frozen
recipe. The resulting checkpoint then INITIALIZES a standard supervised run
(train.py `init_from`, diag-only). Total compute = 2x the baseline's; the
champion aux costs ~1.02x. If MomentAux >= SimCLR-init at HALF the compute,
the positioning claim is strong. C100@5%: baseline 25.23, champion 30.53.

  python scripts/simclr_pretrain.py --config configs/<cell>.yaml --seed 0 \
      --out runs/simclr_pre_5pct/seed0/pretrain.pt

cifar100-only by design (the comparison lives at C100@5%); extend the
transform plumbing before pointing it elsewhere.
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
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import data as data_mod
from momentstem import build_model
from train import set_seed


class TwoCrop:
    def __init__(self, tf):
        self.tf = tf

    def __call__(self, img):
        return self.tf(img), self.tf(img)


def simclr_transform(dataset, augment=None):
    """The standard SimCLR view generator (crop/flip/jitter/grayscale).

    :param augment "deit" ADDS the two DeiT components that are MEANINGFUL for
        a contrastive objective -- RandAugment `rand-m9-mstd0.5-inc1` and
        RandomErasing p=0.25 -- so the pretraining stage gets the same view
        strength as the supervised stage in the deit cells. The other three
        DeiT components are INAPPLICABLE and deliberately omitted: Mixup and
        CutMix blend two images, which makes the NT-Xent positive pair
        ambiguous, and label smoothing needs labels NT-Xent does not have.
        So this is "stronger contrastive views", NOT "the DeiT recipe" -- do
        not describe it as the latter.
    """
    mean, std = data_mod.STATS[dataset]
    size = data_mod.IMAGE_SIZE[dataset]
    base = [
        transforms.RandomResizedCrop(size, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply(
            [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
    ]
    normalize = [transforms.ToTensor(), transforms.Normalize(mean, std)]
    if not augment:
        return transforms.Compose(base + normalize)
    if augment != "deit":
        raise ValueError(f"unknown augment {augment!r} (only 'deit')")
    from timm.data.auto_augment import rand_augment_transform
    from timm.data.random_erasing import RandomErasing

    aa = rand_augment_transform(
        "rand-m9-mstd0.5-inc1",
        {"translate_const": int(size * 0.45),
         "img_mean": tuple(int(255 * m) for m in mean)},
    )
    erase = RandomErasing(probability=0.25, mode="pixel", device="cpu")
    return transforms.Compose(base + [aa] + normalize + [erase])


def nt_xent(z, temp):
    """z: (2B, d) L2-normalized; positives are (i, i+B)."""
    n = z.shape[0]
    sim = z @ z.T / temp
    sim.fill_diagonal_(float("-inf"))
    b = n // 2
    target = torch.cat([torch.arange(b, n), torch.arange(0, b)]).to(z.device)
    return F.cross_entropy(sim, target)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True,
                    help="supervised cell config; dataset/subset_pct are read "
                         "from it so pretrain sees EXACTLY that cell's images")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=0.3, help="SGD lr (conv)")
    ap.add_argument("--adamw-lr", type=float, default=1e-3,
                    help="AdamW lr, used when the cell's optimizer is adamw (ViT)")
    ap.add_argument("--augment", default=None, choices=[None, "deit"],
                    help="'deit' strengthens the CONTRASTIVE VIEWS with "
                         "RandAugment + RandomErasing (see simclr_transform); "
                         "Mixup/CutMix/label-smoothing are inapplicable here")
    ap.add_argument("--temp", type=float, default=0.2)
    ap.add_argument("--proj-dim", type=int, default=128)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    # Datasets verified to follow the Subset(base).transform swap pattern and
    # to have STATS/IMAGE_SIZE/NUM_CLASSES entries. tin added 2026-07-23 for
    # the generalization campaign; the domain datasets added same day (their
    # loaders all expose .transform at the top level -- EuroSAT64, Squash64,
    # PathMNIST64 -- so the two-view swap lands correctly; each smoke-tested).
    SUPPORTED = ("cifar100", "tin", "eurosat", "dtd", "pathmnist", "food101")
    if cfg["dataset"] not in SUPPORTED:
        raise ValueError(f"simclr_pretrain supports {SUPPORTED}; got "
                         f"{cfg['dataset']!r} — verify the transform-swap "
                         f"pattern before whitelisting a new dataset")
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = data_mod.build_dataset(cfg["dataset"], args.data_root, train=True,
                                subset_pct=cfg.get("subset_pct"))
    # Swap in the two-view SimCLR transform (Subset(CIFAR100) or CIFAR100).
    base = ds.dataset if hasattr(ds, "dataset") else ds
    base.transform = TwoCrop(simclr_transform(cfg["dataset"], args.augment))
    loader = DataLoader(ds, batch_size=cfg.get("batch_size", 128), shuffle=True,
                        num_workers=cfg.get("num_workers", 4), drop_last=True,
                        generator=torch.Generator().manual_seed(args.seed))

    # Backbone follows the CELL's config so the pretrain and the supervised
    # run share an architecture exactly (resnet18, vit_tiny, ...).
    model = build_model(cfg["backbone"], "none",
                        num_classes=data_mod.NUM_CLASSES[cfg["dataset"]],
                        small_input=cfg.get("small_input", True),
                        image_size=data_mod.IMAGE_SIZE[cfg["dataset"]]).to(device)
    # Swap the classifier for the SimCLR projection MLP. timm names it .fc on
    # ResNets and .head on ViTs; forward_head applies pooling before it either
    # way, so replacing it makes model(x) return the projection.
    if hasattr(model.net, "fc"):
        clf_attr, feat_dim = "fc", model.net.fc.in_features
    elif hasattr(model.net, "head"):
        clf_attr, feat_dim = "head", model.net.head.in_features
    else:
        raise ValueError(f"no classifier found on {cfg['backbone']}")
    setattr(model.net, clf_attr, nn.Sequential(
        nn.Linear(feat_dim, feat_dim), nn.ReLU(inplace=True),
        nn.Linear(feat_dim, args.proj_dim),
    ).to(device))

    # Optimizer follows the cell's recipe: ViTs do not train under SGD lr=0.1
    # (same reason the supervised diagvit cells use AdamW).
    if cfg.get("optimizer", "sgd").lower() == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=args.adamw_lr,
                                weight_decay=cfg.get("weight_decay", 0.05))
        print(f"pretrain optimizer: adamw lr={args.adamw_lr}")
    else:
        opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                              weight_decay=1e-4)
        print(f"pretrain optimizer: sgd lr={args.lr}")
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    steps = 0
    for epoch in range(args.epochs):
        model.train()
        total, nb = 0.0, 0
        for (v1, v2), _ in loader:
            x = torch.cat([v1, v2]).to(device, non_blocking=True)
            with torch.autocast("cuda", enabled=device.type == "cuda"):
                z = F.normalize(model(x), dim=1)
                loss = nt_xent(z.float(), args.temp)
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += loss.item(); nb += 1; steps += 1
        sched.step()
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"pretrain epoch {epoch+1}/{args.epochs} "
                  f"loss {total/max(nb,1):.4f}", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    sd = {k: v for k, v in model.state_dict().items()
          if not k.startswith(f"net.{clf_attr}")}
    torch.save(sd, args.out)
    print(f"saved {len(sd)} tensors ({steps} steps) -> {args.out}")


if __name__ == "__main__":
    main()
