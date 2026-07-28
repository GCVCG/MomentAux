"""DINO pretraining on a committed subset's IMAGES (no labels) -- the
ATTENTION-NATIVE SSL comparator (2026-07-23, "modern baselines" batch).

WHY: the ViT positioning claim ("the prior matches/beats SSL on attention")
was measured against SimCLR-on-ViT, but DINO (Caron et al. 2021) is what the
field actually uses to self-supervise ViTs -- self-distillation with EMA
teacher, centering, and multi-crop, no negatives. If the prior's attention
claim survives DINO, the "wrong SSL for ViTs" objection dies.

Same contract as simclr/simsiam_pretrain: same committed subset images, same
epoch budget, checkpoint -> train.py init_from (diag-only). Same CLI.

Faithful pieces: DINO head (3-layer MLP -> l2-normalized bottleneck ->
weight-normed prototype layer, K=4096 for this scale), EMA teacher (cosine
0.996 -> 1.0), teacher centering (EMA m=0.9), temperatures 0.04 (teacher) /
0.1 (student), 2 global + 4 local crops, teacher sees globals only.
DELIBERATE DEVIATION, stated: local crops are SMALL-SCALE RandomResizedCrops
(scale 0.05-0.4) RESIZED TO FULL RESOLUTION rather than half-res -- our ViT
has a fixed patch grid per image_size, so mixed-resolution crops would need
pos-embed interpolation. Content of the multi-crop signal (global-local scale
asymmetry) is preserved; resolution asymmetry is not. Do not describe this
as verbatim DINO.
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
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import data as data_mod
from momentstem import build_model
from train import set_seed


class MultiCrop:
    def __init__(self, size, mean, std, n_local=4):
        flip_jitter = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
        ]
        norm = [transforms.ToTensor(), transforms.Normalize(mean, std)]
        self.global_tf = transforms.Compose(
            [transforms.RandomResizedCrop(size, scale=(0.4, 1.0))] + flip_jitter + norm)
        self.local_tf = transforms.Compose(
            [transforms.RandomResizedCrop(size, scale=(0.05, 0.4))] + flip_jitter + norm)
        self.n_local = n_local

    def __call__(self, img):
        return ([self.global_tf(img), self.global_tf(img)]
                + [self.local_tf(img) for _ in range(self.n_local)])


class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim=4096, hidden=2048, bottleneck=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, bottleneck),
        )
        self.last = nn.utils.weight_norm(nn.Linear(bottleneck, out_dim, bias=False))
        self.last.weight_g.data.fill_(1)
        self.last.weight_g.requires_grad = False

    def forward(self, x):
        return self.last(F.normalize(self.mlp(x), dim=-1))


def strip_classifier(model, backbone):
    attr = "fc" if hasattr(model.net, "fc") else "head"
    dim = getattr(model.net, attr).in_features
    setattr(model.net, attr, nn.Identity())
    return attr, dim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=5e-4, help="AdamW (DINO default scale)")
    ap.add_argument("--out-dim", type=int, default=4096)
    ap.add_argument("--teacher-temp", type=float, default=0.04)
    ap.add_argument("--student-temp", type=float, default=0.1)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    SUPPORTED = ("cifar100", "cifar10", "stl10", "tin", "eurosat", "dtd", "pathmnist", "food101", "cub")
    if cfg["dataset"] not in SUPPORTED:
        raise ValueError(f"dino_pretrain supports {SUPPORTED}")
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mean, std = data_mod.STATS[cfg["dataset"]]
    size = data_mod.IMAGE_SIZE[cfg["dataset"]]
    ds = data_mod.build_dataset(cfg["dataset"], args.data_root, train=True,
                                subset_pct=cfg.get("subset_pct"))
    base = ds.dataset if hasattr(ds, "dataset") else ds
    base.transform = MultiCrop(size, mean, std)
    bs = cfg.get("batch_size", 128)
    loader = DataLoader(ds, batch_size=bs, shuffle=True,
                        num_workers=cfg.get("num_workers", 4), drop_last=True,
                        generator=torch.Generator().manual_seed(args.seed))

    def make():
        m = build_model(cfg["backbone"], "none",
                        num_classes=data_mod.NUM_CLASSES[cfg["dataset"]],
                        small_input=cfg.get("small_input", True),
                        image_size=size).to(device)
        _, dim = strip_classifier(m, cfg["backbone"])
        return m, DINOHead(dim, args.out_dim).to(device)

    student, s_head = make()
    teacher, t_head = make()
    teacher.load_state_dict(student.state_dict())
    t_head.load_state_dict(s_head.state_dict())
    for p in list(teacher.parameters()) + list(t_head.parameters()):
        p.requires_grad = False

    params = list(student.parameters()) + list(s_head.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.04)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    center = torch.zeros(1, args.out_dim, device=device)
    scaler = torch.cuda.amp.GradScaler()
    total_epochs, steps = args.epochs, 0

    for epoch in range(total_epochs):
        student.train(); s_head.train()
        m = 0.996 + (1 - 0.996) * (1 - math.cos(math.pi * epoch / total_epochs)) / 2
        tot = 0.0
        for crops, _ in loader:
            crops = [c.to(device, non_blocking=True) for c in crops]
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    t_out = [t_head(teacher(c)) for c in crops[:2]]
                s_out = [s_head(student(c)) for c in crops]
                loss, n = 0.0, 0
                for ti, t in enumerate(t_out):
                    t_prob = F.softmax((t - center) / args.teacher_temp, dim=-1)
                    for si, s in enumerate(s_out):
                        if si == ti:
                            continue
                        loss = loss + torch.sum(
                            -t_prob * F.log_softmax(s / args.student_temp, dim=-1),
                            dim=-1).mean()
                        n += 1
                loss = loss / n
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 3.0)
            scaler.step(opt); scaler.update()
            with torch.no_grad():
                for ps, pt in zip(student.parameters(), teacher.parameters()):
                    pt.mul_(m).add_(ps, alpha=1 - m)
                for ps, pt in zip(s_head.parameters(), t_head.parameters()):
                    pt.mul_(m).add_(ps, alpha=1 - m)
                batch_center = torch.cat(t_out).mean(dim=0, keepdim=True)
                center = center * 0.9 + batch_center * 0.1
            tot += loss.item(); steps += 1
        sched.step()
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"dino epoch {epoch+1}/{total_epochs} loss {tot/max(1,len(loader)):.4f} m {m:.4f}")

    # Save the TEACHER backbone (DINO convention) minus head/classifier.
    sd = teacher.state_dict()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(sd, args.out)
    print(f"saved {len(sd)} tensors ({steps} steps) -> {args.out}")


if __name__ == "__main__":
    main()
