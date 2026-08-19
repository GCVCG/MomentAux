"""MAE pretraining on a committed subset's IMAGES (no labels) -- the
masked/predictive SSL comparator (requested at review; the study's SSL set
was contrastive SimCLR, negative-free SimSiam, and self-distilled DINO, all
invariance-based; masked reconstruction may supply a DIFFERENT currency).

Design mirrors simclr_pretrain.py's contract exactly: pretrain the CELL's
own backbone on the SAME committed subset images for 200 epochs (2x the
baseline's compute), save encoder weights under the supervised model's key
names, and let train.py `init_from` (diag-only) consume the checkpoint.

DEVIATIONS FROM THE MAE PAPER, stated rather than hidden (He et al. train
ViT-L at 224px on ImageNet for 1600 epochs with a large decoder):
  - ViT-tiny at the study's input size (32px, patch 4, 8x8 = 64 tokens);
  - decoder is 2 blocks at dim 128 with 4 heads (a small decoder is the
    paper's own recommendation direction, ours is smaller still);
  - 200 epochs at the recipe's batch size, matching the study's SSL budget
    convention, not MAE's;
  - per-patch normalized-pixel targets (the paper's norm_pix_loss=True).
So this is "MAE-style masked reconstruction under the study's budget", not
a verbatim MAE reproduction -- never describe it as the latter.

  python scripts/mae_pretrain.py --config configs/<cell>.yaml --seed 0 \
      --out runs/mae_pre_5pct/seed0/pretrain.pt
"""

import argparse
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


def mae_transform(dataset):
    """MAE uses light augmentation (crop + flip only), per the paper."""
    mean, std = data_mod.STATS[dataset]
    size = data_mod.IMAGE_SIZE[dataset]
    return transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), transforms.Normalize(mean, std),
    ])


class MAEDecoder(nn.Module):
    def __init__(self, enc_dim, n_tokens, patch_dim, dim=128, depth=2, heads=4):
        super().__init__()
        self.embed = nn.Linear(enc_dim, dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.zeros(1, n_tokens, dim))
        layer = nn.TransformerEncoderLayer(dim, heads, dim * 4,
                                           batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, depth)
        self.norm = nn.LayerNorm(dim)
        self.pred = nn.Linear(dim, patch_dim)
        nn.init.normal_(self.mask_token, std=0.02)
        nn.init.normal_(self.pos, std=0.02)

    def forward(self, z_vis, ids_keep, ids_restore):
        B, n_tok = z_vis.shape[0], self.pos.shape[1]
        x = self.embed(z_vis)
        mask = self.mask_token.expand(B, n_tok - x.shape[1], -1)
        full = torch.cat([x, mask], dim=1)
        full = torch.gather(
            full, 1, ids_restore.unsqueeze(-1).expand(-1, -1, full.shape[-1]))
        full = full + self.pos
        return self.pred(self.norm(self.blocks(full)))


def patchify(imgs, p):
    B, C, H, W = imgs.shape
    h = H // p
    x = imgs.reshape(B, C, h, p, h, p).permute(0, 2, 4, 3, 5, 1)
    return x.reshape(B, h * h, p * p * C)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--lr", type=float, default=1.5e-3)
    ap.add_argument("--mask-ratio", type=float, default=0.75)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if cfg["dataset"] not in ("cifar100",):
        raise ValueError("mae_pretrain is whitelisted for cifar100 only; "
                         "verify the transform-swap pattern before extending")
    if cfg["backbone"] != "vit_tiny":
        raise ValueError("mae_pretrain requires a ViT backbone (token access)")
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ds = data_mod.build_dataset(cfg["dataset"], args.data_root, train=True,
                                subset_pct=cfg.get("subset_pct"))
    base = ds.dataset if hasattr(ds, "dataset") else ds
    base.transform = mae_transform(cfg["dataset"])
    loader = DataLoader(ds, batch_size=cfg.get("batch_size", 128), shuffle=True,
                        num_workers=cfg.get("num_workers", 4), drop_last=True,
                        generator=torch.Generator().manual_seed(args.seed))

    size = data_mod.IMAGE_SIZE[cfg["dataset"]]
    model = build_model(cfg["backbone"], "none",
                        num_classes=data_mod.NUM_CLASSES[cfg["dataset"]],
                        small_input=cfg.get("small_input", True),
                        image_size=size).to(device)
    vit = model.net
    p = vit.patch_embed.patch_size[0]
    n_tok = (size // p) ** 2
    enc_dim = vit.embed_dim
    dec = MAEDecoder(enc_dim, n_tok, p * p * 3).to(device)

    params = [q for q in vit.parameters()] + [q for q in dec.parameters()]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.05,
                            betas=(0.9, 0.95))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    n_keep = int(n_tok * (1 - args.mask_ratio))

    steps = 0
    for epoch in range(args.epochs):
        vit.train(); dec.train()
        total, nb = 0.0, 0
        for imgs, _ in loader:
            imgs = imgs.to(device, non_blocking=True)
            B = imgs.shape[0]
            noise = torch.rand(B, n_tok, device=device)
            ids_shuffle = noise.argsort(dim=1)
            ids_restore = ids_shuffle.argsort(dim=1)
            ids_keep = ids_shuffle[:, :n_keep]
            with torch.autocast("cuda", enabled=device.type == "cuda"):
                x = vit.patch_embed(imgs)              # (B, n_tok, D)
                x = x + vit.pos_embed[:, 1:, :]
                x = torch.gather(
                    x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, enc_dim))
                cls = vit.cls_token + vit.pos_embed[:, :1, :]
                x = torch.cat([cls.expand(B, -1, -1), x], dim=1)
                for blk in vit.blocks:
                    x = blk(x)
                x = vit.norm(x)
                pred = dec(x[:, 1:, :], ids_keep, ids_restore)
                tgt = patchify(imgs, p)
                mu = tgt.mean(dim=-1, keepdim=True)
                var = tgt.var(dim=-1, keepdim=True)
                tgt = (tgt - mu) / (var + 1e-6).sqrt()
                loss_map = ((pred.float() - tgt.float()) ** 2).mean(dim=-1)
                masked = torch.ones(B, n_tok, device=device)
                masked.scatter_(1, ids_keep, 0.0)
                loss = (loss_map * masked).sum() / masked.sum()
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += loss.item(); nb += 1; steps += 1
        sched.step()
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"mae epoch {epoch+1}/{args.epochs} "
                  f"loss {total/max(nb,1):.4f}", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    sd = {k: v for k, v in model.state_dict().items()
          if k.startswith("net.") and not k.startswith("net.head")}
    torch.save(sd, args.out)
    print(f"saved {len(sd)} tensors ({steps} steps) -> {args.out}")


if __name__ == "__main__":
    main()
