"""Does the prior give a small ViT the local structure it fails to learn?

The paper's largest claim is that attention at small scale carries a feature
deficit a fixed oriented-energy target can fill. The evidence so far is
entirely accuracy and linear evaluation. This measures the mechanism the claim
implies, using the standard ViT diagnostic:

  MEAN ATTENTION DISTANCE -- for every head, the average spatial distance (in
  patch units) between a query patch and the patches it attends to, weighted by
  the attention itself. A convolution-like head has small distance; a head that
  attends everywhere uniformly sits near the distance of a random pair.

Dosovitskiy et al. report that well-trained ViTs have a MIX: some early heads
are highly local, and locality decreases with depth. A ViT trained from scratch
on a few thousand images has no reason to discover that. If the prior supplies
the missing structure, its early heads should be MORE local than the baseline's
at matched depth. If instead the prior helps by some route that has nothing to
do with spatial structure, the two curves will lie on top of each other -- which
is a real possible outcome and is why this is worth measuring rather than
asserting.

Attention weights are recomputed from each block's own qkv projection rather
than read from timm's forward, which uses a fused kernel that never
materializes them.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data as data_mod
from momentstem import build_model


@torch.no_grad()
def attention_distance(model, loader, device, max_batches=8):
    """Per-block, per-head mean attention distance in patch units."""
    net = model.net
    blocks = net.blocks
    captured = {}

    def mk_hook(i):
        def hook(_mod, inp):
            captured[i] = inp[0]
        return hook

    handles = [b.register_forward_pre_hook(mk_hook(i)) for i, b in enumerate(blocks)]

    sums = None
    n_seen = 0
    try:
        for bi, (x, _) in enumerate(loader):
            if bi >= max_batches:
                break
            x = x.to(device)
            model(x)

            per_block = []
            for i, blk in enumerate(blocks):
                h = blk.norm1(captured[i])
                attn = blk.attn
                B, N, C = h.shape
                nh = attn.num_heads
                qkv = attn.qkv(h).reshape(B, N, 3, nh, C // nh).permute(2, 0, 3, 1, 4)
                q, k = qkv[0], qkv[1]
                # timm keeps optional q/k norms; they are Identity unless used.
                q = attn.q_norm(q) if hasattr(attn, "q_norm") else q
                k = attn.k_norm(k) if hasattr(attn, "k_norm") else k
                a = (q * attn.scale) @ k.transpose(-2, -1)
                a = a.softmax(dim=-1)                      # B, heads, N, N

                n_pref = N - int(np.sqrt(N - 1)) ** 2      # CLS (and any extra)
                a = a[:, :, n_pref:, n_pref:]              # patch-to-patch only
                g = int(np.sqrt(a.shape[-1]))
                if g * g != a.shape[-1]:
                    raise RuntimeError(f"token grid not square: {a.shape[-1]}")
                a = a / a.sum(-1, keepdim=True).clamp_min(1e-9)

                idx = torch.arange(g, device=device)
                rr, cc = torch.meshgrid(idx, idx, indexing="ij")
                pos = torch.stack([rr.flatten(), cc.flatten()], 1).float()
                d = torch.cdist(pos, pos)                  # g^2 x g^2
                # weighted mean distance, averaged over queries and images
                per_block.append((a * d).sum(-1).mean(dim=(0, 2)).float().cpu())

            stacked = torch.stack(per_block)               # blocks x heads
            sums = stacked if sums is None else sums + stacked
            n_seen += 1
    finally:
        for h in handles:
            h.remove()
    return (sums / max(n_seen, 1)).numpy()


def run_cell(cell, cfg_path, run_root, data_root, device, ckpt="best.pt"):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    ds = cfg["dataset"]
    test_ds = data_mod.build_dataset(ds, data_root, train=False)
    loader = DataLoader(test_ds, batch_size=128, num_workers=3, shuffle=False)

    out = []
    run_dir = os.path.join(run_root, cell)
    for seed_dir in sorted(d for d in os.listdir(run_dir) if d.startswith("seed")):
        path = os.path.join(run_dir, seed_dir, ckpt)
        if not os.path.exists(path):
            continue
        model = build_model(
            cfg["backbone"], cfg.get("stem", "none"),
            num_classes=data_mod.NUM_CLASSES[ds],
            small_input=cfg.get("small_input", True),
            stem_kernel_size=cfg.get("stem_kernel_size", 11),
            stem_kwargs=cfg.get("stem_kwargs"), head_pool=cfg.get("head_pool"),
            head=cfg.get("head"), moment_aux=cfg.get("moment_aux"),
            image_size=data_mod.IMAGE_SIZE[ds],
            in_channels=data_mod.INPUT_CHANNELS.get(ds, 3),
        ).to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        out.append(attention_distance(model, loader, device))
        print(f"  {cell}/{seed_dir}: "
              f"block means {np.round(out[-1].mean(1), 2)}", flush=True)
        del model
        torch.cuda.empty_cache()
    return np.stack(out)          # seeds x blocks x heads


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--run-root", default="runs")
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.spec) as f:
        spec = json.load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    res = {"_spec": spec, "arms": {}}
    for label, d in spec["arms"].items():
        print(f"arm {label}: {d['cell']}", flush=True)
        a = run_cell(d["cell"], d["config"], args.run_root, args.data_root, device)
        res["arms"][label] = {
            "cell": d["cell"],
            "per_seed_block_head": a.tolist(),
            "block_mean": a.mean(axis=(0, 2)).tolist(),
            "block_min_head": a.min(axis=2).mean(axis=0).tolist(),
            "overall_mean": float(a.mean()),
        }

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)

    print("\n=== mean attention distance (patch units), by block ===")
    for label, r in res["arms"].items():
        print(f"  {label:14s} " + " ".join(f"{v:5.2f}" for v in r["block_mean"]))
    print("\n=== most local head in each block ===")
    for label, r in res["arms"].items():
        print(f"  {label:14s} " + " ".join(f"{v:5.2f}" for v in r["block_min_head"]))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
