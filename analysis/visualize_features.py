"""Observability for MomentAux checkpoint pairs: what does the prior DO?

Four figures per baseline/aux pair (one seed, deterministic):

1. tsne_<cell>.png      -- t-SNE of penultimate test features, baseline vs aux
                           side by side, colored by class (Okabe-Ito, CVD-safe,
                           8 classes subsampled for legibility). Panel titles
                           carry the FULL-space silhouette score over ALL
                           classes -- the quantitative clustering statement; the
                           2-D map is only an illustration of it.
2. heatmaps_<cell>.png  -- per image: input | moment magnitude target (channel
                           mean) | baseline layer3 energy | aux layer3 energy.
                           Shows the spatial structure the aux head pulls the
                           tapped features toward.
3. cam_<cell>.png       -- CAM overlays (GAP->fc makes CAM exact for ResNet):
                           where each classifier looks for its predicted class.
4. bank_gabor.png       -- the 9 quadrature pairs (even/odd) of the energy-
                           magnitude bank: the prior itself.

This is a DIAGNOSTIC (like linear_probe.py): it reads whole-test-set images and
never touches training; nothing here feeds a headline table.

    python analysis/visualize_features.py \
        --pair tin_none_1pct tin_aux_1pct --out docs/viz
"""

import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import data as data_mod
from momentstem import build_model

# Okabe & Ito (2008): the standard CVD-safe categorical palette, fixed order.
OKABE_ITO = ["#E69F00", "#56B4E9", "#009E73", "#F0E442",
             "#0072B2", "#D55E00", "#CC79A7", "#999999"]


def load_pair(none_cell, aux_cell, seed_dir, device):
    models = {}
    for cell in (none_cell, aux_cell):
        for d in ("diagnostics", "ablations_full"):
            path = f"configs/{d}/{cell}.yaml"
            if os.path.exists(path):
                break
        cfg = yaml.safe_load(open(path))
        m = build_model(
            cfg["backbone"], cfg.get("stem", "none"),
            num_classes=data_mod.NUM_CLASSES[cfg["dataset"]],
            small_input=cfg.get("small_input", True),
            stem_kernel_size=cfg.get("stem_kernel_size", 11),
            stem_kwargs=cfg.get("stem_kwargs"),
            moment_aux=cfg.get("moment_aux"),
        ).to(device)
        ckpt = f"runs/{cell}/{seed_dir}/best.pt"
        m.load_state_dict(torch.load(ckpt, map_location=device))
        m.eval()
        models[cell] = (m, cfg)
    return models


@torch.no_grad()
def penultimate(model, loader, device):
    feats, ys = [], []
    for x, y in loader:
        x = x.to(device)
        f = model.net.global_pool(model.net.forward_features(model.stem(x)))
        feats.append(f.float().cpu())
        ys.append(y)
    return torch.cat(feats).numpy(), torch.cat(ys).numpy()


def denorm(x, dataset):
    mean, std = data_mod.STATS[dataset]
    img = x.cpu() * torch.tensor(std).view(3, 1, 1) + torch.tensor(mean).view(3, 1, 1)
    return img.clamp(0, 1).permute(1, 2, 0).numpy()


def fig_tsne(models, loader, device, out, cell, n_classes):
    from sklearn.manifold import TSNE
    from sklearn.metrics import silhouette_score

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    for ax, (name, (model, cfg)) in zip(axes, models.items()):
        feats, ys = penultimate(model, loader, device)
        # silhouette on the FULL feature space, ALL classes (the real statement)
        sil = silhouette_score(feats, ys, metric="cosine", sample_size=None)
        keep_cls = np.arange(n_classes)
        mask = np.isin(ys, keep_cls)
        emb = TSNE(n_components=2, init="pca", perplexity=30,
                   random_state=0).fit_transform(feats[mask])
        for i, c in enumerate(keep_cls):
            m = ys[mask] == c
            ax.scatter(emb[m, 0], emb[m, 1], s=7, color=OKABE_ITO[i],
                       label=f"class {c}", linewidths=0)
        kind = "aux" if cfg.get("moment_aux") else "baseline"
        ax.set_title(f"{kind}: silhouette (all classes, full dim) = {sil:.3f}")
        ax.set_xticks([]), ax.set_yticks([])
        for s in ax.spines.values():
            s.set_alpha(0.25)
    axes[0].legend(loc="best", fontsize=7, framealpha=0.6, handletextpad=0.2)
    fig.suptitle(f"{cell}: penultimate test features, t-SNE "
                 f"(first {n_classes} classes shown)")
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"tsne_{cell}.png"), dpi=150)
    plt.close(fig)
    return sil


@torch.no_grad()
def fig_heatmaps(models, test_ds, device, out, cell, dataset, n_images=4):
    (bname, (base, _)), (aname, (aux, acfg)) = models.items()
    if not hasattr(aux, "target"):
        raise RuntimeError(f"{aname} is not a MomentAux model")
    # capture layer3 on both models
    feats = {}
    hooks = []
    for tag, m in (("base", base), ("aux", aux)):
        mod = dict(m.net.named_modules())["layer3"]
        hooks.append(mod.register_forward_hook(
            lambda _m, _i, o, tag=tag: feats.__setitem__(tag, o)))
    xs = torch.stack([test_ds[i][0] for i in range(n_images)]).to(device)
    base.net(base.stem(xs))
    aux.net(aux.stem(xs))
    tgt = aux.target(xs)                                   # (B, 9, H, W)
    tgt = F.adaptive_avg_pool2d(tgt, feats["aux"].shape[-2:])
    for h in hooks:
        h.remove()

    cols = ["input", "moment target (ch. mean)", "baseline layer3 energy",
            "aux layer3 energy"]
    fig, axes = plt.subplots(n_images, 4, figsize=(10.5, 2.6 * n_images))
    for r in range(n_images):
        maps = [None, tgt[r].mean(0), feats["base"][r].abs().mean(0),
                feats["aux"][r].abs().mean(0)]
        for c in range(4):
            ax = axes[r, c]
            if c == 0:
                ax.imshow(denorm(xs[r], dataset))
            else:
                m = maps[c].float().cpu().numpy()
                ax.imshow((m - m.min()) / (m.ptp() + 1e-9), cmap="magma")
            ax.set_xticks([]), ax.set_yticks([])
            if r == 0:
                ax.set_title(cols[c], fontsize=9)
    fig.suptitle(f"{cell}: what the aux head pulls layer3 toward")
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"heatmaps_{cell}.png"), dpi=150)
    plt.close(fig)


@torch.no_grad()
def fig_cam(models, test_ds, device, out, cell, dataset, n_images=4):
    """GAP->fc ResNets make CAM exact: cam_c = sum_k w[c,k] * feat_k(h,w)."""
    xs = torch.stack([test_ds[i][0] for i in range(n_images)]).to(device)
    fig, axes = plt.subplots(n_images, 3, figsize=(8, 2.6 * n_images))
    cols = ["input", "baseline CAM", "aux CAM"]
    for col, (name, (model, cfg)) in enumerate(models.items(), start=1):
        fmap = model.net.forward_features(model.stem(xs))   # (B, C, h, w)
        logits = model.net.get_classifier()(
            model.net.global_pool(fmap))
        pred = logits.argmax(1)
        w = model.net.get_classifier().weight               # (n_cls, C)
        cam = torch.einsum("bchw,bc->bhw", fmap, w[pred])
        for r in range(n_images):
            m = cam[r].float().cpu().numpy()
            m = (m - m.min()) / (m.ptp() + 1e-9)
            m = np.array(torch.nn.functional.interpolate(
                torch.tensor(m)[None, None], size=xs.shape[-2:],
                mode="bilinear", align_corners=False)[0, 0])
            axes[r, col].imshow(denorm(xs[r], dataset))
            axes[r, col].imshow(m, cmap="magma", alpha=0.5)
            axes[r, col].set_xticks([]), axes[r, col].set_yticks([])
            if r == 0:
                axes[r, col].set_title(cols[col], fontsize=9)
    for r in range(n_images):
        axes[r, 0].imshow(denorm(xs[r], dataset))
        axes[r, 0].set_xticks([]), axes[r, 0].set_yticks([])
        if r == 0:
            axes[r, 0].set_title(cols[0], fontsize=9)
    fig.suptitle(f"{cell}: class-activation maps (predicted class)")
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"cam_{cell}.png"), dpi=150)
    plt.close(fig)


def fig_bank(out):
    """The prior itself: 9 quadrature pairs of the energy-magnitude bank."""
    from momentstem import EnergyStem

    stem = EnergyStem(feature_type="magnitude")
    even, odd = stem.even.squeeze(1), stem.odd.squeeze(1)   # (n_pairs, k, k)
    n = even.shape[0]
    fig, axes = plt.subplots(2, n, figsize=(1.45 * n, 3.4))
    lim = float(max(even.abs().max(), odd.abs().max()))
    for i in range(n):
        for r, bank in enumerate((even, odd)):
            ax = axes[r, i]
            ax.imshow(bank[i].numpy(), cmap="RdBu_r", vmin=-lim, vmax=lim)
            ax.set_xticks([]), ax.set_yticks([])
        axes[0, i].set_title(f"pair {i}", fontsize=8)
    axes[0, 0].set_ylabel("even", fontsize=9)
    axes[1, 0].set_ylabel("odd", fontsize=9)
    fig.suptitle(f"energy-magnitude bank: {n} complex-Gabor quadrature pairs "
                 "(magnitude of pair responses = the aux target)")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "bank_gabor.png"), dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", nargs=2, metavar=("NONE_CELL", "AUX_CELL"),
                    required=True)
    ap.add_argument("--out", default="docs/viz")
    ap.add_argument("--seed-dir", default="seed0")
    ap.add_argument("--n-classes", type=int, default=8)
    ap.add_argument("--data-root", default="./data")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    none_cell, aux_cell = args.pair
    models = load_pair(none_cell, aux_cell, args.seed_dir, device)
    dataset = models[aux_cell][1]["dataset"]

    # calibrate the aux target exactly as train.py does
    calib = data_mod.calibration_batch(dataset, args.data_root).to(device)
    aux_model = models[aux_cell][0]
    if hasattr(aux_model, "calibrate"):
        aux_model.calibrate(calib)

    test_ds = data_mod.build_dataset(dataset, args.data_root, train=False)
    loader = DataLoader(test_ds, batch_size=512, num_workers=2, shuffle=False)

    cell = aux_cell
    fig_bank(args.out)
    fig_tsne(models, loader, device, args.out, cell, args.n_classes)
    fig_heatmaps(models, test_ds, device, args.out, cell, dataset)
    fig_cam(models, test_ds, device, args.out, cell, dataset)
    print(f"VIZ_DONE {cell} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
