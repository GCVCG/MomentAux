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


# Official CIFAR-100 coarse (superclass) names, in coarse-index order (the
# pickle's alphabetical convention -- matches cifar100_coarse_labels).
CIFAR100_COARSE = [
    "aquatic mammals", "fish", "flowers", "food containers",
    "fruit and vegetables", "household electrical devices",
    "household furniture", "insects", "large carnivores",
    "large man-made outdoor things", "large natural outdoor scenes",
    "large omnivores and herbivores", "medium mammals",
    "non-insect invertebrates", "people", "reptiles", "small mammals",
    "trees", "vehicles 1", "vehicles 2",
]


def _tin_words(data_root):
    """wnid -> first WordNet lemma, from tiny-imagenet's words.txt."""
    root = os.path.join(data_root, "tiny-imagenet-200")
    words = {}
    with open(os.path.join(root, "words.txt")) as f:
        for line in f:
            wnid, names = line.rstrip("\n").split("\t")[:2]
            words[wnid] = names.split(",")[0]
    return root, words


def class_names(dataset, ds, data_root="./data"):
    """Human labels for EVERY dataset in the study (never fall back to bare
    class indices in a committed figure). CIFAR: torchvision .classes;
    cifar100super: the official 20 coarse names; tin family: wnid -> words
    via tiny-imagenet's words.txt (in each variant's own label order);
    cub: classes.txt species names."""
    if dataset == "cifar100super":
        return list(CIFAR100_COARSE)
    if hasattr(ds, "classes") and ds.classes:
        return list(ds.classes)
    try:
        if dataset in ("tin", "tinsuper", "tinsem"):
            root, words = _tin_words(data_root)
            wnids = sorted(d for d in os.listdir(os.path.join(root, "train"))
                           if os.path.isdir(os.path.join(root, "train", d)))
            fine = [words.get(w, w) for w in wnids]
            if dataset == "tinsuper":
                return [f"group {g} ({fine[g*10]},{fine[g*10+1]},..)"
                        for g in range(20)]
            if dataset == "tinsem":
                order = json.load(open(os.path.join(
                    data_mod.SUBSET_DIR, "tin_semantic_order.json")))["order"]
                sem = [words.get(w, w) for w in order]
                return [f"group {g} ({sem[g*10]},{sem[g*10+1]},..)"
                        for g in range(20)]
            return fine
        if dataset in ("tin20", "tin20b"):
            root, words = _tin_words(data_root)
            wn = (data_mod.tin20_wnids(root) if dataset == "tin20"
                  else data_mod.tin20b_wnids(root))
            return [words.get(w, w) for w in wn]
        if dataset == "cub":
            path = os.path.join(data_root, "CUB_200_2011", "classes.txt")
            names = []
            with open(path) as f:
                for line in f:
                    _, name = line.split()
                    names.append(name.split(".", 1)[-1].replace("_", " "))
            return names
    except OSError:
        return None
    return None


@torch.no_grad()
def predictions(model, loader, device):
    preds = []
    for x, _ in loader:
        x = x.to(device)
        f = model.net.forward_features(model.stem(x))
        preds.append(model.net.get_classifier()(model.net.global_pool(f)).argmax(1).cpu())
    return torch.cat(preds).numpy()


def _corr(a, b):
    """Pearson r between two flattened maps."""
    a = a.flatten().float(); b = b.flatten().float()
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / (a.norm() * b.norm() + 1e-9))


@torch.no_grad()
def target_alignment(base, aux, test_ds, device, n=512):
    """Mean Pearson r between the layer3 channel-mean energy map and the
    moment target's channel-mean map, over n test images. THE quantitative
    claim behind the heatmap figure: how much of the target's spatial
    structure do the tapped features carry?"""
    feats, hooks = {}, []
    for tag, m in (("base", base), ("aux", aux)):
        hooks.append(dict(m.net.named_modules())["layer3"].register_forward_hook(
            lambda _m, _i, o, tag=tag: feats.__setitem__(tag, o)))
    rs = {"base": [], "aux": []}
    for i0 in range(0, n, 128):
        xs = torch.stack([test_ds[i][0] for i in range(i0, min(i0 + 128, n))]).to(device)
        base.net(base.stem(xs)); aux.net(aux.stem(xs))
        tgt = F.adaptive_avg_pool2d(aux.target(xs), feats["aux"].shape[-2:]).mean(1)
        for tag in ("base", "aux"):
            emap = feats[tag].abs().mean(1)
            rs[tag] += [_corr(emap[j].cpu(), tgt[j].cpu()) for j in range(len(xs))]
    for h in hooks:
        h.remove()
    return {t: sum(v) / len(v) for t, v in rs.items()}


def denorm(x, dataset):
    mean, std = data_mod.STATS[dataset]
    img = x.cpu() * torch.tensor(std).view(3, 1, 1) + torch.tensor(mean).view(3, 1, 1)
    return img.clamp(0, 1).permute(1, 2, 0).numpy()


def fig_tsne(models, loader, device, out, cell, n_classes, names=None):
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
            lbl = names[c] if names else f"class {c}"
            ax.scatter(emb[m, 0], emb[m, 1], s=7, color=OKABE_ITO[i],
                       label=lbl, linewidths=0)
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
def fig_heatmaps(models, test_ds, device, out, cell, dataset, loader,
                 names=None, n_images=4):
    """Rows are ADVANTAGE CASES: test images the aux model classifies
    correctly and the baseline gets wrong (first n such, deterministic).
    Each map panel is annotated with its Pearson r against the target map;
    the suptitle carries the test-set-wide mean alignment for both models."""
    (bname, (base, _)), (aname, (aux, acfg)) = models.items()
    if not hasattr(aux, "target"):
        raise RuntimeError(f"{aname} is not a MomentAux model")

    ys = np.asarray(getattr(test_ds, "targets"))
    pb = predictions(base, loader, device)
    pa = predictions(aux, loader, device)
    adv = np.flatnonzero((pa == ys) & (pb != ys))[:n_images]
    if len(adv) < n_images:                       # fallback: first images
        adv = np.arange(n_images)
    align = target_alignment(base, aux, test_ds, device)

    feats, hooks = {}, []
    for tag, m in (("base", base), ("aux", aux)):
        hooks.append(dict(m.net.named_modules())["layer3"].register_forward_hook(
            lambda _m, _i, o, tag=tag: feats.__setitem__(tag, o)))
    xs = torch.stack([test_ds[int(i)][0] for i in adv]).to(device)
    base.net(base.stem(xs))
    aux.net(aux.stem(xs))
    tgt = F.adaptive_avg_pool2d(aux.target(xs), feats["aux"].shape[-2:])
    for h in hooks:
        h.remove()

    def nm(c):
        return names[c] if names else f"class {c}"

    cols = ["input (true label)", "moment target", "baseline layer3", "aux layer3"]
    fig, axes = plt.subplots(len(adv), 4, figsize=(11.5, 2.9 * len(adv)))
    axes = np.atleast_2d(axes)
    for r, idx in enumerate(adv):
        t = tgt[r].mean(0).cpu()
        maps = {2: feats["base"][r].abs().mean(0).cpu(),
                3: feats["aux"][r].abs().mean(0).cpu()}
        for c in range(4):
            ax = axes[r, c]
            if c == 0:
                ax.imshow(denorm(xs[r], dataset))
                ax.set_xlabel(nm(int(ys[idx])), fontsize=8)
            elif c == 1:
                m = t.numpy()
                ax.imshow((m - m.min()) / (m.ptp() + 1e-9), cmap="magma")
            else:
                m = maps[c].float().numpy()
                ax.imshow((m - m.min()) / (m.ptp() + 1e-9), cmap="magma")
                r_t = _corr(maps[c], t)
                pred = pb[idx] if c == 2 else pa[idx]
                ok = "\u2713" if pred == ys[idx] else "\u2717"
                ax.set_xlabel(f"pred: {nm(int(pred))} {ok}   r(target)={r_t:+.2f}",
                              fontsize=8)
            ax.set_xticks([]), ax.set_yticks([])
            if r == 0:
                ax.set_title(cols[c], fontsize=9)
    fig.suptitle(
        f"{cell} — rows: aux RIGHT / baseline WRONG.  Test-set mean alignment "
        f"r(layer3, target): baseline {align['base']:+.3f}  vs  aux "
        f"{align['aux']:+.3f}", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"heatmaps_{cell}.png"), dpi=150)
    plt.close(fig)
    return align


@torch.no_grad()
def fig_cam(models, test_ds, device, out, cell, dataset, loader,
            names=None, n_images=4):
    """CAM overlays on the same ADVANTAGE CASES (aux right, baseline wrong),
    each panel labeled with that model's prediction."""
    ys = np.asarray(getattr(test_ds, "targets"))
    items = list(models.items())
    pb = predictions(items[0][1][0], loader, device)
    pa = predictions(items[1][1][0], loader, device)
    adv = np.flatnonzero((pa == ys) & (pb != ys))[:n_images]
    if len(adv) < n_images:
        adv = np.arange(n_images)
    xs = torch.stack([test_ds[int(i)][0] for i in adv]).to(device)

    def nm(c):
        return names[c] if names else f"class {c}"

    fig, axes = plt.subplots(len(adv), 3, figsize=(9, 2.9 * len(adv)))
    axes = np.atleast_2d(axes)
    cols = ["input (true label)", "baseline CAM", "aux CAM"]
    for col, (name, (model, cfg)) in enumerate(items, start=1):
        fmap = model.net.forward_features(model.stem(xs))
        pred = model.net.get_classifier()(model.net.global_pool(fmap)).argmax(1)
        w = model.net.get_classifier().weight
        cam = torch.einsum("bchw,bc->bhw", fmap, w[pred])
        for r, idx in enumerate(adv):
            m = cam[r].float().cpu().numpy()
            m = (m - m.min()) / (m.ptp() + 1e-9)
            m = np.array(torch.nn.functional.interpolate(
                torch.tensor(m)[None, None], size=xs.shape[-2:],
                mode="bilinear", align_corners=False)[0, 0])
            ax = axes[r, col]
            ax.imshow(denorm(xs[r], dataset))
            ax.imshow(m, cmap="magma", alpha=0.5)
            p = int(pred[r]); ok = "\u2713" if p == ys[idx] else "\u2717"
            ax.set_xlabel(f"pred: {nm(p)} {ok}", fontsize=8)
            ax.set_xticks([]), ax.set_yticks([])
            if r == 0:
                ax.set_title(cols[col], fontsize=9)
    for r, idx in enumerate(adv):
        ax = axes[r, 0]
        ax.imshow(denorm(xs[r], dataset))
        ax.set_xlabel(nm(int(ys[idx])), fontsize=8)
        ax.set_xticks([]), ax.set_yticks([])
        if r == 0:
            ax.set_title(cols[0], fontsize=9)
    fig.suptitle(f"{cell}: CAMs on aux-right / baseline-wrong cases", fontsize=10)
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
    names = class_names(dataset, test_ds, args.data_root)
    fig_bank(args.out)
    fig_tsne(models, loader, device, args.out, cell, args.n_classes, names=names)
    align = fig_heatmaps(models, test_ds, device, args.out, cell, dataset,
                         loader, names=names)
    fig_cam(models, test_ds, device, args.out, cell, dataset, loader, names=names)
    print(f"ALIGNMENT {cell}: baseline {align['base']:+.3f} aux {align['aux']:+.3f}",
          flush=True)
    print(f"VIZ_DONE {cell} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
