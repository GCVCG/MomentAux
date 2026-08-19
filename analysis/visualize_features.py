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


# Authored at the CAS single-COLUMN width, which is where these figures are
# placed. Authoring wide (9-11.6in) and letting \includegraphics shrink into a
# 3.33in column rendered 8pt labels at under 3pt. Keeping them single-column
# rather than promoting them to figure* matters for placement: full-width
# floats this tall cannot be set near the text that discusses them.
FIGW = 3.33
# The bank is the one panel authored at the FULL text width: it is only ~2in
# tall, so spanning both columns costs nothing in placement, while at column
# width each of the 16 filters was under half an inch and unreadable. The tall
# panels stay single-column for exactly the opposite reason.
FIGW_WIDE = 6.86

def denorm(x, dataset):
    mean, std = data_mod.STATS[dataset]
    img = x.cpu() * torch.tensor(std).view(3, 1, 1) + torch.tensor(mean).view(3, 1, 1)
    return img.clamp(0, 1).permute(1, 2, 0).numpy()


def fig_tsne(models, loader, device, out, cell, n_classes, names=None):
    """Two panels STACKED, single column: side by side at 3.33in each panel was
    1.6in wide, too small to read a projection from, and the silhouette had to
    be abbreviated to fit. Stacking gives each panel the full column width and
    room for the statement that makes the panel scientific rather than
    decorative -- the silhouette is computed on ALL classes in the FULL feature
    space, not on the two projected dimensions being displayed."""
    from sklearn.manifold import TSNE
    from sklearn.metrics import silhouette_score

    fig, axes = plt.subplots(2, 1, figsize=(FIGW, FIGW * 1.02))
    sils, handles = {}, None
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
            lbl = (names[c].replace("_", " ") if names else f"class {c}")
            ax.scatter(emb[m, 0], emb[m, 1], s=3.5, color=OKABE_ITO[i],
                       label=lbl, linewidths=0)
        kind = "prior" if cfg.get("moment_aux") else "baseline"
        sils[kind] = sil
        ax.set_title(f"{kind}: silhouette (all classes, full dim) $=$ "
                     f"{sil:+.3f}", fontsize=6.5, pad=2)
        ax.set_xticks([]), ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_alpha(0.25)
        handles = ax.get_legend_handles_labels()

    fig.legend(*handles, loc="lower center", ncol=4, fontsize=5.4,
               frameon=False, handletextpad=0.15, columnspacing=0.8,
               markerscale=2.2, bbox_to_anchor=(0.5, -0.005))
    fig.subplots_adjust(left=0.02, right=0.98, top=0.955, bottom=0.115,
                        hspace=0.16)
    # pad_inches=0.02 rather than matplotlib's default 0.1in: the default
    # leaves a blank border that becomes a visible white band once the
    # panel is placed at \linewidth in the manuscript.
    fig.savefig(os.path.join(out, f"tsne_{cell}.png"), dpi=300,
                bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return sils.get("baseline", 0.0)


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

    # Four advantage cases as a 2x2 grid of sample BLOCKS, two per row,
    # each block the same four panels the single-column layout used. Short
    # per-panel titles; the caption carries the long forms.
    cols = ["input", "target", "base tap", "prior tap"]
    import matplotlib.gridspec as gridspec
    fig = plt.figure(figsize=(FIGW, FIGW / 8 * 1.52 * 2 + 0.26))
    outer = gridspec.GridSpec(2, 2, figure=fig, wspace=0.10, hspace=0.46,
                              left=0.005, right=0.995, top=0.860,
                              bottom=0.095)
    for k, idx in enumerate(adv):
        inner = gridspec.GridSpecFromSubplotSpec(
            1, 4, subplot_spec=outer[k // 2, k % 2], wspace=0.04)
        t = tgt[k].mean(0).cpu()
        maps = {2: feats["base"][k].abs().mean(0).cpu(),
                3: feats["aux"][k].abs().mean(0).cpu()}
        for c in range(4):
            ax = fig.add_subplot(inner[0, c])
            if c == 0:
                ax.imshow(denorm(xs[k], dataset))
                ax.text(0.5, -0.06, nm(int(ys[idx])), transform=ax.transAxes,
                        ha="center", va="top", fontsize=4.6)
            elif c == 1:
                m = t.numpy()
                ax.imshow((m - m.min()) / (np.ptp(m) + 1e-9), cmap="magma")
            else:
                m = maps[c].float().numpy()
                ax.imshow((m - m.min()) / (np.ptp(m) + 1e-9), cmap="magma")
                r_t = _corr(maps[c], t)
                pred = pb[idx] if c == 2 else pa[idx]
                ok = "\u2713" if pred == ys[idx] else "\u2717"
                ax.text(0.5, -0.06, f"{nm(int(pred))} {ok}\n$r$={r_t:+.2f}",
                        transform=ax.transAxes, ha="center", va="top",
                        fontsize=4.0, linespacing=1.2)
            ax.set_xticks([]), ax.set_yticks([])
            if k < 2:
                ax.set_title(cols[c], fontsize=5.2, pad=2.0)
    fig.suptitle(
        f"test-set mean alignment $r$:  baseline {align['base']:+.3f}"
        f"    prior {align['aux']:+.3f}", fontsize=6.5)
    fig.savefig(os.path.join(out, f"heatmaps_{cell}.png"), dpi=300,
                bbox_inches="tight", pad_inches=0.02)
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

    import matplotlib.gridspec as gridspec
    fig = plt.figure(figsize=(FIGW, FIGW / 6 * 1.30 * 2 + 0.10))
    outer = gridspec.GridSpec(2, 2, figure=fig, wspace=0.09, hspace=0.30,
                              left=0.005, right=0.995, top=0.90,
                              bottom=0.065)
    axes = np.empty((len(adv), 3), dtype=object)
    for k in range(len(adv)):
        inner = gridspec.GridSpecFromSubplotSpec(
            1, 3, subplot_spec=outer[k // 2, k % 2], wspace=0.04)
        for c in range(3):
            axes[k, c] = fig.add_subplot(inner[0, c])
    cols = ["input", "baseline CAM", "prior CAM"]
    for col, (name, (model, cfg)) in enumerate(items, start=1):
        fmap = model.net.forward_features(model.stem(xs))
        pred = model.net.get_classifier()(model.net.global_pool(fmap)).argmax(1)
        w = model.net.get_classifier().weight
        cam = torch.einsum("bchw,bc->bhw", fmap, w[pred])
        for r, idx in enumerate(adv):
            m = cam[r].float().cpu().numpy()
            m = (m - m.min()) / (np.ptp(m) + 1e-9)
            m = np.array(torch.nn.functional.interpolate(
                torch.tensor(m)[None, None], size=xs.shape[-2:],
                mode="bilinear", align_corners=False)[0, 0])
            ax = axes[r, col]
            ax.imshow(denorm(xs[r], dataset))
            ax.imshow(m, cmap="magma", alpha=0.5)
            p = int(pred[r]); ok = "\u2713" if p == ys[idx] else "\u2717"
            ax.text(0.5, -0.03, f"{nm(p)} {ok}", transform=ax.transAxes,
                    ha="center", va="top", fontsize=4.4)
            ax.set_xticks([]), ax.set_yticks([])
            if r < 2:
                ax.set_title(cols[col], fontsize=5.2, pad=2.0)
    for r, idx in enumerate(adv):
        ax = axes[r, 0]
        ax.imshow(denorm(xs[r], dataset))
        ax.text(0.5, -0.03, nm(int(ys[idx])), transform=ax.transAxes,
                ha="center", va="top", fontsize=4.4)
        ax.set_xticks([]), ax.set_yticks([])
        if r < 2:
            ax.set_title(cols[0], fontsize=5.2, pad=2.0)
    fig.savefig(os.path.join(out, f"cam_{cell}.png"), dpi=300,
                bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


# The bank's committed layout, scale-major then orientation-major, exactly
# the order quadrature_bank() emits: pair i = scale (i // 4), theta (i % 4).
# sigma = pi / f for the two committed frequencies (pi/2, pi/(2*sqrt(2))).
_BANK_SIGMA = [r"$\sigma{=}2$", r"$\sigma{=}2\sqrt{2}$"]
_BANK_THETA = [r"$0^\circ$", r"$45^\circ$", r"$90^\circ$", r"$135^\circ$"]


def _pick_bank_sample(stem, dataset, data_root, device, n_scan=2000):
    """Deterministically choose the test image that best exercises ALL eight
    pairs, in the sense a reader can SEE: every orientation channel must be
    the locally dominant one somewhere in the image, in a sizeable region.
    A plain max-min on channel means selects dense texture (every filter
    fires, but the maps read as noise); requiring each orientation to WIN a
    region selects images with distinct oriented structures instead. Score =
    min over the four orientations of the pixel fraction where that
    orientation's scale-pooled energy is the argmax, tie-broken toward
    higher overall energy; candidates below median total energy are dropped
    so a flat image cannot win on evenly-split noise. One image, no
    cherry-pick: the criterion is the requirement itself."""
    ds = data_mod.build_dataset(dataset, data_root, train=False)
    n_scan = min(n_scan, len(ds))
    frac_min, tot = [], []
    with torch.no_grad():
        for i0 in range(0, n_scan, 256):
            xs = torch.stack([ds[i][0] for i in range(i0, min(i0 + 256, n_scan))])
            e = stem._energy(stem._luma(xs.to(device)))
            e = e * stem.calib_scale.view(1, -1, 1, 1)
            B, _, Hh, Ww = e.shape
            o = e.view(B, 2, 4, Hh, Ww).mean(1)        # scale-pooled per theta
            win = o.argmax(1)                          # (B, H, W)
            f = torch.stack([(win == t).float().mean(dim=(1, 2))
                             for t in range(4)], 1)    # (B, 4)
            frac_min.append(f.min(dim=1).values.cpu())
            tot.append(e.mean(dim=(1, 2, 3)).cpu())
    frac_min, tot = torch.cat(frac_min), torch.cat(tot)
    ok = tot >= tot.median()
    score = frac_min + 1e-3 * (tot / tot.max())
    score[~ok] = -1.0
    idx = int(score.argmax())
    names = class_names(dataset, ds, data_root)
    label = names[int(ds[idx][1])] if names else f"class {int(ds[idx][1])}"
    return ds, idx, label


def fig_bank(out, dataset="stl10", data_root="./data", device="cpu",
             second_dataset="eurosat"):
    """The prior itself, and what it does to real images.

    Top block: the eight complex Gabor quadrature pairs (even/odd), labelled
    by their (sigma, theta). Below: TWO test images from the study's own
    data -- one photographic (STL-10) and one non-photographic (EuroSAT
    satellite) -- each with its eight magnitude-response maps m(x), computed
    by the ACTUAL pinned bank with the study's own calibration (unit
    per-channel std on the committed calibration batch), on one honest
    colour scale per example. Two domains because the bank is identical for
    every dataset in the study: the same fixed kernels respond to trunks and
    to field boundaries alike. Samples are chosen by a deterministic max-min
    criterion so every pair responds visibly (_pick_bank_sample); a dataset
    that is not present is skipped with a note.

    Single-column. The response tiles are the point of the figure, so they
    take the full column width; the input thumbnail sits in each example's
    header row.
    """
    from momentstem import EnergyStem

    samples = []
    for ds_name in (dataset, second_dataset):
        try:
            stem = EnergyStem(feature_type="magnitude").to(device)
            calib = data_mod.calibration_batch(ds_name, data_root)
            stem.calibrate(calib.to(device))
            ds, idx, label = _pick_bank_sample(stem, ds_name, data_root, device)
            x = ds[idx][0].unsqueeze(0).to(device)
            with torch.no_grad():
                resp = (stem._energy(stem._luma(x))
                        * stem.calib_scale.view(1, -1, 1, 1))[0].cpu()
            px = data_mod.IMAGE_SIZE[ds_name]
            samples.append((ds_name, denorm(x[0], ds_name), resp, label, idx, px))
        except Exception as e:                                # noqa: BLE001
            print(f"bank sample unavailable for {ds_name} ({e}); skipped")

    stem = EnergyStem(feature_type="magnitude")
    even, odd = stem.even.squeeze(1), stem.odd.squeeze(1)     # (8, k, k)
    n = even.shape[0]
    lim = float(max(even.abs().max(), odd.abs().max()))

    # All vertical layout in INCHES, top-down, so nothing lands off-canvas.
    kx0_in, gap = 0.24, 0.035
    kw_in = (FIGW - kx0_in - 0.02 - (n - 1) * gap) / n        # kernel panels
    rw_in = (FIGW - 0.06 - 3 * gap) / 4                       # response tiles,
    rx0_in = 0.04                                             # full width
    top_hdr, theta_hdr = 0.145, 0.105
    hdr_in = 0.46                                             # per-example header
    rw2_h = ((FIGW - 0.06 - 0.10) / 2 - 3 * 0.025) / 4        # side-by-side tile
    blk_in = hdr_in + 2 * rw2_h + 0.025 + 0.04                # ONE block row
    H = (0.02 + top_hdr + theta_hdr + 2 * kw_in + gap
         + (blk_in + 0.02 if samples else 0.04))
    fig = plt.figure(figsize=(FIGW, H))

    def ax_at(x_in, y_top_in, w_in, h_in):
        return fig.add_axes([x_in / FIGW, (H - y_top_in - h_in) / H,
                             w_in / FIGW, h_in / H])

    # ---- kernel mosaic, (sigma, theta)-labelled ------------------------
    k_y0 = 0.02 + top_hdr + theta_hdr
    for i in range(n):
        for r, bank in enumerate((even, odd)):
            ax = ax_at(kx0_in + i * (kw_in + gap), k_y0 + r * (kw_in + gap),
                       kw_in, kw_in)
            ax.imshow(bank[i].numpy(), cmap="RdBu_r", vmin=-lim, vmax=lim)
            ax.set_xticks([]), ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_linewidth(0.3)
            if r == 0:
                ax.set_title(_BANK_THETA[i % 4], fontsize=5.5, pad=1.5)
            if i == 0:
                ax.set_ylabel(("even", "odd")[r], fontsize=6, labelpad=1.5)
    for sgrp in range(2):                  # scale group headers with rules
        gx0 = (kx0_in + sgrp * 4 * (kw_in + gap)) / FIGW
        gx1 = (kx0_in + (sgrp * 4 + 3) * (kw_in + gap) + kw_in) / FIGW
        fig.text((gx0 + gx1) / 2, 1 - (0.02 + 0.055) / H,
                 _BANK_SIGMA[sgrp] + " px", ha="center", fontsize=6)
        fig.add_artist(plt.Line2D([gx0, gx1],
                                  [1 - (0.02 + 0.115) / H] * 2,
                                  color="#888888", lw=0.5))

    if not samples:
        fig.savefig(os.path.join(out, "bank_gabor.png"), dpi=300,
                    bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)
        return

    # ---- example blocks, SIDE BY SIDE: two columns, each = header row
    # (input thumb + label) over a 2x4 map grid at column width ------------
    col_w = (FIGW - 0.06 - 0.10) / 2          # two columns, small mid-gap
    rw2 = (col_w - 3 * 0.025) / 4             # per-column map tile
    y0 = k_y0 + 2 * kw_in + gap + 0.03
    for ci, (ds_name, img, resp, label, idx, px) in enumerate(samples):
        cx = 0.03 + ci * (col_w + 0.10)
        vmax = float(torch.quantile(resp, 0.99))
        iw_in = hdr_in - 0.06
        axi = ax_at(cx, y0 + 0.02, iw_in, iw_in)
        axi.imshow(img)
        axi.set_xticks([]), axi.set_yticks([])
        for sp in axi.spines.values():
            sp.set_linewidth(0.4)
        nice = {"stl10": "STL-10", "eurosat": "EuroSAT"}.get(ds_name, ds_name)
        if ds_name == "eurosat" and str(label).startswith("class "):
            _E = ["annual crop", "forest", "herbaceous veg.", "highway",
                  "industrial", "pasture", "permanent crop", "residential",
                  "river", "sea/lake"]
            label = _E[int(str(label).split()[-1])]
        fig.text((cx + iw_in + 0.07) / FIGW, 1 - (y0 + hdr_in / 2 - 0.10) / H,
                 f"$x$: {nice} {label} ({px} px)", fontsize=5.0,
                 ha="left", va="center")
        fig.text((cx + iw_in + 0.07) / FIGW, 1 - (y0 + hdr_in / 2 + 0.07) / H,
                 r"$m_{\sigma,\theta}(x)$, shared scale", fontsize=4.6,
                 ha="left", va="center")
        r_y0 = y0 + hdr_in
        for i in range(n):
            r, c = divmod(i, 4)
            ax = ax_at(cx + c * (rw2 + 0.025), r_y0 + r * (rw2 + 0.025),
                       rw2, rw2)
            ax.imshow(resp[i].numpy(), cmap="magma", vmin=0, vmax=vmax)
            ax.set_xticks([]), ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_linewidth(0.3)
            ax.text(0.04, 0.04, _BANK_SIGMA[r] + ", " + _BANK_THETA[i % 4],
                    transform=ax.transAxes, fontsize=3.6, color="white",
                    ha="left", va="bottom")
        print(f"bank figure: sample = {ds_name} test index {idx} ({label})")

    fig.savefig(os.path.join(out, "bank_gabor.png"), dpi=300,
                bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", nargs=2, metavar=("NONE_CELL", "AUX_CELL"))
    ap.add_argument("--out", default="docs/viz")
    ap.add_argument("--seed-dir", default="seed0")
    ap.add_argument("--n-classes", type=int, default=8)
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--bank-only", action="store_true",
                    help="write bank_gabor.png only (no checkpoints needed)")
    ap.add_argument("--bank-dataset", default="stl10",
                    help="dataset supplying the bank figure's sample image")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.bank_only:
        fig_bank(args.out, dataset=args.bank_dataset,
                 data_root=args.data_root, device=device)
        print(f"BANK_DONE -> {args.out}", flush=True)
        return
    if args.pair is None:
        ap.error("--pair is required unless --bank-only is given")
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
