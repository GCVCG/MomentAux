#!/usr/bin/env python3
"""Is the prior's class evidence really more concentrated, or did we pick four images?

The CAM panel shows four test images on which the prior arm is right and the
baseline wrong, and its caption says the prior's evidence "concentrates on
object contours where the baseline's is diffuse". As presented that is an
impression from a selected sample -- the weakest evidence in the paper, which is
why it sits in the appendix. This measures the same property over the WHOLE test
set so the claim can be checked rather than eyeballed.

THE STATISTIC. Normalize each class-activation map to sum 1 and take its Gini
coefficient: 0 when evidence is spread uniformly over the map, approaching 1
when it concentrates on a few locations. Gini needs no segmentation masks, which
CIFAR-100 does not have, and it measures exactly the word the caption uses.

Both arms are scored on the SAME images -- every test image, not the advantage
cases -- because a statistic computed only where the prior wins would build the
selection back into the number it is meant to replace.
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import visualize_features as VF  # noqa: E402
import data as data_mod  # noqa: E402


def gini(x):
    """Gini of a non-negative vector; 0 = uniform, ->1 = all mass in one place."""
    x = np.sort(np.asarray(x, dtype=np.float64).ravel())
    x = x - x.min() if x.min() < 0 else x
    n = x.size
    s = x.sum()
    if s <= 0:
        return 0.0
    return float((2.0 * np.arange(1, n + 1) - n - 1).dot(x) / (n * s))


@torch.no_grad()
def cam_gini(model, ds, device, n, batch=128):
    """Mean CAM Gini over n test images, using each image's PREDICTED class --
    the map a reader would actually be shown."""
    out = []
    for i0 in range(0, n, batch):
        xs = torch.stack([ds[i][0] for i in range(i0, min(i0 + batch, n))]).to(device)
        fmap = model.net.forward_features(model.stem(xs))
        logits = model.net.get_classifier()(model.net.global_pool(fmap))
        w = model.net.get_classifier().weight
        cam = torch.einsum("bchw,bc->bhw", fmap, w[logits.argmax(1)])
        cam = F.relu(cam)                      # CAM is read as evidence-for
        for j in range(cam.shape[0]):
            out.append(gini(cam[j].float().cpu().numpy()))
    return float(np.mean(out)), float(np.std(out) / max(len(out), 1) ** 0.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", nargs="+",
                    default=["abl5_none:auxmag_5pct_sched0"],
                    help="baseline:aux cell pairs")
    ap.add_argument("--dataset", default="cifar100")
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/cam_concentration.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    dev = torch.device(a.device)
    ds = data_mod.build_dataset(a.dataset, a.data_root, train=False)
    recs = []
    for pair in a.pairs:
        base, aux = pair.split(":")
        models = VF.load_pair(base, aux, f"seed{a.seed}", dev)
        gb, sb = cam_gini(models[base][0], ds, dev, a.n)
        ga, sa = cam_gini(models[aux][0], ds, dev, a.n)
        recs.append({"baseline": base, "aux": aux, "n_images": a.n,
                     "gini_base": gb, "gini_base_sem": sb,
                     "gini_aux": ga, "gini_aux_sem": sa, "delta": ga - gb})
        print(f"  {aux}\n    baseline Gini {gb:.4f} +-{sb:.4f}   "
              f"prior {ga:.4f} +-{sa:.4f}   delta {ga-gb:+.4f}  (n={a.n} images)")
    if recs:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(recs, open(a.out, "w"), indent=2)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
