#!/usr/bin/env python3
"""AP50 collapses at 1-2% data. The runs do not -- so measure what AP50 destroys.

THE PROBLEM. The pre-declared floor rule says a cell where both arms land below
1.0 AP50 is uninterpretable, because a difference between two near-zero
detectors is not a measurement. VOC detection at 1% (107 images) and 2% returns
0.30 and 0.61 AP50, so the rule fires and T1 -- the delta at ~5 images per class,
the detection analogue of the dense study's headline negative -- has no answer.

But the same cells return fg_acc 15.3 and 21.6 against a 5% chance rate, with
SEMs of 0.13-0.84. The detectors are demonstrably doing something; AP50 is a
ranked-precision integral, so when precision is poor everywhere the integral is
near zero however much better one arm's boxes are. It is COMPRESSIVE near the
floor, the same failure mIoU has on dense prediction (a +1.62 point gain in
pixel accuracy reported as "+0.09 mIoU") and the same failure the pathmnist
probe has as a measuring stick.

THE FIX IS TO DECOMPOSE, NOT TO SWAP METRICS. AP50 confounds two abilities, and
both are measurable CONDITIONED ON GROUND-TRUTH FOREGROUND, where neither can
collapse:

    fg_acc  -- 20-way classification accuracy at the locations GT assigns to an
               object. Already logged, and pre-registered as the readout scale
               before any cell ran, so using it here is not a post-hoc choice.

    fg_iou  -- mean IoU between the predicted box and the GT box AT THOSE SAME
               locations. New here, and the quantity detection was added to the
               study for: this is the only task with a coordinate REGRESSION
               head, and fg_iou measures that head directly.

The conditioning comes from GT boxes through the same assign_targets() the
training loss uses, never from predictions, so it is not circular -- a detector
that predicts nothing still has a well-defined fg_iou.

WHY THIS MATTERS ALREADY, before the low cells are rescued: at 5% and 10% the
prior gains +0.84 and +0.56 AP50 while its fg_acc delta is +0.43 and -0.32. So
the AP50 gain is NOT coming from classification, and the only other component
is localization, which nothing in the study has measured. fg_iou is the
measurement.

AP25 is computed alongside as a ROBUSTNESS CHECK and never as a headline.
Choosing a looser threshold after seeing that the stricter one was inconvenient
is precisely the move a referee should distrust; its only legitimate use is to
show the ORDERING of the arms does not depend on the threshold.

Runs on existing checkpoints. No retraining.
"""
import argparse
import glob
import json
import os
import re
import statistics as st
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import data_det as dq  # noqa: E402
import train_det as TD  # noqa: E402
from momentstem.detection import assign_targets, build_det_model, locations  # noqa: E402


@torch.no_grad()
def fg_iou(model, loader, device, n_classes, stride):
    """Mean IoU of the predicted box against the GT box, at GT foreground
    locations only.

    Both boxes are built the same way -- from an (l, t, r, b) offset quadruple
    at the location's own centre -- so the comparison is between the head's
    regression and the target that regression was trained on, with nothing in
    between. Locations, not detections: no score threshold, no NMS, no ranking,
    which is what keeps this off the floor.
    """
    model.eval()
    tot, n, hit = 0.0, 0, 0
    for imgs, tgts in loader:
        for img, tg in zip(imgs if isinstance(imgs, list) else list(imgs), tgts):
            x = img.unsqueeze(0).to(device)
            H, W = x.shape[-2:]
            ph, pw = (-H) % stride, (-W) % stride
            if ph or pw:
                x = F.pad(x, (0, pw, 0, ph))
            out = model(x)
            cls_l, _ctr, reg = out if not isinstance(out, dict) else out["out"]
            Hf, Wf = cls_l.shape[-2:]
            locs = locations(Hf, Wf, stride, device)
            ct, rt, _ = assign_targets(locs, tg["boxes"].to(device),
                                       tg["labels"].to(device), n_classes,
                                       stride=stride)
            m = ct < n_classes
            if not m.any():
                continue
            pr = reg[0].permute(1, 2, 0).reshape(-1, 4)[m]      # predicted ltrb
            gt = rt[m]                                          # target ltrb
            xy = locs[m]
            def box(d):
                return torch.stack([xy[:, 0] - d[:, 0], xy[:, 1] - d[:, 1],
                                    xy[:, 0] + d[:, 2], xy[:, 1] + d[:, 3]], 1)
            pb, gb = box(pr), box(gt)
            x1 = torch.max(pb[:, 0], gb[:, 0]); y1 = torch.max(pb[:, 1], gb[:, 1])
            x2 = torch.min(pb[:, 2], gb[:, 2]); y2 = torch.min(pb[:, 3], gb[:, 3])
            inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
            ap = (pb[:, 2] - pb[:, 0]).clamp(min=0) * (pb[:, 3] - pb[:, 1]).clamp(min=0)
            ag = (gb[:, 2] - gb[:, 0]) * (gb[:, 3] - gb[:, 1])
            iou = inter / (ap + ag - inter).clamp(min=1e-6)
            tot += float(iou.sum()); n += int(iou.numel())
            hit += int((iou >= 0.5).sum())
    return (tot / max(n, 1)), (hit / max(n, 1)), n


def cell_seeds(runs, cell):
    return sorted(glob.glob(os.path.join(runs, cell, "seed*")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs_det")
    ap.add_argument("--configs", default="configs/det")
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--ap25", action="store_true",
                    help="also compute AP at IoU 0.25 (robustness check only)")
    ap.add_argument("--out", default="results/det_decompose.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    dev = torch.device(a.device)
    va = dq.VOCDetection(a.data_root, "val", train=False)
    lva = torch.utils.data.DataLoader(va, batch_size=1, shuffle=False,
                                      num_workers=4, collate_fn=dq.collate)

    cells = sorted(os.path.basename(p)[:-5] for p in glob.glob(os.path.join(a.configs, "*.yaml")))
    recs = []
    import yaml
    for cell in cells:
        cfg = yaml.safe_load(open(os.path.join(a.configs, cell + ".yaml")))
        stride = cfg.get("output_stride", 8)
        for sd in cell_seeds(a.runs, cell):
            ck = os.path.join(sd, a.ckpt)
            if not os.path.exists(ck):
                continue
            model = build_det_model(cfg.get("backbone", "resnet18"), dq.NUM_CLASSES,
                                    stride, cfg.get("pretrained", False),
                                    cfg.get("moment_aux"), cfg.get("crop", 512)).to(dev)
            sd_state = torch.load(ck, map_location=dev)
            model.load_state_dict(sd_state.get("model", sd_state), strict=False)
            miou, rate, nloc = fg_iou(model, lva, dev, dq.NUM_CLASSES, stride)
            r = {"cell": cell, "seed": os.path.basename(sd),
                 "pct": int(re.search(r"(\d+)pct", cell).group(1)),
                 "arm": "aux" if "_aux_" in cell else "none",
                 "fg_iou": miou, "fg_iou50_rate": rate, "n_fg_locations": nloc}
            if a.ap25:
                r["ap25"] = TD.evaluate(model, lva, dev, dq.NUM_CLASSES,
                                        stride, iou_thr=0.25)[0]
            recs.append(r)
            print(f"  {cell:24s} {r['seed']}  fg_iou {miou:.4f}  "
                  f"IoU>=0.5 {100*rate:5.1f}%  ({nloc} locations)"
                  + (f"  AP25 {r['ap25']:.2f}" if a.ap25 else ""))

    if not recs:
        raise SystemExit("no checkpoints found under %s" % a.runs)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(recs, open(a.out, "w"), indent=2)
    print(f"\nwrote {a.out} ({len(recs)} runs)\n")

    print(f"{'pct':>4s} | {'fg_iou none':>13s} {'fg_iou aux':>13s} {'delta':>9s}")
    for p in sorted({r["pct"] for r in recs}):
        def arm(k):
            xs = [r["fg_iou"] for r in recs if r["pct"] == p and r["arm"] == k]
            if not xs:
                return None, None
            return st.mean(xs), (st.stdev(xs) / len(xs) ** .5 if len(xs) > 1 else 0.0)
        mn, sn = arm("none"); ma, sa = arm("aux")
        if mn is None or ma is None:
            continue
        print(f"{p:>3d}% | {mn:8.4f}+-{sn:<4.4f} {ma:8.4f}+-{sa:<4.4f} {ma-mn:+9.4f}")


if __name__ == "__main__":
    main()
