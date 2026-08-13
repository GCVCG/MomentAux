#!/usr/bin/env python3
"""The detection analog of G: what a FROZEN backbone supports under a minimal head.

    python analysis/det_probe.py --cell vocdet_aux_1pct [--seeds 0 1 2]

WHY THIS IS WORTH RUNNING AFTER A NULL. The detection grid came back at zero on
every measure -- AP50, fg_acc and fg_iou all flat at every fraction. That admits
two readings the e2e numbers cannot separate:

  H-NO-FEATURES     the prior genuinely buys a detection backbone nothing, so
                    the null is feature-side and detection is simply outside
                    what the prior helps.
  H-READOUT         the features ARE better and the detection head cannot cash
                    them -- which is precisely the pattern the classification
                    study documents on its left flank, where at 5 images per
                    class the probe sees +4.70 of feature gain and the trained
                    classifier realizes +1.91 of it.

Only a frozen-feature probe distinguishes them, and the answer decides whether
detection is a limitation of the prior or a limitation of the head.

AND IT COULD SUPPLY SOMETHING THE DENSE STUDY FAILED TO. Baseline fg_acc is
15.3 / 21.6 / 28.9 at 1 / 2 / 5%, all clearly BELOW the crossing bracket
[31.8, 40.3], where the sign law demands a NEGATIVE readout. Since e2e Delta is
~0 at those fractions, readout = -G. So a resolvably positive G makes detection
a resolvable NEGATIVE-BRANCH law cell -- the test all nine resolvable
segmentation cells could not provide (every one sat above the crossing) and
that Pascal-Context, built for exactly this, returned 0.00 on.

PROTOCOL, mirroring analysis/dense_probe.py so the three tasks are comparable:
freeze everything, fit a MINIMAL head on the FULL train_aug split (10,582
images, far more labels than any probed cell saw), score on val. Both arms get
an identical probe.

THE HEAD IS 1x1 CONVOLUTIONS ONLY. The trained FCOSHead has two 3x3 towers with
GroupNorm and ReLU; that capacity can partly compensate for a worse backbone,
which is what the probe exists to rule out. Stripping it to per-location linear
maps is the detection equivalent of the dense probe's single 1x1 conv. The
exp(scale * pred) parameterization and the prior-probability bias init are kept,
because without them the regression cannot stay positive and the focal loss
starts with an enormous background gradient -- neither is capacity.

100% CELLS ARE EXCLUDED BY THE PROBE-CEILING RULE: there the probe's labels ARE
the cell's labels, so no G/readout split is interpretable (the cub@100%
precedent). 10% is probed but makes no sign call -- its baseline fg_acc of 36.5
sits INSIDE the bracket (the mnet precedent).
"""
import argparse
import json
import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import data_det as dq  # noqa: E402
import train_det as TD  # noqa: E402
from momentstem.detection import (assign_targets, build_det_model,  # noqa: E402
                                  giou_loss, locations)
from train_det import focal_loss  # noqa: E402  (defined in the trainer)


class LinearFCOSHead(nn.Module):
    """FCOSHead with the towers removed: three 1x1 convolutions off the frozen
    feature map. Per-location linear, no hidden layer, no normalization."""

    def __init__(self, in_ch, n_classes):
        super().__init__()
        self.cls_logits = nn.Conv2d(in_ch, n_classes, 1)
        self.centerness = nn.Conv2d(in_ch, 1, 1)
        self.bbox_pred = nn.Conv2d(in_ch, 4, 1)
        self.scale = nn.Parameter(torch.tensor(1.0))
        for m in (self.cls_logits, self.centerness, self.bbox_pred):
            nn.init.normal_(m.weight, std=0.01)
            nn.init.constant_(m.bias, 0)
        nn.init.constant_(self.cls_logits.bias, -math.log((1 - 0.01) / 0.01))

    def forward(self, f):
        return (self.cls_logits(f), self.centerness(f),
                torch.exp(self.scale * self.bbox_pred(f)))


class Probed(nn.Module):
    """Frozen trunk + probe head, shaped like the trained model so that
    train_det.evaluate() and det_decompose.fg_iou() can score it unchanged."""

    def __init__(self, trunk, head, stride):
        super().__init__()
        self.backbone, self.head, self.stride = trunk, head, stride

    def features(self, x):
        with torch.no_grad():
            return self.backbone.forward_features(x)

    def forward(self, x):
        return self.head(self.features(x))


def load_trunk(cell, seed, runs, device, ckpt="last.pt", configs="configs/det"):
    cfg = yaml.safe_load(open(os.path.join(configs, cell + ".yaml")))
    path = os.path.join(runs, cell, f"seed{seed}", ckpt)
    if not os.path.exists(path):
        return None, cfg
    model = build_det_model(cfg.get("backbone", "resnet18"), dq.NUM_CLASSES,
                            cfg.get("output_stride", 8), cfg.get("pretrained", False),
                            cfg.get("moment_aux"), cfg.get("crop", 512)).to(device)
    state = torch.load(path, map_location=device)
    model.load_state_dict(state.get("model", state))
    # the auxiliary wrapper is training-only; the deployed object is the
    # detector, and inside it only the backbone is probed
    net = getattr(model, "net", model)
    trunk = net.backbone
    trunk.eval()
    for p in trunk.parameters():
        p.requires_grad_(False)
    return trunk, cfg


def probe(trunk, cfg, data_root, device, epochs=10, lr=0.01, batch=16,
          crop=512, seed=0, nw=4):
    torch.manual_seed(seed)
    stride = cfg.get("output_stride", 8)
    # FULL train split: subset_pct is deliberately NOT passed through.
    tr = dq.VOCDetection(data_root, "train_aug", crop=crop, train=True)
    va = dq.VOCDetection(data_root, "val", train=False)
    g = torch.Generator(); g.manual_seed(seed)
    ltr = DataLoader(tr, batch_size=batch, shuffle=True, num_workers=nw,
                     collate_fn=dq.collate, drop_last=True, generator=g,
                     pin_memory=True)
    lva = DataLoader(va, batch_size=1, shuffle=False, num_workers=nw,
                     collate_fn=dq.collate)

    with torch.no_grad():
        c = trunk.forward_features(torch.zeros(1, 3, crop, crop, device=device)).shape[1]
    model = Probed(trunk, LinearFCOSHead(c, dq.NUM_CLASSES).to(device), stride)
    opt = torch.optim.SGD(model.head.parameters(), lr=lr, momentum=0.9)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    C = dq.NUM_CLASSES
    for ep in range(epochs):
        model.head.train()
        tot = 0.0
        for imgs, tgts in ltr:
            imgs = imgs.to(device, non_blocking=True)
            cls_l, ctr, reg = model(imgs)
            locs = locations(cls_l.shape[-2], cls_l.shape[-1], stride, device)
            ct, rt, wt = [], [], []
            for tg in tgts:
                c_, r_, w_ = assign_targets(locs, tg["boxes"].to(device),
                                            tg["labels"].to(device), C, stride=stride)
                ct.append(c_); rt.append(r_); wt.append(w_)
            ct = torch.cat(ct); rt = torch.cat(rt); wt = torch.cat(wt)
            cl = cls_l.permute(0, 2, 3, 1).reshape(-1, C)
            cr = reg.permute(0, 2, 3, 1).reshape(-1, 4)
            cc = ctr.reshape(-1)
            npos = int((ct < C).sum())
            loss = focal_loss(cl, ct, C) / max(npos, 1)
            if npos:
                m = ct < C
                loss = loss + giou_loss(cr[m], rt[m], wt[m]) + \
                    F.binary_cross_entropy_with_logits(cc[m], wt[m])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.head.parameters(), 10.0)
            opt.step()
            tot += float(loss)
        sched.step()
        print(f"    probe ep {ep:2d}  loss {tot/max(len(ltr),1):.4f}", flush=True)

    model.head.eval()
    ap50, loc_acc, fg_acc = TD.evaluate(model, lva, device, C, stride)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from det_decompose import fg_iou as _fg_iou
    miou, rate, _ = _fg_iou(model, lva, device, C, stride)
    return {"probe_ap50": ap50, "probe_fg_acc": fg_acc,
            "probe_loc_acc": loc_acc, "probe_fg_iou": miou,
            "probe_fg_iou50_rate": rate, "probe_epochs": epochs,
            "n_probe_train": len(tr)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--runs", default="runs_det")
    ap.add_argument("--configs", default="configs/det")
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--ckpt", default="last.pt")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    if a.cell.endswith("100pct"):
        raise SystemExit("REFUSING %s: at 100%% the probe's labels ARE the "
                         "cell's labels, so no G/readout split is "
                         "interpretable (probe-ceiling rule)." % a.cell)

    dev = torch.device(a.device)
    for sd in a.seeds:
        out = os.path.join(a.runs, a.cell, f"seed{sd}", "det_probe.json")
        if os.path.exists(out):
            print(f"  SKIP {a.cell}/seed{sd}: already probed"); continue
        trunk, cfg = load_trunk(a.cell, sd, a.runs, dev, a.ckpt, a.configs)
        if trunk is None:
            print(f"  SKIP {a.cell}/seed{sd}: no {a.ckpt}"); continue
        print(f"  {a.cell}/seed{sd}")
        r = probe(trunk, cfg, a.data_root, dev, epochs=a.epochs, seed=sd,
                  nw=a.num_workers, crop=cfg.get("crop", 512))
        r.update({"cell": a.cell, "seed": sd, "ckpt": a.ckpt})
        json.dump(r, open(out, "w"), indent=2)
        print(f"    -> AP50 {r['probe_ap50']:.2f}  fg_acc {r['probe_fg_acc']:.2f}  "
              f"fg_iou {r['probe_fg_iou']:.4f}   wrote {out}", flush=True)


if __name__ == "__main__":
    main()
