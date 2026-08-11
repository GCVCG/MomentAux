"""The dense analog of G: what a FROZEN backbone supports under a linear head.

    python analysis/dense_probe.py --cell voc_aux_5pct [--seeds 0 1 2]

The classification study measures G by freezing the network and fitting a
multinomial logistic head on the full training split -- far more labels than
the cell itself saw -- so the number is a property of the FEATURES rather than
of what the classifier could cash in. The dense analog is the same idea with
the same protocol: freeze everything, fit a single 1x1 convolution from the
backbone's last feature map to the 21 classes, upsample, and score mIoU on
val.

WHY 1x1 AND NOTHING ELSE. The FCN head used in training has a 3x3 conv, a
norm and a nonlinearity; that is enough capacity to partly compensate for a
worse backbone, which is exactly what the probe exists to rule out. A single
1x1 conv is a per-location linear classifier on the features -- the dense
equivalent of a linear probe, no more.

WHAT IT LICENSES, and the limit. Delta = G + readout was derived on top-1
accuracy. The readout crossing bracket [31.8, 40.3] is an ACCURACY bracket and
is NOT transplanted here: mIoU is a different scale, so the numeric bracket
would be meaningless. What these cells can test is the FORM -- readout
negative where the baseline is low and rising through zero as it rises -- and,
with enough fractions, they can estimate where that crossing sits in mIoU.
"""
import argparse
import json
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import data_dense as dd
from momentstem.segmentation import build_seg_model


def load_backbone(cell, seed, runs, data_root, device, ckpt="best.pt"):
    """Rebuild the cell's model and load its weights, then strip to features."""
    cfg = yaml.safe_load(open(os.path.join("configs", "dense", f"{cell}.yaml")))
    path = os.path.join(runs, cell, f"seed{seed}", ckpt)
    if not os.path.exists(path):
        return None, cfg
    model = build_seg_model(cfg.get("backbone", "resnet18"),
                            n_classes=dd.NUM_CLASSES[cfg.get("dataset", "voc_seg")],
                            output_stride=cfg.get("output_stride", 8),
                            moment_aux=cfg.get("moment_aux"),
                            image_size=cfg.get("crop", 512)).to(device)
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    # The auxiliary wrapper is training-only; the deployed object is the
    # segmentation net itself, which is what must be probed.
    net = getattr(model, "net", model)
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net, cfg


@torch.no_grad()
def _features(net, x):
    return net.features(x)


def probe(net, data_root, device, epochs=10, lr=0.01, batch=8, crop=512,
          seed=0, ds="voc_seg"):
    """Fit a 1x1 conv on frozen features over the FULL augmented train split."""
    torch.manual_seed(seed)
    train = dd.SegmentationDataset(data_root, train=True, crop=crop, ds=ds)
    val = dd.SegmentationDataset(data_root, train=False, ds=ds)
    tl = DataLoader(train, batch_size=batch, shuffle=True, num_workers=4,
                    drop_last=True, pin_memory=True)
    vl = DataLoader(val, batch_size=1, shuffle=False, num_workers=4)

    with torch.no_grad():
        c = _features(net, torch.zeros(1, 3, crop, crop, device=device)).shape[1]
    head = nn.Conv2d(c, dd.NUM_CLASSES[ds], 1).to(device)
    opt = torch.optim.SGD(head.parameters(), lr=lr, momentum=0.9)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    for ep in range(epochs):
        head.train()
        for x, y in tl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=device.type == "cuda"):
                f = _features(net, x)
                logits = F.interpolate(head(f), size=y.shape[-2:],
                                       mode="bilinear", align_corners=False)
                loss = F.cross_entropy(logits, y, ignore_index=dd.IGNORE_INDEX)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        sched.step()

    head.eval()
    cm = dd.ConfusionMatrix(dd.NUM_CLASSES[ds])
    with torch.no_grad():
        for x, y in vl:
            x = x.to(device, non_blocking=True)
            with torch.autocast("cuda", enabled=device.type == "cuda"):
                f = _features(net, x)
                logits = F.interpolate(head(f), size=y.shape[-2:],
                                       mode="bilinear", align_corners=False)
            cm.update(logits.argmax(1).flatten(), y.flatten())
    miou, per_class = cm.miou()
    return miou, per_class


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--runs", default="runs_dense")
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=8,
                    help="probe batch size. The protocol value is 8 and every "
                         "recorded dense G used it; it is exposed only so a "
                         "shared GPU can be worked around, and it is written "
                         "into the output JSON so a cell probed at a "
                         "different batch is never silently compared with one "
                         "probed at 8.")
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    device = torch.device(a.device)

    out = {"cell": a.cell, "epochs": a.epochs, "batch": a.batch,
           "ckpt": a.ckpt, "seeds": {}}
    vals = []
    for s in a.seeds:
        net, cfg = load_backbone(a.cell, s, a.runs, a.data_root, device, a.ckpt)
        if net is None:
            # SKIPPING A MISSING CHECKPOINT SILENTLY is how a probe pass ends
            # up under-seeded without anyone noticing; say so.
            print(f"  seed {s}: checkpoint missing, SKIPPED", flush=True)
            continue
        miou, per_class = probe(net, a.data_root, device, epochs=a.epochs,
                                batch=a.batch, crop=512, seed=s,
                                ds=cfg.get("dataset", "voc_seg"))
        out["seeds"][str(s)] = {"probe_miou": miou, "per_class_iou": per_class}
        vals.append(miou)
        print(f"  seed {s}: probe mIoU {miou:.2f}", flush=True)
    if vals:
        import statistics
        out["probe_miou_mean"] = statistics.fmean(vals)
        out["probe_miou_std"] = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        out["n_seeds"] = len(vals)
        d = os.path.join(a.runs, a.cell)
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "dense_probe.json"), "w") as f:
            json.dump(out, f, indent=1)
        print(f"PROBE {a.cell}: {out['probe_miou_mean']:.2f} "
              f"+- {out['probe_miou_std']:.2f} (n={len(vals)})")


if __name__ == "__main__":
    main()
