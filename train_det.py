"""Detection entry point: VOC boxes on the study's backbone and recipe.

Nothing here imports into train.py or train_dense.py and nothing in them
changes. A detection cell cannot affect a classification or segmentation cell
even if this file is wrong.

THE RECIPE IS THE DENSE ONE, VERBATIM: SGD 0.9, lr 0.01, wd 1e-4, cosine,
200 epochs, batch 16, 512 crops with random scale 0.5-2.0 and horizontal flip.
That is deliberate and it is the point: detection then differs from
segmentation in the head and the loss and in nothing else, so a difference
between the two tasks is attributable. 200 epochs is not tuned -- it is the
same budget classification and segmentation use, and choosing a detection-
specific budget from convergence curves is exactly the per-task tuning the
frozen-recipe discipline exists to prevent (see the 50-epoch dense re-pin,
which produced two reversed conclusions before it was caught).

THE READOUT SCALE, decided before any cell ran. Readout is a function of the
classification the head actually performs, not of the task metric -- scoring it
on mIoU is what put every segmentation cell on the wrong flank. Detection's raw
per-location accuracy is ~99% because ~99.7% of locations are background, so it
saturates and tests nothing. The scale we register as PRIMARY is instead
accuracy over the locations ground truth assigns to an object: a genuine 20-way
accuracy, conditioned on GT boxes rather than on predictions. Both are logged
so the saturation claim is measured rather than assumed.

This matters beyond bookkeeping. All nine resolvable cells in the segmentation
grid sat ABOVE the crossing, so that task confirmed the law's positive branch
and tested its negative branch not at all. If foreground accuracy at low
fractions lands below the bracket, detection supplies the test segmentation
could not.
"""
import argparse
import fcntl
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import data_det as dq
from momentstem.detection import (assign_targets, giou_loss, locations)
from momentstem.detection import build_det_model

RECIPE = {"epochs": 200, "batch_size": 16, "lr": 0.01,
          "weight_decay": 1e-4, "momentum": 0.9, "crop": 512}


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def seed_worker(wid):
    s = torch.initial_seed() % 2 ** 32
    np.random.seed(s); random.seed(s)


def atomic_save(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def focal_loss(logits, targets, n_classes, alpha=0.25, gamma=2.0):
    """Sigmoid focal loss over C classes; `targets` uses index n_classes for
    background, which becomes an all-zeros one-hot row."""
    t = torch.zeros_like(logits)
    fg = targets < n_classes
    if fg.any():
        t[fg, targets[fg]] = 1.0
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, t, reduction="none")
    pt = p * t + (1 - p) * (1 - t)
    w = alpha * t + (1 - alpha) * (1 - t)
    return (w * (1 - pt) ** gamma * ce).sum()


def decode(cls_logits, ctr, reg, stride, score_thr=0.05, topk=1000):
    """Per-image decode to (boxes, scores, labels) in input pixels."""
    C, H, W = cls_logits.shape
    locs = locations(H, W, stride, cls_logits.device)
    scores = (torch.sigmoid(cls_logits.permute(1, 2, 0).reshape(-1, C)) *
              torch.sigmoid(ctr.reshape(-1, 1)))
    reg = reg.permute(1, 2, 0).reshape(-1, 4)
    flat = scores.reshape(-1)
    keep = flat > score_thr
    if keep.sum() == 0:
        return (torch.zeros((0, 4), device=cls_logits.device),
                torch.zeros((0,), device=cls_logits.device),
                torch.zeros((0,), dtype=torch.int64, device=cls_logits.device))
    idx = torch.nonzero(keep).squeeze(1)
    if idx.numel() > topk:
        idx = idx[flat[idx].topk(topk).indices]
    loc_i, cls_i = idx // C, idx % C
    l, t, r, b = reg[loc_i].unbind(1)
    x, y = locs[loc_i, 0], locs[loc_i, 1]
    boxes = torch.stack([x - l, y - t, x + r, y + b], dim=1)
    return boxes, flat[idx], cls_i


def voc_ap(rec, prec):
    """All-points interpolated AP (the VOC2010+ definition)."""
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    i = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1]))


def evaluate(model, loader, device, n_classes, stride, iou_thr=0.5):
    """VOC AP50, plus the two candidate scales for the readout term.

    WHY TWO. The dense pass established that readout is a property of the
    classification the head ACTUALLY PERFORMS, not of the task metric -- scoring
    it on mIoU put every segmentation cell on the wrong flank. Detection needs
    the same treatment, and it has two defensible scalars:

      loc_acc  -- per-location accuracy over ALL locations, the direct analogue
                  of segmentation's pixel accuracy (which also counts
                  background). Detection is ~99.7% background, so we expect this
                  to saturate near 99 and be NON-DISCRIMINATING: the law would
                  predict readout ~ 0 everywhere, which is true but tests
                  nothing. Recorded so that claim is measured rather than
                  assumed.

      fg_acc   -- accuracy over the locations ground truth assigns to an object,
                  i.e. a genuine 20-way classification accuracy. The
                  conditioning comes from GT boxes and not from predictions, so
                  it is not circular.

    PRE-REGISTERED: fg_acc is the PRIMARY readout scale for detection, chosen
    before any cell ran and for the stated reason. If it lands below the
    crossing bracket at low fractions it supplies the negative-branch test that
    segmentation could not -- all nine resolvable dense cells sat above the
    crossing."""
    from torchvision.ops import batched_nms, box_iou
    model.eval()
    dets = {c: [] for c in range(n_classes)}
    gts = {}
    acc_all = [0, 0]; acc_fg = [0, 0]
    with torch.no_grad():
        for imgs, tgts in loader:
            imgs_l = imgs if isinstance(imgs, list) else list(imgs)
            for img, tg in zip(imgs_l, tgts):
                x = img.unsqueeze(0).to(device)
                # pad to a multiple of the stride so the feature grid covers
                # the whole image; predictions are in input pixels either way
                H, W = x.shape[-2:]
                ph, pw = (-H) % stride, (-W) % stride
                if ph or pw:
                    x = F.pad(x, (0, pw, 0, ph))
                out = model(x)
                cls_l, ctr, reg = out if not isinstance(out, dict) else out["out"]
                b, s, c = decode(cls_l[0], ctr[0], reg[0], stride)
                if b.numel():
                    k = batched_nms(b, s, c, 0.5)[:100]
                    b, s, c = b[k], s[k], c[k]
                # readout scales: compare the head's argmax against the
                # SAME assignment the training targets use, so the quantity is
                # defined identically at train and eval time.
                Hf, Wf = cls_l.shape[-2:]
                locs_e = locations(Hf, Wf, stride, device)
                ct_e, _, _ = assign_targets(locs_e, tg["boxes"].to(device),
                                            tg["labels"].to(device), n_classes,
                                            stride=stride)
                pred_e = cls_l[0].permute(1, 2, 0).reshape(-1, n_classes)
                bg = pred_e.sigmoid().max(dim=1).values < 0.05
                arg = pred_e.argmax(dim=1)
                arg[bg] = n_classes
                acc_all[0] += int((arg == ct_e).sum()); acc_all[1] += ct_e.numel()
                fgm = ct_e < n_classes
                if fgm.any():
                    acc_fg[0] += int((arg[fgm] == ct_e[fgm]).sum())
                    acc_fg[1] += int(fgm.sum())

                iid = tg["image_id"]
                gts[iid] = tg
                for bb, ss, cc in zip(b.cpu().numpy(), s.cpu().numpy(), c.cpu().numpy()):
                    dets[int(cc)].append((iid, float(ss), bb))
    aps = []
    for c in range(n_classes):
        npos = 0
        gtc = {}
        for iid, tg in gts.items():
            m = tg["labels"] == c
            gtc[iid] = {"boxes": tg["boxes"][m], "diff": tg["difficult"][m],
                        "used": torch.zeros(int(m.sum()), dtype=torch.bool)}
            npos += int((~tg["difficult"][m]).sum())
        if npos == 0:
            continue
        d = sorted(dets[c], key=lambda t: -t[1])
        tp, fp = np.zeros(len(d)), np.zeros(len(d))
        for i, (iid, _, bb) in enumerate(d):
            g = gtc.get(iid)
            if g is None or len(g["boxes"]) == 0:
                fp[i] = 1; continue
            ious = box_iou(torch.tensor(bb).view(1, 4).float(), g["boxes"])[0]
            j = int(ious.argmax())
            if float(ious[j]) >= iou_thr:
                if bool(g["diff"][j]):
                    continue          # difficult: neither TP nor FP
                if not bool(g["used"][j]):
                    tp[i] = 1; g["used"][j] = True
                else:
                    fp[i] = 1
            else:
                fp[i] = 1
        tp, fp = np.cumsum(tp), np.cumsum(fp)
        aps.append(voc_ap(tp / max(npos, 1), tp / np.maximum(tp + fp, 1e-9)))
    ap50 = 100.0 * float(np.mean(aps)) if aps else 0.0
    loc_acc = 100.0 * acc_all[0] / max(acc_all[1], 1)
    fg_acc = 100.0 * acc_fg[0] / max(acc_fg[1], 1)
    return ap50, loc_acc, fg_acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--out-root", default="runs_det")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch-size", type=int)
    a = ap.parse_args()

    import yaml
    cfg = yaml.safe_load(open(a.config))
    name = cfg["name"]
    epochs = a.epochs or cfg.get("epochs", RECIPE["epochs"])
    bs = a.batch_size or cfg.get("batch_size", RECIPE["batch_size"])
    eval_every = cfg.get("eval_every", 20)
    out_dir = os.path.join(a.out_root, name, f"seed{a.seed}")
    os.makedirs(out_dir, exist_ok=True)

    # Same two guards the other trainers carry: an idempotent skip so a stale
    # worklist cannot redo finished work, and an exclusive lock so a second
    # trainer aborts instead of racing checkpoint writes. NOTE the lock is
    # node-local on GPFS -- it does not protect across nodes.
    if os.path.exists(os.path.join(out_dir, "final.json")) and not os.environ.get("MS_FORCE_RERUN"):
        print(f"SKIP: {out_dir}/final.json exists -- this cell is complete."); return
    lock_fd = open(os.path.join(out_dir, ".runlock"), "w")
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        raise SystemExit(f"ABORT: another trainer holds {out_dir}/.runlock")

    set_seed(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    nw = cfg.get("num_workers", 4)

    tr = dq.VOCDetection(a.data_root, "train_aug", crop=cfg.get("crop", 512),
                         train=True, pct=cfg.get("subset_pct"))
    va = dq.VOCDetection(a.data_root, "val", train=False)
    g = torch.Generator(); g.manual_seed(a.seed)
    ltr = DataLoader(tr, batch_size=bs, shuffle=True, num_workers=nw,
                     collate_fn=dq.collate, drop_last=True, generator=g,
                     worker_init_fn=seed_worker, pin_memory=True)
    lva = DataLoader(va, batch_size=1, shuffle=False, num_workers=nw,
                     collate_fn=dq.collate)

    model = build_det_model(cfg.get("backbone", "resnet18"), dq.NUM_CLASSES,
                            cfg.get("output_stride", 8), cfg.get("pretrained", False),
                            cfg.get("moment_aux"), cfg.get("crop", 512)).to(dev)
    stride = cfg.get("output_stride", 8)
    opt = torch.optim.SGD(model.parameters(), lr=cfg.get("lr", RECIPE["lr"]),
                          momentum=RECIPE["momentum"],
                          weight_decay=cfg.get("weight_decay", RECIPE["weight_decay"]))
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    lam0 = (cfg.get("moment_aux") or {}).get("weight", 0.0)
    lam_f = (cfg.get("moment_aux") or {}).get("weight_final", 0.0)

    mpath = os.path.join(out_dir, "metrics.csv")
    with open(mpath, "w") as f:
        f.write("epoch,loss,cls,reg,ctr,ap50,loc_acc,fg_acc,lr,aux_lambda\n")
    best, t0 = 0.0, time.time()
    for ep in range(epochs):
        # lambda(t): cosine to EXACTLY lam_f at the final epoch. That is what
        # makes high-data neutrality structural rather than tuned, and it is
        # identical to the other two tasks.
        lam = lam_f + 0.5 * (lam0 - lam_f) * (1 + math.cos(math.pi * ep / max(epochs - 1, 1)))
        if hasattr(model, "aux_weight"):
            model.aux_weight = lam
        model.train()
        agg = np.zeros(4)
        for imgs, tgts in ltr:
            imgs = imgs.to(dev, non_blocking=True)
            out = model(imgs)
            cls_l, ctr, reg = out
            B, C, H, W = cls_l.shape
            locs = locations(H, W, stride, dev)
            ct, rt, wt = [], [], []
            for tg in tgts:
                c_, r_, w_ = assign_targets(locs, tg["boxes"].to(dev),
                                            tg["labels"].to(dev), C, stride=stride)
                ct.append(c_); rt.append(r_); wt.append(w_)
            ct = torch.cat(ct); rt = torch.cat(rt); wt = torch.cat(wt)
            cl = cls_l.permute(0, 2, 3, 1).reshape(-1, C)
            cr = reg.permute(0, 2, 3, 1).reshape(-1, 4)
            cc = ctr.reshape(-1)
            npos = int((ct < C).sum())
            l_cls = focal_loss(cl, ct, C) / max(npos, 1)
            if npos:
                m = ct < C
                l_reg = giou_loss(cr[m], rt[m], wt[m])
                l_ctr = F.binary_cross_entropy_with_logits(cc[m], wt[m])
            else:
                l_reg = cr.sum() * 0.0; l_ctr = cc.sum() * 0.0
            loss = l_cls + l_reg + l_ctr
            if hasattr(model, "last_aux") and lam > 0:
                loss = loss + lam * model.last_aux
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            agg += [float(loss), float(l_cls), float(l_reg), float(l_ctr)]
        sched.step()
        agg /= max(len(ltr), 1)
        if ep % eval_every == 0 or ep == epochs - 1:
            ap50, loc_acc, fg_acc = evaluate(model, lva, dev, dq.NUM_CLASSES, stride)
        else:
            ap50 = loc_acc = fg_acc = float("nan")
        with open(mpath, "a") as f:
            f.write(f"{ep},{agg[0]:.4f},{agg[1]:.4f},{agg[2]:.4f},{agg[3]:.4f},"
                    f"{ap50:.4f},{loc_acc:.4f},{fg_acc:.4f},"
                    f"{opt.param_groups[0]['lr']:.6f},{lam:.4f}\n")
        print(f"ep {ep:3d}  loss {agg[0]:.4f}  AP50 {ap50:6.2f}  "
              f"fgAcc {fg_acc:5.1f}  lam {lam:.3f}", flush=True)
        if ap50 == ap50 and ap50 > best:
            best = ap50
            atomic_save(model.state_dict(), os.path.join(out_dir, "best.pt"))
    atomic_save(model.state_dict(), os.path.join(out_dir, "last.pt"))
    json.dump({"config": cfg, "seed": a.seed, "epochs": epochs,
               "final_ap50": ap50, "best_ap50": best,
               "final_loc_acc": loc_acc, "final_fg_acc": fg_acc,
               "n_train": len(tr), "n_val": len(va),
               "wall_seconds": time.time() - t0,
               "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
               "torch": torch.__version__},
              open(os.path.join(out_dir, "final.json"), "w"), indent=2)
    print(f"FINAL AP50 {ap50:.2f}  best {best:.2f}  "
          f"({time.time()-t0:.0f}s, {len(tr)} train images)")


if __name__ == "__main__":
    main()
