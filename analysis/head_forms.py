"""Matched-budget HEAD-FORM comparison on frozen features (readout axis).

Q7.3 showed e2e realizes exactly what a same-label-budget LINEAR probe
realizes. This script asks whether the head FORM (not the label budget) is
what leaves the left-flank feature gain unclaimed: on the same frozen
penultimate features, at the SAME labels-per-class budget the cell itself
had, compare three readouts:

  linear -- multinomial logistic (linear_probe.probe, the study standard)
  cosine -- same LBFGS logistic on L2-NORMALIZED features, no bias
            (the convex analogue of the diagcos e2e head)
  ncm    -- nearest class mean on L2-normalized features (zero trainable
            params beyond the k*C means; the strongest few-shot baseline)

If cosine/ncm read MORE aux-vs-baseline gap than linear at 5 img/class, the
readout penalty is partly head-expressivity and the e2e diagcos cells should
show it too. If all three agree, the penalty is label-INFORMATION and the
left flank stays classifier-bound. Diagnostic only, never a headline cell.

  python analysis/head_forms.py --run runs/<cell> --config <cfg> --shots 5
"""

import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import data as data_mod
from momentstem import build_model
from linear_probe import extract, probe, set_eval_transform, shots_subset


def probe_cosine(train_f, train_y, test_f, test_y, num_classes, device,
                 max_iter=200):
    """LBFGS logistic on L2-normalized features, bias-free -- the convex
    counterpart of the CosineClassifier e2e head (scale absorbed into W)."""
    train_f = F.normalize(train_f, dim=1).to(device)
    test_f = F.normalize(test_f, dim=1).to(device)
    train_y, test_y = train_y.to(device), test_y.to(device)
    clf = torch.nn.Linear(train_f.shape[1], num_classes, bias=False).to(device)
    opt = torch.optim.LBFGS(clf.parameters(), lr=1.0, max_iter=max_iter,
                            history_size=10, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(clf(train_f), train_y) \
            + 1e-4 * clf.weight.pow(2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        tr = (clf(train_f).argmax(1) == train_y).float().mean().item()
        te = (clf(test_f).argmax(1) == test_y).float().mean().item()
    return tr, te


def probe_ncm(train_f, train_y, test_f, test_y, num_classes, device):
    """Nearest class mean under cosine distance on L2-normalized features."""
    train_f = F.normalize(train_f, dim=1)
    test_f = F.normalize(test_f, dim=1)
    means = torch.zeros(num_classes, train_f.shape[1])
    for c in range(num_classes):
        means[c] = train_f[train_y == c].mean(0)
    means = F.normalize(means, dim=1).to(device)
    test_f, test_y = test_f.to(device), test_y.to(device)
    train_fd, train_yd = train_f.to(device), train_y.to(device)
    tr = ((train_fd @ means.T).argmax(1) == train_yd).float().mean().item()
    te = ((test_f @ means.T).argmax(1) == test_y).float().mean().item()
    return tr, te


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="runs/<cell> (every seed)")
    ap.add_argument("--config", required=True)
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--shots", type=int, required=True,
                    help="labels per class for every head (match the cell's "
                         "own budget, e.g. 5 for a 1%% C100/tin cell)")
    ap.add_argument("--draws", type=int, default=5,
                    help="stratified label-draws averaged per seed")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds_name = cfg["dataset"]
    num_classes = data_mod.NUM_CLASSES[ds_name]

    train_ds = set_eval_transform(
        data_mod.build_dataset(ds_name, args.data_root, train=True), ds_name
    )
    test_ds = data_mod.build_dataset(ds_name, args.data_root, train=False)
    tr_loader = DataLoader(train_ds, batch_size=512, num_workers=4,
                           shuffle=False)
    te_loader = DataLoader(test_ds, batch_size=512, num_workers=4,
                           shuffle=False)

    heads = {"linear": probe, "cosine": probe_cosine, "ncm": probe_ncm}
    out = []
    for seed_dir in sorted(d for d in os.listdir(args.run)
                           if d.startswith("seed")):
        ckpt = os.path.join(args.run, seed_dir, args.ckpt)
        if not os.path.exists(ckpt):
            continue
        model = build_model(
            cfg["backbone"], cfg.get("stem", "none"),
            num_classes=num_classes,
            small_input=cfg.get("small_input", True),
            stem_kernel_size=cfg.get("stem_kernel_size", 11),
            stem_kwargs=cfg.get("stem_kwargs"),
            head=cfg.get("head"),
            moment_aux=cfg.get("moment_aux"),
        ).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.eval()
        with torch.no_grad():
            trf, trY = extract(model, tr_loader, device)
            tef, teY = extract(model, te_loader, device)

        rec = {"seed": seed_dir, "shots": args.shots}
        for hname, fn in heads.items():
            tes = []
            for draw in range(args.draws):
                idx = shots_subset(trY, args.shots, seed=draw)
                sub_f, sub_y = trf[idx], trY[idx]
                _, te = fn(sub_f, sub_y, tef, teY, num_classes, device)
                tes.append(te)
            t = torch.tensor(tes)
            rec[hname] = {"test_mean": t.mean().item(),
                          "test_std": t.std().item()}
            print(f"{os.path.basename(args.run)} {seed_dir} {hname:6s} "
                  f"@{args.shots}/cls: {t.mean()*100:.2f} +/- {t.std()*100:.2f}",
                  flush=True)
        out.append(rec)

    path = os.path.join(args.run, f"head_forms_{args.shots}shot.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"-> {path}")


if __name__ == "__main__":
    main()
