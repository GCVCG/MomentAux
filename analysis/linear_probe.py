"""Linear-probe a trained checkpoint's frozen features on the FULL train set.

Why this exists (2026-07-16). MomentAux's gain collapses at extreme scarcity:
CIFAR-100 @1% gives only +1.91 where @5% gives +5.30, and the 1% tap sweep
falsified the "layer3 is too late" explanation (layer1 recovers nothing). The
surviving account is that at 5 img/class the CLASSIFIER, not the features, is
the bottleneck -- 5 examples cannot define a decision boundary however good the
representation is.

That account makes a sharp, falsifiable split which this script measures:

    if the aux model's FROZEN features probe BETTER than the baseline's when
    given plenty of labels, while its 1% end-to-end accuracy barely moved,
    then the prior DID improve the representation and the 1% classifier simply
    could not exploit it.  -> the left flank becomes a mechanism, not a defect.

    if the aux features probe NO better, the prior genuinely fails to shape
    features at 5 img/class and the account is wrong.

The probe deliberately uses the FULL labelled train set: the point is to remove
the classifier-data bottleneck so that only feature quality is measured. This is
a DIAGNOSTIC, not a headline cell -- it reads labels the training cell never saw,
and must never be mixed into a headline table.

    python analysis/linear_probe.py --run runs/auxmag_1pct_sched2 \
        --config configs/diagnostics/auxmag_1pct_sched2.yaml
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


def set_eval_transform(ds, dataset):
    """Force the deterministic (no crop/flip) transform. Handles the wrappers
    build_dataset may return: Subset(...) and Relabelled(...)."""
    tf = data_mod.build_transforms(dataset, train=False)
    node = ds
    for _ in range(4):
        if hasattr(node, "transform"):
            node.transform = tf
            return ds
        node = getattr(node, "dataset", None) or getattr(node, "ds", None)
        if node is None:
            break
    raise RuntimeError(f"could not reach .transform on {type(ds).__name__}")


@torch.no_grad()
def extract(model, loader, device):
    """Penultimate (post-GAP) features for every image. Goes through model.stem
    so this stays correct for forward-path stems, not just identity ones."""
    feats, ys = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast("cuda", enabled=device.type == "cuda"):
            f = model.net.forward_features(model.stem(x))
            gp = getattr(model.net, "global_pool", None)
            if gp is None:
                # timm ConvNeXt (2026-08-05): has NO top-level `global_pool` at
                # all -- pooling lives inside NormMlpClassifierHead -- so the
                # bare attribute access this line used to do raised
                # AttributeError before any branch was reached. Defer to timm's
                # own pre_logits path, as for Swin/MobileNetV3. Gated on the
                # attribute's ABSENCE, so every backbone that HAS a global_pool
                # keeps the exact branch its recorded G was measured under.
                f = model.net.forward_head(f, pre_logits=True)
            elif callable(gp):
                pooled = gp(f)
                if pooled.ndim > 2:
                    # timm MobileNetV3 (2026-07-29): global_pool IS callable,
                    # but its flatten is Identity AND a conv_head+act2 sits
                    # between pooling and the classifier -- so pooling alone
                    # is NOT the penultimate feature (576ch, 4D, vs
                    # classifier.in_features 1024). ResNets pool to 2D and
                    # keep the branch above, so every recorded conv G is
                    # measured under the identical code path.
                    f = model.net.forward_head(f, pre_logits=True)
                else:
                    f = pooled
            elif not hasattr(model.net, "fc_norm"):
                # timm ClassifierHead backbones (Swin, 2026-07-28): global_pool
                # is the STRING 'avg' exactly as on ViT, but the pooling module
                # lives inside .head and forward_features returns NHWC -- so the
                # ViT branches below would both mis-pool and reach for a
                # non-existent fc_norm. timm's own pre_logits path is the
                # authority; verified bit-identical to head.flatten(
                # head.global_pool(f)) on swin_tiny. Gated on fc_norm's ABSENCE
                # so every ViT cell keeps the exact branch its recorded G was
                # measured under.
                f = model.net.forward_head(f, pre_logits=True)
            elif gp == "token":  # timm ViT: global_pool is a STRING; the
                # penultimate feature is the CLS token (+ fc_norm, Identity
                # unless pool='avg'), mirroring timm's forward_head.
                f = model.net.fc_norm(f[:, 0])
            elif gp == "avg":
                f = model.net.fc_norm(f.mean(1))
            else:
                raise ValueError(f"unhandled global_pool {gp!r}")
        feats.append(f.float().cpu())
        ys.append(y)
    return torch.cat(feats), torch.cat(ys)


def shots_subset(y, per_class, seed=0):
    """Indices of a stratified `per_class`-shot subset (deterministic)."""
    import numpy as np

    ynp = y.numpy() if hasattr(y, "numpy") else np.asarray(y)
    out = []
    for c in np.unique(ynp):
        idx = np.flatnonzero(ynp == c)
        rs = np.random.RandomState(seed * 100003 + int(c))
        out.extend(idx[rs.permutation(len(idx))][:per_class].tolist())
    return sorted(out)


def probe(train_f, train_y, test_f, test_y, num_classes, device, max_iter=200):
    """Multinomial logistic regression on frozen, standardized features.
    Identical hyper-parameters for every cell -- only the features differ."""
    mu = train_f.mean(0, keepdim=True)
    sd = train_f.std(0, keepdim=True).clamp_min(1e-6)
    train_f = ((train_f - mu) / sd).to(device)
    test_f = ((test_f - mu) / sd).to(device)
    train_y, test_y = train_y.to(device), test_y.to(device)

    clf = torch.nn.Linear(train_f.shape[1], num_classes).to(device)
    opt = torch.optim.LBFGS(
        clf.parameters(), lr=1.0, max_iter=max_iter, history_size=10,
        line_search_fn="strong_wolfe",
    )

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(clf(train_f), train_y) + 1e-4 * clf.weight.pow(2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        tr = (clf(train_f).argmax(1) == train_y).float().mean().item()
        te = (clf(test_f).argmax(1) == test_y).float().mean().item()
    return tr, te


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="runs/<cell> (probes every seed)")
    ap.add_argument("--config", required=True)
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--shots", default=None,
                    help="comma-separated labels-per-class to refit the head on, "
                         "e.g. '5,10,25,50,100,500'. Measures how much of a "
                         "feature gain a classifier with N labels can actually "
                         "cash in. Omit to probe on the full train set only.")
    ap.add_argument("--shots-only", action="store_true",
                    help="skip the full-train-set head fit and run ONLY the "
                         "--shots budgets. Required at ImageNet64 scale: a "
                         "full-train LBFGS at 1.28M x feat_dim is impractical, "
                         "so those G values use a FIXED SHOTS budget and are "
                         "comparable ONLY to other same-budget probes, never "
                         "to the full-train G curve (pre-registered protocol, "
                         "2026-08-05). Extraction still covers every image; "
                         "only the head fit is restricted.")
    ap.add_argument("--probe-dataset", default=None,
                    help="probe the checkpoints' FEATURES under a different "
                         "dataset/label space than they were trained on (the "
                         "MODEL is still built from --config so the state_dict "
                         "loads; features are label-free, so any dataset with "
                         "the same input size works). Used for the 2x2 "
                         "train-space x probe-space crossing that separates "
                         "'training label space changed the features' from "
                         "'the probe label space changed the measuring stick'. "
                         "Output goes to linear_probe_<probe-dataset>.json, "
                         "never clobbering the default probe.")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds_name = args.probe_dataset or cfg["dataset"]
    if data_mod.IMAGE_SIZE[ds_name] != data_mod.IMAGE_SIZE[cfg["dataset"]]:
        raise ValueError(f"probe dataset {ds_name!r} image size != model's")
    num_classes = data_mod.NUM_CLASSES[ds_name]

    # FULL train split (no subset_pct) with the eval transform: we are measuring
    # feature quality with the classifier's data bottleneck removed.
    train_ds = set_eval_transform(
        data_mod.build_dataset(ds_name, args.data_root, train=True), ds_name
    )
    test_ds = data_mod.build_dataset(ds_name, args.data_root, train=False)
    # num_workers is SAFE to tune here (unlike training): shuffle=False and we
    # extract every feature, so worker count cannot change the result -- only
    # CPU pressure. Lowered from a hardcoded 8 and made overridable so many
    # probes can share a node: 2026-08-05 a 16-probe node ran 3 of 4 GPUs at
    # 0% while 128 loader workers fought over cores.
    _nw = int(os.environ.get("MS_PROBE_WORKERS", "3"))
    tr_loader = DataLoader(train_ds, batch_size=512, num_workers=_nw, shuffle=False)
    te_loader = DataLoader(test_ds, batch_size=512, num_workers=_nw, shuffle=False)

    out = []
    for seed_dir in sorted(d for d in os.listdir(args.run) if d.startswith("seed")):
        ckpt = os.path.join(args.run, seed_dir, args.ckpt)
        if not os.path.exists(ckpt):
            continue
        model = build_model(
            cfg["backbone"], cfg.get("stem", "none"),
            num_classes=data_mod.NUM_CLASSES[cfg["dataset"]],
            small_input=cfg.get("small_input", True),
            stem_kernel_size=cfg.get("stem_kernel_size", 11),
            stem_kwargs=cfg.get("stem_kwargs"),
            # head_pool was never plumbed (2026-08-05): MultiMaskPool cells
            # replace global_pool with a masked 4096-d readout, so the model
            # built here did not match their state_dict at all. Defaults to
            # None for every other cell, i.e. exactly the previous behaviour.
            head_pool=cfg.get("head_pool"),
            head=cfg.get("head"),
            moment_aux=cfg.get("moment_aux"),
            image_size=data_mod.IMAGE_SIZE[cfg["dataset"]],
            # >3 only for the multispectral sensor-fusion cells; without this
            # their checkpoints fail to load (conv1 is 13-channel, not 3).
            in_channels=data_mod.INPUT_CHANNELS.get(cfg["dataset"], 3),
        ).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.eval()

        trf, trY = extract(model, tr_loader, device)
        tef, teY = extract(model, te_loader, device)
        if args.shots_only:
            if not args.shots:
                raise SystemExit("--shots-only requires --shots")
            rec = {"seed": seed_dir, "shots_only": True}
            print(f"{os.path.basename(args.run)} {seed_dir}: shots-only "
                  f"({len(trY)} imgs extracted)", flush=True)
        else:
            tr_acc, te_acc = probe(trf, trY, tef, teY, num_classes, device)
            rec = {"seed": seed_dir, "probe_train": tr_acc, "probe_test": te_acc}
            print(f"{os.path.basename(args.run)} {seed_dir}: "
                  f"probe train {tr_acc*100:.2f}  test {te_acc*100:.2f}", flush=True)

        if args.shots:
            # Same frozen features, head refit on progressively more labels.
            # The whole point: if a feature gain is real but unrealisable at
            # 5 shots, the gap vs baseline must GROW with shots.
            rec["shots"] = {}
            for k in (int(s) for s in args.shots.split(",")):
                sel = shots_subset(trY, k)
                _, acc = probe(trf[sel], trY[sel], tef, teY, num_classes, device)
                rec["shots"][k] = acc
                print(f"    {k:>4} shots/class (n={len(sel):>5}): test {acc*100:.2f}",
                      flush=True)
        out.append(rec)

    # shots-only results live in their OWN file: they are same-budget-
    # comparable numbers, not the standard full-train probe, and writing them
    # to linear_probe.json would CLOBBER the recorded G-curve values (caught
    # live 2026-08-06 -- the first smoke of --shots-only overwrote
    # abl1_none's probe record, restored immediately after).
    if args.shots_only:
        fname = ("linear_probe_shots.json" if args.probe_dataset is None
                 else f"linear_probe_shots_{args.probe_dataset}.json")
    else:
        fname = ("linear_probe.json" if args.probe_dataset is None
                 else f"linear_probe_{args.probe_dataset}.json")
    path = os.path.join(args.run, fname)
    with open(path, "w") as f:
        json.dump({"config": args.config, "ckpt": args.ckpt, "results": out}, f, indent=2)
    if out:
        if args.shots_only:
            k = max(int(s) for s in args.shots.split(","))
            vals = [r["shots"][k] * 100 for r in out]
            tag = f"shots{k}"
        else:
            vals = [r["probe_test"] * 100 for r in out]
            tag = "test"
        mean = sum(vals) / len(vals)
        sd = (sum((v - mean) ** 2 for v in vals) / max(len(vals) - 1, 1)) ** 0.5
        print(f"PROBE_RESULT {os.path.basename(args.run)} "
              f"{tag} {mean:.2f}+/-{sd:.2f} over {len(vals)} seeds -> {path}", flush=True)


if __name__ == "__main__":
    main()
