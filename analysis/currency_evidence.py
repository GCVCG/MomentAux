"""Does 'same currency' show up in WHAT an intervention changes, not just how much?

The taxonomy in the paper is inferred from the MAGNITUDE of each source's
feature gain: sources with similar G are called substitutes, sources with
dissimilar G are called complementary. That is indirect. Two interventions
could each add +14 points of feature gain while improving completely
different images, in which case calling them "the same currency" would be an
artifact of comparing scalars.

This script tests the claim directly, at two levels:

  (1) PREDICTION LEVEL. Which test images does each intervention FIX relative
      to a shared baseline? Two sources of the same currency should fix
      overlapping sets; different currencies should fix different images.

  (2) REPRESENTATION LEVEL. Linear CKA between the arms' frozen penultimate
      features. Same currency should produce more similar representations.

THE CONFOUND, and why raw overlap would be meaningless: some test images are
simply easier, so ANY two interventions overlap above chance. The control is
the intervention's overlap WITH ITSELF ACROSS SEEDS -- the most same-currency
pair that exists. Every cross-intervention number is normalized by it:

    S(A,B) = across(A,B) / sqrt(within(A) * within(B))

S = 1 means "A and B differ no more than two seeds of A differ", i.e. as
same-currency as measurable. S = 0 means no shared structure at all.

The design is controlled: all arms share ONE backbone (ViT-tiny), ONE
dataset/fraction, and ONE baseline, so the only thing that varies is which
information source was fused in.

Usage:
  python analysis/currency_evidence.py --spec analysis/currency_spec_vit.json
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data as data_mod
from momentstem import build_model
from analysis.linear_probe import extract


@torch.no_grad()
def predict(model, loader, device):
    """Test-set predictions and penultimate features from one checkpoint."""
    preds, ys = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast("cuda", enabled=device.type == "cuda"):
            logits = model(x)
            if isinstance(logits, (tuple, list)):  # aux models return (logits, aux)
                logits = logits[0]
        preds.append(logits.float().argmax(1).cpu())
        ys.append(y)
    return torch.cat(preds).numpy(), torch.cat(ys).numpy()


def load_arm(cell, cfg_path, run_root, data_root, device, ckpt="best.pt"):
    """Every seed of one cell: correctness mask + penultimate features."""
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    ds = cfg["dataset"]
    test_ds = data_mod.build_dataset(ds, data_root, train=False)
    loader = DataLoader(test_ds, batch_size=256, num_workers=3, shuffle=False)

    correct, feats, accs = [], [], []
    run_dir = os.path.join(run_root, cell)
    for seed_dir in sorted(d for d in os.listdir(run_dir) if d.startswith("seed")):
        path = os.path.join(run_dir, seed_dir, ckpt)
        if not os.path.exists(path):
            continue
        model = build_model(
            cfg["backbone"], cfg.get("stem", "none"),
            num_classes=data_mod.NUM_CLASSES[ds],
            small_input=cfg.get("small_input", True),
            stem_kernel_size=cfg.get("stem_kernel_size", 11),
            stem_kwargs=cfg.get("stem_kwargs"),
            head_pool=cfg.get("head_pool"),
            head=cfg.get("head"),
            moment_aux=cfg.get("moment_aux"),
            image_size=data_mod.IMAGE_SIZE[ds],
            in_channels=data_mod.INPUT_CHANNELS.get(ds, 3),
        ).to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()

        p, y = predict(model, loader, device)
        f, _ = extract(model, loader, device)
        correct.append(p == y)
        feats.append(f.numpy())
        accs.append(float((p == y).mean() * 100))
        del model
        torch.cuda.empty_cache()
        print(f"  {cell}/{seed_dir}: acc {accs[-1]:.2f}", flush=True)
    return np.array(correct), feats, accs


def jaccard(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / union) if union else 0.0


def overlap_coef(a, b):
    """|A and B| / min(|A|,|B|). Unlike Jaccard this is insensitive to the two
    sets having different SIZES, which matters here: a weaker intervention
    fixes fewer images, and Jaccard would penalize it for that alone."""
    m = min(a.sum(), b.sum())
    return float(np.logical_and(a, b).sum() / m) if m else 0.0


def lift(a, b):
    """P(B fixes | A fixes) / P(B fixes). Size-normalized by construction:
    1.0 means the two interventions fix images independently, higher means
    they concentrate on the same images."""
    pa = a.mean()
    if pa == 0 or b.mean() == 0:
        return 0.0
    return float(b[a].mean() / b.mean())


def linear_cka(X, Y):
    """Linear CKA between two feature matrices (rows = the same images)."""
    X = X - X.mean(0, keepdims=True)
    Y = Y - Y.mean(0, keepdims=True)
    # ||Y^T X||_F^2 / (||X^T X||_F ||Y^T Y||_F), the standard linear form.
    xty = X.T @ Y
    num = float((xty ** 2).sum())
    den = float(np.linalg.norm(X.T @ X) * np.linalg.norm(Y.T @ Y))
    return num / den if den else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True,
                    help="JSON: {baseline: {cell, config}, arms: {label: {cell, config}}}")
    ap.add_argument("--run-root", default="runs")
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.spec) as f:
        spec = json.load(f)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("baseline:", spec["baseline"]["cell"], flush=True)
    base_correct, base_feats, base_accs = load_arm(
        spec["baseline"]["cell"], spec["baseline"]["config"],
        args.run_root, args.data_root, device)

    arms = {}
    for label, d in spec["arms"].items():
        print(f"arm {label}: {d['cell']}", flush=True)
        c, f, a = load_arm(d["cell"], d["config"], args.run_root,
                           args.data_root, device)
        # FIXED set, seed-matched to the baseline: correct here, wrong there.
        n = min(len(c), len(base_correct))
        fixed = np.array([np.logical_and(c[i], ~base_correct[i]) for i in range(n)])
        arms[label] = {"correct": c, "feats": f, "accs": a, "fixed": fixed,
                       "cell": d["cell"]}

    labels = list(arms)
    res = {
        "baseline": {"cell": spec["baseline"]["cell"],
                     "acc_mean": float(np.mean(base_accs)), "accs": base_accs},
        "arms": {l: {"cell": arms[l]["cell"],
                     "acc_mean": float(np.mean(arms[l]["accs"])),
                     "accs": arms[l]["accs"],
                     "delta": float(np.mean(arms[l]["accs"]) - np.mean(base_accs)),
                     "n_fixed_mean": float(arms[l]["fixed"].sum(1).mean())}
                 for l in labels},
    }

    # ---- within-intervention agreement (the normalizer) --------------------
    within_fix, within_cka, within_ovl, within_lift = {}, {}, {}, {}
    for l in labels:
        fx, ft = arms[l]["fixed"], arms[l]["feats"]
        pairs = [(i, j) for i in range(len(fx)) for j in range(i + 1, len(fx))]
        within_fix[l] = float(np.mean([jaccard(fx[i], fx[j]) for i, j in pairs]))
        within_ovl[l] = float(np.mean([overlap_coef(fx[i], fx[j]) for i, j in pairs]))
        within_lift[l] = float(np.mean([lift(fx[i], fx[j]) for i, j in pairs]))
        within_cka[l] = float(np.mean([linear_cka(ft[i], ft[j]) for i, j in pairs]))

    # ---- across-intervention agreement ------------------------------------
    pairwise = {}
    for a in range(len(labels)):
        for b in range(a + 1, len(labels)):
            la, lb = labels[a], labels[b]
            fa, fb = arms[la]["fixed"], arms[lb]["fixed"]
            ta, tb = arms[la]["feats"], arms[lb]["feats"]
            acr_fix = float(np.mean([jaccard(x, y) for x in fa for y in fb]))
            acr_ovl = float(np.mean([overlap_coef(x, y) for x in fa for y in fb]))
            acr_lift = float(np.mean([lift(x, y) for x in fa for y in fb]))
            acr_cka = float(np.mean([linear_cka(x, y) for x in ta for y in tb]))
            pairwise[f"{la}|{lb}"] = {
                "fixed_jaccard_across": acr_fix,
                "fixed_similarity": acr_fix / float(np.sqrt(within_fix[la] * within_fix[lb])),
                "overlap_coef_across": acr_ovl,
                "overlap_similarity": acr_ovl / float(np.sqrt(within_ovl[la] * within_ovl[lb])),
                "lift_across": acr_lift,
                "lift_similarity": acr_lift / float(np.sqrt(within_lift[la] * within_lift[lb])),
                "cka_across": acr_cka,
                "cka_similarity": acr_cka / float(np.sqrt(within_cka[la] * within_cka[lb])),
            }
    res["within"] = {"fixed_jaccard": within_fix, "overlap_coef": within_ovl,
                     "lift": within_lift, "cka": within_cka}
    res["pairwise"] = pairwise

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)

    print("\n=== accuracy ===")
    print(f"  baseline {res['baseline']['acc_mean']:.2f}")
    for l in labels:
        print(f"  {l:14s} {res['arms'][l]['acc_mean']:6.2f}  "
              f"delta {res['arms'][l]['delta']:+6.2f}  "
              f"fixes {res['arms'][l]['n_fixed_mean']:.0f} images")
    print("\n=== within-intervention (seed-to-seed) agreement ===")
    for l in labels:
        print(f"  {l:14s} fixed-J {within_fix[l]:.3f}   CKA {within_cka[l]:.3f}")
    print("\n=== across-intervention, normalized by the above ===")
    for k, v in pairwise.items():
        print(f"  {k:22s} fixed-sim {v['fixed_similarity']:.3f}  "
              f"ovl-sim {v['overlap_similarity']:.3f}  "
              f"lift {v['lift_across']:.2f} (sim {v['lift_similarity']:.3f})  "
              f"CKA-sim {v['cka_similarity']:.3f}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
