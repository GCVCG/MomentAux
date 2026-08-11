"""Decision-level fusion of trained arms, as a second test of the currency rule.

The stack/substitute/tax taxonomy was derived from TRAINING-time fusion: two
information sources injected into one network. Multi-classifier / decision
fusion is a different level of fusion entirely, and it is one of the
Information Fusion journal's own listed topics. If the currency account is
about the sources rather than about the injection mechanism, it should
predict decision-level outcomes too:

  two arms whose gains come from the SAME currency should make CORRELATED
  errors and ensemble poorly; arms carrying DIFFERENT currencies should
  decorrelate and ensemble well.

This script needs no training. It loads existing checkpoints, caches each
model's test-set probabilities, and reports, for every pair of arms:
  - the ensemble accuracy (mean of softmax over the two arms' seeds)
  - the gain over the better single arm
  - the disagreement rate between the arms (the classical ensemble
    diversity measure, cf. Brown et al., Information Fusion 2005)

Usage:
  python analysis/decision_fusion.py --cells cells.txt --out results/decision_fusion.json
"""
import argparse, itertools, json, os, sys
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data as data_mod
from momentstem import build_model


def find_config(cell):
    for root in ("configs/grid", "configs/diagnostics", "configs/ablations_full",
                 "configs", "configs/expansion"):
        p = os.path.join(root, cell + ".yaml")
        if os.path.isfile(p):
            return p
    hits = []
    for dirpath, _, files in os.walk("configs"):
        if cell + ".yaml" in files:
            hits.append(os.path.join(dirpath, cell + ".yaml"))
    return hits[0] if hits else None


@torch.no_grad()
def probs_for_cell(cell, device, batch=256):
    """Mean softmax over the cell's seeds, plus per-seed argmax predictions."""
    cfgp = find_config(cell)
    if cfgp is None:
        return None
    cfg = yaml.safe_load(open(cfgp))
    ds = cfg["dataset"]
    test_ds = data_mod.build_dataset(ds, "data", train=False)
    loader = DataLoader(test_ds, batch_size=batch, num_workers=2, shuffle=False)
    seeds = sorted(d for d in os.listdir(os.path.join("runs", cell))
                   if d.startswith("seed")
                   and os.path.isfile(os.path.join("runs", cell, d, "best.pt")))
    if len(seeds) < 3:
        return None
    acc_p, y_ref = [], None
    for sd in seeds:
        model = build_model(
            cfg["backbone"], cfg.get("stem", "none"),
            num_classes=data_mod.NUM_CLASSES[ds],
            small_input=cfg.get("small_input", True),
            stem_kernel_size=cfg.get("stem_kernel_size", 11),
            stem_kwargs=cfg.get("stem_kwargs"),
            head_pool=cfg.get("head_pool"), head=cfg.get("head"),
            moment_aux=cfg.get("moment_aux"),
            image_size=data_mod.IMAGE_SIZE[ds]).to(device)
        model.load_state_dict(torch.load(
            os.path.join("runs", cell, sd, "best.pt"), map_location=device))
        model.eval()
        ps, ys = [], []
        for xb, yb in loader:
            out = model(xb.to(device, non_blocking=True))
            if isinstance(out, (tuple, list)):
                out = out[0]
            ps.append(F.softmax(out.float(), dim=1).cpu())
            ys.append(yb)
        acc_p.append(torch.cat(ps).numpy())
        y = torch.cat(ys).numpy()
        y_ref = y if y_ref is None else y_ref
        del model
        torch.cuda.empty_cache()
    return np.stack(acc_p), y_ref



def fit_gain_model(pairs):
    """OLS of ensemble gain on disagreement and accuracy asymmetry.

    Eq. (5) of the paper was fitted once, by hand, and reported without
    uncertainty -- the only equation in the study with no script behind it.
    It is refitted here from the released pair records, so a reader can
    reproduce the coefficients and see how well determined they are.

        g_ij = b0 + b_d * d_ij + b_a * a_ij

    with g the gain over the better single arm (points), d the disagreement
    rate (percent of test images on which the two arms predict different
    classes) and a = |acc_i - acc_j| the accuracy asymmetry (points).
    """
    X = np.array([[1.0, r["disagreement"], abs(r["acc_a"] - r["acc_b"])]
                  for r in pairs])
    y = np.array([r["gain_over_best"] for r in pairs])
    n, k = X.shape
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    rss = float(resid @ resid)
    tss = float(((y - y.mean()) ** 2).sum())
    s2 = rss / (n - k)
    se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    return {"n": n, "r2": 1 - rss / tss, "residual_sd": float(np.sqrt(s2)),
            "terms": {name: {"coef": float(b), "se": float(e), "t": float(b / e)}
                      for name, b, e in zip(("intercept", "disagreement",
                                             "acc_asymmetry"), beta, se)}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", help="file, one cell name per line")
    ap.add_argument("--fit", metavar="JSON",
                    help="refit the gain model from an existing results file "
                         "and print the coefficients; runs no networks")
    ap.add_argument("--out", default="results/decision_fusion.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    if a.fit:
        with open(a.fit) as f:
            r = fit_gain_model(json.load(f)["pairs"])
        print(f"n = {r['n']}   R^2 = {r['r2']:.3f}   "
              f"residual SD = {r['residual_sd']:.3f} points")
        for name, t in r["terms"].items():
            print(f"  {name:<14s} {t['coef']:+8.4f} +- {t['se']:.4f}  "
                  f"(t = {t['t']:+.2f})")
        return

    if not a.cells:
        ap.error("--cells is required unless --fit is given")
    cells = [l.strip() for l in open(a.cells) if l.strip() and not l.startswith("#")]
    dev = torch.device(a.device)

    P, Y = {}, None
    for c in cells:
        try:
            r = probs_for_cell(c, dev)
        except Exception as e:
            print(f"  {c}: FAILED ({type(e).__name__}: {e})", flush=True)
            continue
        if r is None:
            print(f"  {c}: skipped (no config or <3 seeds)", flush=True)
            continue
        P[c], Y = r[0], r[1]
        acc = (P[c].mean(0).argmax(1) == Y).mean() * 100
        print(f"  {c:<44} seeds={P[c].shape[0]}  seed-ens acc={acc:5.2f}", flush=True)

    out = {"cells": {}, "pairs": []}
    for c, p in P.items():
        singles = [(p[i].argmax(1) == Y).mean() * 100 for i in range(p.shape[0])]
        out["cells"][c] = {"mean_single": float(np.mean(singles)),
                           "seed_ensemble": float((p.mean(0).argmax(1) == Y).mean() * 100)}
    for c1, c2 in itertools.combinations(P, 2):
        p1, p2 = P[c1].mean(0), P[c2].mean(0)
        a1 = (p1.argmax(1) == Y).mean() * 100
        a2 = (p2.argmax(1) == Y).mean() * 100
        ens = (((p1 + p2) / 2).argmax(1) == Y).mean() * 100
        disagree = (p1.argmax(1) != p2.argmax(1)).mean() * 100
        out["pairs"].append({"a": c1, "b": c2, "acc_a": float(a1), "acc_b": float(a2),
                             "ensemble": float(ens), "gain_over_best": float(ens - max(a1, a2)),
                             "disagreement": float(disagree)})
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"\nwrote {a.out}: {len(P)} cells, {len(out['pairs'])} pairs")
    for r in sorted(out["pairs"], key=lambda r: -r["gain_over_best"])[:10]:
        print(f"  +{r['gain_over_best']:5.2f}  {r['a'][:34]:<34} + {r['b'][:34]:<34} "
              f"(disagree {r['disagreement']:.1f}%)")


if __name__ == "__main__":
    main()
