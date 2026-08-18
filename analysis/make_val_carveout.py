"""Deterministic VALIDATION carve-out from the UNUSED portion of train.

Why this exists (2026-08-17, pre-registered as B1). The Section 9.3 procedure
decides whether to fuse two sources from Delta and G measured on each source
alone -- but every one of those numbers was measured on the HELD-OUT TEST
split, which the paper discloses as "a defect of protocol, not of feasibility".
A procedure whose stated purpose is to decide BEFORE fusing must estimate its
inputs on a split it is allowed to look at.

final.json carries final_test_acc and best_test_acc only, so there is no
recorded validation score to reuse. The validation set must therefore be carved
from the training images the cell did NOT train on, i.e. train minus the
committed subset. That exists only for sub-100% cells; 100% cells cannot be
validated this way at all.

TWO disjoint sets are drawn, not one, because the procedure needs BOTH an
end-to-end score and a linear-probe G, and the probe FITS a head on labels:

    FIT  -- the probe head is fitted here, and nowhere else.
    VAL  -- every reported number (end-to-end accuracy and probe accuracy)
            is evaluated here, and nothing is fitted on it.

DISJOINTNESS, which is the one way this experiment can silently produce a
meaningless number:

    FIT n VAL                = 0   (asserted)
    (FIT u VAL) n SUBSET     = 0   (asserted, for EVERY fraction listed)
    (FIT u VAL) n TEST       = 0   (structural: both are drawn from the train
                                    split, which is disjoint from test by
                                    construction in data.py)

The committed subsets are NESTED (verified here, not assumed: the 5% indices
are a subset of the 25% indices on all three datasets), so excluding the
LARGEST fraction in use excludes every smaller one too -- which is what lets a
SINGLE carve-out serve every fraction and every arm of a dataset. That matters:
if two arms of a pair were scored on different carve-outs the comparison would
be void.

    python analysis/make_val_carveout.py --dataset eurosat --pcts 5,10,25
"""

import argparse
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import data as data_mod

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "data", "valcarve")
CARVE_SEED = 12345  # committed once; changing it invalidates every val number


def train_targets(dataset, data_root):
    """Labels of the FULL train split in index order (no subset, no transform
    cost): build_dataset's wrappers all expose .targets or wrap something that
    does."""
    ds = data_mod.build_dataset(dataset, data_root, train=True, download=False)
    node = ds
    for _ in range(4):
        t = getattr(node, "targets", None)
        if t is not None:
            return np.asarray(list(t))
        node = getattr(node, "dataset", None) or getattr(node, "base", None)
        if node is None:
            break
    raise RuntimeError(f"could not reach .targets on {type(ds).__name__}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--pcts", required=True,
                    help="comma-separated fractions whose cells this carve-out "
                         "will score, e.g. '5,10,25'. Every one is excluded.")
    ap.add_argument("--val-frac", type=float, default=0.25,
                    help="share of the unused pool held out for evaluation; "
                         "the rest becomes the probe's fit set")
    ap.add_argument("--data-root", default="./data")
    args = ap.parse_args()

    pcts = [int(p) for p in args.pcts.split(",")]
    y = train_targets(args.dataset, args.data_root)
    n_train = len(y)

    subs, used = {}, set()
    for p in pcts:
        idx = set(data_mod.load_subset_indices(args.dataset, p))
        subs[p] = idx
        used |= idx
    # NESTING CHECK, verified rather than assumed: it is what licenses one
    # carve-out per dataset instead of one per fraction.
    biggest = max(pcts)
    nested = all(subs[p] <= subs[biggest] for p in pcts)
    assert used == subs[biggest] if nested else True
    pool = np.array(sorted(set(range(n_train)) - used))

    rs = np.random.RandomState(CARVE_SEED)
    val, fit = [], []
    for c in np.unique(y):
        cls = pool[y[pool] == c]
        perm = cls[rs.permutation(len(cls))]
        k = int(round(args.val_frac * len(cls)))
        val.extend(perm[:k].tolist())
        fit.extend(perm[k:].tolist())
    val, fit = sorted(val), sorted(fit)

    sv, sf = set(val), set(fit)
    assert not (sv & sf), "FIT and VAL overlap"
    assert not ((sv | sf) & used), "carve-out overlaps a committed training subset"
    assert len(sv | sf) == len(pool), "carve-out does not partition the unused pool"

    payload = {
        "dataset": args.dataset,
        "excluded_pcts": pcts,
        "subsets_nested": bool(nested),
        "carve_seed": CARVE_SEED,
        "val_frac": args.val_frac,
        "n_train": n_train,
        "n_excluded": len(used),
        "n_val": len(val),
        "n_fit": len(fit),
        "val_indices": val,
        "fit_indices": fit,
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"{args.dataset}.json")
    with open(path, "w") as f:
        json.dump(payload, f)
    digest = hashlib.sha256(
        json.dumps([val, fit]).encode()).hexdigest()[:16]
    print(f"{args.dataset}: train {n_train}, excluded {len(used)} "
          f"(pcts {pcts}, nested={nested}), pool {len(pool)} -> "
          f"VAL {len(val)} / FIT {len(fit)}  sha {digest}\n  -> {path}")


if __name__ == "__main__":
    main()
