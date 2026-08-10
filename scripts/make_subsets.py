"""Generate the committed label-subset index files (run once per dataset, then
commit data/subsets/*.json). Every stem trains on these identical indices.

    python scripts/make_subsets.py [--data-root ./data] [--dataset cifar100]
    python scripts/make_subsets.py --check      # verify committed files match

Deterministic: make_subset_indices(labels, pct, SUBSET_SEED) is a pure function
of (labels, pct, seed), so re-running reproduces byte-identical files. --check
asserts exactly that against what is committed, which is what makes "the subsets
are committed" a real guarantee rather than a hope.

cifar100super deliberately has NO subsets of its own -- it reuses cifar100's
indices (see SUBSET_ALIAS in data.py) so that the two differ only in label
granularity.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torchvision import datasets

from data import SUBSET_DIR, SUBSET_SEED, make_subset_indices, subset_path

# STL-10's train split is only 5000 images (500/class), so its fractions are
# chosen to stay above the frozen recipe's step floor: at batch 128 with
# drop_last=True, 10% = 500 imgs = 3 batches/epoch = 600 steps -- which exactly
# matches cifar10 @1% (500 imgs, 50/class, 600 steps), making the pair a clean
# resolution-only comparison (32x32 vs 96x96).
# Tiny-ImageNet: 100k train / 200 classes / 500 per class at 64x64. Same
# fractions as the CIFAR pair so the envelopes line up point-for-point; note
# tin @1% = 5 img/class exactly like cifar100 @1%, but with 2x the classes and
# 2x the images (1000 vs 500) -- another angle on the granularity confound.
# FULL GRID (2026-07-22, user decision "no missing data for any configuration"):
# every dataset carries every fraction whose subset still yields at least one
# batch of 128 with drop_last=True. Below that the loader is EMPTY and the cell
# cannot train at all -- those cells are impossible, not merely unrun:
#   stl10 @1-2% (50/100 imgs), cub @1-2% (60/120), tin20/tin20b @1% (100).
# cifar100super/tinsuper/tinsem reuse cifar100/tin indices via SUBSET_ALIAS.
FULL = (1, 2, 3, 5, 7, 10, 15, 20, 25, 50)
PCTS = {
    "cifar100": FULL,
    "cifar10": FULL,
    "stl10": (3, 5, 7, 10, 15, 20, 25, 50),
    "tin": FULL,
    "tin20": (2, 3, 5, 7, 10, 15, 20, 25, 50),
    "tin20b": (2, 3, 5, 7, 10, 15, 20, 25, 50),
    "cub": (3, 5, 7, 10, 15, 20, 25, 50),
    # Domain-generalization datasets (2026-07-23). Sub-one-batch floors:
    # eurosat 21600 train -> 1% = 216 >= 128, all fractions live.
    # dtd 3760 train -> 1%/2%/3% = 37/75/112 < 128 impossible; 5% = 188 ok.
    # pathmnist 89996 -> all fractions live (1% = 899).
    # food101 75750 -> all fractions live (1% = 757).
    "eurosat": FULL,
    "dtd": (5, 7, 10, 15, 20, 25, 50),
    "pathmnist": FULL,
    "food101": FULL,
    # ImageNet stages (2026-08-10). These were originally run at 100% only, to
    # test whether the scale falsifiers fire; the envelope -- peak location and
    # left-flank suppression -- was left untested above 100k images, which the
    # limitations section flags as the cheapest open gap. It is cheap because
    # the diagnostic epoch budget is FIXED (40 for imagenet64, 100 for
    # imagenet100) regardless of fraction, so a 1% cell costs ~1% of its 100%
    # cell.
    # imagenet64: 1,281,167 train / 1000 classes -> 1% = 12,811 images, 13 per
    # class. That is the FINEST label space in the study at genuine scarcity,
    # which is exactly where the readout term is largest. 50% is omitted: it
    # would cost as much as every other fraction combined while the 100% cell
    # already anchors the right flank.
    "imagenet64": (1, 2, 3, 5, 7, 10, 15, 20, 25),
    # imagenet100: 126,689 train / 100 classes at NATIVE 224px, so each cell is
    # far more expensive per image; carry the five fractions that resolve the
    # envelope shape rather than the full grid. 1% = 1,267 images (13 per
    # class), still ten batches of 128.
    "imagenet100": (1, 2, 5, 10, 25),
}


def get_labels(dataset, data_root):
    if dataset == "cifar100":
        return datasets.CIFAR100(data_root, train=True, download=True).targets
    if dataset == "cifar10":
        return datasets.CIFAR10(data_root, train=True, download=True).targets
    if dataset == "stl10":
        return datasets.STL10(data_root, split="train", download=True).labels
    if dataset == "tin":
        import os

        from data import tin_root

        return datasets.ImageFolder(os.path.join(tin_root(data_root), "train")).targets
    if dataset in ("tin20", "tin20b"):
        import os

        from data import tin20_filter, tin20_wnids, tin20b_wnids, tin_root

        root = tin_root(data_root)
        base = datasets.ImageFolder(os.path.join(root, "train")).targets
        wn = tin20_wnids(root) if dataset == "tin20" else tin20b_wnids(root)
        _, new_targets = tin20_filter(base, root, keep_wnids=wn)
        return new_targets
    if dataset == "cub":
        from data import CUB200

        return CUB200(data_root, train=True).targets
    if dataset in ("eurosat", "dtd", "pathmnist", "food101"):
        from data import build_dataset

        return build_dataset(dataset, data_root, train=True,
                             download=False).targets
    if dataset == "imagenet64":
        # Read the label array directly rather than constructing ImageNet64:
        # the constructor memory-maps a 15.7GB image array this does not need.
        # `os` is re-imported because the tin branches above import it inside
        # this function, which makes the name function-local throughout.
        import os

        import numpy as np

        return [int(t) for t in
                np.load(os.path.join(data_root, "imagenet64", "train_y.npy"))]
    if dataset == "imagenet100":
        from data import ImageNet100

        return ImageNet100(data_root, train=True).targets
    raise ValueError(f"unknown dataset {dataset!r}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--dataset", choices=sorted(PCTS), default=None,
                    help="default: all datasets")
    ap.add_argument("--check", action="store_true",
                    help="verify committed files reproduce; write nothing")
    args = ap.parse_args()

    os.makedirs(SUBSET_DIR, exist_ok=True)
    targets = [args.dataset] if args.dataset else sorted(PCTS)
    failures = 0
    for ds_name in targets:
        labels = get_labels(ds_name, args.data_root)
        for pct in PCTS[ds_name]:
            indices = make_subset_indices(labels, pct, seed=SUBSET_SEED)
            path = subset_path(ds_name, pct)
            payload = {
                "dataset": ds_name, "pct": pct, "seed": SUBSET_SEED,
                "n": len(indices), "indices": indices,
            }
            if args.check:
                if not os.path.exists(path):
                    print(f"MISSING  {path}")
                    failures += 1
                    continue
                with open(path) as f:
                    have = json.load(f)
                ok = have["indices"] == indices
                print(f"{'ok      ' if ok else 'MISMATCH'} {path} (n={len(indices)})")
                failures += not ok
            else:
                with open(path, "w") as f:
                    json.dump(payload, f)
                print(f"wrote {path}: {len(indices)} indices")
    if args.check:
        print("all committed subsets reproduce" if not failures
              else f"{failures} FILE(S) DO NOT REPRODUCE")
        sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
