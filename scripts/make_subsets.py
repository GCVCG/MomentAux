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
PCTS = {
    "cifar100": (1, 2, 3, 5, 7, 10, 15, 25),
    "cifar10": (1, 2, 3, 5, 7, 10, 15, 25),
    "stl10": (10, 20, 50),
    "tin": (1, 2, 3, 5, 7, 10, 15, 25),
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
