"""Generate the committed CIFAR-100 label-subset index files (run ONCE, then
commit data/subsets/*.json). Every stem trains on these identical indices.

    python scripts/make_subsets.py [--data-root ./data]
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torchvision import datasets

from data import SUBSET_DIR, SUBSET_SEED, make_subset_indices, subset_path

PCTS = (1, 5, 10, 25)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="./data")
    args = ap.parse_args()

    ds = datasets.CIFAR100(args.data_root, train=True, download=True)
    labels = ds.targets
    os.makedirs(SUBSET_DIR, exist_ok=True)
    for pct in PCTS:
        indices = make_subset_indices(labels, pct, seed=SUBSET_SEED)
        payload = {
            "dataset": "cifar100",
            "pct": pct,
            "seed": SUBSET_SEED,
            "n": len(indices),
            "indices": indices,
        }
        path = subset_path("cifar100", pct)
        with open(path, "w") as f:
            json.dump(payload, f)
        print(f"wrote {path}: {len(indices)} indices")


if __name__ == "__main__":
    main()
