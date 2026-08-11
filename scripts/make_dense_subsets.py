"""Generate the committed VOC segmentation subset indices.

Same contract as scripts/make_subsets.py: a pure function of (keys, pct, seed),
written once and committed, so every arm of every dense cell consumes
byte-identical images and --check proves it.

The stratification key is each image's DOMINANT non-background class, because
a segmentation image has no single label (see data_dense.py). At 1% that is
106 images over 20 classes; an unstratified draw would leave several classes
unrepresented and the cell would measure the draw rather than the method.

    python scripts/make_dense_subsets.py [--data-root ./data]
    python scripts/make_dense_subsets.py --check
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import data_dense as dd

PCTS = (1, 2, 5, 10, 25)     # 100% needs no index file

# ADE20K is 20,210 training images, so its 1% cell (202 images over 150
# classes) is far LEFT of anything VOC can express -- roughly 1.3 per class.
# Cityscapes is only 2,975, so its 1% is 30 images and its fractions climb
# fast. Same fraction grid on all three so the envelopes are directly
# comparable as fractions, which is the axis the tin@25% budget result showed
# actually governs the ordering.


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--dataset", default="voc_seg",
                    choices=sorted(dd.NUM_CLASSES))
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    ds = args.dataset
    split = dd.TRAIN_SPLIT[ds]
    ids = dd.read_split(args.data_root, split, ds)
    keys = dd.dominant_classes(args.data_root, ids, ds, split)
    os.makedirs(dd.SUBSET_DIR, exist_ok=True)

    failures = 0
    for pct in PCTS:
        idx = dd.make_subset_indices(keys, pct, seed=dd.SUBSET_SEED)
        path = dd.subset_path(pct, ds)
        payload = {"dataset": ds, "pct": pct, "seed": dd.SUBSET_SEED,
                   "n": len(idx), "n_pool": len(ids), "indices": idx}
        if args.check:
            if not os.path.exists(path):
                print(f"MISSING  {path}")
                failures += 1
                continue
            with open(path) as f:
                have = json.load(f)
            ok = have["indices"] == idx
            print(f"{'ok      ' if ok else 'MISMATCH'} {path} (n={len(idx)})")
            failures += not ok
        else:
            with open(path, "w") as f:
                json.dump(payload, f)
            # Report classes-per-cell, because that is what decides the regime
            # and it is the number a reader needs to judge the 1% cell.
            # Report STRATA covered, not "classes represented". They are not
            # the same thing and conflating them understates the draw: on
            # FoodSeg103 only 79 of 104 classes are ever an image's DOMINANT
            # class, so strata coverage reads 79/104 while the 1% subset
            # actually contains 88 of the 104 classes in its pixels -- a
            # segmentation image carries many classes, only one of which is
            # the stratification key.
            n_cls = dd.NUM_CLASSES[ds]
            n_strata = len(set(keys))
            covered = len({keys[i] for i in idx})
            print(f"wrote {path}: {len(idx)} images, {covered}/{n_strata} "
                  f"strata covered (of {n_cls} classes), "
                  f"{len(idx) / n_cls:.1f} images per class")
    if args.check:
        print("all committed dense subsets reproduce" if not failures
              else f"{failures} FILE(S) DO NOT REPRODUCE")
        sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
