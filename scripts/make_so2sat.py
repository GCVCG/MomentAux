"""Pack So2Sat LCZ42 into a memmappable array with SAR and optical separated.

So2Sat is this study's CROSS-MODALITY fusion population. Each 32x32 patch
carries both Sentinel-1 SAR (8 channels: VV/VH real and imaginary, their
intensities, and the PolSAR covariance terms) and Sentinel-2 optical
(10 surface-reflectance bands), co-registered, with one of 17 Local Climate
Zone labels. Unlike the multispectral EuroSAT grid, where the two "sources"
are band groups from a single instrument, here they are different physical
sensing principles: active radar backscatter against passive reflectance.

Splits: we use the v4 VALIDATION split as the training pool and the v4
TESTING split as the test set. Both are held out from the training cities,
so this is a harder generalisation setting than the canonical split, and
the study trains on 1-25% subsets in any case.

Channels are stored as float32 (the source is float64, which doubles the
footprint for no precision that matters here), and per-channel statistics
are computed on the training pool only.

Usage:  python scripts/make_so2sat.py
"""
import argparse, os
import numpy as np
import h5py


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/so2sat")
    ap.add_argument("--out", default="data/so2sat_32")
    a = ap.parse_args()

    parts = {}
    for split, fn in (("train", "validation.h5"), ("test", "testing.h5")):
        with h5py.File(os.path.join(a.root, fn), "r") as h:
            s1 = np.asarray(h["sen1"], dtype=np.float32)   # (N,32,32,8)
            s2 = np.asarray(h["sen2"], dtype=np.float32)   # (N,32,32,10)
            y = np.asarray(h["label"]).argmax(1).astype(np.int64)
        x = np.concatenate([s1, s2], axis=3)               # (N,32,32,18)
        x = np.transpose(x, (0, 3, 1, 2))                  # (N,18,32,32)
        parts[split] = (x, y)
        print(f"  {split:<6} {x.shape} labels {y.shape} "
              f"classes {len(np.unique(y))}", flush=True)

    xtr, ytr = parts["train"]
    xte, yte = parts["test"]
    mean = xtr.mean(axis=(0, 2, 3))
    std = xtr.std(axis=(0, 2, 3))
    print("  per-channel mean (SAR 0-7, optical 8-17):", np.round(mean, 4))
    print("  per-channel std :", np.round(std, 4))

    n_tr = len(ytr)
    allx = np.concatenate([xtr, xte], axis=0)
    ally = np.concatenate([ytr, yte], axis=0)
    np.save(a.out + "_images.npy", allx)
    np.savez(a.out + "_meta.npz", labels=ally,
             train_idx=np.arange(n_tr),
             test_idx=np.arange(n_tr, len(ally)),
             mean=mean, std=std)
    print(f"  wrote {a.out}_images.npy "
          f"({os.path.getsize(a.out + '_images.npy')/1e9:.2f} GB) and _meta.npz")


if __name__ == "__main__":
    main()
