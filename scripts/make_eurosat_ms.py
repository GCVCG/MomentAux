"""Pack EuroSAT multispectral (13-band Sentinel-2 GeoTIFF) into one array.

EuroSAT_MS is the same 27,000 tiles as the RGB EuroSAT we already use, but
with all 13 Sentinel-2 bands instead of the 3 visible ones. That makes it a
genuine MULTI-SENSOR setting: the visible bands and the near/short-wave
infrared and red-edge bands come from different detectors on the same
platform and carry different physical information.

Band order (Sentinel-2 L1C, as shipped in EuroSAT_MS):
  0 B01 coastal aerosol   1 B02 blue      2 B03 green    3 B04 red
  4 B05 red edge 1        5 B06 red edge 2  6 B07 red edge 3
  7 B08 NIR               8 B08A narrow NIR 9 B09 water vapour
 10 B10 cirrus           11 B11 SWIR 1    12 B12 SWIR 2

So RGB = bands (3, 2, 1) and the ten non-visible bands are the complement.

The train/test split reuses the SAME deterministic rule as the RGB loader so
the two are directly comparable.

Usage:  python scripts/make_eurosat_ms.py [--root data/eurosat_ms] [--out data/eurosat_ms_64.npz]
"""
import argparse, os, sys
import numpy as np
import tifffile

RGB_BANDS = (3, 2, 1)
NONVIS_BANDS = (0, 4, 5, 6, 7, 8, 9, 10, 11, 12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/eurosat_ms")
    ap.add_argument("--out", default="data/eurosat_ms_64.npz")
    a = ap.parse_args()

    base = os.path.join(a.root, "EuroSAT_MS")
    classes = sorted(d for d in os.listdir(base)
                     if os.path.isdir(os.path.join(base, d)))
    print(f"{len(classes)} classes: {classes}")

    paths, labels = [], []
    for ci, c in enumerate(classes):
        d = os.path.join(base, c)
        # sort by the numeric suffix so the order is stable and matches the
        # RGB loader's ImageFolder ordering as closely as possible
        fs = sorted(os.listdir(d),
                    key=lambda f: int(f.rsplit("_", 1)[1].split(".")[0]))
        for f in fs:
            paths.append(os.path.join(d, f))
            labels.append(ci)
    n = len(paths)
    print(f"{n} tiles")

    X = np.zeros((n, 13, 64, 64), dtype=np.uint16)
    for i, p in enumerate(paths):
        img = tifffile.imread(p)           # (64, 64, 13)
        X[i] = np.transpose(img, (2, 0, 1))
        if (i + 1) % 5000 == 0:
            print(f"  {i+1}/{n}", flush=True)
    y = np.asarray(labels, dtype=np.int64)

    # deterministic 80/20 split, seeded, stratified by class
    rng = np.random.RandomState(0)
    tr, te = [], []
    for ci in range(len(classes)):
        idx = np.where(y == ci)[0]
        rng.shuffle(idx)
        k = int(round(0.8 * len(idx)))
        tr.append(idx[:k]); te.append(idx[k:])
    tr = np.sort(np.concatenate(tr)); te = np.sort(np.concatenate(te))
    print(f"train {len(tr)}  test {len(te)}")

    # per-band mean/std over the TRAIN split only, in reflectance-ish units
    Xf = X[tr].astype(np.float32) / 10000.0
    mean = Xf.mean(axis=(0, 2, 3))
    std = Xf.std(axis=(0, 2, 3))
    print("per-band mean:", np.round(mean, 4))
    print("per-band std :", np.round(std, 4))

    np.savez_compressed(a.out, images=X, labels=y, train_idx=tr, test_idx=te,
                        mean=mean, std=std, classes=np.array(classes),
                        rgb_bands=np.array(RGB_BANDS),
                        nonvis_bands=np.array(NONVIS_BANDS))
    print(f"wrote {a.out} ({os.path.getsize(a.out)/1e9:.2f} GB)")


if __name__ == "__main__":
    main()
