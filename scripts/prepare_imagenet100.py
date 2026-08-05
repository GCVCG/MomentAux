"""Pack the HF imagenet-100 parquet shards into a flat JPEG blob + offsets.

    python scripts/prepare_imagenet100.py --src <hf_snapshot> --out <data_dir>

Unlike imagenet64 (fixed 64x64, so a uint8 array works), these are VARIABLE
SIZE native-resolution JPEGs, and RandomResizedCrop needs the original -- so
the bytes stay JPEG-encoded and are decoded per __getitem__, exactly like a
normal ImageNet pipeline.

Layout: one <split>.bin of concatenated JPEG bytes plus <split>_off.npy
(int64 offsets, N+1 entries) and <split>_y.npy. Random access is a slice, the
blob memory-maps, and it is ONE file to rsync instead of 17 parquet shards.

STATS ARE COMPUTED HERE, NEVER RECALLED -- on a random subsample at the
training resolution, which is what the transform actually sees.
"""

import argparse
import glob
import io
import os

import numpy as np
import pyarrow.parquet as pq
from PIL import Image


def pack(files, out_bin, out_off, out_lab):
    offs = [0]
    labs = []
    with open(out_bin, "wb") as fh:
        for f in files:
            for batch in pq.ParquetFile(f).iter_batches(batch_size=2048):
                d = batch.to_pydict()
                for rec, lab in zip(d["image"], d["label"]):
                    b = rec["bytes"]
                    fh.write(b)
                    offs.append(offs[-1] + len(b))
                    labs.append(lab)
                if len(labs) % 20000 < 2048:
                    print("  %d" % len(labs), flush=True)
    np.save(out_off, np.asarray(offs, dtype=np.int64))
    np.save(out_lab, np.asarray(labs, dtype=np.int16))
    return len(labs)


def stats(bin_path, off_path, size=224, n=4000, seed=0):
    """Channel mean/std at the TRAINING resolution, on a random subsample."""
    offs = np.load(off_path)
    blob = np.memmap(bin_path, dtype=np.uint8, mode="r")
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(offs) - 1, size=min(n, len(offs) - 1), replace=False)
    ssum = np.zeros(3, dtype=np.float64)
    ssq = np.zeros(3, dtype=np.float64)
    cnt = 0
    for i in idx:
        raw = blob[offs[i]:offs[i + 1]].tobytes()
        im = Image.open(io.BytesIO(raw)).convert("RGB").resize((size, size),
                                                               Image.BILINEAR)
        a = np.asarray(im, dtype=np.float64) / 255.0
        ssum += a.sum(axis=(0, 1))
        ssq += (a ** 2).sum(axis=(0, 1))
        cnt += size * size
    mean = ssum / cnt
    return mean, np.sqrt(ssq / cnt - mean ** 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for split, pat in (("train", "train-*.parquet"), ("val", "validation-*.parquet")):
        files = sorted(glob.glob(os.path.join(args.src, "data", pat)))
        print("%s: %d shards" % (split, len(files)), flush=True)
        n = pack(files,
                 os.path.join(args.out, "%s.bin" % split),
                 os.path.join(args.out, "%s_off.npy" % split),
                 os.path.join(args.out, "%s_y.npy" % split))
        print("%s packed: %d images" % (split, n), flush=True)
    m, s = stats(os.path.join(args.out, "train.bin"),
                 os.path.join(args.out, "train_off.npy"))
    print("train STATS @224 mean=%s std=%s" % (
        tuple(round(float(v), 4) for v in m), tuple(round(float(v), 4) for v in s)),
        flush=True)


if __name__ == "__main__":
    main()
