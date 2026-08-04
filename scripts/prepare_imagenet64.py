"""Convert the HF imagenet-1k-64x64 parquet shards into memmap-able uint8
arrays, and COMPUTE the channel statistics in the same pass.

    python scripts/prepare_imagenet64.py --src <hf_snapshot_dir> --out <data_dir>

Why not read the parquet directly at train time: the images are PNG-encoded,
so every epoch would re-decode 1.28M PNGs. A flat uint8 array (N,64,64,3) is
15.7 GB for train, memory-maps cleanly, and stages into /dev/shm on a compute
node exactly like the tin ZIP does -- the same trick that took tin from ~17s to
~2.2s per epoch on BeeGFS.

STATS ARE COMPUTED HERE, NEVER RECALLED. Placeholder statistics silently
corrupted all four domain datasets earlier in this study; the numbers this
prints are the ones that get pinned in data.py.

NOTE the shards are CLASS-ORDERED (shard 0 holds labels 0..200), so anything
that samples a subset must shuffle across the whole array, never take a prefix.
"""

import argparse
import glob
import io
import os

import numpy as np
import pyarrow.parquet as pq
from PIL import Image


def convert(files, out_img, out_lab, count):
    imgs = np.lib.format.open_memmap(out_img, mode="w+", dtype=np.uint8,
                                     shape=(count, 64, 64, 3))
    labs = np.zeros(count, dtype=np.int16)
    # accumulate in float64 over uint8 values; /255 happens at the end
    ssum = np.zeros(3, dtype=np.float64)
    ssq = np.zeros(3, dtype=np.float64)
    i = 0
    for f in files:
        for batch in pq.ParquetFile(f).iter_batches(batch_size=4096):
            d = batch.to_pydict()
            for rec, lab in zip(d["image"], d["label"]):
                a = np.asarray(Image.open(io.BytesIO(rec["bytes"])).convert("RGB"),
                               dtype=np.uint8)
                imgs[i] = a
                labs[i] = lab
                i += 1
            chunk = imgs[max(0, i - len(d["label"])):i].astype(np.float64) / 255.0
            ssum += chunk.sum(axis=(0, 1, 2))
            ssq += (chunk ** 2).sum(axis=(0, 1, 2))
            if i % 200000 < 4096:
                print("  %d/%d" % (i, count), flush=True)
    assert i == count, (i, count)
    imgs.flush()
    np.save(out_lab, labs)
    n = count * 64 * 64
    mean = ssum / n
    std = np.sqrt(ssq / n - mean ** 2)
    return mean, std, labs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    for split, pat in (("train", "train-*.parquet"), ("val", "validation-*.parquet")):
        files = sorted(glob.glob(os.path.join(args.src, "data", pat)))
        count = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
        print("%s: %d rows from %d shards" % (split, count, len(files)), flush=True)
        mean, std, labs = convert(
            files,
            os.path.join(args.out, "%s_x.npy" % split),
            os.path.join(args.out, "%s_y.npy" % split),
            count,
        )
        print("%s STATS mean=%s std=%s" % (
            split, tuple(round(float(v), 4) for v in mean),
            tuple(round(float(v), 4) for v in std)), flush=True)
        print("%s classes=%d min=%d max=%d" % (split, len(np.unique(labs)),
                                               labs.min(), labs.max()), flush=True)


if __name__ == "__main__":
    main()
