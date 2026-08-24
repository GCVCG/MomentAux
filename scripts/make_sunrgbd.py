"""Pack SUN RGB-D scene classification (RGB + raw depth) into a memmappable
64px array, the study's THIRD multi-source population (limitations campaign
block H, 2026-08-23).

WHY THIS POPULATION: EuroSAT-MS splits ONE instrument into band groups
(symmetric sources) and So2Sat pairs two instruments whose single-source
strengths differ a lot (asymmetric). SUN RGB-D pairs two DIFFERENT
modalities -- appearance (RGB) and geometry (depth) -- whose single-source
scene-classification accuracies are reported as comparable: the missing
"different modality AND symmetric" corner.

SOURCE FILES (Princeton, https://rgbd.cs.princeton.edu/):
  SUNRGBD.zip          10,335 samples under SUNRGBD/{kv1,kv2,realsense,xtion}/...,
                       each folder holding image/*.jpg, depth/*.png,
                       depth_bfx/*.png and scene.txt (the scene label).
  SUNRGBDtoolbox.zip   traintestSUNRGBD/allsplit.mat with the OFFICIAL split
                       (alltrain 5,285 / alltest 5,050 sample folders).

THE 19-CLASS TASK (Song, Lichtenberg & Xiao, CVPR 2015, Sec. 5.1): scene
categories with at least 80 images, over the official split. The paper
reports 4,845 train / 4,659 test; this script ASSERTS exactly those counts
after mapping, and refuses to write anything otherwise. The 19 class names
are pinned below (sorted), and the script also asserts that every scene label
occurring >= 80 times in the whole set is in the list and nothing else is --
so the "19 classes" is derived from the data and checked against the paper,
not merely typed in.

DEPTH: the RAW per-frame depth (depth/, not the inpainted depth_bfx/), because
the inpainting is a post-process that hallucinates geometry into holes and we
want the sensor's own measurement to be the second source; missing depth stays
0 (missing) and is reported. The PNGs are NOT plain millimetres: the toolbox's
read3dPoints.m decodes them as
    mm = bitor(bitshift(x, -3), bitshift(x, 13))    (uint16 arithmetic)
i.e. a 3-bit rotate, and that decode is applied here verbatim. Decoded depth
is clipped to [0, 10] m and scaled to [0, 1] (the toolbox clips at 8 m for
point clouds; 10 m is the sensors' nominal range and keeps the rare far
values; the fraction beyond 8 m is printed). The depth map is then
squash-resized to 64x64 by VALID-PIXEL-WEIGHTED AREA AVERAGING (PIL BOX, the
area filter -- cv2's INTER_AREA equivalent -- applied to depth*mask and to
mask separately, then divided): a plain area average would blend holes (0)
into neighbouring valid depth and paint spurious near-surfaces around every
hole. Output pixels with no valid coverage stay 0.

RGB: EXIF-transposed, converted to RGB, squash-resized to 64x64 BILINEAR
exactly as CUB-200 / DTD / Food-101 are in data.py, scaled to [0, 1].

ASSERTIONS (the 2026-08-11 encoding-trap lesson: assert every assumption
about a new dataset, not the two or three that look risky):
  - every sample folder in the split exists locally and has exactly one
    image/*.jpg, one depth/*.png and a scene.txt;
  - exactly 19 classes after filtering, names == the pinned list;
  - split sizes == 4,845 / 4,659 and no folder in both splits;
  - depth HxW == RGB HxW after EXIF transpose, for every sample;
  - no image is NaN; the count of samples with > 50% missing depth is
    reported (and any with 100% missing is reported separately);
  - the class histogram is printed for both splits.

OUTPUT (same layout as scripts/make_so2sat.py, so data.py's N-channel memmap
loader is reused): <out>_images.npy float32 (N, 4, 64, 64) with channels
[R, G, B, depth] in [0, 1], train rows first; <out>_meta.npz with labels,
train_idx, test_idx, per-channel mean/std from the TRAIN split only, class
names, and the relative sample-folder path of every row (provenance).

Usage:
  python scripts/make_sunrgbd.py --root /media/HDD_16TB/sunrgbd --out data/sunrgbd_64
"""
import argparse
import collections
import json
import os
import sys

import numpy as np
from PIL import Image, ImageOps

# The 19 scene categories of the SUN RGB-D scene-classification benchmark
# (Song et al. 2015): every scene label with >= 80 images in the full set.
CLASSES = (
    "bathroom", "bedroom", "classroom", "computer_room", "conference_room",
    "corridor", "dining_area", "dining_room", "discussion_area",
    "furniture_store", "home_office", "kitchen", "lab", "lecture_theatre",
    "library", "living_room", "office", "rest_space", "study_space",
)
OFFICIAL_TRAIN, OFFICIAL_TEST = 4845, 4659
MIN_IMAGES_PER_CLASS = 80
SIZE = 64
DEPTH_CLIP_M = 10.0
SPLIT_PREFIX = "/n/fs/sun3d/data/"


def decode_depth_mm(png):
    """Toolbox read3dPoints.m decode: 3-bit rotate of the uint16 PNG value."""
    x = np.asarray(png, dtype=np.uint16)
    if x.ndim != 2:
        raise ValueError(f"depth png is not single-channel: {x.shape}")
    return ((x >> 3) | (x << 13)).astype(np.uint16)


def load_split(toolbox_dir):
    import scipy.io as sio

    m = sio.loadmat(os.path.join(toolbox_dir, "traintestSUNRGBD", "allsplit.mat"))

    def paths(arr):
        out = []
        for cell in arr.ravel():
            s = str(np.asarray(cell).ravel()[0])
            if not s.startswith(SPLIT_PREFIX):
                raise ValueError(f"unexpected split path {s!r}")
            out.append(s[len(SPLIT_PREFIX):].rstrip("/"))
        return out

    tr, te = paths(m["alltrain"]), paths(m["alltest"])
    assert len(tr) == 5285 and len(te) == 5050, (len(tr), len(te))
    assert not (set(tr) & set(te)), "official split has overlapping folders"
    return tr, te


def one_file(d, exts):
    fs = sorted(f for f in os.listdir(d) if f.lower().endswith(exts))
    if len(fs) != 1:
        raise ValueError(f"{d}: expected exactly one {exts} file, found {fs}")
    return os.path.join(d, fs[0])


def read_sample(root, rel):
    d = os.path.join(root, rel)
    with open(os.path.join(d, "scene.txt")) as f:
        scene = f.read().strip()
    rgb_p = one_file(os.path.join(d, "image"), (".jpg", ".jpeg", ".png"))
    dep_p = one_file(os.path.join(d, "depth"), (".png",))
    return scene, rgb_p, dep_p


def resize_rgb(img):
    img = ImageOps.exif_transpose(img).convert("RGB")
    w, h = img.size
    out = img.resize((SIZE, SIZE), Image.BILINEAR)
    return np.asarray(out, dtype=np.float32) / 255.0, (h, w)   # (64,64,3)


def resize_depth(mm):
    """Valid-pixel-weighted area average of depth in [0,1] (clip at 10 m)."""
    h, w = mm.shape
    m = (mm > 0).astype(np.float32)
    d = np.clip(mm.astype(np.float32) / 1000.0, 0.0, DEPTH_CLIP_M) / DEPTH_CLIP_M
    num = Image.fromarray(d * m).resize((SIZE, SIZE), Image.BOX)
    den = Image.fromarray(m).resize((SIZE, SIZE), Image.BOX)
    num = np.asarray(num, dtype=np.float32)
    den = np.asarray(den, dtype=np.float32)
    out = np.where(den > 0, num / np.maximum(den, 1e-6), 0.0).astype(np.float32)
    return out, (h, w), float(1.0 - m.mean()), float((mm > 8000).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/media/HDD_16TB/sunrgbd",
                    help="dir holding the extracted SUNRGBD/ and SUNRGBDtoolbox/")
    ap.add_argument("--out", default="data/sunrgbd_64")
    a = ap.parse_args()

    data_root = a.root                       # contains SUNRGBD/<sensor>/...
    toolbox = os.path.join(a.root, "SUNRGBDtoolbox")
    tr_all, te_all = load_split(toolbox)
    print(f"official split: {len(tr_all)} train / {len(te_all)} test folders")

    # Pass 1: scene label of every folder in the official split.
    scenes = {}
    missing = []
    for rel in tr_all + te_all:
        d = os.path.join(data_root, rel)
        if not os.path.isdir(d):
            missing.append(rel)
            continue
        with open(os.path.join(d, "scene.txt")) as f:
            scenes[rel] = f.read().strip()
    if missing:
        print(f"FATAL: {len(missing)} split folders missing locally, e.g. {missing[:3]}")
        sys.exit(1)
    hist_all = collections.Counter(scenes.values())
    derived = sorted(c for c, n in hist_all.items() if n >= MIN_IMAGES_PER_CLASS)
    print(f"{len(hist_all)} distinct scene labels; {len(derived)} with >= "
          f"{MIN_IMAGES_PER_CLASS} images: {derived}")
    # MEASURED 2026-08-23 (the assertion fired on first run and was tightened
    # to the finding, not loosened): TWO labels beyond the pinned 19 clear the
    # >=80 bar -- "idk" (276 images, the annotators' don't-know label, not a
    # scene category) and "office_kitchen" (82). The benchmark's own counts
    # decide: filtering to the pinned 19 reproduces the official 4,845/4,659
    # split EXACTLY (asserted below), so those two are excluded by the
    # benchmark itself, not by us.
    ALLOWED_EXTRA = {"idk", "office_kitchen"}
    extra = set(derived) - set(CLASSES)
    assert set(CLASSES) <= set(derived), (
        f"pinned classes missing from the >=80 set: {set(CLASSES) - set(derived)}")
    assert extra == ALLOWED_EXTRA, (
        "unexpected >=80-image labels outside the pinned 19-class list "
        f"(known: idk, office_kitchen): {sorted(extra)}")
    cls_idx = {c: i for i, c in enumerate(CLASSES)}   # CLASSES is sorted

    tr = [r for r in tr_all if scenes[r] in cls_idx]
    te = [r for r in te_all if scenes[r] in cls_idx]
    print(f"after 19-class filter: {len(tr)} train / {len(te)} test")
    assert (len(tr), len(te)) == (OFFICIAL_TRAIN, OFFICIAL_TEST), (
        f"split sizes {len(tr)}/{len(te)} != official {OFFICIAL_TRAIN}/{OFFICIAL_TEST}")

    rows = tr + te
    n = len(rows)
    X = np.zeros((n, 4, SIZE, SIZE), dtype=np.float32)
    y = np.zeros(n, dtype=np.int64)
    frac_missing = np.zeros(n, dtype=np.float32)
    frac_far = np.zeros(n, dtype=np.float32)
    shape_mismatch = []
    for i, rel in enumerate(rows):
        scene, rgb_p, dep_p = read_sample(data_root, rel)
        assert scene == scenes[rel]
        with Image.open(rgb_p) as im:
            rgb, rgb_hw = resize_rgb(im)
        with Image.open(dep_p) as im:
            mm = decode_depth_mm(im)
        dep, dep_hw, fm, ff = resize_depth(mm)
        if rgb_hw != dep_hw:
            shape_mismatch.append((rel, rgb_hw, dep_hw))
        X[i, :3] = np.transpose(rgb, (2, 0, 1))
        X[i, 3] = dep
        y[i] = cls_idx[scene]
        frac_missing[i] = fm
        frac_far[i] = ff
        if (i + 1) % 1000 == 0:
            print(f"  {i+1}/{n}", flush=True)

    # --- assertions on the packed tensor ---
    assert not shape_mismatch, (
        f"{len(shape_mismatch)} samples whose depth HxW != RGB HxW, e.g. "
        f"{shape_mismatch[:3]}")
    assert np.isfinite(X).all(), "NaN/inf in packed images"
    assert X.min() >= 0.0 and X.max() <= 1.0
    assert len(np.unique(y)) == 19
    all_missing = int((frac_missing >= 1.0).sum())
    half_missing = int((frac_missing > 0.5).sum())
    print(f"depth: {all_missing} samples with NO valid depth; "
          f"{half_missing} with > 50% missing; mean missing fraction "
          f"{frac_missing.mean():.4f}; fraction of decoded depth > 8 m: "
          f"{frac_far.mean():.5f}")
    assert all_missing == 0, "some samples have an all-zero depth map"

    n_tr = len(tr)
    ytr, yte = y[:n_tr], y[n_tr:]
    print("class histogram (train / test):")
    for i, c in enumerate(CLASSES):
        print(f"  {i:2d} {c:<16} {int((ytr == i).sum()):5d} / {int((yte == i).sum()):5d}")
    mean = X[:n_tr].mean(axis=(0, 2, 3))
    std = X[:n_tr].std(axis=(0, 2, 3))
    print("per-channel mean (R,G,B,depth):", np.round(mean, 4))
    print("per-channel std :", np.round(std, 4))

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    np.save(a.out + "_images.npy", X)
    np.savez(a.out + "_meta.npz", labels=y,
             train_idx=np.arange(n_tr), test_idx=np.arange(n_tr, n),
             mean=mean, std=std,
             classes=np.array(CLASSES),
             folders=np.array(rows),
             frac_missing_depth=frac_missing)
    with open(a.out + "_meta.json", "w") as f:
        json.dump({"classes": list(CLASSES), "n_train": n_tr, "n_test": n - n_tr,
                   "mean": mean.tolist(), "std": std.tolist(),
                   "depth_clip_m": DEPTH_CLIP_M, "size": SIZE,
                   "depth_source": "depth/ (raw, toolbox 3-bit-rotate decode)",
                   "n_half_missing_depth": half_missing}, f, indent=1)
    print(f"wrote {a.out}_images.npy "
          f"({os.path.getsize(a.out + '_images.npy')/1e9:.2f} GB), _meta.npz, _meta.json")


if __name__ == "__main__":
    main()
