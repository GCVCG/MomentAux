"""Stage ADE20K and Cityscapes into the flat layout data_dense.py expects.

    python scripts/prepare_seg_data.py --dataset ade20k     --src <unzipped dir>
    python scripts/prepare_seg_data.py --dataset cityscapes --src <parquet dir>
    python scripts/prepare_seg_data.py --dataset ade20k --verify

Output, identical in shape for both:

    <data-root>/<ds>/images/<split>/<id>.{jpg,png}
    <data-root>/<ds>/masks/<split>/<id>.png       # train ids, 255 = ignore

WHY CONVERT AT ALL rather than map on the fly. Both datasets encode their
labels in a form the loader must not see: ADE20K uses 1..150 with 0 meaning
"unlabelled", and Cityscapes ships 34 labelIds of which only 19 are evaluated.
Doing that mapping inside __getitem__ would put a dataset-specific branch on
the hot path of every cell and, worse, make it invisible -- a wrong LUT would
train happily and produce a plausible number. Converting once, here, means the
mapping is applied in exactly one place and can be CHECKED, which is what
--verify does.

That is not a hypothetical concern in this study: the VOC palette bug
(2026-08-10) converted class 1 to 38 and class 15 to 147 by pasting a mode-P
image into a mode-L canvas, and it broke training while leaving validation
correct -- a plausible number from garbage labels. Every assumption about
encoding below is therefore ASSERTED against the actual pixels rather than
trusted.
"""
import argparse
import glob
import io
import os
import sys

import numpy as np
from PIL import Image, ImageOps

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import data_dense as dd

# --- Cityscapes: the 19 evaluated classes -----------------------------------
# labelId -> trainId. Everything absent from this map is ignore (255), which
# is the official protocol: the other 15 labelIds are void or rare classes
# excluded from the benchmark.
CS_LABELID_TO_TRAIN = {7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7,
                       21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
                       28: 15, 31: 16, 32: 17, 33: 18}
# The same 19 classes as the official colour palette, for mirrors that ship
# colour-coded masks instead of labelIds.
CS_COLOR_TO_TRAIN = {
    (128, 64, 128): 0, (244, 35, 232): 1, (70, 70, 70): 2, (102, 102, 156): 3,
    (190, 153, 153): 4, (153, 153, 153): 5, (250, 170, 30): 6, (220, 220, 0): 7,
    (107, 142, 35): 8, (152, 251, 152): 9, (70, 130, 180): 10, (220, 20, 60): 11,
    (255, 0, 0): 12, (0, 0, 142): 13, (0, 0, 70): 14, (0, 60, 100): 15,
    (0, 80, 100): 16, (0, 0, 230): 17, (119, 11, 32): 18}

# Cityscapes ships 1024x2048. Stored at HALF resolution, and this is a
# deliberate, stated deviation rather than an optimisation: the frozen dense
# recipe crops 512 with scale 0.5-2.0, which on a 2048-wide image samples a
# far smaller fraction of the scene than the same crop does on a ~500px VOC
# image. Half resolution makes the crop cover a comparable share of the frame,
# so the three populations are being trained under a comparable recipe rather
# than only a nominally identical one. Applied identically to both arms.
CS_SIZE = (1024, 512)          # (W, H)


def _open_image(path):
    """Open an RGB image with EXIF orientation applied.

    Not defensive boilerplate: exactly one FoodSeg103 training image carries an
    orientation tag, so its raw stored size is (384, 512) while its mask is
    (512, 384). Without this the pair trains 90 degrees out of alignment --
    one image in ~5,000, far too few to move a metric visibly, and invisible
    to any check that compares only counts. The size assertion in each
    preparation function is what surfaced it, which is the argument for
    asserting rather than assuming.
    """
    return ImageOps.exif_transpose(Image.open(path)).convert("RGB")


def _out_dirs(root, ds, split):
    im = os.path.join(root, ds, "images", split)
    mk = os.path.join(root, ds, "masks", split)
    os.makedirs(im, exist_ok=True)
    os.makedirs(mk, exist_ok=True)
    return im, mk


# ---------------------------------------------------------------- ADE20K ----
def prepare_ade20k(src, root, limit=None):
    written = 0
    for split, sub in (("training", "training"), ("validation", "validation")):
        im_dir, mk_dir = _out_dirs(root, "ade20k", split)
        ann = sorted(glob.glob(os.path.join(src, "annotations", sub, "*.png")))
        if not ann:
            raise FileNotFoundError(
                f"no annotations under {src}/annotations/{sub} -- point --src "
                "at the unzipped ADEChallengeData2016 directory")
        if limit:
            ann = ann[:limit]
        for k, a in enumerate(ann):
            stem = os.path.splitext(os.path.basename(a))[0]
            raw = np.array(Image.open(a))
            if raw.ndim != 2:
                raise ValueError(f"{a}: expected a single-channel mask")
            hi = int(raw.max())
            if hi > 150:
                raise ValueError(
                    f"{a}: value {hi} above 150 -- this is not the "
                    "SceneParse150 annotation format")
            # 0 means unlabelled in ADE20K, and classes run 1..150. Shift to
            # 0..149 and send 0 to the ignore index. Done in int16 so the
            # 0 -> -1 step cannot wrap around in uint8 and silently become 255
            # for a genuine class.
            lab = raw.astype(np.int16) - 1
            lab[lab < 0] = dd.IGNORE_INDEX
            Image.fromarray(lab.astype(np.uint8)).save(
                os.path.join(mk_dir, f"{stem}.png"))
            jpg = os.path.join(src, "images", sub, f"{stem}.jpg")
            img = _open_image(jpg)
            if img.size != Image.open(a).size:
                raise ValueError(f"{stem}: image {img.size} vs mask "
                                 f"{Image.open(a).size}")
            img.save(os.path.join(im_dir, f"{stem}.jpg"), quality=95)
            written += 1
            if k % 2000 == 0:
                print(f"  {split}: {k}/{len(ann)}", flush=True)
        print(f"  {split}: {len(ann)} images", flush=True)
    return written


# ------------------------------------------------------------ Cityscapes ----
def _cs_mask_to_train(arr):
    """Map a Cityscapes mask to train ids, detecting which encoding it is."""
    if arr.ndim == 3:
        rgb = arr[:, :, :3]
        # A 3-channel mask is NOT automatically a colour palette. This mirror
        # stores labelIds REPLICATED across all three channels -- (7,7,7) is
        # road, (11,11,11) building, (26,26,26) car -- and running the palette
        # LUT over it matched nothing, mapping every pixel to ignore. The
        # staged masks then contained no classes at all, and the first version
        # of verify() below passed them, because "no train id >= n_classes" is
        # trivially true of an all-ignore mask. Check for the grey case first.
        if (rgb[:, :, 0] == rgb[:, :, 1]).all() and (rgb[:, :, 1] == rgb[:, :, 2]).all():
            arr = rgb[:, :, 0]
        else:
            out = np.full(rgb.shape[:2], dd.IGNORE_INDEX, np.uint8)
            for colour, t in CS_COLOR_TO_TRAIN.items():
                out[(rgb == np.array(colour, np.uint8)).all(-1)] = t
            if (out == dd.IGNORE_INDEX).all():
                raise ValueError(
                    "3-channel mask matched neither replicated labelIds nor "
                    "the official colour palette; refusing to stage masks "
                    "that would be entirely ignore")
            return out, "colour"
    vals = set(np.unique(arr).tolist())
    if vals - set(range(34)):
        raise ValueError(f"mask values {sorted(vals)[:8]}... are neither "
                         "labelIds (0-33) nor a known palette")
    out = np.full(arr.shape, dd.IGNORE_INDEX, np.uint8)
    for lid, t in CS_LABELID_TO_TRAIN.items():
        out[arr == lid] = t
    return out, "labelId"


def prepare_cityscapes(src, root, limit=None):
    import pyarrow.parquet as pq
    written, kinds = 0, set()
    for split, pat in (("train", "train-*.parquet"),
                       ("val", "validation-*.parquet")):
        im_dir, mk_dir = _out_dirs(root, "cityscapes", split)
        shards = sorted(glob.glob(os.path.join(src, "data", pat))) or \
                 sorted(glob.glob(os.path.join(src, pat)))
        if not shards:
            raise FileNotFoundError(f"no {pat} under {src}")
        n = 0
        for sh in shards:
            t = pq.read_table(sh)
            cols = t.column_names
            ic = "image" if "image" in cols else cols[0]
            mc = ("semantic_segmentation" if "semantic_segmentation" in cols
                  else [c for c in cols if c != ic][0])
            for row in t.to_pylist():
                if limit and n >= limit:
                    break
                img = Image.open(io.BytesIO(row[ic]["bytes"])).convert("RGB")
                m = np.array(Image.open(io.BytesIO(row[mc]["bytes"])))
                lab, kind = _cs_mask_to_train(m)
                kinds.add(kind)
                stem = f"{split}_{n:05d}"
                img.resize(CS_SIZE, Image.BILINEAR).save(
                    os.path.join(im_dir, f"{stem}.png"))
                # NEAREST on the mask, always: bilinear would invent train ids
                # that do not exist, which is the same class of silent error
                # the joint-transform tests exist to catch.
                Image.fromarray(lab).resize(CS_SIZE, Image.NEAREST).save(
                    os.path.join(mk_dir, f"{stem}.png"))
                n += 1
            if limit and n >= limit:
                break
        print(f"  {split}: {n} images", flush=True)
        written += n
    print(f"  mask encoding detected: {sorted(kinds)}")
    return written


# ------------------------------------------------------------ FoodSeg103 ----
def prepare_foodseg103(src, root, limit=None):
    """Stage FoodSeg103, which needs no label remapping at all.

    Its masks are already mode-L with values 0..103 -- 0 is background, a real
    class, exactly like VOC -- so this is a pure restaging into the flat
    layout. Two things are checked rather than assumed:

      * n_classes is 104, not the 103 the name and `wc -l category_id.txt`
        both suggest. The file's last line carries no trailing newline, so the
        obvious count is short by one and would silently drop the last class
        ("other ingredients") out of the confusion matrix.
      * 2 of the 4,985 training images have no annotation. Ids come from the
        MASK directory for exactly this reason, so an image without a label
        can never enter a split.
    """
    written = 0
    for split in ("train", "test"):
        im_dir, mk_dir = _out_dirs(root, "foodseg103", split)
        anns = sorted(glob.glob(os.path.join(src, "Images", "ann_dir", split, "*")))
        if not anns:
            raise FileNotFoundError(
                f"no masks under {src}/Images/ann_dir/{split} -- point --src "
                "at the FoodSeg103 directory")
        if limit:
            anns = anns[:limit]
        n_img = len(glob.glob(os.path.join(src, "Images", "img_dir", split, "*")))
        if n_img != len(anns) and not limit:
            print(f"  note {split}: {n_img} images vs {len(anns)} masks -- "
                  "ids come from the masks, so the extras are dropped")
        for k, a in enumerate(anns):
            stem = os.path.splitext(os.path.basename(a))[0]
            m = np.array(Image.open(a))
            if m.ndim != 2 or m.max() > 103:
                raise ValueError(f"{a}: unexpected mask (ndim {m.ndim}, "
                                 f"max {m.max()}); expected 2-D, <= 103")
            img_p = None
            for ext in (".jpg", ".png", ".jpeg"):
                c = os.path.join(src, "Images", "img_dir", split, stem + ext)
                if os.path.exists(c):
                    img_p = c
                    break
            if img_p is None:
                print(f"  skipping {stem}: no image for this mask")
                continue
            im = _open_image(img_p)
            if im.size != Image.open(a).size:
                raise ValueError(f"{stem}: image {im.size} vs mask "
                                 f"{Image.open(a).size}")
            Image.fromarray(m).save(os.path.join(mk_dir, f"{stem}.png"))
            im.save(os.path.join(im_dir, f"{stem}.jpg"), quality=95)
            written += 1
            if k % 1000 == 0:
                print(f"  {split}: {k}/{len(anns)}", flush=True)
        print(f"  {split}: staged", flush=True)
    return written


# --------------------------------------------------------- Pascal-Context ---
def prepare_pascalcontext(src, root, limit=None):
    """Stage Pascal-Context: VOC's own images, re-annotated with 459 classes.

    WHY THIS POPULATION EXISTS, and it is two things at once:

    (1) THE SAME-PIXELS CONTROL. It re-annotates the identical VOC images
        voc_seg trains on, so it is the dense analog of cifar100super: change
        ONLY the label space and watch what moves. Nothing else in the dense
        study can separate a pixel effect from a label-space effect.
    (2) THE LOW-PIXEL-ACCURACY POPULATION. ADE20K was chosen to sit below the
        readout crossing bracket [31.8, 40.3] and, once trained, did not --
        34.0 and 38.5, inside it, where the no-call rule applies. 254 classes
        over these images should push trained pixel accuracy clearly below,
        which is what E-C actually needs.

    TWO PROTOCOL CHOICES, both stated rather than buried:

    * ONLY 254 OF THE 459 CLASSES are kept, ranked by pixel frequency on the
      TRAIN split alone (never val), with the tail mapped to ignore. The
      reason is mundane -- an 8-bit mask holds 255 values and 255 is the
      ignore index -- and the cost is nil: the top 254 classes cover 100.00%
      of labelled pixels in a 1,200-image sample, because only ~253 of the
      459 ever occur. This is the same protocol SHAPE as Cityscapes' standard
      19-of-34, which is the field norm rather than an invention here.
    * THE SPLIT IS THE OFFICIAL ONE, not a re-draw: VOC2012 Main/train and
      Main/val intersected with the Pascal-Context ids give exactly 4,998 and
      5,105, the canonical Pascal-Context split.

    Only MASKS are staged. The images are read from VOCdevkit directly, which
    is what makes the same-pixels claim literal rather than approximate.
    """
    from scipy.io import loadmat

    mats = {os.path.splitext(os.path.basename(f))[0]: f
            for f in glob.glob(os.path.join(src, "trainval", "*.mat"))}
    if not mats:
        raise FileNotFoundError(f"no trainval/*.mat under {src}")
    voc_sets = os.path.join(root, "VOCdevkit", "VOC2012", "ImageSets", "Main")
    splits = {}
    for name in ("train", "val"):
        p = os.path.join(voc_sets, f"{name}.txt")
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"{p} missing: Pascal-Context uses VOC's own split lists")
        ids = {l.split()[0] for l in open(p) if l.strip()}
        splits[name] = sorted(ids & set(mats))
    print(f"  official split: train {len(splits['train'])}, "
          f"val {len(splits['val'])}")

    # Rank classes by pixel frequency on TRAIN ONLY -- ranking on val too would
    # leak held-out statistics into the label space itself.
    counts = np.zeros(512, np.int64)
    for i in splits["train"]:
        a = loadmat(mats[i])["LabelMap"]
        counts += np.bincount(a.ravel(), minlength=512)[:512]
    counts[0] = 0                                   # 0 is unlabelled
    order = np.argsort(-counts)
    keep = [int(c) for c in order[:254] if counts[c] > 0]
    lut = np.full(512, dd.IGNORE_INDEX, np.uint8)
    for new, old in enumerate(keep):
        lut[old] = new
    cov = counts[keep].sum() / max(counts.sum(), 1)
    print(f"  keeping {len(keep)} classes, covering {100*cov:.2f}% of "
          "labelled train pixels")

    written = 0
    for split in ("train", "val"):
        _, mk_dir = _out_dirs(root, "pascalcontext", split)
        ids = splits[split][:limit] if limit else splits[split]
        for k, i in enumerate(ids):
            a = loadmat(mats[i])["LabelMap"]
            if a.max() >= 512:
                raise ValueError(f"{i}: label {a.max()} outside the LUT")
            lab = lut[a]
            img = _open_image(os.path.join(root, "VOCdevkit", "VOC2012",
                                           "JPEGImages", f"{i}.jpg"))
            if img.size != (lab.shape[1], lab.shape[0]):
                raise ValueError(f"{i}: image {img.size} vs mask "
                                 f"{(lab.shape[1], lab.shape[0])}")
            Image.fromarray(lab).save(os.path.join(mk_dir, f"{i}.png"))
            written += 1
            if k % 1000 == 0:
                print(f"  {split}: {k}/{len(ids)}", flush=True)
        print(f"  {split}: {len(ids)} masks", flush=True)
    return written


# ----------------------------------------------------------------- verify ---
def verify(ds, root):
    """Check the staged tree really says what the loader assumes it says."""
    ok = True
    n_cls = dd.NUM_CLASSES[ds]
    for split in (dd.TRAIN_SPLIT[ds], dd.VAL_SPLIT[ds]):
        ids = dd.read_split(root, split, ds)
        seen, missing_img = set(), 0
        for i in ids[:400]:
            m = np.array(Image.open(dd._mask_path(root, i, ds, split)))
            v = set(np.unique(m).tolist()) - {dd.IGNORE_INDEX}
            bad = {x for x in v if x >= n_cls}
            if bad:
                print(f"  BAD {ds}/{split}/{i}: train ids {sorted(bad)[:5]} "
                      f">= n_classes {n_cls}")
                ok = False
            seen |= v
            if not os.path.exists(dd._image_path(root, i, ds, split)):
                missing_img += 1
        # Shapes must agree pixel for pixel; a mask that is merely the right
        # size is not the same as a mask that matches its image.
        i0 = ids[0]
        im = Image.open(dd._image_path(root, i0, ds, split))
        mk = Image.open(dd._mask_path(root, i0, ds, split))
        if im.size != mk.size:
            print(f"  BAD {ds}/{split}: image {im.size} vs mask {mk.size}")
            ok = False
        print(f"  {ds}/{split}: {len(ids)} ids, {len(seen)}/{n_cls} classes in "
              f"the first 400, {missing_img} images missing, size {im.size}")
        if missing_img:
            ok = False
        # THE HOLE THIS PLUGS: the original check only rejected train ids at
        # or above n_classes, which an ALL-IGNORE mask passes trivially. The
        # Cityscapes mirror's replicated-labelId encoding produced exactly
        # that -- every pixel ignore, zero classes present -- and verify()
        # printed OK. A staged split must actually contain most of its label
        # space; anything less than half is a mapping failure, not a rare
        # dataset.
        if len(seen) < max(2, n_cls // 2):
            print(f"  BAD {ds}/{split}: only {len(seen)} of {n_cls} classes "
                  "appear in 400 masks -- the label mapping is wrong")
            ok = False
    print(("VERIFY OK " if ok else "VERIFY FAILED ") + ds)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    choices=["ade20k", "cityscapes", "foodseg103", "pascalcontext"])
    ap.add_argument("--src", default=None)
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--limit", type=int, default=None,
                    help="stage only the first N per split (smoke tests)")
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()

    if a.verify:
        sys.exit(0 if verify(a.dataset, a.data_root) else 1)
    if not a.src:
        ap.error("--src is required unless --verify")
    fn = {"ade20k": prepare_ade20k, "cityscapes": prepare_cityscapes,
          "foodseg103": prepare_foodseg103,
          "pascalcontext": prepare_pascalcontext}[a.dataset]
    n = fn(a.src, a.data_root, a.limit)
    print(f"staged {n} images for {a.dataset}")
    sys.exit(0 if verify(a.dataset, a.data_root) else 1)


if __name__ == "__main__":
    main()
