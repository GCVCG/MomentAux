"""Dense-prediction data: PASCAL VOC-2012 segmentation, with the augmented
(SBD) training set, plus the fixed subset machinery the rest of the study uses.

SEPARATE FROM data.py ON PURPOSE. The classification loader is consumed by
~9,000 finished runs; nothing here is imported by it, so a mistake in this
file cannot change a classification number.

THREE THINGS DIFFER FROM THE CLASSIFICATION LOADER, and each is a deliberate
choice rather than an oversight:

1. TRANSFORMS ARE JOINT. Every geometric operation has to be applied to the
   mask as well as the image, with NEAREST interpolation on the mask (bilinear
   would invent label values that do not exist) and 255 -- the ignore index --
   as the fill for anything that falls outside the original image. A crop that
   is applied to the image but not the mask is silent and produces a plausible
   but meaningless number, which is why the smoke test checks alignment
   explicitly rather than just shapes.

2. THE SUBSET IS STRATIFIED BY DOMINANT CLASS. Classification subsets stratify
   on the label; a segmentation image has no single label. The nearest honest
   analog is the largest non-background class present, which is what is used
   here. It matters at the small fractions the study cares about: 1% of the
   augmented set is 106 images over 20 classes, and an unstratified draw would
   leave several classes with none at all, so the cell would be measuring
   which classes were sampled rather than what the prior does.

3. EVALUATION IS AT NATIVE RESOLUTION, batch size 1. VOC images are small and
   variable-sized; the FCN head upsamples its logits to whatever the input
   was, so no resizing or padding is needed and none is done. Both arms are
   scored by the identical procedure.

The RECIPE the crops follow (random scale 0.5-2.0, 512 crop, horizontal flip)
is the standard VOC segmentation recipe, not the study's 32px crop+flip. It is
frozen across every dense cell in exactly the same sense: no cell tunes it.
"""
import json
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

VOC_CLASSES = 21          # 20 objects + background
IGNORE_INDEX = 255
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)

# --- the dense dataset registry ---------------------------------------------
# VOC was the only population when this file was written, and its behaviour is
# reproduced EXACTLY by the voc_seg entry: 36 finished cells and 10 probes
# depend on it, so every VOC-specific path, cache name and subset filename
# below is the original one. New datasets are additions, never edits.
#
# The three populations are chosen to vary the two things VOC cannot:
#   voc_seg     21 classes, object-centric, background-dominated
#   ade20k     150 classes, scene-centric -- pixel accuracy and mIoU decouple
#              far more sharply here, which is the axis the units finding
#              (readout reads on the head's own classification scale, not
#              mIoU) actually turns on
#   cityscapes  19 classes, driving domain, near-fixed geometry -- the domain
#              transfer test rather than the label-space one
#   foodseg103 104 classes, texture-dominated food. The one population with a
#              CLASSIFICATION twin already in the study (food101), so it is
#              the only place a same-domain dense-vs-classification comparison
#              is possible. Recorded 2026-07-23 as "wrong task for the frozen
#              classification recipe" -- true then, and exactly right now that
#              a dense task exists.
#   pascalcontext  254 classes on the SAME VOC images voc_seg uses -- the
#              dense analog of the cifar100super control (identical pixels,
#              different label space), and the only population whose trained
#              pixel accuracy is expected BELOW the readout crossing bracket,
#              which is what E-C needs and ADE20K turned out not to supply.
NUM_CLASSES = {"voc_seg": 21, "ade20k": 150, "cityscapes": 19,
               "foodseg103": 104,      # 103 food classes + background
               "pascalcontext": 254}   # top-254 by train frequency; see below
TRAIN_SPLIT = {"voc_seg": "train_aug", "ade20k": "training",
               "cityscapes": "train", "foodseg103": "train",
               "pascalcontext": "train"}
VAL_SPLIT = {"voc_seg": "val", "ade20k": "validation",
             "cityscapes": "val", "foodseg103": "test",
             "pascalcontext": "val"}

SUBSET_PREFIX = {"voc_seg": "voc_seg", "ade20k": "ade20k_seg",
                 "cityscapes": "cityscapes_seg",
                 "foodseg103": "foodseg103_seg",
                 "pascalcontext": "pascalcontext_seg"}
DOMINANT_CACHE = {"voc_seg": "voc_dominant.json",
                  "ade20k": "ade20k_dominant.json",
                  "cityscapes": "cityscapes_dominant.json",
                  "foodseg103": "foodseg103_dominant.json",
                  "pascalcontext": "pascalcontext_dominant.json"}

SUBSET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "data", "subsets")
SUBSET_SEED = 1234        # same constant as the classification subsets


def _voc_root(data_root):
    return os.path.join(data_root, "VOCdevkit", "VOC2012")


def _flat_root(data_root, ds):
    """ADE20K and Cityscapes are staged into one flat, dataset-agnostic layout.

    Both arrive in their own idiosyncratic trees (ADE20K as
    images/training + annotations/training; Cityscapes as parquet shards from
    a mirror, since the official download is account-gated). Rather than teach
    the loader two more directory conventions, scripts/prepare_seg_data.py
    stages both into

        <root>/<ds>/images/<split>/<id>.{jpg,png}
        <root>/<ds>/masks/<split>/<id>.png        # ALREADY train-ids

    so the loader has one path rule and the dataset-specific mapping lives in
    exactly one place -- the preparation script -- where it is checked once
    rather than on every __getitem__.
    """
    return os.path.join(data_root, ds)


def read_split(data_root, split, ds="voc_seg"):
    """Image ids for a split.

    `train_aug` is the standard augmented training set: the SBD annotations
    unioned with VOC's own train ids, minus anything in VOC val, so that val
    stays genuinely held out. Built once and cached beside the data.
    """
    if ds != "voc_seg":
        d = os.path.join(_flat_root(data_root, ds), "masks", split)
        if not os.path.isdir(d):
            raise FileNotFoundError(
                f"{d} missing: run scripts/prepare_seg_data.py --dataset {ds}")
        return sorted(os.path.splitext(f)[0] for f in os.listdir(d)
                      if f.endswith(".png"))

    voc = _voc_root(data_root)
    seg = os.path.join(voc, "ImageSets", "Segmentation")
    if split in ("train", "val"):
        with open(os.path.join(seg, f"{split}.txt")) as f:
            return [l.strip() for l in f if l.strip()]
    if split != "train_aug":
        raise ValueError(f"unknown split {split!r}")
    cache = os.path.join(seg, "train_aug.txt")
    if os.path.exists(cache):
        with open(cache) as f:
            return [l.strip() for l in f if l.strip()]
    sbd_cls = os.path.join(data_root, "sbd", "img")          # SBD images
    if not os.path.isdir(sbd_cls):
        raise FileNotFoundError(
            f"{sbd_cls} missing: the augmented split needs SBD. Fetch it with "
            "torchvision.datasets.SBDataset(root='data/sbd', "
            "image_set='train_noval', mode='segmentation', download=True)")
    val = set(read_split(data_root, "val"))
    sbd_ids = {os.path.splitext(f)[0] for f in os.listdir(sbd_cls)}
    ids = sorted((sbd_ids | set(read_split(data_root, "train"))) - val)
    with open(cache, "w") as f:
        f.write("\n".join(ids) + "\n")
    return ids


def _mask_path(data_root, img_id, ds="voc_seg", split=None):
    """VOC's own mask if present, else SBD's converted one."""
    if ds != "voc_seg":
        return os.path.join(_flat_root(data_root, ds), "masks", split,
                            f"{img_id}.png")
    p = os.path.join(_voc_root(data_root), "SegmentationClass", f"{img_id}.png")
    if os.path.exists(p):
        return p
    return os.path.join(data_root, "sbd", "cls_png", f"{img_id}.png")


def _image_path(data_root, img_id, ds="voc_seg", split=None):
    if ds == "pascalcontext":
        # Pascal-Context re-annotates VOC images, so it reads VOC's JPEGs
        # directly. Only the masks are staged.
        return os.path.join(_voc_root(data_root), "JPEGImages", f"{img_id}.jpg")
    if ds == "voc_seg":
        return os.path.join(_voc_root(data_root), "JPEGImages", f"{img_id}.jpg")
    base = os.path.join(_flat_root(data_root, ds), "images", split, img_id)
    for ext in (".jpg", ".png"):
        if os.path.exists(base + ext):
            return base + ext
    return base + ".jpg"


def dominant_classes(data_root, ids, ds="voc_seg", split=None):
    """Largest non-background class per image; the stratification key.

    Cached, because it reads every mask once and the answer is a property of
    the dataset rather than of a run.
    """
    cache = os.path.join(SUBSET_DIR, DOMINANT_CACHE[ds])
    if os.path.exists(cache):
        with open(cache) as f:
            have = json.load(f)
        if all(i in have for i in ids):
            return [have[i] for i in ids]
    n_cls = NUM_CLASSES[ds]
    out = {}
    for i in ids:
        m = np.array(Image.open(_mask_path(data_root, i, ds, split)))
        # VOC reserves 0 for background, so the dominant NON-background class
        # is the honest key. ADE20K and Cityscapes have no background class --
        # index 0 is a real class (wall / road) -- so excluding it there would
        # silently mis-key every road-dominated image. Only the ignore label
        # is dropped for those two.
        valid = (m != IGNORE_INDEX) & ((m > 0) if ds == "voc_seg" else True)
        counts = np.bincount(m[valid].ravel(), minlength=n_cls)
        out[i] = int(counts.argmax()) if counts.sum() else 0
    os.makedirs(SUBSET_DIR, exist_ok=True)
    with open(cache, "w") as f:
        json.dump(out, f)
    return [out[i] for i in ids]


def make_subset_indices(keys, pct, seed=SUBSET_SEED):
    """Deterministic stratified subset over the dominant-class key.

    Identical in form to data.make_subset_indices, so the two kinds of cell
    draw their data the same way.
    """
    keys = np.asarray(keys)
    idx = []
    for c in sorted(np.unique(keys)):
        cls = np.flatnonzero(keys == c)
        n = max(1, int(round(len(cls) * pct / 100.0)))
        rs = np.random.RandomState(seed * 100003 + int(c))
        idx.extend(cls[rs.permutation(len(cls))][:n].tolist())
    return sorted(int(i) for i in idx)


def subset_path(pct, ds="voc_seg"):
    return os.path.join(SUBSET_DIR, f"{SUBSET_PREFIX[ds]}_{pct}pct.json")


def load_subset_indices(pct, ds="voc_seg"):
    path = subset_path(pct, ds)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Committed subset file missing: {path}. Run "
            "scripts/make_dense_subsets.py once and commit the result.")
    with open(path) as f:
        return json.load(f)["indices"]


class SegmentationDataset(Dataset):
    """Semantic segmentation with joint image/mask transforms.

    One class for all three populations. The dataset-specific parts -- where
    the files live and how raw annotation values map to train ids -- are
    resolved before this point (registry above, preparation script for the two
    new sets), so the transform pipeline that every cell trains under is
    literally the same code on every population. That is what makes a
    cross-dataset Delta comparison mean anything.
    """

    def __init__(self, data_root, split=None, train=True,
                 crop=512, scale=(0.5, 2.0), subset_pct=None, ds="voc_seg"):
        self.root = data_root
        self.ds = ds
        if split is None:
            split = TRAIN_SPLIT[ds] if train else VAL_SPLIT[ds]
        self.split = split
        self.ids = read_split(data_root, split, ds)
        if subset_pct is not None and subset_pct != 100:
            if not train:
                raise ValueError("subsets apply to the train split only")
            keep = load_subset_indices(subset_pct, ds)
            self.ids = [self.ids[i] for i in keep]
        self.train, self.crop, self.scale = train, crop, scale
        self.mean = torch.tensor(MEAN).view(3, 1, 1)
        self.std = torch.tensor(STD).view(3, 1, 1)

    def __len__(self):
        return len(self.ids)

    def _to_tensor(self, img, mask):
        x = torch.from_numpy(np.array(img, np.uint8).transpose(2, 0, 1)).float() / 255.0
        x = (x - self.mean) / self.std
        return x, torch.from_numpy(np.array(mask, np.uint8)).long()

    def __getitem__(self, i):
        img_id = self.ids[i]
        img = Image.open(_image_path(self.root, img_id, self.ds,
                                     self.split)).convert("RGB")
        # VOC masks are PALETTE pngs. Read them once into an index-valued
        # mode-L image: np.array() on a mode-P image gives the class indices,
        # and Image.fromarray puts them back as grey levels of the same value.
        # Without this, pasting the mask into the mode-L pad canvas below
        # CONVERTS the palette to luminance -- class 1 becomes 38, class 15
        # becomes 147 -- and training silently optimizes garbage labels while
        # validation (which never pads) stays correct. A train/eval mismatch
        # that produces a plausible number is the worst kind, so it is fixed
        # at the source and asserted in the smoke test.
        mask = Image.fromarray(np.array(Image.open(
            _mask_path(self.root, img_id, self.ds, self.split))))
        if not self.train:
            # Native resolution, no resizing: the head upsamples to the input
            # size, so nothing needs to be padded or cropped to score it.
            return self._to_tensor(img, mask)

        # Random scale, then a fixed crop with ignore-fill -- the standard VOC
        # recipe. The mask uses NEAREST so no label is interpolated into
        # existence, and 255 (ignore) fills anything outside the original.
        s = np.random.uniform(*self.scale)
        w, h = img.size
        nw, nh = int(round(w * s)), int(round(h * s))
        img = img.resize((nw, nh), Image.BILINEAR)
        mask = mask.resize((nw, nh), Image.NEAREST)

        pad_w, pad_h = max(0, self.crop - nw), max(0, self.crop - nh)
        if pad_w or pad_h:
            canvas = Image.new("RGB", (nw + pad_w, nh + pad_h), (124, 116, 104))
            canvas.paste(img, (0, 0))
            img = canvas
            mcanvas = Image.new("L", (nw + pad_w, nh + pad_h), IGNORE_INDEX)
            mcanvas.paste(mask, (0, 0))
            mask = mcanvas
        W, H = img.size
        x0 = np.random.randint(0, W - self.crop + 1)
        y0 = np.random.randint(0, H - self.crop + 1)
        box = (x0, y0, x0 + self.crop, y0 + self.crop)
        img, mask = img.crop(box), mask.crop(box)

        if np.random.rand() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        return self._to_tensor(img, mask)


class VOCSegmentation(SegmentationDataset):
    """Backwards-compatible alias, and a real subclass rather than a factory.

    The 36 finished VOC cells and their 10 probes were run against this name,
    and the transform tests SUBCLASS it -- so a factory function would have
    been a silent API break. Keeping it a class costs nothing and means no
    already-recorded cell or test depends on a rename.
    """

    def __init__(self, *a, **kw):
        kw.setdefault("ds", "voc_seg")
        super().__init__(*a, **kw)


class ConfusionMatrix:
    """Standard mIoU over a running confusion matrix, ignoring index 255."""

    def __init__(self, n_classes=VOC_CLASSES):
        self.n = n_classes
        self.mat = torch.zeros(n_classes, n_classes, dtype=torch.int64)

    def update(self, pred, target):
        # Accumulate on whatever device the predictions are already on. This
        # is a pure speedup and NOT a change of measurement: every operation
        # here is integer (mask, multiply-add, bincount), so the result is
        # bit-identical to accumulating on the CPU -- asserted in
        # tests/test_dense_transforms.py rather than argued.
        #
        # It matters because the caller used to move every prediction to the
        # CPU first, and ADE20K validates on 2,000 native-resolution images of
        # ~350k pixels each; that transfer, not the forward pass, was the cost
        # of a validation pass.
        if self.mat.device != pred.device:
            self.mat = self.mat.to(pred.device)
        target = target.to(pred.device)
        k = (target >= 0) & (target < self.n)      # drops the ignore label
        idx = self.n * target[k].to(torch.int64) + pred[k].to(torch.int64)
        self.mat += torch.bincount(idx, minlength=self.n ** 2).reshape(
            self.n, self.n)

    def miou(self):
        h = self.mat.float()
        iu = torch.diag(h) / (h.sum(1) + h.sum(0) - torch.diag(h)).clamp_min(1)
        return float(iu.mean() * 100), [float(v * 100) for v in iu]

    def pixel_acc(self):
        h = self.mat.float()
        return float(torch.diag(h).sum() / h.sum().clamp_min(1) * 100)
