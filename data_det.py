"""PASCAL VOC detection, on the SAME images and splits the segmentation cells use.

WHY NOT VOC 07+12. The standard detection protocol trains on VOC2007+2012
trainval and tests on VOC2007 test. We deliberately do not use it. This study
compares arms under one recipe rather than competing with published detection
numbers, and the far more valuable property here is a CROSS-TASK CONTROL: every
one of the 10,582 train_aug images and all 1,449 val images that voc_seg uses
also carries box annotations (verified, zero missing), so detection can run on
byte-identical images, byte-identical splits and the same committed subset
indices as segmentation. Classification, segmentation and detection on the same
pixels is a comparison nothing else in the study can make, and it is worth more
than comparability with a leaderboard we are not on.

CONSEQUENCE, STATED: absolute AP here is not comparable to published VOC
numbers, both because of the split and because these models train from scratch.
Only the arm-vs-arm difference is claimed, exactly as everywhere else.

The 20 detection classes are VOC's, in the canonical order, which is the
segmentation label order minus background -- so class k here is segmentation
class k+1, and a cross-task per-class comparison is a lookup rather than a
mapping.
"""
import json
import os
import xml.etree.ElementTree as ET

import torch
from torch.utils.data import Dataset

import data_dense as dd

VOC_CLASSES = (
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
)
NUM_CLASSES = len(VOC_CLASSES)
CLASS_TO_IDX = {c: i for i, c in enumerate(VOC_CLASSES)}


def _ann_path(data_root, img_id):
    return os.path.join(dd._voc_root(data_root), "Annotations", img_id + ".xml")


def parse_boxes(data_root, img_id, keep_difficult=False):
    """Boxes as (N,4) xyxy float and (N,) int64 labels.

    `difficult` objects are dropped from TRAINING by default and, following the
    VOC protocol, are neither counted as false positives nor required as
    recalls at evaluation. Both arms see the identical treatment, so this is a
    protocol choice rather than a variable.
    """
    root = ET.parse(_ann_path(data_root, img_id)).getroot()
    boxes, labels, difficult = [], [], []
    for obj in root.findall("object"):
        name = obj.findtext("name")
        if name not in CLASS_TO_IDX:
            continue
        d = int(obj.findtext("difficult") or 0)
        if d and not keep_difficult:
            continue
        bb = obj.find("bndbox")
        # VOC boxes are 1-indexed inclusive; convert to 0-indexed half-open so
        # that width = x2 - x1 rather than x2 - x1 + 1. Getting this wrong
        # shifts every box by a pixel and silently costs AP at strict IoU.
        x1 = float(bb.findtext("xmin")) - 1.0
        y1 = float(bb.findtext("ymin")) - 1.0
        x2 = float(bb.findtext("xmax")) - 1.0
        y2 = float(bb.findtext("ymax")) - 1.0
        boxes.append([x1, y1, x2, y2])
        labels.append(CLASS_TO_IDX[name])
        difficult.append(d)
    if not boxes:
        return (torch.zeros((0, 4), dtype=torch.float32),
                torch.zeros((0,), dtype=torch.int64),
                torch.zeros((0,), dtype=torch.bool))
    return (torch.tensor(boxes, dtype=torch.float32),
            torch.tensor(labels, dtype=torch.int64),
            torch.tensor(difficult, dtype=torch.bool))


class VOCDetection(Dataset):
    """VOC detection over the segmentation splits and subset indices.

    The geometric augmentation is the dense recipe's -- random scale in
    [0.5, 2.0], horizontal flip, 512 crop -- applied jointly to image and
    boxes. It has to be reimplemented for boxes rather than reused, because
    the mask transform interpolates and boxes must be transformed
    analytically; the CONTENT test in tests/ checks the two agree on where a
    pixel lands.
    """

    def __init__(self, data_root, split, crop=512, train=True, pct=None,
                 scale_range=(0.5, 2.0)):
        self.root = data_root
        self.ids = dd.read_split(data_root, split, ds="voc_seg")
        if pct is not None and pct < 100:
            # The SAME committed indices the segmentation cells use, so a
            # detection cell at 5% sees exactly the images its segmentation
            # counterpart saw.
            idx = dd.load_subset_indices(pct, ds="voc_seg")
            self.ids = [self.ids[i] for i in idx]
        self.crop = crop
        self.train = train
        self.scale_range = scale_range

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        import random
        import numpy as np
        from PIL import Image
        import torchvision.transforms.functional as TF

        img_id = self.ids[i]
        img = Image.open(dd._image_path(self.root, img_id, ds="voc_seg")).convert("RGB")
        boxes, labels, difficult = parse_boxes(self.root, img_id)

        if self.train:
            s = random.uniform(*self.scale_range)
            w, h = img.size
            nw, nh = int(round(w * s)), int(round(h * s))
            img = img.resize((nw, nh), Image.BILINEAR)
            boxes = boxes * s

            if random.random() < 0.5:
                img = TF.hflip(img)
                if len(boxes):
                    x1 = boxes[:, 0].clone()
                    boxes[:, 0] = nw - boxes[:, 2]
                    boxes[:, 2] = nw - x1

            # pad-then-crop, so small images are not upscaled to fill the crop
            pw, ph = max(0, self.crop - nw), max(0, self.crop - nh)
            if pw or ph:
                img = TF.pad(img, [0, 0, pw, ph], fill=0)
                nw, nh = nw + pw, nh + ph
            ox = random.randint(0, nw - self.crop)
            oy = random.randint(0, nh - self.crop)
            img = img.crop((ox, oy, ox + self.crop, oy + self.crop))
            if len(boxes):
                boxes[:, [0, 2]] -= ox
                boxes[:, [1, 3]] -= oy
                boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(0, self.crop)
                boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(0, self.crop)
                # A box cropped to nothing must be dropped, not kept as a
                # degenerate zero-area target -- it would otherwise contribute
                # an unsatisfiable regression target at every location.
                keep = ((boxes[:, 2] - boxes[:, 0]) > 1) & ((boxes[:, 3] - boxes[:, 1]) > 1)
                boxes, labels, difficult = boxes[keep], labels[keep], difficult[keep]

        # Identical normalization to the segmentation path, so the two tasks
        # feed the backbone the same pixel statistics.
        t = TF.normalize(TF.to_tensor(img), dd.MEAN, dd.STD)
        return t, {"boxes": boxes, "labels": labels, "difficult": difficult,
                   "image_id": img_id,
                   "size": torch.tensor([img.size[1], img.size[0]])}


def collate(batch):
    """Images stack only when training (fixed crop); at eval they are native
    size and variable, so the batch stays a list."""
    imgs = [b[0] for b in batch]
    tgts = [b[1] for b in batch]
    if all(i.shape == imgs[0].shape for i in imgs):
        return torch.stack(imgs), tgts
    return imgs, tgts
