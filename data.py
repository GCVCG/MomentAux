"""Datasets, transforms, and FIXED label subsets.

Subset protocol: the {1,5,10,25}% CIFAR-100 subsets are stratified, drawn once
with a committed seed, and stored as JSON index files under data/subsets/ in
the repo. Every stem variant trains on the identical indices; train.py refuses
to run a subset cell without the committed file.
"""

import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms

SUBSET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "subsets")
SUBSET_SEED = 0

STATS = {
    "cifar100": ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
    "cifar100super": ((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
    "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    "stl10": ((0.4467, 0.4398, 0.4066), (0.2603, 0.2566, 0.2713)),
    "tin": ((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821)),
}

NUM_CLASSES = {"cifar100": 100, "cifar100super": 20, "cifar10": 10, "stl10": 10,
               "tin": 200}
IMAGE_SIZE = {"cifar100": 32, "cifar100super": 32, "cifar10": 32, "stl10": 96,
              "tin": 64}

# cifar100super is CIFAR-100's IMAGES with its 20 official coarse labels, and it
# deliberately reuses CIFAR-100's COMMITTED subset indices (data/subsets/
# cifar100_*.json). That is the whole point: at a given pct the two datasets see
# byte-identical images, the same count, and the same number of optimizer steps,
# so ONLY per-class count changes (x5: a 100-fine-class-stratified subset gives
# exactly 5 fine classes x n per coarse class). It is the one design that breaks
# the CIFAR-10-vs-CIFAR-100 confound -- both of those are 50,000 images, so
# matching per-class count there NECESSARILY unmatches total data/steps by 10x.
SUBSET_ALIAS = {"cifar100super": "cifar100"}

# The 15 standard CIFAR-C corruptions of Hendrycks & Dietterich (ICLR 2019).
CIFAR_C_CORRUPTIONS = (
    "gaussian_noise", "shot_noise", "impulse_noise",
    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
    "snow", "frost", "fog", "brightness",
    "contrast", "elastic_transform", "pixelate", "jpeg_compression",
)


def build_transforms(dataset, train):
    """Standard crop+flip only (recipe v1: minimal and identical for all)."""
    mean, std = STATS[dataset]
    normalize = [transforms.ToTensor(), transforms.Normalize(mean, std)]
    if not train:
        return transforms.Compose(normalize)
    size = IMAGE_SIZE[dataset]
    pad = size // 8  # 4 px at 32, 12 px at 96
    return transforms.Compose(
        [transforms.RandomCrop(size, padding=pad), transforms.RandomHorizontalFlip()]
        + normalize
    )


def make_subset_indices(labels, pct, seed=SUBSET_SEED):
    """Deterministic stratified subset: for each class, a seeded permutation
    of its indices, truncated to pct%. Same (labels, pct, seed) -> identical
    indices on every machine and run."""
    labels = np.asarray(labels)
    indices = []
    for c in sorted(np.unique(labels)):
        cls_idx = np.flatnonzero(labels == c)
        n = int(round(len(cls_idx) * pct / 100.0))
        if n < 1:
            raise ValueError(f"pct={pct} leaves zero samples for class {c}")
        rs = np.random.RandomState(seed * 100003 + int(c))
        indices.extend(cls_idx[rs.permutation(len(cls_idx))][:n].tolist())
    return sorted(int(i) for i in indices)


def subset_path(dataset, pct):
    return os.path.join(SUBSET_DIR, f"{SUBSET_ALIAS.get(dataset, dataset)}_{pct}pct.json")


def load_subset_indices(dataset, pct):
    dataset = SUBSET_ALIAS.get(dataset, dataset)
    path = subset_path(dataset, pct)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Committed subset file missing: {path}. "
            "Run scripts/make_subsets.py once and commit the result -- "
            "subsets must be identical for every stem."
        )
    with open(path) as f:
        payload = json.load(f)
    assert payload["dataset"] == dataset and payload["pct"] == pct
    return payload["indices"]


def cifar100_coarse_labels(data_root, train):
    """CIFAR-100's 20 official coarse labels, in dataset order.

    torchvision's CIFAR100 exposes only ``fine_labels``, so read the raw pickle
    it already downloaded. Returns (coarse, fine); ``fine`` is returned so the
    caller can assert the ordering matches torchvision's ``targets``.
    """
    import pickle

    path = os.path.join(data_root, "cifar-100-python", "train" if train else "test")
    with open(path, "rb") as f:
        d = pickle.load(f, encoding="latin1")
    return d["coarse_labels"], d["fine_labels"]


class TinyImageNetVal(Dataset):
    """Tiny-ImageNet's val split: images are flat under val/images/ with labels
    in val_annotations.txt, so ImageFolder cannot read it directly.

    Class indices are sorted(wnids) -- identical to what ImageFolder derives for
    the train split, so train and val agree. (If they didn't, val accuracy would
    be a silent permutation of the truth.)
    """

    def __init__(self, root, transform=None):
        from torchvision.datasets.folder import default_loader

        wnids = sorted(
            d for d in os.listdir(os.path.join(root, "train"))
            if os.path.isdir(os.path.join(root, "train", d))
        )
        self.class_to_idx = {w: i for i, w in enumerate(wnids)}
        self.loader = default_loader  # converts to RGB (TIN has grayscale files)
        self.transform = transform
        self.samples = []
        with open(os.path.join(root, "val", "val_annotations.txt")) as f:
            for line in f:
                fn, wnid = line.split("\t")[:2]
                self.samples.append(
                    (os.path.join(root, "val", "images", fn), self.class_to_idx[wnid])
                )
        self.targets = [t for _, t in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, y = self.samples[i]
        img = self.loader(path)
        return (self.transform(img) if self.transform else img), y


def tin_root(data_root):
    return os.path.join(data_root, "tiny-imagenet-200")


class Relabelled(Dataset):
    """Same images, different label map (see SUBSET_ALIAS note above)."""

    def __init__(self, ds, targets):
        if len(targets) != len(ds):
            raise ValueError(f"{len(targets)} targets for {len(ds)} images")
        self.ds = ds
        self.targets = list(targets)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        return self.ds[i][0], self.targets[i]


def build_dataset(dataset, data_root, train, subset_pct=None, download=True):
    tf = build_transforms(dataset, train)
    if dataset == "cifar100":
        ds = datasets.CIFAR100(data_root, train=train, transform=tf, download=download)
    elif dataset == "cifar100super":
        ds = datasets.CIFAR100(data_root, train=train, transform=tf, download=download)
        coarse, fine = cifar100_coarse_labels(data_root, train)
        if list(ds.targets) != list(fine):
            raise RuntimeError("coarse-label pickle order != torchvision targets")
        ds = Relabelled(ds, coarse)
    elif dataset == "cifar10":
        ds = datasets.CIFAR10(data_root, train=train, transform=tf, download=download)
    elif dataset == "stl10":
        split = "train" if train else "test"
        ds = datasets.STL10(data_root, split=split, transform=tf, download=download)
    elif dataset == "tin":
        root = tin_root(data_root)
        if not os.path.isdir(root):
            raise FileNotFoundError(
                f"{root} missing. Tiny-ImageNet has no torchvision downloader; get it "
                "once with:\n  cd data && wget http://cs231n.stanford.edu/"
                "tiny-imagenet-200.zip && unzip -q tiny-imagenet-200.zip"
            )
        ds = (
            datasets.ImageFolder(os.path.join(root, "train"), transform=tf)
            if train
            else TinyImageNetVal(root, transform=tf)
        )
    else:
        raise ValueError(f"unknown dataset {dataset!r}")
    if subset_pct is not None and subset_pct != 100:
        if not train:
            raise ValueError("subsets apply to the train split only")
        ds = Subset(ds, load_subset_indices(dataset, subset_pct))
    return ds


def calibration_batch(dataset, data_root, n=1024):
    """Deterministic batch for stem response calibration: the first n train
    images in index order, eval transform (no augmentation). Identical for
    every stem, subset, and seed of a dataset; uses image statistics only
    (no labels), so it leaks nothing into the low-label protocol."""
    tf = build_transforms(dataset, train=False)
    if dataset in ("cifar100", "cifar100super"):  # labels unused; same images
        ds = datasets.CIFAR100(data_root, train=True, transform=tf, download=False)
    elif dataset == "cifar10":
        ds = datasets.CIFAR10(data_root, train=True, transform=tf, download=False)
    elif dataset == "stl10":
        ds = datasets.STL10(data_root, split="train", transform=tf, download=False)
    elif dataset == "tin":
        ds = datasets.ImageFolder(os.path.join(tin_root(data_root), "train"), transform=tf)
    else:
        raise ValueError(f"unknown dataset {dataset!r}")
    return torch.stack([ds[i][0] for i in range(min(n, len(ds)))])


class CIFARCorrupted(Dataset):
    """CIFAR-10-C / CIFAR-100-C (Hendrycks & Dietterich, ICLR 2019).

    Expects the Zenodo layout: <root>/<corruption>.npy of shape
    (50000, 32, 32, 3) uint8 HWC -- 10000 test images x severities 1..5
    stacked -- plus labels.npy.
    """

    def __init__(self, root, corruption, severity, dataset="cifar100"):
        if not 1 <= severity <= 5:
            raise ValueError("severity must be in 1..5")
        images = np.load(os.path.join(root, f"{corruption}.npy"), mmap_mode="r")
        labels = np.load(os.path.join(root, "labels.npy"))
        assert len(images) % 5 == 0 and len(images) == len(labels), (
            "CIFAR-C file must stack 5 severities of equal size"
        )
        n_per = len(images) // 5
        lo, hi = (severity - 1) * n_per, severity * n_per
        self.images = images[lo:hi]
        self.labels = labels[lo:hi].astype(np.int64)
        assert self.images.shape[1:] == (32, 32, 3), (
            f"expected HWC uint8 CIFAR-C layout, got {self.images.shape}"
        )
        mean, std = STATS[dataset]
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        img = np.asarray(self.images[i])  # HWC uint8
        # Explicit HWC -> CHW via permute (never reshape!) + content checks.
        x = torch.from_numpy(img.copy())
        assert x.shape == (32, 32, 3)
        x = x.permute(2, 0, 1).contiguous().float() / 255.0
        x = (x - self.mean) / self.std
        return x, int(self.labels[i])
