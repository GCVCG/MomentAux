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
    # tin20 keeps tin's normalization so the pixel pipeline is IDENTICAL --
    # the whole point is that only the label space changes.
    "tin20": ((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821)),
    "tin20b": ((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821)),
    # tinsuper: tin's IMAGES (all 200 classes) with 20 coarse labels
    # (sorted-wnid index // 10) -- the byte-identical-pixel granularity
    # control on tin, mirroring cifar100super. Groups are positional, not
    # semantic. OUTCOME (Q6.9k): the incoherent groups keep the baseline
    # low (14.08), so no readout boost -- label COUNT alone buys nothing;
    # what matters is baseline task performance, as the sign law says.
    "tinsuper": ((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821)),
    # CUB-200-2011 at 64x64 (squash-resize; see CUB200): the first GENUINELY
    # fine-grained dataset (200 bird species, ~30 train img/cls at 100%) --
    # the "prior substitutes for fine-grained weak supervision" test.
    # Stats computed 2026-07-20 on the 5994-image train split at 64x64 after
    # the squash-resize (per-pixel mean/std over all images), then pinned.
    "cub": ((0.4857, 0.4995, 0.4324), (0.2159, 0.2112, 0.2509)),
}

NUM_CLASSES = {"cifar100": 100, "cifar100super": 20, "cifar10": 10, "stl10": 10,
               "tin": 200, "tin20": 20, "tin20b": 20, "tinsuper": 20,
               "cub": 200}
IMAGE_SIZE = {"cifar100": 32, "cifar100super": 32, "cifar10": 32, "stl10": 96,
              "tin": 64, "tin20": 64, "tin20b": 64, "tinsuper": 64,
              "cub": 64}

# cifar100super is CIFAR-100's IMAGES with its 20 official coarse labels, and it
# deliberately reuses CIFAR-100's COMMITTED subset indices (data/subsets/
# cifar100_*.json). That is the whole point: at a given pct the two datasets see
# byte-identical images, the same count, and the same number of optimizer steps,
# so ONLY per-class count changes (x5: a 100-fine-class-stratified subset gives
# exactly 5 fine classes x n per coarse class). It is the one design that breaks
# the CIFAR-10-vs-CIFAR-100 confound -- both of those are 50,000 images, so
# matching per-class count there NECESSARILY unmatches total data/steps by 10x.
SUBSET_ALIAS = {"cifar100super": "cifar100", "tinsuper": "tin"}

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


class CiFAIRTest(Dataset):
    """ciFAIR-100 test set (Barz & Denzler 2020, arXiv:1902.00423).

    CIFAR-100's test set with the ~9% of images that DUPLICATE training images
    replaced by fresh ones (927/10000 here); labels and order are unchanged, so
    it is a drop-in swap. Its only job is to answer "how much of a low-data
    result rests on train/test duplication?" -- a question any reviewer asks of
    a paper whose headline cells live at 1-5% data.

    Note the contamination is against the FULL train set, so at 1% (500 imgs)
    most duplicate sources are not even in the training subset -- expect the
    effect to be smaller at low data, and smaller still on the aux-vs-baseline
    DELTA, since both cells eat the identical contamination.
    """

    def __init__(self, data_root, transform=None):
        import pickle

        path = os.path.join(data_root, "ciFAIR-100", "test")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{path} missing. Get it once with:\n  cd data && curl -sLO "
                "https://github.com/cvjena/cifair/releases/download/v1.0/"
                "ciFAIR-100.zip && unzip -q ciFAIR-100.zip"
            )
        with open(path, "rb") as f:
            d = pickle.load(f, encoding="latin1")
        # same layout as CIFAR's pickle: (N, 3072) flat CHW -> HWC uint8 for PIL
        self.data = np.asarray(d["data"]).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        self.targets = list(d["fine_labels"])
        self.transform = transform

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        from PIL import Image

        img = Image.fromarray(self.data[i])
        return (self.transform(img) if self.transform else img), self.targets[i]


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


class CUB200(Dataset):
    """CUB-200-2011 (Wah et al. 2011) at 64x64: 200 bird species, 5994 train /
    5794 test images, ~30 train images per class -- a NATURALLY low-label,
    genuinely fine-grained task (tin20/tinsuper showed fine-grained weak
    supervision is where the prior does the most feature work).

    Images are squash-resized to 64x64 (BILINEAR) BEFORE the standard
    crop+flip transform so the pipeline downstream of the resize is byte-
    identical to tin's (RandomCrop(64, padding=8) + flip). Squash rather than
    shortest-side+center-crop keeps the whole bird in frame; the choice is
    fixed and identical for every cell, so it cancels in every Δ.

    Expects <data_root>/CUB_200_2011/{images/, images.txt,
    image_class_labels.txt, train_test_split.txt} (the official tgz layout).
    """

    def __init__(self, data_root, train, transform=None):
        from PIL import Image

        self.root = os.path.join(data_root, "CUB_200_2011")
        if not os.path.isdir(self.root):
            raise FileNotFoundError(
                f"{self.root} missing. Get it once with:\n  cd data && wget "
                "https://data.caltech.edu/records/65de6-vp158/files/"
                "CUB_200_2011.tgz && tar xzf CUB_200_2011.tgz"
            )
        self.transform = transform
        self._Image = Image

        def read_pairs(name):
            with open(os.path.join(self.root, name)) as f:
                return [line.split() for line in f if line.strip()]

        paths = {i: p for i, p in read_pairs("images.txt")}
        labels = {i: int(c) - 1 for i, c in read_pairs("image_class_labels.txt")}
        is_train = {i: t == "1" for i, t in read_pairs("train_test_split.txt")}
        ids = sorted((i for i in paths if is_train[i] == train), key=int)
        self.samples = [
            (os.path.join(self.root, "images", paths[i]), labels[i]) for i in ids
        ]
        self.targets = [t for _, t in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, y = self.samples[i]
        img = self._Image.open(path).convert("RGB").resize(
            (64, 64), self._Image.BILINEAR
        )
        return (self.transform(img) if self.transform else img), y


def tin_root(data_root):
    return os.path.join(data_root, "tiny-imagenet-200")


def _tin_wnid_slice(root, offset):
    wnids = sorted(
        d for d in os.listdir(os.path.join(root, "train"))
        if os.path.isdir(os.path.join(root, "train", d))
    )
    return wnids[offset::10]


def tin20_wnids(root):
    """The 20 classes of tin20: every 10th of the 200 sorted wnids. Purely
    positional -- deterministic, spread across the sorted list, and immune to
    cherry-picking accusations. tin20 exists to move ONLY label granularity
    within Tiny-ImageNet (5 img/cls 200-way at 1000 imgs vs 50 img/cls 20-way
    at the same image count): the within-tin mirror of cifar100super.
    """
    return _tin_wnid_slice(root, 0)


def tin20b_wnids(root):
    """tin20's class-draw control: a DISJOINT positional slice (offset 5). If
    tin20's +4.07 were a property of its particular 20 classes rather than of
    granularity, tin20b should differ materially."""
    return _tin_wnid_slice(root, 5)


def tin20_filter(ds_targets, root, keep_wnids=None):
    """(keep_indices, new_targets): positions of the 20 kept classes in a
    200-class tin split, and their 0..19 relabels. Shared by build_dataset and
    make_subsets so the two can never disagree about what tin20 contains."""
    if keep_wnids is None:
        keep_wnids = tin20_wnids(root)
    all_wnids = sorted(
        d for d in os.listdir(os.path.join(root, "train"))
        if os.path.isdir(os.path.join(root, "train", d))
    )
    old_idx = {all_wnids.index(w): new for new, w in enumerate(keep_wnids)}
    keep = [i for i, t in enumerate(ds_targets) if t in old_idx]
    return keep, [old_idx[ds_targets[i]] for i in keep]


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
    elif dataset in ("tin20", "tin20b"):
        root = tin_root(data_root)
        base = (
            datasets.ImageFolder(os.path.join(root, "train"), transform=tf)
            if train
            else TinyImageNetVal(root, transform=tf)
        )
        wn = tin20_wnids(root) if dataset == "tin20" else tin20b_wnids(root)
        keep, new_targets = tin20_filter(base.targets, root, keep_wnids=wn)
        ds = Relabelled(Subset(base, keep), new_targets)
    elif dataset == "tinsuper":
        # tin's images and index order untouched; only the label map changes.
        # ImageFolder and TinyImageNetVal both index classes by sorted(wnids),
        # so fine_idx // 10 groups 10 consecutive sorted wnids per coarse
        # class -- deterministic, and tin20's classes (wnids[::10]) are one
        # representative per block.
        root = tin_root(data_root)
        base = (
            datasets.ImageFolder(os.path.join(root, "train"), transform=tf)
            if train
            else TinyImageNetVal(root, transform=tf)
        )
        ds = Relabelled(base, [t // 10 for t in base.targets])
    elif dataset == "cub":
        ds = CUB200(data_root, train=train, transform=tf)
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
    elif dataset in ("tin", "tin20", "tin20b", "tinsuper"):
        # tin20/tin20b/tinsuper calibrate on FULL tin deliberately (labels unused): the aux
        # target pipeline stays byte-identical to tin@1%'s, which is the whole
        # point of the within-tin granularity control. Mirrors cifar100super
        # calibrating on cifar100's images.
        ds = datasets.ImageFolder(os.path.join(tin_root(data_root), "train"), transform=tf)
    elif dataset == "cub":
        ds = CUB200(data_root, train=True, transform=tf)
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
