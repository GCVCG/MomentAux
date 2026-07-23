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
    # tinsem: tinsuper's one-variable SEMANTIC control -- same images (tin@1%
    # committed subset via SUBSET_ALIAS), same block-of-10 coarse construction,
    # but blocks are taken in WordNet HYPERNYM-PATH order (committed in
    # data/subsets/tin_semantic_order.json, scripts/make_tin_semantic_order.py)
    # instead of lexicographic wnid order. Adjudicates the Q6.9j caveat:
    # does semantic coherence of the coarse groups matter, at byte-identical
    # pixels and identical label-count structure?
    "tinsem": ((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821)),
    # CUB-200-2011 at 64x64 (squash-resize; see CUB200): the first GENUINELY
    # fine-grained dataset (200 bird species, ~30 train img/cls at 100%) --
    # the "prior substitutes for fine-grained weak supervision" test.
    # Stats computed 2026-07-20 on the 5994-image train split at 64x64 after
    # the squash-resize (per-pixel mean/std over all images), then pinned.
    "cub": ((0.4857, 0.4995, 0.4324), (0.2159, 0.2112, 0.2509)),
    # --- DOMAIN-GENERALIZATION datasets (2026-07-23, user-approved: the five
    # existing populations are all small natural-photo sets; these test the
    # prior where image statistics differ fundamentally). Stats pinned from
    # each train split at 64px, same procedure as cub.
    # EuroSAT: Sentinel-2 satellite RGB, 27000 imgs, 64px native, 10 classes.
    # No official split -> deterministic per-class 80/20 (see EuroSAT64).
    # Computed 2026-07-23 on the 21600-image deterministic train split.
    "eurosat": ((0.3445, 0.3805, 0.4079), (0.2040, 0.1369, 0.1151)),
    # DTD textures: 47 classes, train+val (3760) as train / test (1880) as
    # test, partition 1, squash-resized to 64px like cub.
    # Computed 2026-07-23 on the 3760-image train+val split at 64px.
    "dtd": ((0.5273, 0.4702, 0.4235), (0.2455, 0.2331, 0.2431)),
    # PathMNIST (MedMNIST+ 64px variant): colon histopathology, 9 classes,
    # 89996 train / 7180 test, 64px native npz.
    # Computed 2026-07-23 on the 89996-image train split.
    "pathmnist": ((0.7405, 0.5330, 0.7058), (0.1404, 0.1952, 0.1388)),
    # Food-101 at 64px (squash-resize): 101 classes, 750 train/cls.
    # Computed 2026-07-23 on the 75750-image train split at 64px.
    "food101": ((0.5450, 0.4435, 0.3436), (0.2612, 0.2627, 0.2675)),
}

NUM_CLASSES = {"cifar100": 100, "cifar100super": 20, "cifar10": 10, "stl10": 10,
               "tin": 200, "tin20": 20, "tin20b": 20, "tinsuper": 20,
               "tinsem": 20, "cub": 200,
               "eurosat": 10, "dtd": 47, "pathmnist": 9, "food101": 101}
IMAGE_SIZE = {"cifar100": 32, "cifar100super": 32, "cifar10": 32, "stl10": 96,
              "tin": 64, "tin20": 64, "tin20b": 64, "tinsuper": 64,
              "tinsem": 64, "cub": 64,
              "eurosat": 64, "dtd": 64, "pathmnist": 64, "food101": 64}

# cifar100super is CIFAR-100's IMAGES with its 20 official coarse labels, and it
# deliberately reuses CIFAR-100's COMMITTED subset indices (data/subsets/
# cifar100_*.json). That is the whole point: at a given pct the two datasets see
# byte-identical images, the same count, and the same number of optimizer steps,
# so ONLY per-class count changes (x5: a 100-fine-class-stratified subset gives
# exactly 5 fine classes x n per coarse class). It is the one design that breaks
# the CIFAR-10-vs-CIFAR-100 confound -- both of those are 50,000 images, so
# matching per-class count there NECESSARILY unmatches total data/steps by 10x.
SUBSET_ALIAS = {"cifar100super": "cifar100", "tinsuper": "tin", "tinsem": "tin"}

# The 15 standard CIFAR-C corruptions of Hendrycks & Dietterich (ICLR 2019).
CIFAR_C_CORRUPTIONS = (
    "gaussian_noise", "shot_noise", "impulse_noise",
    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
    "snow", "frost", "fog", "brightness",
    "contrast", "elastic_transform", "pixelate", "jpeg_compression",
)


def build_transforms(dataset, train, augment=None):
    """Standard crop+flip only (recipe v1: minimal and identical for all).

    :param augment "deit" ADDS the DeiT (Touvron et al. 2021) augmentation
        stack on top of the study's base crop+flip, with DeiT's PUBLISHED
        hyper-parameters verbatim: RandAugment `rand-m9-mstd0.5-inc1` and
        RandomErasing p=0.25 mode=pixel. (Mixup 0.8 / CutMix 1.0 / label
        smoothing 0.1 are batch-level and live in train.py.)
        The base RandomCrop+flip is DELIBERATELY retained so the only
        difference from every other cell in the study is the ADDED
        augmentation -- a one-variable change. Diag-only.
    """
    mean, std = STATS[dataset]
    normalize = [transforms.ToTensor(), transforms.Normalize(mean, std)]
    if not train:
        return transforms.Compose(normalize)
    size = IMAGE_SIZE[dataset]
    pad = size // 8  # 4 px at 32, 12 px at 96
    base = [transforms.RandomCrop(size, padding=pad),
            transforms.RandomHorizontalFlip()]
    if not augment:
        return transforms.Compose(base + normalize)
    if augment != "deit":
        raise ValueError(f"unknown augment {augment!r} (only 'deit')")
    from timm.data.auto_augment import rand_augment_transform
    from timm.data.random_erasing import RandomErasing

    aa = rand_augment_transform(
        "rand-m9-mstd0.5-inc1",
        {"translate_const": int(size * 0.45),
         "img_mean": tuple(int(255 * m) for m in mean)},
    )
    erase = RandomErasing(probability=0.25, mode="pixel", device="cpu")
    return transforms.Compose(base + [aa] + normalize + [erase])


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


class Squash64(Dataset):
    """Wrap a (PIL, label) dataset with a 64x64 BILINEAR squash-resize before
    the transform -- the exact CUB200 convention (whole subject stays in
    frame; fixed choice, identical for every cell, cancels in every Delta)."""

    def __init__(self, base, transform=None):
        from PIL import Image
        self.base, self.transform, self._Image = base, transform, Image
        self.targets = list(getattr(base, "targets", []) or
                            [base[i][1] for i in range(len(base))])

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        img, y = self.base[i]
        img = img.convert("RGB").resize((64, 64), self._Image.BILINEAR)
        return (self.transform(img) if self.transform else img), y


class EuroSAT64(Dataset):
    """EuroSAT RGB (Helber et al. 2019): 27000 Sentinel-2 patches, 64px
    native, 10 land-use classes. Satellite optics -- the sharpest test of
    whether the moment prior encodes generic early vision or natural-photo
    statistics specifically.

    EuroSAT ships with NO official split. Deterministic 80/20 per-class
    split: torchvision's ImageFolder orders samples by sorted path (stable
    across machines), and each class's index list is permuted by
    RandomState(SUBSET_SEED) with the last 20%% held out as test. Pure
    function of (SUBSET_SEED, sorted paths), so it reproduces byte-identically
    everywhere without a committed file."""

    def __init__(self, data_root, train, transform=None):
        from torchvision import datasets as tvd
        base = tvd.EuroSAT(data_root, download=False)
        rng_targets = np.asarray(base.targets)
        keep = []
        for c in np.unique(rng_targets):
            idx = np.where(rng_targets == c)[0]          # sorted-path order
            perm = np.random.RandomState(SUBSET_SEED).permutation(len(idx))
            cut = int(round(len(idx) * 0.8))
            keep.extend(idx[perm[:cut]] if train else idx[perm[cut:]])
        self.indices = sorted(keep)
        self.base, self.transform = base, transform
        self.targets = [int(rng_targets[i]) for i in self.indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        img, y = self.base[self.indices[i]]
        return (self.transform(img) if self.transform else img), y


class PathMNIST64(Dataset):
    """PathMNIST (MedMNIST+ 64px variant, Yang et al. 2023): colon-pathology
    patches, 9 tissue classes, 89996 train / 7180 test. Histopathology stain
    statistics are the furthest from ImageNet-like photos of any population
    in the study -- the biomedical domain test.

    Expects <data_root>/pathmnist_64.npz (Zenodo record 10519652)."""

    def __init__(self, data_root, train, transform=None):
        from PIL import Image
        path = os.path.join(data_root, "pathmnist_64.npz")
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"{path} missing. Get it once with:\n  cd data && wget "
                "https://zenodo.org/records/10519652/files/pathmnist_64.npz")
        z = np.load(path)
        split = "train" if train else "test"
        self.images = z[f"{split}_images"]           # (N, 64, 64, 3) uint8
        self.targets = [int(t) for t in z[f"{split}_labels"].ravel()]
        self.transform, self._Image = transform, Image

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        img = self._Image.fromarray(self.images[i])
        return (self.transform(img) if self.transform else img), self.targets[i]


def dtd64(data_root, train, transform=None):
    """DTD (Cimpoi et al. 2014) at 64px: 47 texture classes, partition 1,
    train+val (3760, 80/cls) as train -- the standard DTD protocol -- and
    test (1880) as test. Texture-dominated images: the moment prior IS an
    oriented-energy descriptor, so this is the domain where its inductive
    bias should be strongest."""
    from torch.utils.data import ConcatDataset
    from torchvision import datasets as tvd
    if train:
        tr = tvd.DTD(data_root, split="train", partition=1, download=False)
        va = tvd.DTD(data_root, split="val", partition=1, download=False)
        base = ConcatDataset([tr, va])
        base.targets = list(tr._labels) + list(va._labels)
    else:
        base = tvd.DTD(data_root, split="test", partition=1, download=False)
        base.targets = list(base._labels)
    return Squash64(base, transform)


def food101_64(data_root, train, transform=None):
    """Food-101 (Bossard et al. 2014) at 64px squash-resize: 101 dishes,
    750 train / 250 test per class. Fine-grained, texture-rich natural
    photos -- the food-domain population (user request 2026-07-23; the
    segmentation set FoodSeg103 does not fit the classification recipe)."""
    from torchvision import datasets as tvd
    split = "train" if train else "test"
    base = tvd.Food101(data_root, split=split, download=False)
    base.targets = list(base._labels)
    return Squash64(base, transform)


def tin_root(data_root):
    return os.path.join(data_root, "tiny-imagenet-200")


def _tin_wnid_slice(root, offset):
    wnids = sorted(
        d for d in os.listdir(os.path.join(root, "train"))
        if os.path.isdir(os.path.join(root, "train", d))
    )
    return wnids[offset::10]


def tin_semantic_coarse_map(root):
    """fine_idx (sorted-wnid order, as ImageFolder assigns) -> tinsem coarse
    label: the wnid's rank in the COMMITTED WordNet hypernym-path order
    (data/subsets/tin_semantic_order.json) // 10. Mirrors tinsuper's
    fine_idx // 10 with only the sort key changed."""
    path = os.path.join(SUBSET_DIR, "tin_semantic_order.json")
    with open(path) as f:
        order = json.load(f)["order"]
    all_wnids = sorted(
        d for d in os.listdir(os.path.join(root, "train"))
        if os.path.isdir(os.path.join(root, "train", d))
    )
    if sorted(order) != all_wnids:
        raise RuntimeError("committed tin_semantic_order.json wnids != dataset's")
    rank = {w: i for i, w in enumerate(order)}
    return [rank[w] // 10 for w in all_wnids]


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


def build_dataset(dataset, data_root, train, subset_pct=None, download=True,
                  augment=None):
    tf = build_transforms(dataset, train, augment=augment)
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
    elif dataset == "tinsem":
        # tinsuper with the SEMANTIC sort key: same images and index order,
        # coarse label = committed-hypernym-path rank // 10.
        root = tin_root(data_root)
        base = (
            datasets.ImageFolder(os.path.join(root, "train"), transform=tf)
            if train
            else TinyImageNetVal(root, transform=tf)
        )
        cmap = tin_semantic_coarse_map(root)
        ds = Relabelled(base, [cmap[t] for t in base.targets])
    elif dataset == "cub":
        ds = CUB200(data_root, train=train, transform=tf)
    elif dataset == "eurosat":
        ds = EuroSAT64(data_root, train=train, transform=tf)
    elif dataset == "dtd":
        ds = dtd64(data_root, train=train, transform=tf)
    elif dataset == "pathmnist":
        ds = PathMNIST64(data_root, train=train, transform=tf)
    elif dataset == "food101":
        ds = food101_64(data_root, train=train, transform=tf)
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
    elif dataset in ("tin", "tin20", "tin20b", "tinsuper", "tinsem"):
        # tin20/tin20b/tinsuper calibrate on FULL tin deliberately (labels unused): the aux
        # target pipeline stays byte-identical to tin@1%'s, which is the whole
        # point of the within-tin granularity control. Mirrors cifar100super
        # calibrating on cifar100's images.
        ds = datasets.ImageFolder(os.path.join(tin_root(data_root), "train"), transform=tf)
    elif dataset == "cub":
        ds = CUB200(data_root, train=True, transform=tf)
    elif dataset in ("eurosat", "dtd", "pathmnist", "food101"):
        ds = build_dataset(dataset, data_root, train=True, subset_pct=None,
                           download=False)
        ds = _with_eval_transform(ds, tf)
    else:
        raise ValueError(f"unknown dataset {dataset!r}")
    return torch.stack([ds[i][0] for i in range(min(n, len(ds)))])


def _with_eval_transform(ds, tf):
    """Return ds with its transform swapped to tf (calibration uses the eval
    transform). The new datasets all expose .transform at the top level."""
    ds.transform = tf
    return ds


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
