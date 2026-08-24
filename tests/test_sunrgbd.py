"""SUN RGB-D (limitations campaign block H): the third multi-source
population. Three sources -- rgb (3ch), depth (1ch), all (4ch) -- are the SAME
frames read through different channels of one packed array, and the three
share ONE committed subset file. These tests pin that contract, the table
entries, and that the 1- and 4-channel stems build and run on both backbones.

The pack-dependent tests skip when data/sunrgbd_64_images.npy is absent (it is
677 MB and not committed); the table/model tests always run.
"""

import os

import numpy as np
import pytest
import torch

import data as data_mod
from data import (
    IMAGE_SIZE,
    INPUT_CHANNELS,
    NUM_CLASSES,
    SUBSET_ALIAS,
    SUNRGBD_BANDS,
    build_dataset,
    load_subset_indices,
    subset_path,
)

PACK = os.path.join("data", "sunrgbd_64_images.npy")
META = os.path.join("data", "sunrgbd_64_meta.npz")
needs_pack = pytest.mark.skipif(not (os.path.isfile(PACK) and os.path.isfile(META)),
                                reason="SUN RGB-D pack not built (scripts/make_sunrgbd.py)")

SOURCES = ("sunrgbd_rgb", "sunrgbd_depth", "sunrgbd_all")


# ---------------------------------------------------------------- tables

def test_tables_are_consistent():
    for d in SOURCES:
        assert NUM_CLASSES[d] == 19
        assert IMAGE_SIZE[d] == 64
        assert SUBSET_ALIAS[d] == "sunrgbd"
        assert subset_path(d, 5) == subset_path("sunrgbd_all", 5)
    assert INPUT_CHANNELS["sunrgbd_rgb"] == 3
    assert INPUT_CHANNELS["sunrgbd_depth"] == 1
    assert INPUT_CHANNELS["sunrgbd_all"] == 4
    assert SUNRGBD_BANDS["rgb"] == (0, 1, 2)
    assert SUNRGBD_BANDS["depth"] == (3,)
    assert SUNRGBD_BANDS["all"] == (0, 1, 2, 3)
    for d in SOURCES:
        assert data_mod.build_transforms(d, train=True) is None   # tensor-native path


def test_committed_subsets_exist_and_are_shared():
    """One file per pct (sunrgbd_<pct>pct.json), the same indices whichever
    source asks for them. 1% and 2% are deliberately ABSENT: 4,845 frames
    over 19 classes put both below one batch of 128 (and 1% below one sample
    for the smallest classes), the same floor as stl10/cub @1-2%."""
    for pct in (3, 5, 10, 25):
        a = load_subset_indices("sunrgbd_rgb", pct)
        assert a == load_subset_indices("sunrgbd_depth", pct)
        assert a == load_subset_indices("sunrgbd_all", pct)
        assert len(a) == len(set(a))
        assert len(a) >= 128, "a committed fraction must yield >= one batch"
    for pct in (1, 2):
        assert not os.path.exists(subset_path("sunrgbd_all", pct))


# ---------------------------------------------------------------- models

@pytest.mark.parametrize("in_ch", [1, 4])
def test_energy_stem_accepts_1_and_4_channels(in_ch):
    from momentstem import EnergyStem

    stem = EnergyStem("magnitude", in_channels=in_ch, kernel_size=11)
    x = torch.rand(2, in_ch, 64, 64)
    y = stem(x)
    assert y.shape[:2] == (2, stem.out_channels)
    assert y.shape[1] == in_ch + (stem.out_channels - in_ch)
    assert torch.isfinite(y).all()


@pytest.mark.parametrize("backbone,tap", [("resnet18", "layer3"), ("vit_tiny", "blocks.8")])
@pytest.mark.parametrize("in_ch", [1, 4])
def test_aux_model_builds_and_runs_on_n_channels(backbone, tap, in_ch):
    """The aux model must accept 1- and 4-channel input on both backbones
    used by block H (ResNet-18 frozen recipe, ViT-tiny diag) at 64px."""
    from momentstem import build_model

    model = build_model(
        backbone, "none", num_classes=19, small_input=True,
        moment_aux={"stem": "energy-magnitude", "tap": tap, "weight": 1.0,
                    "head_norm": True},
        image_size=64, in_channels=in_ch,
    )
    x = torch.rand(2, in_ch, 64, 64)
    model.calibrate(x)
    model.train()
    out = model(x)
    logits = out[0] if isinstance(out, tuple) else out
    assert logits.shape == (2, 19)
    model.eval()
    with torch.no_grad():
        out = model(x)
    logits = out[0] if isinstance(out, tuple) else out
    assert logits.shape == (2, 19)


# ---------------------------------------------------------------- the pack

@needs_pack
def test_pack_shape_classes_and_split():
    z = np.load(META, allow_pickle=False)
    x = np.load(PACK, mmap_mode="r")
    assert x.ndim == 4 and x.shape[1:] == (4, 64, 64)
    assert x.dtype == np.float32            # the so2sat pack dtype
    assert len(z["labels"]) == x.shape[0]
    assert len(z["train_idx"]) == 4845 and len(z["test_idx"]) == 4659
    assert len(np.unique(z["labels"])) == 19
    assert len(z["classes"]) == 19
    assert z["mean"].shape == (4,) and z["std"].shape == (4,)
    # the pinned STATS entry must be the pack's own train-split statistics
    mean, std = data_mod.STATS["sunrgbd_all"]
    assert np.allclose(mean, z["mean"], atol=5e-4)
    assert np.allclose(std, z["std"], atol=5e-4)
    assert data_mod.STATS["sunrgbd_rgb"] == (tuple(mean[:3]), tuple(std[:3]))
    assert data_mod.STATS["sunrgbd_depth"] == ((mean[3],), (std[3],))


@needs_pack
def test_sources_are_channel_views_of_the_same_frames():
    """CONTENT test, not shape: rgb[i] == all[i][:3] and depth[i] == all[i][3],
    for both splits, under the deterministic eval transform."""
    for train in (True, False):
        rgb = build_dataset("sunrgbd_rgb", "./data", train=train)
        dep = build_dataset("sunrgbd_depth", "./data", train=train)
        al = build_dataset("sunrgbd_all", "./data", train=train)
        for ds in (rgb, dep, al):
            ds.transform = None
        assert len(rgb) == len(dep) == len(al) == (4845 if train else 4659)
        for i in (0, 7, 1234, len(al) - 1):
            xr, yr = rgb[i]
            xd, yd = dep[i]
            xa, ya = al[i]
            assert yr == yd == ya
            assert xr.shape == (3, 64, 64) and xd.shape == (1, 64, 64) and xa.shape == (4, 64, 64)
            assert torch.equal(xr, xa[:3])
            assert torch.equal(xd, xa[3:4])
            assert torch.isfinite(xa).all()


@needs_pack
def test_subset_is_stratified_over_the_train_labels():
    z = np.load(META, allow_pickle=False)
    ytr = z["labels"][z["train_idx"]]
    for pct in (3, 5, 10, 25):
        idx = load_subset_indices("sunrgbd_all", pct)
        counts = np.bincount(ytr[idx], minlength=19)
        expect = np.array([int(round((ytr == c).sum() * pct / 100.0)) for c in range(19)])
        assert (counts == expect).all()
        assert counts.min() >= 1


@needs_pack
def test_calibration_batch_has_the_source_channel_count():
    for d, c in (("sunrgbd_rgb", 3), ("sunrgbd_depth", 1), ("sunrgbd_all", 4)):
        b = data_mod.calibration_batch(d, "./data", n=8)
        assert b.shape == (8, c, 64, 64)
