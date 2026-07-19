"""tinsuper: tin's images with 20 coarse labels (sorted-wnid index // 10).

The byte-identical-pixel granularity control on tin (mirror of cifar100super
on CIFAR-100). Exists to disentangle the tin 2x2's checkpoint-set effect:
tin20/tin20b change the label space AND the pixel population; tinsuper
changes ONLY the label space. FINDINGS Q6.9j follow-up.
"""

import os

import numpy as np
import pytest

from data import build_dataset, calibration_batch, load_subset_indices, subset_path

needs_tin = pytest.mark.skipif(
    not os.path.isdir("./data/tiny-imagenet-200"), reason="Tiny-ImageNet not downloaded"
)


@needs_tin
def test_coarse_labels_are_fine_div_10_on_identical_images():
    """Same images in the same order as tin; targets are exactly fine // 10.
    Both facts matter: index-order identity is what makes tin's committed
    subset files valid for tinsuper via SUBSET_ALIAS."""
    fine = build_dataset("tin", "./data", train=True)
    coarse = build_dataset("tinsuper", "./data", train=True)
    assert len(fine) == len(coarse) == 100000
    f = np.asarray(fine.targets)
    c = np.asarray(coarse.targets)
    assert (c == f // 10).all()

    fv = build_dataset("tin", "./data", train=False)
    cv = build_dataset("tinsuper", "./data", train=False)
    assert (np.asarray(cv.targets) == np.asarray(fv.targets) // 10).all()


@needs_tin
def test_subset_alias_reuses_tin_indices():
    """tinsuper@1% must be byte-identical images to tin@1%: the committed tin
    subset file is resolved via SUBSET_ALIAS, giving 1000 imgs stratified 5
    per fine class = 50 per coarse class."""
    assert subset_path("tinsuper", 1) == subset_path("tin", 1)
    idx = load_subset_indices("tinsuper", 1)
    assert idx == load_subset_indices("tin", 1)
    tr = build_dataset("tinsuper", "./data", train=True)
    t = np.asarray(tr.targets)[idx]
    assert np.bincount(t).tolist() == [50] * 20


@needs_tin
def test_calibration_matches_tin():
    """Aux target pipeline byte-identical to tin@1%'s (same rule as tin20)."""
    import torch

    a = calibration_batch("tin", "./data", n=64)
    b = calibration_batch("tinsuper", "./data", n=64)
    assert torch.equal(a, b)
