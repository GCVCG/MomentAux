"""cifar100super contract: CIFAR-100's images with its 20 official coarse
labels, reusing CIFAR-100's committed subset indices.

The whole experimental value of this dataset rests on the images and the step
count being IDENTICAL to cifar100 at the same pct, so that per-class count is
the only thing that varies. These tests pin exactly that.
"""

import collections

import numpy as np
import pytest
import torch

from data import (
    IMAGE_SIZE,
    NUM_CLASSES,
    build_dataset,
    build_transforms,
    cifar100_coarse_labels,
    load_subset_indices,
    subset_path,
)

pytestmark = pytest.mark.skipif(
    not __import__("os").path.exists("./data/cifar-100-python"),
    reason="CIFAR-100 not downloaded",
)


def test_subset_indices_are_literally_cifar100s():
    for pct in (1, 5, 10, 25):
        assert load_subset_indices("cifar100super", pct) == load_subset_indices(
            "cifar100", pct
        )
        assert subset_path("cifar100super", pct) == subset_path("cifar100", pct)


def test_images_are_byte_identical_to_cifar100():
    tf = build_transforms("cifar100", train=False)  # deterministic
    fine = build_dataset("cifar100", "./data", train=True, subset_pct=5)
    super_ = build_dataset("cifar100super", "./data", train=True, subset_pct=5)
    fine.dataset.transform = tf
    super_.dataset.ds.transform = tf
    assert len(fine) == len(super_) == 2500
    for i in range(100):
        assert torch.equal(fine[i][0], super_[i][0])


def test_label_map_is_the_official_20_superclasses():
    idx = load_subset_indices("cifar100super", 5)
    coarse, fine = cifar100_coarse_labels("./data", train=True)
    groups = collections.defaultdict(set)
    for f, c in zip(fine, coarse):
        groups[c].add(f)
    assert len(groups) == 20
    assert all(len(v) == 5 for v in groups.values())  # 5 fine per coarse
    # a fine-stratified subset is therefore exactly coarse-stratified too
    counts = collections.Counter(np.asarray(coarse)[idx])
    assert set(counts.values()) == {125}  # 25 per fine x 5 fine = 125 per coarse


def test_metadata():
    assert NUM_CLASSES["cifar100super"] == 20
    assert IMAGE_SIZE["cifar100super"] == 32
    assert len(build_dataset("cifar100super", "./data", train=False)) == 10000
    y = build_dataset("cifar100super", "./data", train=False)[0][1]
    assert 0 <= y < 20
