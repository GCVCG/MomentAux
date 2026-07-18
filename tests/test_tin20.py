"""tin20: 20 of tin's 200 classes, labels remapped 0..19. The within-tin
granularity control (FINDINGS Q6.9d) -- only valid if the class choice is
deterministic, train and val agree on the mapping, and the committed subset
is balanced."""

import os

import numpy as np
import pytest

from data import (NUM_CLASSES, build_dataset, load_subset_indices, subset_path,
                  tin20_wnids, tin_root)

TIN = tin_root("./data")
needs_tin = pytest.mark.skipif(not os.path.isdir(TIN), reason="tiny-imagenet not downloaded")


@needs_tin
def test_class_choice_is_deterministic_and_positional():
    w = tin20_wnids(TIN)
    assert len(w) == len(set(w)) == 20 == NUM_CLASSES["tin20"]
    assert w == tin20_wnids(TIN)  # pure function of the directory listing
    all_w = sorted(d for d in os.listdir(os.path.join(TIN, "train"))
                   if os.path.isdir(os.path.join(TIN, "train", d)))
    assert w == all_w[::10]  # every 10th, no cherry-picking


@needs_tin
def test_train_and_val_are_filtered_and_relabelled_consistently():
    tr = build_dataset("tin20", "./data", train=True)
    va = build_dataset("tin20", "./data", train=False)
    t, v = np.asarray(tr.targets), np.asarray(va.targets)
    assert len(tr) == 10000 and len(va) == 1000
    assert set(np.unique(t)) == set(np.unique(v)) == set(range(20))
    assert np.bincount(t).tolist() == [500] * 20
    assert np.bincount(v).tolist() == [50] * 20


@needs_tin
def test_committed_subset_is_balanced_at_50_per_class():
    if not os.path.exists(subset_path("tin20", 10)):
        pytest.skip("tin20 subset not generated")
    idx = load_subset_indices("tin20", 10)
    tr = build_dataset("tin20", "./data", train=True)
    t = np.asarray(tr.targets)[idx]
    assert len(idx) == len(set(idx)) == 1000
    assert np.bincount(t).tolist() == [50] * 20  # 1400 steps, tin@1%'s twin


@needs_tin
def test_calibration_batch_covers_tin20_and_matches_tin():
    """calibration_batch has its own dataset dispatch, so a new dataset can
    train-crash despite build_dataset working (this exact bug shipped once).
    tin20 must calibrate, and on the SAME images as tin -- the aux target
    pipeline is deliberately byte-identical to tin@1%'s."""
    import torch

    from data import calibration_batch

    a = calibration_batch("tin", "./data", n=64)
    b = calibration_batch("tin20", "./data", n=64)
    assert torch.equal(a, b)
