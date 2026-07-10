"""Fixed label subsets: identical across runs, seeds, and machines, and the
committed JSON files are internally consistent."""

import json
import os

import numpy as np
import pytest

from data import SUBSET_DIR, load_subset_indices, make_subset_indices, subset_path


def _fake_cifar100_labels():
    # 100 classes x 500 samples, shuffled the same way CIFAR isn't -- the
    # generator must not care about label order.
    rs = np.random.RandomState(42)
    return rs.permutation(np.repeat(np.arange(100), 500))


def test_generation_is_deterministic():
    labels = _fake_cifar100_labels()
    a = make_subset_indices(labels, 10)
    b = make_subset_indices(labels, 10)
    assert a == b
    assert make_subset_indices(labels, 10, seed=1) != a  # seed actually matters


def test_generation_is_stratified_and_unique():
    labels = _fake_cifar100_labels()
    for pct, per_class in ((1, 5), (5, 25), (10, 50), (25, 125)):
        idx = make_subset_indices(labels, pct)
        assert len(idx) == len(set(idx)) == per_class * 100
        counts = np.bincount(labels[idx], minlength=100)
        assert (counts == per_class).all(), f"{pct}% subset is not stratified"


def test_subsets_nest_is_not_assumed():
    """Document (not require) the protocol: subsets are drawn per-pct from the
    same seeded permutation, so smaller subsets ARE prefixes of larger ones.
    This makes the H1 data-efficiency curve monotone in data, not in draws."""
    labels = _fake_cifar100_labels()
    small = set(make_subset_indices(labels, 5))
    large = set(make_subset_indices(labels, 25))
    assert small <= large


@pytest.mark.parametrize("pct", [1, 5, 10, 25])
def test_committed_files_are_valid(pct):
    path = subset_path("cifar100", pct)
    if not os.path.exists(path):
        pytest.skip(f"{path} not generated yet (run scripts/make_subsets.py)")
    idx = load_subset_indices("cifar100", pct)
    assert len(idx) == len(set(idx)) == int(50000 * pct / 100)
    assert min(idx) >= 0 and max(idx) < 50000
    with open(path) as f:
        payload = json.load(f)
    assert payload["seed"] is not None  # provenance recorded


def test_committed_files_match_regeneration_when_cifar_present():
    cifar_dir = os.path.join(os.path.dirname(SUBSET_DIR), "cifar-100-python")
    if not os.path.exists(cifar_dir):
        pytest.skip("CIFAR-100 not downloaded; regeneration check skipped")
    import pickle

    with open(os.path.join(cifar_dir, "train"), "rb") as f:
        labels = pickle.load(f, encoding="latin1")["fine_labels"]
    for pct in (1, 5, 10, 25):
        if not os.path.exists(subset_path("cifar100", pct)):
            pytest.skip("subset files not generated yet")
        assert load_subset_indices("cifar100", pct) == make_subset_indices(labels, pct)
