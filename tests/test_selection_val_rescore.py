"""Block E (2026-08-23): the selection re-scoring's axis table and helpers.

No GPU, no checkpoints: these check that every arm the re-scoring claims to
re-run actually names a CIFAR-100 config at the fraction its axis says, that
the per-fraction VAL restriction is disjoint from the training subset, and
that the paired/unpaired margin helper does what the .md says it does.
"""
import json
import os

import numpy as np
import pytest
import yaml

from analysis import selection_val_rescore as svr


def test_every_arm_is_a_cifar100_config_at_its_axis_fraction():
    for ax in svr.AXES:
        for label, cell in list(ax["arms"].items()) + [("baseline", ax["baseline"])]:
            p = svr.find_config(cell)
            assert p is not None, (ax["id"], label, cell)
            cfg = yaml.safe_load(open(p))
            assert cfg["dataset"] == "cifar100", (ax["id"], cell)
            assert cfg.get("subset_pct", 100) == ax["pct"], (ax["id"], cell)
        assert ax["selected"] is not None
    assert len({ax["id"] for ax in svr.AXES}) == len(svr.AXES)


@pytest.mark.skipif(not os.path.exists(svr.CARVE), reason="carve-out not present")
def test_val_indices_disjoint_from_subset_and_never_touch_test():
    carve = json.load(open(svr.CARVE))
    for p in (1, 10, 15, 25):
        idx, dropped = svr.val_indices_for(p, carve)
        sub = json.load(open(f"data/subsets/cifar100_{p}pct.json"))
        sub = set(sub["indices"] if isinstance(sub, dict) else sub)
        assert not (set(idx.tolist()) & sub)
        assert idx.max() < carve["n_train"]        # train-split indices only
        if p <= 10:
            assert dropped == 0 and len(idx) == carve["n_val"]
        else:
            assert dropped > 0 and len(idx) == carve["n_val"] - dropped


def test_margin_pairs_by_seed_when_it_can():
    a = {"seed0": 10.0, "seed1": 12.0, "seed2": 14.0}
    b = {"seed0": 9.0, "seed1": 11.0, "seed2": 13.0}
    d, sem, paired = svr.margin(a, b)
    assert paired and d == pytest.approx(1.0) and sem == pytest.approx(0.0)
    # disjoint seed labels -> independent SEM
    c = {"seed5": 9.0, "seed6": 11.0}
    d2, sem2, paired2 = svr.margin(a, c)
    assert not paired2 and d2 == pytest.approx(2.0)
    assert sem2 == pytest.approx(np.sqrt((2.0 / np.sqrt(3)) ** 2 + (np.sqrt(2) / np.sqrt(2)) ** 2))
