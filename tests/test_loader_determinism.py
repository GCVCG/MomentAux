"""What `num_workers` does and does not change.

Subsets are committed so that every stem sees identical indices (CLAUDE.md
non-negotiable). That guarantee reaches training only if the DataLoader's
sampler order is worker-count-independent -- it is, because the sampler draws
from `generator=gen` in the MAIN process. This file pins that.

It also documents the converse, which is a live footgun: AUGMENTATION is drawn
per-worker (PyTorch seeds each worker's torch RNG to `base_seed + worker_id`,
and RandomCrop/RandomHorizontalFlip draw from it), so changing num_workers
re-draws the augmentation stream. That is UNBIASED -- same crop+flip
distribution, it acts like a different augmentation seed -- so a Delta measured
across a worker-count boundary stays valid, but seeds are NOT paired across it
and the cell is not byte-reproducible. Hence: never change num_workers on a cell
that already has completed seeds.

The trap this file exists to prevent: batches are dispatched round-robin, so
batch i is built by worker (i % num_workers) and the FIRST num_workers batches
match trivially between any two worker counts. A 2-batch comparison therefore
reports a FALSE match. Divergence can only begin at batch index == num_workers.
"""

import numpy as np
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from data import build_transforms

N_IMAGES = 2048
BATCH = 128


class _FakeImages(torch.utils.data.Dataset):
    """Deterministic RGB images; content depends only on the index, so any
    difference between two loaders comes from augmentation, never the data."""

    def __init__(self, transform):
        self.transform = transform

    def __len__(self):
        return N_IMAGES

    def __getitem__(self, i):
        rs = np.random.RandomState(i)
        arr = rs.randint(0, 256, (32, 32, 3), dtype=np.uint8)
        return self.transform(Image.fromarray(arr)), i % 10


def _seed_worker(worker_id):  # mirrors train.py
    s = torch.initial_seed() % 2 ** 32
    np.random.seed(s)
    import random

    random.seed(s)


def _batches(num_workers, seed=0, n=12):
    torch.manual_seed(seed)
    ds = _FakeImages(build_transforms("cifar10", train=True))
    gen = torch.Generator().manual_seed(seed)
    dl = DataLoader(
        ds, batch_size=BATCH, shuffle=True, num_workers=num_workers, drop_last=True,
        generator=gen, worker_init_fn=_seed_worker,
        persistent_workers=num_workers > 0,
    )
    out = []
    for i, (x, y) in enumerate(dl):
        out.append((x.clone(), y.clone()))
        if i + 1 >= n:
            break
    return out


def test_sample_order_is_worker_count_independent():
    """THE LOAD-BEARING PROPERTY: committed subsets reach the model in the same
    order regardless of num_workers, so cells run at different worker counts are
    still comparable. If this ever fails, every cross-cell Delta is suspect."""
    a, b = _batches(2), _batches(8)
    for i, ((_, ya), (_, yb)) in enumerate(zip(a, b)):
        assert torch.equal(ya, yb), f"sample order diverged at batch {i}"


def test_sample_order_is_seed_reproducible():
    a, b = _batches(2, seed=0), _batches(2, seed=0)
    for (_, ya), (_, yb) in zip(a, b):
        assert torch.equal(ya, yb)
    c = _batches(2, seed=1)
    assert not all(torch.equal(ya, yc) for (_, ya), (_, yc) in zip(a, c)), \
        "seed does not affect sample order -- shuffling is not seeded"


def test_augmentation_is_NOT_worker_count_independent():
    """Documents the footgun (does not endorse it). Divergence starts at batch
    index == num_workers, never before -- round-robin gives batch i to worker
    (i % nw), so batches 0..nw-1 match trivially between the two loaders."""
    nw_small = 2
    a, b = _batches(nw_small), _batches(8)

    for i in range(nw_small):  # same worker id -> same RNG -> identical crop/flip
        assert torch.equal(a[i][0], b[i][0]), (
            f"batch {i} < num_workers should match trivially; if this fails, "
            f"batches are no longer dispatched round-robin"
        )

    diverged = [i for i in range(nw_small, len(a)) if not torch.equal(a[i][0], b[i][0])]
    assert diverged, (
        "augmentation no longer depends on num_workers -- if PyTorch changed "
        "this, the num_workers reproducibility caveat in CLAUDE.md can be dropped"
    )
    assert diverged[0] == nw_small, (
        f"divergence began at batch {diverged[0]}, expected {nw_small}"
    )


@pytest.mark.parametrize("num_workers", [2, 8])
def test_augmentation_is_reproducible_at_a_fixed_worker_count(num_workers):
    """The contract we actually rely on: hold num_workers fixed and a cell
    reproduces exactly, augmentation included."""
    a, b = _batches(num_workers, seed=0), _batches(num_workers, seed=0)
    for i, ((xa, _), (xb, _)) in enumerate(zip(a, b)):
        assert torch.equal(xa, xb), f"batch {i} not reproducible at nw={num_workers}"
