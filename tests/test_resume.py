"""`resume_every` (train.py): the full-state save/restore round trip.

Block F of the limitations campaign (2026-08-23) needs ViT-L cells that
outlive the cluster's 24h MaxWall, so train.py can checkpoint and resume.
These tests pin the contract stated in train.py's docstring:
  * with num_workers=0 a resumed run is BYTE-IDENTICAL to an uninterrupted
    one (same losses, same parameters);
  * with persistent workers the DATA ORDER is restored exactly (the label
    sequence after resume equals the uninterrupted run's);
  * metrics.csv is truncated to the saved row count and best.pt is rolled
    back to the linked copy;
  * resume_every is refused on a non-diag config name (train.py guard, checked
    by reading the guard's source rather than launching a run).
All on CPU, synthetic data, a few seconds."""

import csv
import os

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import train as train_mod


def _loader(n=64, num_workers=0, seed=0, batch=8):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, 3, 8, 8, generator=torch.Generator().manual_seed(123))
    y = torch.arange(n) % 4
    return DataLoader(TensorDataset(x, y), batch_size=batch, shuffle=True,
                      num_workers=num_workers, drop_last=True, generator=g,
                      worker_init_fn=train_mod.seed_worker,
                      persistent_workers=num_workers > 0)


def _setup(seed=0, num_workers=0):
    train_mod.set_seed(seed)
    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 16), nn.ReLU(), nn.Linear(16, 4))
    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=6)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    loader = _loader(num_workers=num_workers, seed=seed)
    return model, opt, sched, scaler, loader


def _epoch(model, opt, sched, scaler, loader):
    losses, labels = [], []
    model.train()
    for x, y in loader:
        opt.zero_grad()
        loss = nn.functional.cross_entropy(model(x), y)
        # a main-process numpy draw, as Mixup would make
        loss = loss * (1.0 + 0.01 * float(np.random.rand()))
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        losses.append(loss.item())
        labels.append(y.tolist())
    sched.step()
    return losses, labels


def _write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "x"])
        for r in rows:
            w.writerow([r, "v"])


def test_resume_round_trip_is_byte_identical_with_no_workers(tmp_path):
    # uninterrupted reference: 6 epochs
    model, opt, sched, scaler, loader = _setup()
    ref = [_epoch(model, opt, sched, scaler, loader)[0] for _ in range(6)]
    ref_params = [p.detach().clone() for p in model.parameters()]

    # interrupted: 3 epochs, save, then a FRESH process-equivalent restores
    model, opt, sched, scaler, loader = _setup()
    got = [_epoch(model, opt, sched, scaler, loader)[0] for _ in range(3)]
    out = tmp_path
    csv_path = os.path.join(out, "metrics.csv")
    _write_csv(csv_path, [0, 1, 2, 3, 4])            # 2 rows past the save point
    torch.save({"w": 1}, os.path.join(out, "best.pt"))
    path = os.path.join(out, "resume.pt")
    train_mod._save_resume_state(path, model, opt, sched, scaler, loader, 3, 0.5,
                                 12.0, [], str(out))
    assert os.path.exists(path) and os.path.exists(os.path.join(out, "best.pt.resume"))
    # a post-save "improvement", written the way train.py writes best.pt
    # (tmp + os.replace = a NEW inode, which is what lets the hard link
    # best.pt.resume keep the old contents)
    torch.save({"w": 2}, os.path.join(out, "best.pt.tmp"))
    os.replace(os.path.join(out, "best.pt.tmp"), os.path.join(out, "best.pt"))
    # perturb every RNG, then restore into fresh objects (same seed, as train.py does)
    _epoch(model, opt, sched, scaler, loader); np.random.rand(5); torch.rand(3)
    model, opt, sched, scaler, loader = _setup()
    state = torch.load(path, map_location="cpu", weights_only=False)
    start, best, wall = train_mod._restore_resume_state(
        state, model, opt, sched, scaler, loader, 0, str(out), csv_path,
        torch.device("cpu"))
    assert (start, best, wall) == (3, 0.5, 12.0)
    got += [_epoch(model, opt, sched, scaler, loader)[0] for _ in range(3)]
    assert got == ref, "resumed losses differ from the uninterrupted run"
    for a, b in zip(model.parameters(), ref_params):
        assert torch.equal(a.detach(), b)
    # metrics.csv truncated to header + 3 rows; best.pt rolled back
    with open(csv_path) as f:
        rows = list(csv.reader(f))
    assert [r[0] for r in rows] == ["epoch", "0", "1", "2"]
    assert torch.load(os.path.join(out, "best.pt"))["w"] == 1


def test_resume_restores_data_order_with_persistent_workers(tmp_path):
    """With persistent workers the sampler generator is restored and the
    one-off worker-base-seed draw is redirected, so the label ORDER after
    resume equals the uninterrupted run's. (Worker-side augmentation RNG is
    NOT restorable -- the docstring says so -- but a TensorDataset has none,
    so order is the whole content here.)"""
    model, opt, sched, scaler, loader = _setup(num_workers=2)
    ref = [_epoch(model, opt, sched, scaler, loader)[1] for _ in range(5)]

    model, opt, sched, scaler, loader = _setup(num_workers=2)
    got = [_epoch(model, opt, sched, scaler, loader)[1] for _ in range(2)]
    csv_path = os.path.join(tmp_path, "metrics.csv")
    _write_csv(csv_path, [0, 1])
    path = os.path.join(tmp_path, "resume.pt")
    train_mod._save_resume_state(path, model, opt, sched, scaler, loader, 2, 0.1,
                                 1.0, [], str(tmp_path))
    model, opt, sched, scaler, loader = _setup(num_workers=2)   # fresh loader: new workers
    state = torch.load(path, map_location="cpu", weights_only=False)
    train_mod._restore_resume_state(state, model, opt, sched, scaler, loader, 0,
                                    str(tmp_path), csv_path, torch.device("cpu"))
    got += [_epoch(model, opt, sched, scaler, loader)[1] for _ in range(3)]
    assert got == ref, "data order after resume differs from the uninterrupted run"
    # and the NEXT save records the sampler's generator, not the swapped one
    train_mod._save_resume_state(path, model, opt, sched, scaler, loader, 5, 0.1,
                                 1.0, [2], str(tmp_path))
    st2 = torch.load(path, map_location="cpu", weights_only=False)
    assert torch.equal(st2["rng"]["sampler_gen"], loader.sampler.generator.get_state())
    assert st2["resume_history"] == [2]


def test_resume_refuses_inconsistent_csv(tmp_path):
    import pytest
    model, opt, sched, scaler, loader = _setup()
    csv_path = os.path.join(tmp_path, "metrics.csv")
    _write_csv(csv_path, [0])                       # only 1 row but state says 3
    path = os.path.join(tmp_path, "resume.pt")
    train_mod._save_resume_state(path, model, opt, sched, scaler, loader, 3, 0.5,
                                 1.0, [], str(tmp_path))
    state = torch.load(path, map_location="cpu", weights_only=False)
    with pytest.raises(RuntimeError):
        train_mod._restore_resume_state(state, model, opt, sched, scaler, loader, 0,
                                        str(tmp_path), csv_path, torch.device("cpu"))


def test_resume_is_diag_only_in_train_py():
    src = open(os.path.join(os.path.dirname(train_mod.__file__), "train.py")).read()
    assert 'raise ValueError("resume_every requires a diag* config name")' in src
    assert "resume_every = cfg.get(\"resume_every\")" in src
