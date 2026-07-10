"""Moment kernels must be bit-identical after optimization; gabor-learn must
train. This is the core 'zero trainable parameters in the stem' contract."""

import torch

from momentstem import build_model
from momentstem.stem import MomentStem, gabor_bank, zernike_bank


def _train_steps(model, steps=3, image_size=32):
    opt = torch.optim.SGD(
        [p for p in model.parameters() if p.requires_grad], lr=0.5, momentum=0.9
    )
    criterion = torch.nn.CrossEntropyLoss()
    torch.manual_seed(0)
    for _ in range(steps):
        x = torch.randn(8, 3, image_size, image_size)
        y = torch.randint(0, 10, (8,))
        opt.zero_grad()
        criterion(model(x), y).backward()
        opt.step()


def _stem_filters(stem):
    return {
        n: t.detach().clone()
        for n, t in (("gabor", stem.gabor_weight), ("zernike", stem.zernike_weight))
        if t is not None
    }


def test_moment_stem_frozen_after_training():
    for stem_name in ("moments-sum", "moments-cat", "random-fixed"):
        model = build_model("resnet18", stem_name, num_classes=10)
        before = _stem_filters(model.stem)
        _train_steps(model)
        after = _stem_filters(model.stem)
        for key in before:
            assert torch.equal(before[key], after[key]), (
                f"{stem_name}/{key} filters changed during training"
            )


def test_moment_stem_has_no_trainable_params():
    stem = MomentStem(mode="concat")
    assert sum(p.numel() for p in stem.parameters()) == 0
    assert not stem.gabor_weight.requires_grad
    assert not stem.zernike_weight.requires_grad
    assert stem.filter_numel() == stem.gabor_weight.numel() + stem.zernike_weight.numel()


def test_gabor_learn_kernels_do_change():
    model = build_model("resnet18", "gabor-learn", num_classes=10)
    assert model.stem.gabor_weight.requires_grad
    assert model.stem.zernike_weight.requires_grad
    before = _stem_filters(model.stem)
    _train_steps(model)
    after = _stem_filters(model.stem)
    assert not torch.equal(before["gabor"], after["gabor"]), "gabor-learn gabor frozen"
    assert not torch.equal(before["zernike"], after["zernike"]), "gabor-learn zernike frozen"


def test_gabor_learn_initialised_to_moment_bank():
    fixed = MomentStem(mode="concat", trainable=False)
    learn = MomentStem(mode="concat", trainable=True)
    assert torch.equal(fixed.gabor_weight, learn.gabor_weight.detach())
    assert torch.equal(fixed.zernike_weight, learn.zernike_weight.detach())


def test_banks_are_deterministic_constants():
    b1, b2 = gabor_bank(), gabor_bank()
    assert torch.equal(b1, b2)
    z1, z2 = zernike_bank(), zernike_bank()
    assert torch.equal(z1, z2)
    # Two stems built independently share the identical bank.
    s1, s2 = MomentStem(mode="concat"), MomentStem(mode="concat")
    assert torch.equal(s1.gabor_weight, s2.gabor_weight)
    assert torch.equal(s1.zernike_weight, s2.zernike_weight)


def test_random_fixed_differs_from_moments_but_matches_norms():
    moments = MomentStem(mode="concat", init="moments")
    rand = MomentStem(mode="concat", init="random", seed=7)
    assert not torch.allclose(moments.gabor_weight, rand.gabor_weight)
    for m, r in (
        (moments.gabor_weight, rand.gabor_weight),
        (moments.zernike_weight, rand.zernike_weight),
    ):
        m_norms = m.flatten(2).norm(dim=2)
        r_norms = r.flatten(2).norm(dim=2)
        assert torch.allclose(m_norms, r_norms, rtol=1e-4), "per-kernel norms must match"
    # Different seeds give different draws (one draw per training seed).
    rand2 = MomentStem(mode="concat", init="random", seed=8)
    assert not torch.allclose(rand.gabor_weight, rand2.gabor_weight)
