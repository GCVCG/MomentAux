"""Shape AND content contracts for every stem, at 32x32 (CIFAR) and 96x96
(STL-10). Content checks make channel-ordering/reshape bugs impossible to
ship silently (see 'non-negotiable hygiene' in README)."""

import pytest
import torch

from momentstem import MomentStem, build_model, build_stem
from momentstem.stem import N_ZERNIKE


@pytest.mark.parametrize("size", [32, 96])
def test_sum_mode_preserves_3channel_contract(size):
    stem = MomentStem(mode="sum")
    x = torch.randn(2, 3, size, size)
    out = stem(x)
    assert out.shape == (2, 3, size, size)


@pytest.mark.parametrize("size", [32, 96])
def test_concat_mode_channel_count(size):
    stem = MomentStem(mode="concat")
    x = torch.randn(2, 3, size, size)
    out = stem(x)
    # identity 3 + gabor 9 + zernike 15 = 27
    assert stem.out_channels == 3 + 9 + N_ZERNIKE == 27
    assert out.shape == (2, 27, size, size)


def test_concat_channel_count_matches_config_flags():
    cases = [
        dict(use_gabor=True, use_zernike=False, include_identity=True, expect=12),
        dict(use_gabor=False, use_zernike=True, include_identity=True, expect=18),
        dict(use_gabor=True, use_zernike=True, include_identity=False, expect=24),
        dict(use_gabor=True, use_zernike=False, include_identity=False, expect=9),
    ]
    for case in cases:
        expect = case.pop("expect")
        stem = MomentStem(mode="concat", **case)
        assert stem.out_channels == expect
        assert stem(torch.randn(1, 3, 32, 32)).shape[1] == expect


def test_zernike_indices_pruning():
    from momentstem.stem import zernike_bank

    keep = [1, 2, 3, 7, 11]
    stem = MomentStem(mode="concat", zernike_indices=keep)
    assert stem.out_channels == 3 + 9 + len(keep) == 17
    assert stem(torch.randn(1, 3, 32, 32)).shape[1] == 17
    # pruned weights are exactly the selected kernels (applied to channel mean)
    full = zernike_bank(11)[keep]
    assert torch.allclose(stem.zernike_weight, full.unsqueeze(1).repeat(1, 3, 1, 1) / 3)
    with pytest.raises(ValueError):
        MomentStem(zernike_indices=[0, 0, 1])  # duplicates rejected
    with pytest.raises(ValueError):
        MomentStem(zernike_indices=[15])  # out of range


def test_concat_identity_channels_are_exact_passthrough():
    stem = MomentStem(mode="concat", include_identity=True)
    x = torch.randn(2, 3, 32, 32)
    out = stem(x)
    assert torch.equal(out[:, :3], x), "identity channels must be the input, bitwise"


def test_sum_is_channel_fold_of_concat_gabor():
    """The sum variant must equal the concat variant with responses summed
    back per output index: sum[:, o] == sum_i cat[:, 3i+o]. Pins down both
    the dense/grouped weight layouts and the channel ordering."""
    sum_stem = MomentStem(mode="sum", use_zernike=False)
    cat_stem = MomentStem(mode="concat", use_zernike=False, include_identity=False)
    x = torch.randn(2, 3, 32, 32)
    folded = torch.stack(
        [sum(cat_stem(x)[:, 3 * i + o] for i in range(3)) for o in range(3)], dim=1
    )
    assert torch.allclose(sum_stem(x), folded, atol=1e-5)


@pytest.mark.parametrize("stem_name", ["none", "moments-sum", "moments-cat",
                                       "learned", "random-fixed", "gabor-learn"])
@pytest.mark.parametrize("size", [32, 96])
def test_every_stem_feeds_resnet18(stem_name, size):
    model = build_model("resnet18", stem_name, num_classes=10)
    out = model(torch.randn(2, 3, size, size))
    assert out.shape == (2, 10)


def test_all_concat_stems_share_out_channels():
    """learned/random-fixed/gabor-learn must present the identical channel
    interface to the backbone as moments-cat."""
    ref = build_stem("moments-cat").out_channels
    for name in ("learned", "random-fixed", "gabor-learn"):
        assert build_stem(name).out_channels == ref


def test_stem_rejects_wrong_layout():
    stem = MomentStem(mode="concat")
    with pytest.raises(ValueError):
        stem(torch.randn(2, 32, 32, 3))  # HWC must be rejected, not silently eaten


# ---------------------------------------------------------------------------
# EnergyStem channel_reduce (2026-08-23, the SAR channel-reduction block)
# ---------------------------------------------------------------------------

def _cr_stem(mode, in_channels, feature_type="magnitude"):
    from momentstem import EnergyStem
    return EnergyStem(feature_type=feature_type, in_channels=in_channels,
                      kernel_size=11, channel_reduce=mode)


def test_channel_reduce_mean_is_the_old_luma_path_exactly():
    """'mean' must be byte-identical to the historical behaviour: on 3 channels
    the scalar field IS BT.601 luma and the forward equals _energy(_luma(x))."""
    import torch
    torch.manual_seed(0)
    x = torch.randn(4, 3, 32, 32)
    stem = _cr_stem("mean", 3)
    stem.calibrate(x)
    ref = stem._energy(stem._luma(x)) * stem.calib_scale.view(1, -1, 1, 1)
    out = stem(x)[:, 3:]
    assert torch.equal(out, ref)
    # the default constructor is 'mean', and its state_dict carries no pca keys
    from momentstem import EnergyStem
    d = EnergyStem(feature_type="magnitude", in_channels=3)
    assert d.channel_reduce == "mean"
    assert not any(k.startswith("pc1") for k in d.state_dict())


def test_channel_reduce_perchannel_equals_mean_on_one_channel():
    import torch
    torch.manual_seed(1)
    x = torch.randn(3, 1, 32, 32)
    a = _cr_stem("mean", 1)
    b = _cr_stem("perchannel", 1)
    a.calibrate(x); b.calibrate(x)
    assert torch.equal(a(x), b(x))


def test_channel_reduce_perchannel_averages_energies():
    """perchannel = bank on every channel separately, energies averaged."""
    import torch
    torch.manual_seed(2)
    x = torch.randn(2, 4, 32, 32)
    stem = _cr_stem("perchannel", 4)
    per = torch.stack([stem._energy(x[:, c:c + 1]) for c in range(4)]).mean(0)
    assert torch.allclose(stem._raw_energy(x), per, atol=1e-6)


def test_channel_reduce_pca1_requires_calibration_then_runs():
    import pytest, torch
    torch.manual_seed(3)
    x = torch.randn(16, 8, 32, 32)
    stem = _cr_stem("pca1", 8)
    with pytest.raises(RuntimeError):
        stem(x)
    stem.calibrate(x)
    assert bool(stem.pc1_fitted)
    w = stem.pc1_w.view(-1)
    assert abs(w.norm().item() - 1.0) < 1e-5
    assert w[w.abs().argmax()] > 0  # deterministic sign convention
    out = stem(x)
    assert out.shape == (16, 8 + stem.n_energy, 32, 32)
    # a strongly correlated channel set: pc1 must align with the shared component
    base = torch.randn(16, 1, 32, 32)
    xs = base.repeat(1, 8, 1, 1) + 0.01 * torch.randn(16, 8, 32, 32)
    s2 = _cr_stem("pca1", 8).calibrate(xs)
    w2 = s2.pc1_w.view(-1)
    assert (w2 - w2.mean()).abs().max() < 0.05
    # the buffers travel in the state_dict
    assert "pc1_w" in s2.state_dict() and "pc1_fitted" in s2.state_dict()


def test_channel_reduce_shapes_on_eight_channels_all_modes():
    import torch
    from momentstem.energy import CHANNEL_REDUCE, ENERGY_TYPES
    torch.manual_seed(4)
    x = torch.randn(5, 8, 32, 32)
    for mode in CHANNEL_REDUCE:
        for ft in ENERGY_TYPES:
            stem = _cr_stem(mode, 8, feature_type=ft)
            stem.calibrate(x)
            out = stem(x)
            assert out.shape == (5, 8 + stem.n_energy, 32, 32), (mode, ft)
            assert torch.isfinite(out).all(), (mode, ft)
            # calibration leaves each energy channel at ~unit std
            e = out[:, 8:]
            std = e.permute(1, 0, 2, 3).reshape(e.shape[1], -1).std(1)
            assert torch.allclose(std, torch.ones_like(std), atol=5e-2), (mode, ft)


def test_channel_reduce_logmean_is_documented_transform():
    import torch, torch.nn.functional as F
    x = torch.randn(2, 8, 32, 32)
    stem = _cr_stem("logmean", 8)
    ref = torch.log1p(F.softplus(x)).mean(1, keepdim=True)
    assert torch.equal(stem._scalar(x), ref)


def test_channel_reduce_rejects_unknown():
    import pytest
    with pytest.raises(ValueError):
        _cr_stem("median", 3)


def test_moment_aux_channel_reduce_is_plumbed():
    """moment_aux.channel_reduce reaches the energy target; absent => 'mean'."""
    from momentstem.backbones import build_model
    base = dict(stem="energy-magnitude", tap="layer3", weight=1.0, kernel_size=11)
    m0 = build_model("resnet18", "none", num_classes=17, in_channels=8, image_size=32,
                     moment_aux=dict(base))
    assert m0.target.stem.channel_reduce == "mean"
    m1 = build_model("resnet18", "none", num_classes=17, in_channels=8, image_size=32,
                     moment_aux=dict(base, channel_reduce="perchannel"))
    assert m1.target.stem.channel_reduce == "perchannel"
