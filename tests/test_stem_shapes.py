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
