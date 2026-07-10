"""Response calibration: unit-variance moment channels, deterministic,
identity untouched, frozen afterwards, and fair to the random control."""

import torch

from momentstem import MomentStem


def _batch(n=64, size=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    # anisotropic, non-white fake images so calibration has real work to do
    x = torch.randn(n, 3, size, size, generator=g)
    x = torch.nn.functional.avg_pool2d(x, 3, stride=1, padding=1)  # correlate
    return x * torch.tensor([1.0, 2.0, 0.5]).view(1, 3, 1, 1)


def test_concat_calibration_unit_variance_and_identity_untouched():
    x = _batch()
    stem = MomentStem(mode="concat").calibrate(x)
    out = stem(x)
    std = out.std(dim=(0, 2, 3))
    assert torch.equal(out[:, :3], x), "identity channels must not be rescaled"
    assert torch.allclose(std[3:], torch.ones(24), atol=1e-4), (
        f"moment channels not unit-std after calibration: {std[3:]}"
    )


def test_sum_calibration_unit_variance():
    x = _batch()
    stem = MomentStem(mode="sum").calibrate(x)
    std = stem(x).std(dim=(0, 2, 3))
    assert torch.allclose(std, torch.ones(3), atol=1e-4)


def test_calibration_is_deterministic():
    a = MomentStem(mode="concat").calibrate(_batch())
    b = MomentStem(mode="concat").calibrate(_batch())
    assert torch.equal(a.gabor_weight, b.gabor_weight)
    assert torch.equal(a.zernike_weight, b.zernike_weight)


def test_calibrated_filters_stay_frozen():
    stem = MomentStem(mode="concat").calibrate(_batch())
    assert sum(p.numel() for p in stem.parameters()) == 0
    assert not stem.gabor_weight.requires_grad


def test_calibration_applies_to_random_control():
    x = _batch()
    stem = MomentStem(mode="concat", init="random", seed=3).calibrate(x)
    std = stem(x).std(dim=(0, 2, 3))
    assert torch.allclose(std[3:], torch.ones(24), atol=1e-4)


def test_calibration_survives_state_dict_roundtrip():
    x = _batch()
    src = MomentStem(mode="concat").calibrate(x)
    dst = MomentStem(mode="concat")
    dst.load_state_dict(src.state_dict())
    assert torch.equal(src.gabor_weight, dst.gabor_weight)
    assert torch.equal(src.zernike_weight, dst.zernike_weight)
