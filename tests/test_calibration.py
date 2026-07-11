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


def test_zca_output_is_whitened_and_information_preserving():
    x = _batch(n=128)
    plain = MomentStem(mode="concat", use_zernike=False)
    plain.calibrate(x)
    ref = plain(x)
    stem = MomentStem(mode="concat", use_zernike=False).calibrate_zca(x)
    out = stem(x)
    # whitened: identity covariance on the calibration batch
    f = out.permute(1, 0, 2, 3).flatten(1)
    f = f - f.mean(dim=1, keepdim=True)
    cov = (f @ f.T) / f.shape[1]
    assert (cov - torch.eye(12)).abs().max() < 0.05
    # information preserved: fused output is an affine map of the plain output
    # (recoverable by least squares to numerical precision)
    A = torch.linalg.lstsq(
        torch.cat([ref.permute(0, 2, 3, 1).flatten(0, 2),
                   torch.ones(ref.numel() // 12, 1)], dim=1),
        out.permute(0, 2, 3, 1).flatten(0, 2),
    ).solution
    recon = torch.cat([ref.permute(0, 2, 3, 1).flatten(0, 2),
                       torch.ones(ref.numel() // 12, 1)], dim=1) @ A
    assert (recon - out.permute(0, 2, 3, 1).flatten(0, 2)).abs().max() < 1e-3


def test_zca_deterministic_frozen_and_loadable():
    x = _batch()
    a = MomentStem(mode="concat", use_zernike=False).calibrate_zca(x)
    b = MomentStem(mode="concat", use_zernike=False).calibrate_zca(x)
    assert torch.equal(a.fused_weight, b.fused_weight)
    assert torch.equal(a.fused_bias, b.fused_bias)
    assert sum(p.numel() for p in a.parameters()) == 0
    # fresh stem + placeholder buffers can load a ZCA checkpoint
    c = MomentStem(mode="concat", use_zernike=False)
    c._ensure_fused_buffers()
    c.load_state_dict(a.state_dict())
    assert torch.equal(c(x), a(x))


def test_zca_applies_to_random_control():
    x = _batch()
    stem = MomentStem(mode="concat", use_zernike=False, init="random", seed=5)
    stem.calibrate_zca(x)
    f = stem(x).permute(1, 0, 2, 3).flatten(1)
    f = f - f.mean(dim=1, keepdim=True)
    cov = (f @ f.T) / f.shape[1]
    assert (cov - torch.eye(12)).abs().max() < 0.05


def test_calibration_survives_state_dict_roundtrip():
    x = _batch()
    src = MomentStem(mode="concat").calibrate(x)
    dst = MomentStem(mode="concat")
    dst.load_state_dict(src.state_dict())
    assert torch.equal(src.gabor_weight, dst.gabor_weight)
    assert torch.equal(src.zernike_weight, dst.zernike_weight)
