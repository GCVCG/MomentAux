"""Numeric regression pins for the committed filter banks. If any of these
move, the study's filters changed and every existing run is invalidated --
that must be a loud, deliberate decision, not drift."""

import math

import pytest
import torch

from momentstem.stem import GABOR_SEED, gabor_bank, gabor_kernel, zernike_bank

# Committed fingerprints of the exact banks used by every run (kernel_size=11,
# GABOR_SEED=1234). Zernike constants re-pinned 2026-07-10 after correcting
# the j=7/8 coma formulas (the ported table made them exact duplicates of
# j=6/9 -- see PORTING.md); the sum is unchanged because both old and new
# kernels are odd functions summing to zero.
GABOR_SUM = -0.0577797294
GABOR_ABSMEAN = 0.0037828626
ZERNIKE_SUM = 16.7257347107
ZERNIKE_ABSMEAN = 0.0570115298
GRID_GABOR_SUM = -0.0143144000
GRID_GABOR_ABSMEAN = 0.0039181341


def test_gabor_bank_fingerprint():
    g = gabor_bank()
    assert g.shape == (3, 3, 11, 11)
    assert g.sum().item() == pytest.approx(GABOR_SUM, abs=1e-6)
    assert g.abs().mean().item() == pytest.approx(GABOR_ABSMEAN, abs=1e-8)
    assert GABOR_SEED == 1234


def test_zernike_bank_fingerprint():
    z = zernike_bank()
    assert z.shape == (15, 11, 11)
    assert z.sum().item() == pytest.approx(ZERNIKE_SUM, abs=1e-5)
    assert z.abs().mean().item() == pytest.approx(ZERNIKE_ABSMEAN, abs=1e-8)


def test_zernike_kernels_unit_norm_and_disk_masked():
    z = zernike_bank()
    norms = z.flatten(1).norm(dim=1)
    assert torch.allclose(norms, torch.ones(15), atol=1e-5)
    coords = torch.linspace(-1, 1, 11)
    y, x = torch.meshgrid(coords, coords, indexing="ij")
    outside = (x ** 2 + y ** 2) > 1.0
    assert (z[:, outside] == 0).all(), "values outside the unit disk must be zero"


def test_zernike_bank_has_no_duplicate_kernels():
    """The ported table made j=7 a multiple of j=6 and j=8 of j=9; after L2
    normalisation those were bitwise-identical channels. Must never regress."""
    z = zernike_bank().flatten(1)
    zn = z / z.norm(dim=1, keepdim=True)
    cos = zn @ zn.T - torch.eye(15)
    assert cos.abs().max() < 0.99, f"near-duplicate kernels: max |cos| {cos.abs().max():.4f}"


def test_grid_gabor_bank_fingerprint_and_diversity():
    from momentstem.stem import gabor_bank_grid

    g = gabor_bank_grid()
    assert g.shape == (3, 3, 11, 11)
    assert g.sum().item() == pytest.approx(GRID_GABOR_SUM, abs=1e-6)
    assert g.abs().mean().item() == pytest.approx(GRID_GABOR_ABSMEAN, abs=1e-8)
    flat = g.reshape(9, -1)
    fn = flat / flat.norm(dim=1, keepdim=True)
    cos = fn @ fn.T - torch.eye(9)
    # the ported random bank hits 0.67 for the committed seed; grid must stay
    # decisively more diverse
    assert cos.abs().max() < 0.5


def test_pyramid_and_luma_bank_fingerprints():
    from momentstem.stem import gabor_bank_luma, gabor_bank_pyramid

    p = gabor_bank_pyramid()
    assert p.shape == (3, 3, 11, 11)
    assert p.sum().item() == pytest.approx(0.0236303322, abs=1e-6)
    assert p.abs().mean().item() == pytest.approx(0.0039229356, abs=1e-8)
    l = gabor_bank_luma()
    assert l.shape == (16, 11, 11)
    assert l.sum().item() == pytest.approx(0.0252580792, abs=1e-6)
    assert l.abs().mean().item() == pytest.approx(0.0042592874, abs=1e-8)


def test_luma_stem_shapes_and_calibration():
    import torch

    from momentstem import MomentStem

    stem = MomentStem(mode="concat", use_zernike=False, gabor_bank_type="luma")
    assert stem.out_channels == 3 + 16
    x = torch.randn(2, 3, 32, 32)
    out = stem(x)
    assert out.shape == (2, 19, 32, 32)
    assert torch.equal(out[:, :3], x)
    assert sum(p.numel() for p in stem.parameters()) == 0
    stem.calibrate(x)
    std = stem(x).std(dim=(0, 2, 3))
    assert torch.allclose(std[3:], torch.ones(16), atol=1e-3)


def test_energy_kernel_fingerprints():
    """Pins for the nonlinear EnergyStem kernels (magnitude/rotinv/structure).
    Additive banks -- no existing run uses them -- but they are constants of the
    study and must not drift."""
    from momentstem.energy import (
        _MAG_FREQS, _MAG_ORIENTS, _ROT_FREQS, _ROT_ORIENTS, _STRUCT_SIGMAS,
        gaussian_derivative_kernels, quadrature_bank,
    )

    me, mo = quadrature_bank(_MAG_FREQS, _MAG_ORIENTS, 11)
    assert me.shape == (8, 1, 11, 11) and mo.shape == (8, 1, 11, 11)
    assert me.sum().item() == pytest.approx(0.0334119499, abs=1e-6)
    assert mo.abs().mean().item() == pytest.approx(0.0042306492, abs=1e-8)

    re, ro = quadrature_bank(_ROT_FREQS, _ROT_ORIENTS, 11)
    assert re.shape == (24, 1, 11, 11)
    assert re.sum().item() == pytest.approx(0.2056152672, abs=1e-6)
    assert ro.abs().mean().item() == pytest.approx(0.0034957058, abs=1e-8)

    gx = torch.cat([gaussian_derivative_kernels(s, 11)[0] for s in _STRUCT_SIGMAS])
    win = torch.cat([gaussian_derivative_kernels(s, 11)[2] for s in _STRUCT_SIGMAS])
    assert gx.shape == (3, 1, 11, 11)
    assert gx.abs().mean().item() == pytest.approx(0.0041429861, abs=1e-8)
    assert gx.sum().abs().item() < 1e-6, "gradient kernels must be zero-sum"
    assert win.sum().item() == pytest.approx(3.0, abs=1e-6), "each window sums to one"


def test_energy_stem_contracts():
    """RGB passthrough, zero trainable params, and unit-std calibration for
    every energy feature type."""
    from momentstem import EnergyStem
    from momentstem.energy import ENERGY_TYPES

    x = torch.randn(3, 3, 32, 32)
    expected_ch = {"magnitude": 11, "rotinv": 11, "structure": 12}
    for ft in ENERGY_TYPES:
        stem = EnergyStem(feature_type=ft)
        out = stem(x)
        assert out.shape[1] == stem.out_channels == expected_ch[ft]
        assert torch.equal(out[:, :3], x), f"{ft}: identity passthrough broken"
        assert sum(p.numel() for p in stem.parameters()) == 0, f"{ft}: not fixed"
        stem.calibrate(x)
        std = stem(x).std(dim=(0, 2, 3))[3:]
        assert torch.allclose(std, torch.ones_like(std), atol=1e-3), f"{ft}: calib"


def test_gabor_kernel_formula_reference():
    """Independent re-evaluation of the ported Gabor formula at one point,
    against a hand-computed value (guards against grid or rotation drift)."""
    k = gabor_kernel(freq=math.pi / 2, theta=0.0, sigma=2.0, psi=0.0, kernel_size=11)
    # Center of the ported grid linspace(-5, 6, 11) is (x=0.5, y=0.5) at index
    # (5, 5): rotx=0.5, roty=0.5.
    sigma, delta = 2.0, 1e-3
    g = math.exp(-0.5 * (0.25 + 0.25) / (sigma + delta) ** 2)
    g *= math.cos((math.pi / 2) * 0.5)
    g /= 2 * math.pi * sigma ** 2
    assert k[5, 5].item() == pytest.approx(g, rel=1e-5)
