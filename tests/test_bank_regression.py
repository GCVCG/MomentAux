"""Numeric regression pins for the committed filter banks. If any of these
move, the study's filters changed and every existing run is invalidated --
that must be a loud, deliberate decision, not drift."""

import math

import pytest
import torch

from momentstem.stem import GABOR_SEED, gabor_bank, gabor_kernel, zernike_bank

# Committed fingerprints of the exact banks used by every run (kernel_size=11,
# GABOR_SEED=1234). Recorded at repo creation.
GABOR_SUM = -0.0577797294
GABOR_ABSMEAN = 0.0037828626
ZERNIKE_SUM = 16.7257347107
ZERNIKE_ABSMEAN = 0.0563285500


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
