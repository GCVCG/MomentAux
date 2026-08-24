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

    from momentstem.energy import _STEER_FREQS, _STEER_ORIENTS
    se, so = quadrature_bank(_STEER_FREQS, _STEER_ORIENTS, 11)
    assert se.shape == (24, 1, 11, 11)
    assert se.sum().item() == pytest.approx(-0.0543976501, abs=1e-6)
    assert so.abs().mean().item() == pytest.approx(0.0038967307, abs=1e-8)

    # magnitude3 (2026-07-19): committed bank + one lower octave, pinned at
    # its intended kernel_size=17 (sigma=4 envelope needs the +-8 px window).
    from momentstem.energy import _MAG3_FREQS
    m3e, m3o = quadrature_bank(_MAG3_FREQS, _MAG_ORIENTS, 17)
    assert m3e.shape == (12, 1, 17, 17) and m3o.shape == (12, 1, 17, 17)
    assert m3e[:8].shape == (8, 1, 17, 17)  # first 8 pairs = committed layout at k17
    assert m3e.sum().item() == pytest.approx(0.0736529008, abs=1e-6)
    assert m3o.abs().mean().item() == pytest.approx(0.0019418814, abs=1e-8)

    # magnitude6o (2026-07-20): width-matched control for magnitude3 -- the
    # committed 2 octaves x 6 orientations = 12 pairs at k11.
    from momentstem.energy import _MAG6O_ORIENTS
    m6e, m6o = quadrature_bank(_MAG_FREQS, _MAG6O_ORIENTS, 11)
    assert m6e.shape == (12, 1, 11, 11)
    assert m6e.sum().item() == pytest.approx(0.0414302126, abs=1e-6)
    assert m6o.abs().mean().item() == pytest.approx(0.0042441976, abs=1e-8)


def _rotation_drift(stem, x):
    """Relative change in per-channel mean energy under a 90-deg input rotation."""
    e0 = stem._energy(stem._luma(x)).mean(dim=(0, 2, 3))
    er = stem._energy(stem._luma(torch.rot90(x, 1, dims=(2, 3)))).mean(dim=(0, 2, 3))
    return (e0 - er).abs().max().item() / (e0.abs().max().item() + 1e-8)


def test_energy_stem_rotation_invariance():
    """rotinv/steerable/invariants must be (approximately) invariant to a 90-deg
    rotation of the input -- the property that is supposed to survive past 5%.
    'magnitude' and 'structure' are orientation-SELECTIVE and must NOT be
    invariant (guards against accidentally pooling the orientation away).

    THE STIMULUS MUST BE ORIENTED. This probed torch.randn until 2026-07-16,
    which is isotropic in expectation: the spatially-averaged energy of an
    orientation-selective bank is then nearly unchanged by rotation, so the
    selectivity assertion was comparing finite-sample noise (drift ~0.04-0.07,
    varying by draw) against a 0.05 threshold sitting inside that noise -- an
    unseeded test that failed ~1 run in 10. A grating gives an orientation-
    selective bank something to actually be selective about, and separates the
    two families by ~20x instead of ~1.4x:
        grating -> selective 0.94-0.98, invariant 0.0000
        randn   -> selective 0.03-0.07, invariant 0.0000-0.0045
    The banks themselves are unchanged; only the probe stimulus is.
    """
    from momentstem import EnergyStem

    torch.manual_seed(0)
    # horizontal grating: rotating it 90 deg genuinely swaps which oriented
    # filters fire, which is the property under test.
    rows = torch.linspace(0, 6 * math.pi, 32).view(1, 1, 32, 1)
    x = torch.sin(rows).expand(2, 3, 32, 32).contiguous()

    for ft in ("rotinv", "steerable", "invariants"):
        drift = _rotation_drift(EnergyStem(feature_type=ft), x)
        assert drift < 0.02, f"{ft}: not rotation-invariant (drift {drift:.4f})"
    for ft in ("magnitude", "structure"):
        drift = _rotation_drift(EnergyStem(feature_type=ft), x)
        assert drift > 0.5, f"{ft}: must stay orientation-selective (drift {drift:.4f})"


def test_energy_stem_contracts():
    """RGB passthrough, zero trainable params, and unit-std calibration for
    every energy feature type."""
    from momentstem import EnergyStem
    from momentstem.energy import ENERGY_TYPES

    x = torch.randn(3, 3, 32, 32)
    expected_ch = {"magnitude": 11, "magnitude3": 15, "magnitude6o": 15, "rotinv": 11, "structure": 12,
                   "steerable": 12, "invariants": 12,
                   # block C (2026-08-23): 3 + 16 / 3 + 4 / 3 + 24 / 3 + 12
                   "phase": 19, "symmetry": 7, "magnitude+phase": 27, "magnitude+symmetry": 15}
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


# ---------------------------------------------------------------------------
# Block C (2026-08-23): phase / symmetry read-outs of the committed magnitude
# bank, and the two magnitude+<new> concatenations. ADDITIVE pins: the bank
# itself is the one pinned above (test_energy_kernel_fingerprints), so the
# magnitude fingerprints cannot move; these pin the RESPONSES of every
# read-out on a deterministic analytic probe (no RNG, so no torch-version
# dependence), plus the bounds and the exact-concatenation contract.
# ---------------------------------------------------------------------------

def _readout_probe():
    """Deterministic, oriented, multi-frequency probe (2, 3, 32, 32)."""
    i = torch.arange(32, dtype=torch.float32).view(1, 1, 32, 1)
    j = torch.arange(32, dtype=torch.float32).view(1, 1, 1, 32)
    b = torch.arange(2, dtype=torch.float32).view(2, 1, 1, 1)
    c = torch.arange(3, dtype=torch.float32).view(1, 3, 1, 1)
    return (torch.sin(0.9 * i + 0.4 * j + b) + 0.5 * torch.cos(0.3 * i - 0.8 * j + 0.5 * c)
            + 0.25 * torch.sin(1.7 * (i + j))).contiguous()


# (sum, abs-mean) of the RAW (pre-calibration) energy channels on _readout_probe.
READOUT_PINS = {
    "magnitude":          (1711.05619732, 0.1044345824),
    "phase":              (32.88487173, 0.6366319713),
    "symmetry":           (2133.67070281, 0.2604578495),
    "magnitude+phase":    (1743.94106905, 0.4592328416),
    "magnitude+symmetry": (3844.72690013, 0.1564423381),
}


def _raw(ft, x):
    from momentstem import EnergyStem
    st = EnergyStem(feature_type=ft)
    return st, st._energy(st._luma(x)).double()


def test_energy_readout_fingerprints():
    x = _readout_probe()
    for ft, (ssum, amean) in READOUT_PINS.items():
        _, e = _raw(ft, x)
        assert e.sum().item() == pytest.approx(ssum, rel=1e-5), ft
        assert e.abs().mean().item() == pytest.approx(amean, rel=1e-6), ft


def test_energy_readouts_share_the_committed_bank():
    """phase / symmetry / combined must use buffers bitwise-identical to the
    magnitude bank -- the design's whole point is that only the read-out
    differs. Also pins the channel counts: 16 phase (2 x 8 pairs), 4 symmetry
    (one per orientation; the committed bank is 4 orientations x 2 octaves)."""
    from momentstem import EnergyStem
    ref = EnergyStem("magnitude")
    counts = {"phase": 16, "symmetry": 4, "magnitude+phase": 24, "magnitude+symmetry": 12}
    for ft, n in counts.items():
        st = EnergyStem(ft)
        assert torch.equal(st.even, ref.even) and torch.equal(st.odd, ref.odd), ft
        assert st.n_energy == n, (ft, st.n_energy)
        assert st.out_channels == 3 + n
        assert sum(p.numel() for p in st.parameters()) == 0


def test_energy_phase_bounded_unit_circle():
    x = _readout_probe()
    _, ph = _raw("phase", x)
    assert ph.shape[1] == 16
    assert ph.min() >= -1.0 and ph.max() <= 1.0
    # pair-major interleaved: 2p = cos, 2p+1 = sin  => cos^2 + sin^2 == 1
    norm = ph[:, 0::2] ** 2 + ph[:, 1::2] ** 2
    assert torch.allclose(norm, torch.ones_like(norm), atol=1e-5)
    # the probe has oriented structure, so phase genuinely varies
    assert ph.std() > 0.3
    # atan2 convention at zero amplitude: (cos, sin) = (1, 0)
    _, ph0 = _raw("phase", torch.zeros(1, 3, 16, 16))
    assert torch.allclose(ph0[:, 0::2], torch.ones_like(ph0[:, 0::2]))
    assert torch.allclose(ph0[:, 1::2], torch.zeros_like(ph0[:, 1::2]))


def test_energy_symmetry_bounded_unit_interval():
    x = _readout_probe()
    _, sy = _raw("symmetry", x)
    assert sy.shape[1] == 4
    assert sy.min() >= 0.0 and sy.max() <= 1.0
    assert 0.05 < sy.mean() < 0.95  # neither saturated nor dead on a real signal
    # pure zero input: numerator 0, denominator eps  =>  exactly 0
    _, sy0 = _raw("symmetry", torch.zeros(1, 3, 16, 16))
    assert torch.equal(sy0, torch.zeros_like(sy0))


def test_energy_combined_is_exact_concat_before_and_after_calibration():
    from momentstem import EnergyStem
    x = _readout_probe()
    for new in ("phase", "symmetry"):
        _, m = _raw("magnitude", x)
        _, n = _raw(new, x)
        _, c = _raw("magnitude+" + new, x)
        assert torch.equal(c, torch.cat([m, n], dim=1)), new
        # calibration is per channel, so the calibrated combined target equals
        # the concatenation of the two calibrated singles exactly.
        sm = EnergyStem("magnitude").calibrate(x)
        sn = EnergyStem(new).calibrate(x)
        sc = EnergyStem("magnitude+" + new).calibrate(x)
        assert torch.equal(sc.calib_scale, torch.cat([sm.calib_scale, sn.calib_scale]))
        out_c = sc(x)[:, 3:]
        out_s = torch.cat([sm(x)[:, 3:], sn(x)[:, 3:]], dim=1)
        assert torch.equal(out_c, out_s), new
        # the magnitude half of the combined target IS the magnitude target
        assert torch.equal(sc(x)[:, 3:11], sm(x)[:, 3:])
