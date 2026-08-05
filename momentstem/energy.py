"""EnergyStem: fixed NONLINEAR moment features prepended to a CNN backbone.

Motivation (2026-07-13, from the k5/k11 envelope). The linear MomentStem gives
orientation-SELECTIVE first-order Gabor responses -- exactly what a CNN's conv1
learns first on its own. That is why the benefit collapses after ~5% data: once
the network can estimate those filters itself, the fixed copies are redundant
and merely constrain the high-LR commitment phase (the mid-data penalty band).

To leave an impact PAST 5% a fixed prior must encode something the mid-data
network does NOT spontaneously recover. These three feature types each break the
"redundant oriented edge" mould, and all are NONLINEAR (magnitude / pooling /
products), so they cannot be a single fixed conv -- hence a separate module from
the strictly-linear MomentStem (whose calibration and ZCA fusion assume one
equivalent conv).

Feature types (all operate on BT.601 luminance, RGB identity passed through):

* ``magnitude`` -- complex-Gabor quadrature energy sqrt(even^2 + odd^2): the
  phase-invariant "complex cell" step above conv1's simple-cell filters.
  4 orientations x 2 scale octaves = 8 channels.
* ``rotinv``    -- rotation-invariant energy: quadrature energy pooled (mean and
  max) over 6 orientations at each of 4 scales = 8 channels. Injects an
  invariance the flip-only augmentation never teaches at any data scale.
* ``structure`` -- second-order structure tensor: locally-averaged gradient
  products J11=<Ix^2>, J22=<Iy^2>, J12=<Ix Iy> at 3 gradient scales = 9
  channels. A genuinely higher-order (co-occurrence) statistic.
* ``steerable`` -- angular-harmonic energy (principled rotation invariance):
  magnitude of the first 3 Fourier harmonics |c_0|,|c_1|,|c_2| of the
  orientation-energy profile at 3 scales = 9 channels. Rotation-invariant like
  ``rotinv`` but retains the SHAPE of the orientation distribution (isotropic
  vs oriented vs cross), strictly richer than mean/max pooling.
* ``invariants`` -- structure-tensor eigen-invariants (principled 2nd order):
  the two eigenvalues lambda1>=lambda2 (complete rotation-invariant 2nd-order
  content) and coherence (lambda1-lambda2)/(lambda1+lambda2) at 3 scales = 9
  channels. The rotation-invariant refinement of ``structure``.

All kernels are BUFFERS (zero trainable parameters). ``calibrate`` sets a fixed
per-channel gain buffer so every energy channel has unit std on the committed
calibration batch (the nonlinearity means the scale cannot be folded into the
kernels the way MomentStem.calibrate does, so it is a separate multiplier).
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

from .stem import LUMA_WEIGHTS, gabor_kernel

ENERGY_TYPES = ("magnitude", "magnitude3", "magnitude6o", "rotinv",
                "structure", "steerable", "invariants")
_ENERGY_EPS = 1e-6

# Committed layouts (constants of the study, like the Gabor bank seed).
_MAG_FREQS = (math.pi / 2, math.pi / (2 * math.sqrt(2)))          # 2 octaves
_MAG_ORIENTS = 4                                                   # 0, pi/4, pi/2, 3pi/4
# magnitude3: the committed bank PLUS one lower octave (sigma = pi/f = 4 px).
# Exists to answer "does a WIDER bank add aux value?" (2026-07-19): the extra
# octave sees spatial structure the k11 2-octave bank cannot, physically
# meaningful only at 64x64 -- run with kernel_size >= 17 so the sigma=4
# envelope fits (+-8 px = 2 sigma). Orientation count deliberately unchanged:
# extra orientations are near-linear combinations of the existing four
# (angular bandwidth ~30-45 deg), so they add regression rows without adding
# constraint -- the octave is the only non-redundant direction.
_MAG3_FREQS = (math.pi / 2, math.pi / (2 * math.sqrt(2)), math.pi / 4)
# magnitude6o: the WIDTH-MATCHED control for magnitude3 (2026-07-20). Same 12
# target channels, but from 6 ORIENTATIONS x the committed 2 octaves at k11 --
# no new frequency content. If auxmag3 beats champion while this does not,
# the octave (not target width) carries the value; if this also beats, the
# redundancy argument for orientations is wrong.
_MAG6O_ORIENTS = 6
_ROT_FREQS = (math.pi / 2, math.pi / (2 * math.sqrt(2)),
              math.pi / 4, math.pi / (4 * math.sqrt(2)))           # 4 octaves
_ROT_ORIENTS = 6                                                   # pooled away
_STRUCT_SIGMAS = (1.0, 1.6, 2.5)                                   # 3 gradient scales
_STEER_FREQS = (math.pi / 2, math.pi / (2 * math.sqrt(2)), math.pi / 4)  # 3 octaves
_STEER_ORIENTS = 8                                                 # for angular FFT
_STEER_HARMONICS = 3                                              # |c_0|, |c_1|, |c_2|


def quadrature_bank(freqs, n_orient, kernel_size):
    """Even/odd Gabor quadrature pairs, one per (scale, orientation).

    Returns (even, odd), each (n_scale*n_orient, 1, k, k), ordered scale-major
    then orientation-major so the caller can reshape to (n_scale, n_orient)."""
    even, odd = [], []
    for f in freqs:
        for o in range(n_orient):
            theta = o * math.pi / n_orient
            sigma = math.pi / f
            even.append(gabor_kernel(f, theta, sigma, 0.0, kernel_size))
            odd.append(gabor_kernel(f, theta, sigma, math.pi / 2, kernel_size))
    even = torch.stack(even).unsqueeze(1)
    odd = torch.stack(odd).unsqueeze(1)
    return even, odd


def gaussian_derivative_kernels(sigma, kernel_size):
    """Derivative-of-Gaussian dx, dy (each (1,1,k,k)) and the Gaussian
    integration window (1,1,k,k), all at scale sigma. Gradient kernels are
    zero-sum (odd); the window sums to one."""
    r = kernel_size // 2
    c = torch.arange(-r, r + 1, dtype=torch.float64)
    y, x = torch.meshgrid(c, c, indexing="ij")
    g = torch.exp(-(x ** 2 + y ** 2) / (2.0 * sigma ** 2))
    g = g / g.sum()
    gx = -(x / sigma ** 2) * g
    gy = -(y / sigma ** 2) * g
    return (
        gx.to(torch.float32)[None, None],
        gy.to(torch.float32)[None, None],
        g.to(torch.float32)[None, None],
    )


class EnergyStem(nn.Module):
    """Fixed nonlinear energy stem. See module docstring.

    :param feature_type one of ENERGY_TYPES.
    :param kernel_size odd support for the quadrature / gradient kernels.
    """

    def __init__(self, feature_type="magnitude", in_channels=3, kernel_size=11):
        super().__init__()
        if feature_type not in ENERGY_TYPES:
            raise ValueError(f"feature_type must be one of {ENERGY_TYPES}")
        if in_channels != 3:
            raise ValueError("EnergyStem operates on BT.601 luma; needs 3 input channels")
        if kernel_size % 2 != 1:
            raise ValueError("kernel_size must be odd (same-padding contract)")

        self.feature_type = feature_type
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.register_buffer("luma_w", torch.tensor(LUMA_WEIGHTS).view(1, in_channels, 1, 1))

        if feature_type == "magnitude":
            even, odd = quadrature_bank(_MAG_FREQS, _MAG_ORIENTS, kernel_size)
            self.register_buffer("even", even)
            self.register_buffer("odd", odd)
            n_energy = len(_MAG_FREQS) * _MAG_ORIENTS
        elif feature_type == "magnitude3":
            even, odd = quadrature_bank(_MAG3_FREQS, _MAG_ORIENTS, kernel_size)
            self.register_buffer("even", even)
            self.register_buffer("odd", odd)
            n_energy = len(_MAG3_FREQS) * _MAG_ORIENTS
        elif feature_type == "magnitude6o":
            even, odd = quadrature_bank(_MAG_FREQS, _MAG6O_ORIENTS, kernel_size)
            self.register_buffer("even", even)
            self.register_buffer("odd", odd)
            n_energy = len(_MAG_FREQS) * _MAG6O_ORIENTS
        elif feature_type == "rotinv":
            even, odd = quadrature_bank(_ROT_FREQS, _ROT_ORIENTS, kernel_size)
            self.register_buffer("even", even)
            self.register_buffer("odd", odd)
            self.n_scale, self.n_orient = len(_ROT_FREQS), _ROT_ORIENTS
            n_energy = self.n_scale * 2  # mean + max pooled over orientation
        elif feature_type == "steerable":
            even, odd = quadrature_bank(_STEER_FREQS, _STEER_ORIENTS, kernel_size)
            self.register_buffer("even", even)
            self.register_buffer("odd", odd)
            self.n_scale, self.n_orient = len(_STEER_FREQS), _STEER_ORIENTS
            self.n_harm = _STEER_HARMONICS
            # angular-harmonic weights: orientation is pi-periodic, so harmonic k
            # uses angle 2*k*theta (theta = j*pi/n_orient). Buffers so forward is
            # allocation-free and the layout is a committed constant.
            thetas = torch.arange(self.n_orient, dtype=torch.float32) * math.pi / self.n_orient
            ks = torch.arange(self.n_harm, dtype=torch.float32).view(-1, 1)
            self.register_buffer("harm_cos", torch.cos(2.0 * ks * thetas))  # (H, O)
            self.register_buffer("harm_sin", torch.sin(2.0 * ks * thetas))
            n_energy = self.n_scale * self.n_harm
        else:  # structure or invariants (both from the structure tensor)
            gx, gy, win = [], [], []
            for s in _STRUCT_SIGMAS:
                dx, dy, g = gaussian_derivative_kernels(s, kernel_size)
                gx.append(dx); gy.append(dy); win.append(g)
            self.register_buffer("grad_x", torch.cat(gx))   # (S,1,k,k)
            self.register_buffer("grad_y", torch.cat(gy))
            self.register_buffer("window", torch.cat(win))
            n_energy = len(_STRUCT_SIGMAS) * 3              # 3 outputs per scale

        self.n_energy = n_energy
        self.out_channels = in_channels + n_energy
        self.register_buffer("calib_scale", torch.ones(n_energy))

    def _luma(self, x):
        return (x * self.luma_w).sum(dim=1, keepdim=True)

    def _energy(self, luma):
        """Raw (pre-calibration) energy channels (B, n_energy, H, W)."""
        if self.feature_type in ("magnitude", "magnitude3", "magnitude6o"):
            e = F.conv2d(luma, self.even, padding=self.padding)
            o = F.conv2d(luma, self.odd, padding=self.padding)
            return torch.sqrt(e ** 2 + o ** 2 + _ENERGY_EPS)
        if self.feature_type == "rotinv":
            e = F.conv2d(luma, self.even, padding=self.padding)
            o = F.conv2d(luma, self.odd, padding=self.padding)
            energy = torch.sqrt(e ** 2 + o ** 2 + _ENERGY_EPS)
            B, _, H, W = energy.shape
            energy = energy.view(B, self.n_scale, self.n_orient, H, W)
            pooled = torch.cat([energy.mean(dim=2), energy.amax(dim=2)], dim=1)
            return pooled
        if self.feature_type == "steerable":
            e = F.conv2d(luma, self.even, padding=self.padding)
            o = F.conv2d(luma, self.odd, padding=self.padding)
            energy = torch.sqrt(e ** 2 + o ** 2 + _ENERGY_EPS)
            B, _, H, W = energy.shape
            energy = energy.view(B, self.n_scale, self.n_orient, H, W)
            # angular Fourier magnitude per harmonic: |sum_o energy * e^{i 2k theta_o}|
            # einsum over orientation with the committed cos/sin weight tables.
            real = torch.einsum("bsoyx,ko->bksyx", energy, self.harm_cos)
            imag = torch.einsum("bsoyx,ko->bksyx", energy, self.harm_sin)
            mag = torch.sqrt(real ** 2 + imag ** 2 + _ENERGY_EPS)  # (B, K, n_scale, H, W)
            return mag.reshape(B, self.n_harm * self.n_scale, H, W)
        # structure tensor (structure: raw components; invariants: eigen-invariants)
        ix = F.conv2d(luma, self.grad_x, padding=self.padding)
        iy = F.conv2d(luma, self.grad_y, padding=self.padding)
        s = self.window.shape[0]
        comps = []
        for k in range(s):
            w = self.window[k:k + 1]
            j11 = F.conv2d(ix[:, k:k + 1] ** 2, w, padding=self.padding)
            j22 = F.conv2d(iy[:, k:k + 1] ** 2, w, padding=self.padding)
            j12 = F.conv2d(ix[:, k:k + 1] * iy[:, k:k + 1], w, padding=self.padding)
            if self.feature_type == "structure":
                comps += [j11, j22, j12]
            else:  # invariants: eigenvalues + coherence
                tr = j11 + j22
                disc = torch.sqrt((j11 - j22) ** 2 + 4.0 * j12 ** 2 + _ENERGY_EPS)
                lam1 = 0.5 * (tr + disc)
                lam2 = 0.5 * (tr - disc)
                coherence = disc / (tr + _ENERGY_EPS)
                comps += [lam1, lam2, coherence]
        return torch.cat(comps, dim=1)

    @torch.no_grad()
    def calibrate(self, x, chunk=64):
        """Set per-channel gain so each energy channel has unit std on the
        calibration batch. Identity RGB channels are untouched. Deterministic
        given the batch; the gain lives in a buffer and travels with checkpoints.

        Computed in CHUNKS via accumulated sums rather than one forward over
        the whole batch: at 224px a 1024-image calibration batch OOMs even a
        24GB card (the energy maps are B x n_filters x H x W). Chunking is
        numerically equivalent -- std is pooled over exactly the same elements
        -- so every previously calibrated stem is unaffected, which
        tests/test_bank_regression.py checks."""
        n = 0
        s1 = s2 = None
        for i in range(0, x.shape[0], chunk):
            e = self._energy(self._luma(x[i:i + chunk]))
            c = e.shape[1]
            flat = e.permute(1, 0, 2, 3).reshape(c, -1).double()
            s1 = flat.sum(1) if s1 is None else s1 + flat.sum(1)
            s2 = (flat ** 2).sum(1) if s2 is None else s2 + (flat ** 2).sum(1)
            n += flat.shape[1]
        mean = s1 / n
        var = (s2 / n - mean ** 2).clamp_min(0)
        std = var.sqrt().to(self.calib_scale.dtype).clamp_min(1e-8)
        self.calib_scale.copy_(1.0 / std)
        return self

    def filter_numel(self):
        """Fixed-filter elements (the 'effective parameter' overhead; buffers
        are invisible to parameter counters). Matches MomentStem's contract."""
        total = 0
        for name in ("even", "odd", "grad_x", "grad_y", "window"):
            w = getattr(self, name, None)
            if w is not None:
                total += w.numel()
        return total

    def forward(self, x):
        if x.dim() != 4 or x.shape[1] != self.in_channels:
            raise ValueError(
                f"expected (B, {self.in_channels}, H, W) input, got {tuple(x.shape)}"
            )
        e = self._energy(self._luma(x)) * self.calib_scale.view(1, -1, 1, 1)
        return torch.cat([x, e], dim=1)

    def extra_repr(self):
        return (
            f"feature_type={self.feature_type}, out_channels={self.out_channels}, "
            f"kernel_size={self.kernel_size}, n_energy={self.n_energy}"
        )
