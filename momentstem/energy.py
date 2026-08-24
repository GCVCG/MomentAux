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

Two further read-outs of the SAME committed ``magnitude`` quadrature bank
(2026-08-23, limitations campaign block C, "other currencies"). Same
frequencies, orientations, kernel and calibration batch as ``magnitude`` --
the only new variable is WHAT is read off the quadrature pair (e, o):

* ``phase``    -- local phase per (octave, orientation) pair: (cos phi, sin phi)
  with phi = atan2(o, e); the amplitude sqrt(e^2 + o^2) is DISCARDED. Channel
  layout is pair-major interleaved: channel 2p = cos phi_p, 2p + 1 = sin phi_p
  for the p-th quadrature pair in the bank's (scale-major, orientation-minor)
  order, so 2 x 8 = 16 channels for the committed bank. Each channel is
  bounded in [-1, 1] and cos^2 + sin^2 == 1 per pair at every location.
  LIMITATION, stated: at (near-)zero amplitude the phase is undefined; by
  atan2's convention atan2(0, 0) = 0, so such locations read (1, 0). No
  amplitude gate is applied -- weighting by a/(a + eps) would reintroduce the
  magnitude this target exists to discard. NOTE on the aux use: MomentAuxModel
  average-pools the target to the tap's spatial size, so the POOLED phase
  target is the mean resultant vector of the local phases in each cell (norm
  <= 1, i.e. phase COHERENCE), which is what the aux head regresses.
* ``symmetry`` -- Kovesi phase symmetry per ORIENTATION, summed over the
  bank's octaves:  sym_o = sum_s max(|e_s| - |o_s|, 0) /
  (sum_s sqrt(e_s^2 + o_s^2) + eps),  in [0, 1], one channel per orientation
  = 4 channels for the committed bank (4 orientations x 2 octaves). (The
  block-C pre-registration text says "8 channels"; that assumed 8
  orientations. The committed magnitude bank is 4 orientations x 2 octaves,
  and the per-orientation-summed-over-octaves definition is kept, so the
  target has 4 channels. Recorded as a deviation, not a redesign.)
  eps = _SYM_EPS = 1e-3, FIXED (not tied to calibration, which runs later):
  on standardised CIFAR-100 training images the per-orientation amplitude
  sum sum_s sqrt(e^2 + o^2) has median 0.068 and mean 0.097 (1st percentile
  0.006), so eps is ~1.5% of the median -- it damps the ratio only in
  near-flat regions (denominator below ~1e-2) and leaves edges untouched.
* ``magnitude+phase`` / ``magnitude+symmetry`` -- channel CONCATENATION of the
  two single targets (magnitude channels first), 8 + 16 = 24 and 8 + 4 = 12
  channels. Used for the prior-on-prior combination cells: ONE aux head
  regressing both targets under MSE over the concatenated channels, which is
  the SUM of the two single aux losses (up to the per-channel mean) under a
  SHARED lambda. The concatenation is EXACT (asserted by test) both before and
  after calibration, because calibration is per channel.

Calibration applies to ALL of these uniformly (``calibrate`` sets every
energy channel to unit std on the calibration batch). The phase and symmetry
channels are already bounded, so the gain there is a constant ~1.3-3x, not a
normalisation of an unbounded quantity; it is kept so every target the aux
head ever sees has the same unit-std statistics and so the code path stays
identical across families. Pinned additively in tests/test_bank_regression.py;
the existing magnitude fingerprints are unchanged because the bank is shared.

Channel reduction (``channel_reduce``, 2026-08-23, the SAR limitations block).
The bank operates on ONE scalar field per image. How the input channels are
reduced to that field is a design choice that was never examined until the
So2Sat SAR cells returned a null that is ambiguous between "the prior does not
suit radar statistics" and "the channel reduction is wrong for radar":

* ``mean``       -- the historical default, byte-identical to every recorded
  run: BT.601 luma for 3-channel input, the uniform 1/N band mean otherwise.
* ``perchannel`` -- NO reduction before filtering: the bank is applied to every
  input channel separately and the ENERGIES are averaged over channels. The
  output channel count is unchanged. For a 1-channel input this equals ``mean``
  exactly.
* ``pca1``       -- the scalar field is the projection of the per-pixel channel
  vector onto its FIRST PRINCIPAL COMPONENT, fitted on the calibration batch
  inside ``calibrate`` (covariance over all calibration pixels; eigenvector
  sign fixed so its largest-magnitude weight is positive; stored in the buffer
  ``pc1_w`` so it travels with checkpoints). Forward RAISES if pca1 is requested
  and calibration has not run; the buffer holds the uniform mean weights until
  then so a loaded state_dict always has a well-defined value.
* ``logmean``    -- mean over channels of ``log1p(softplus(x_c))``, i.e. a
  monotone heavy-tail compression of each band applied BEFORE averaging.
  DOCUMENTED HONESTLY: this is NOT the dB transform of raw backscatter
  intensity. So2Sat's SAR bands are stored RAW (linear real/imag parts,
  intensities and covariance terms -- scripts/make_so2sat.py; not dB), but the
  dataset standardises every band (mean/std) before the tensor reaches this
  module and the stem has no access to the raw values, so a true
  log-intensity is not recoverable here. softplus maps the standardised band
  to a strictly positive value monotonically (so the transform is invertible
  per band), and log1p then compresses its right tail, which is the functional
  purpose of the SAR log transform (speckle / heavy-tail compression). For
  negative-valued bands (the real/imag channels) it is a soft rectification.
  Any reader of the So2Sat results must read ``logmean`` as "log-compressed
  standardised bands", not "log intensity".

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
                "structure", "steerable", "invariants",
                # block C (2026-08-23): read-outs of the committed magnitude bank
                "phase", "symmetry", "magnitude+phase", "magnitude+symmetry")
_ENERGY_EPS = 1e-6
# Kovesi phase-symmetry denominator floor (see module docstring for how it
# relates to the raw amplitude scale: ~1.5% of the median per-orientation
# amplitude sum on standardised CIFAR-100). Fixed, independent of calibration.
_SYM_EPS = 1e-3
# The quadrature family that shares the committed magnitude bank.
_MAG_BANK_TYPES = ("magnitude", "phase", "symmetry", "magnitude+phase", "magnitude+symmetry")
CHANNEL_REDUCE = ("mean", "perchannel", "pca1", "logmean")

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
    :param channel_reduce one of CHANNEL_REDUCE (see module docstring);
        "mean" is the historical default and leaves every recorded run
        byte-identical.
    """

    def __init__(self, feature_type="magnitude", in_channels=3, kernel_size=11,
                 channel_reduce="mean"):
        super().__init__()
        if feature_type not in ENERGY_TYPES:
            raise ValueError(f"feature_type must be one of {ENERGY_TYPES}")
        if channel_reduce not in CHANNEL_REDUCE:
            raise ValueError(f"channel_reduce must be one of {CHANNEL_REDUCE}")
        if in_channels < 1:
            raise ValueError("in_channels must be >= 1")
        if kernel_size % 2 != 1:
            raise ValueError("kernel_size must be odd (same-padding contract)")

        self.feature_type = feature_type
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        # BT.601 luma for ordinary 3-channel colour. For multispectral input
        # (EuroSAT MS carries 13 Sentinel-2 bands) there is no luma, so we use
        # the unweighted band mean as the achromatic surrogate: the bank
        # measures ORIENTED ENERGY, which is a spatial property, and we do not
        # want an arbitrary photometric weighting of non-visible bands to
        # decide what it sees. Recorded as a deliberate deviation.
        if in_channels == 3:
            _w = torch.tensor(LUMA_WEIGHTS).view(1, 3, 1, 1)
        else:
            _w = torch.full((1, in_channels, 1, 1), 1.0 / in_channels)
        self.register_buffer("luma_w", _w)
        self.channel_reduce = channel_reduce
        # pca1: the first principal component of the per-pixel channel vector,
        # fitted in calibrate(). Initialised to the uniform mean weights and
        # flagged unfitted; forward refuses to run pca1 before calibration.
        # Registered ONLY in pca1 mode so the state_dict of every existing
        # stem (and every recorded checkpoint) keeps exactly its old keys.
        if channel_reduce == "pca1":
            self.register_buffer("pc1_w", torch.full((1, in_channels, 1, 1), 1.0 / in_channels))
            self.register_buffer("pc1_fitted", torch.zeros((), dtype=torch.bool))

        if feature_type in _MAG_BANK_TYPES:
            # ONE committed bank for magnitude / phase / symmetry / the two
            # combined targets: identical buffers, so the fingerprint of
            # "magnitude" pins all five and only the read-out differs.
            even, odd = quadrature_bank(_MAG_FREQS, _MAG_ORIENTS, kernel_size)
            self.register_buffer("even", even)
            self.register_buffer("odd", odd)
            self.n_scale, self.n_orient = len(_MAG_FREQS), _MAG_ORIENTS
            n_pairs = self.n_scale * self.n_orient
            n_energy = {
                "magnitude": n_pairs,                   # 8
                "phase": 2 * n_pairs,                   # 16 (cos, sin per pair)
                "symmetry": self.n_orient,              # 4 (per orientation)
                "magnitude+phase": n_pairs + 2 * n_pairs,        # 24
                "magnitude+symmetry": n_pairs + self.n_orient,   # 12
            }[feature_type]
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
        """The historical scalar field (channel_reduce="mean")."""
        return (x * self.luma_w).sum(dim=1, keepdim=True)

    def _scalar(self, x):
        """Scalar field(s) for the bank under the configured channel_reduce.

        Returns a tensor of shape (B * m, 1, H, W) where m = 1 for mean / pca1 /
        logmean and m = in_channels for perchannel (channels folded into the
        batch dimension so the conv path is unchanged)."""
        if self.channel_reduce == "mean":
            return self._luma(x)
        if self.channel_reduce == "perchannel":
            B, C, H, W = x.shape
            return x.reshape(B * C, 1, H, W)
        if self.channel_reduce == "pca1":
            if not bool(self.pc1_fitted):
                raise RuntimeError(
                    "channel_reduce='pca1' requires calibrate() to have fitted "
                    "the first principal component before forward()"
                )
            return (x * self.pc1_w).sum(dim=1, keepdim=True)
        # logmean: log-compressed standardised bands, averaged (see module doc)
        return torch.log1p(F.softplus(x)).mean(dim=1, keepdim=True)

    def _raw_energy(self, x):
        """Pre-calibration energy (B, n_energy, H, W) under channel_reduce."""
        e = self._energy(self._scalar(x))
        if self.channel_reduce == "perchannel":
            B = x.shape[0]
            e = e.view(B, self.in_channels, e.shape[1], e.shape[2], e.shape[3]).mean(dim=1)
        return e

    def _energy(self, luma):
        """Raw (pre-calibration) energy channels (B, n_energy, H, W)."""
        if self.feature_type in ("magnitude", "magnitude3", "magnitude6o"):
            e = F.conv2d(luma, self.even, padding=self.padding)
            o = F.conv2d(luma, self.odd, padding=self.padding)
            return torch.sqrt(e ** 2 + o ** 2 + _ENERGY_EPS)
        if self.feature_type in ("phase", "symmetry", "magnitude+phase", "magnitude+symmetry"):
            e = F.conv2d(luma, self.even, padding=self.padding)
            o = F.conv2d(luma, self.odd, padding=self.padding)
            parts = []
            if self.feature_type.startswith("magnitude"):
                # byte-identical to the "magnitude" read-out of the same bank
                parts.append(torch.sqrt(e ** 2 + o ** 2 + _ENERGY_EPS))
            if self.feature_type.endswith("phase"):
                phi = torch.atan2(o, e)                 # (B, P, H, W); atan2(0,0) = 0
                # pair-major interleaved: channel 2p = cos, 2p+1 = sin
                parts.append(torch.stack([torch.cos(phi), torch.sin(phi)], dim=2)
                             .flatten(1, 2))
            if self.feature_type.endswith("symmetry"):
                B, _, H, W = e.shape
                es = e.view(B, self.n_scale, self.n_orient, H, W)
                os_ = o.view(B, self.n_scale, self.n_orient, H, W)
                num = F.relu(es.abs() - os_.abs()).sum(dim=1)               # (B, O, H, W)
                den = torch.sqrt(es ** 2 + os_ ** 2).sum(dim=1) + _SYM_EPS
                parts.append(num / den)
            return parts[0] if len(parts) == 1 else torch.cat(parts, dim=1)
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
        if self.channel_reduce == "pca1":
            self._fit_pc1(x, chunk)
        n = 0
        s1 = s2 = None
        for i in range(0, x.shape[0], chunk):
            e = self._raw_energy(x[i:i + chunk])
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

    @torch.no_grad()
    def _fit_pc1(self, x, chunk=64):
        """Fit the first principal component of the per-pixel channel vector on
        the calibration batch (accumulated sums in double over chunks, so it is
        numerically equivalent to one pass over every pixel). Sign convention:
        the largest-magnitude weight is positive, so the fit is deterministic."""
        C = self.in_channels
        n = 0
        s1 = torch.zeros(C, dtype=torch.float64, device=x.device)
        s2 = torch.zeros(C, C, dtype=torch.float64, device=x.device)
        for i in range(0, x.shape[0], chunk):
            flat = x[i:i + chunk].permute(1, 0, 2, 3).reshape(C, -1).double()
            s1 += flat.sum(1)
            s2 += flat @ flat.t()
            n += flat.shape[1]
        mean = s1 / n
        cov = s2 / n - torch.outer(mean, mean)
        evals, evecs = torch.linalg.eigh(cov)          # ascending
        w = evecs[:, -1]
        w = w * torch.sign(w[w.abs().argmax()])
        self.pc1_w.copy_(w.to(self.pc1_w.dtype).view(1, C, 1, 1))
        self.pc1_fitted.fill_(True)
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
        e = self._raw_energy(x) * self.calib_scale.view(1, -1, 1, 1)
        return torch.cat([x, e], dim=1)

    def extra_repr(self):
        return (
            f"feature_type={self.feature_type}, out_channels={self.out_channels}, "
            f"kernel_size={self.kernel_size}, n_energy={self.n_energy}, "
            f"channel_reduce={self.channel_reduce}"
        )
