"""MomentStem: fixed orthogonal-moment filter bank prepended to a CNN backbone.

Ported from the MomentsNeRF encoder (~/projects/momentsnerf), with no
PixelNeRF dependencies. See PORTING.md for a line-by-line account of what was
ported faithfully and what was changed (and why).

Two filter families:

* Gabor bank -- 9 kernels generated with the exact formula of
  ``GaborNet.GaborConv2d.calculate_weights`` (Meshgini et al. parameter
  scheme), drawn once with a fixed committed seed and frozen.
* Zernike bank -- 15 kernels, one per polynomial j=0..14 of the table in
  ``zernet.layers.ComplexZernikeFunction`` (ported verbatim, including its
  deviations from the standard OSA table -- see PORTING.md), evaluated on the
  unit disk (pixels with rho > 1 are zeroed) and L2-normalised.

Modes:

* ``sum``    -- output stays 3-channel: dense 3->3 Gabor conv (the 9 responses
  are summed into 3 channels, exactly the original "sum" variant), followed by
  a depthwise conv with the mean Zernike kernel. Keeps the pretrained-stem
  input contract.
* ``concat`` -- output is ``[identity RGB (3) | Gabor (9) | Zernike (15)]``
  = 27 channels. The Gabor part is the original grouped-conv concat variant
  (channel 3i+o holds input channel i's o-th Gabor response); the identity
  passthrough and the Zernike channels generalise it (the MomentsNeRF concat
  variant was Gabor-only).

All moment kernels are registered as BUFFERS (zero trainable parameters)
unless ``trainable=True`` (the ``gabor-learn`` control).
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

# Committed constants: every run of the study uses this exact filter bank.
GABOR_SEED = 1234
GABOR_DELTA = 1e-3  # small addition to avoid division by zero (ported)


def gabor_kernel(freq, theta, sigma, psi, kernel_size):
    """Single Gabor kernel, exact port of GaborConv2d.calculate_weights.

    Grid construction is ported verbatim: x0 = ceil(k/2) and
    linspace(-x0 + 1, x0, k), which for odd k is the symmetric integer grid
    shifted so it spans [-(k//2), k//2 + 1]; kept as-is for faithfulness.
    """
    x0 = math.ceil(kernel_size / 2)
    y, x = torch.meshgrid(
        torch.linspace(-x0 + 1, x0 + 0, kernel_size, dtype=torch.float64),
        torch.linspace(-x0 + 1, x0 + 0, kernel_size, dtype=torch.float64),
        indexing="ij",
    )
    rotx = x * math.cos(theta) + y * math.sin(theta)
    roty = -x * math.sin(theta) + y * math.cos(theta)
    g = torch.exp(-0.5 * ((rotx ** 2 + roty ** 2) / (sigma + GABOR_DELTA) ** 2))
    g = g * torch.cos(freq * rotx + psi)
    g = g / (2 * math.pi * sigma ** 2)
    return g.to(torch.float32)


def gabor_bank(n_out=3, n_in=3, kernel_size=11, seed=GABOR_SEED):
    """Bank of n_out*n_in Gabor kernels, shape (n_out, n_in, k, k).

    Parameter scheme ported from GaborConv2d.__init__ (Meshgini et al.):
    freq = (pi/2) * sqrt(2)^(-n), n ~ U{0..4}; theta = m*pi/8, m ~ U{0..7};
    sigma = pi/freq; psi ~ U[0, pi). Drawn once from a fixed committed seed --
    the bank is a constant of the study, not a random variable.
    """
    gen = torch.Generator().manual_seed(seed)
    n = torch.randint(0, 5, (n_out, n_in), generator=gen)
    m = torch.randint(0, 8, (n_out, n_in), generator=gen)
    psi = math.pi * torch.rand(n_out, n_in, generator=gen)
    freq = (math.pi / 2) * math.sqrt(2) ** (-n.to(torch.float64))
    theta = (math.pi / 8) * m.to(torch.float64)
    sigma = math.pi / freq

    bank = torch.stack(
        [
            torch.stack(
                [
                    gabor_kernel(
                        freq[i, j].item(),
                        theta[i, j].item(),
                        sigma[i, j].item(),
                        psi[i, j].item(),
                        kernel_size,
                    )
                    for j in range(n_in)
                ]
            )
            for i in range(n_out)
        ]
    )
    return bank


def gabor_bank_grid(n_out=3, n_in=3, kernel_size=11):
    """Deterministic, collision-free Gabor bank (ablation alternative to the
    ported random draw, which produces duplicate (freq, theta) combos --
    measured max pairwise |cos| = 0.67 for the committed seed).

    Slot s = o*n_in + i gets: orientation s * pi / (n_out*n_in) (evenly
    spaced), frequency cycling over 3 octaves with o, phase alternating
    even (bar) / odd (edge), sigma = pi/freq as in the ported scheme.
    """
    freqs = [math.pi / 2, math.pi / (2 * math.sqrt(2)), math.pi / 4]
    n_slots = n_out * n_in
    bank = torch.empty(n_out, n_in, kernel_size, kernel_size)
    for o in range(n_out):
        for i in range(n_in):
            s = o * n_in + i
            freq = freqs[o % len(freqs)]
            theta = s * math.pi / n_slots
            psi = 0.0 if s % 2 == 0 else math.pi / 2
            bank[o, i] = gabor_kernel(freq, theta, math.pi / freq, psi, kernel_size)
    return bank


def zernike_polynomial(j, x, y):
    """Zernike polynomial j evaluated at cartesian coords, j in 0..14.

    Verbatim port of the polynomial table in
    zernet.layers.ComplexZernikeFunction.forward (see PORTING.md for where
    that table deviates from the standard OSA/Wikipedia indexing).
    """
    if j == 0:
        return torch.ones_like(x)
    if j == 1:
        return x
    if j == 2:
        return y
    if j == 3:  # oblique astigmatism
        return 2.0 * x * y
    if j == 4:  # defocus (as ported: no 2r^2 - 1 normalisation)
        return x ** 2 + y ** 2
    if j == 5:  # vertical astigmatism
        return x ** 2 - y ** 2
    r = torch.sqrt(x ** 2 + y ** 2)
    t = torch.atan2(y, x)
    if j == 6:  # vertical trefoil
        return r ** 3 * torch.sin(3.0 * t)
    if j == 7:  # vertical coma (standard OSA form; the momentsnerf table had
        # 3r^3 sin(3t) here, an exact scalar multiple of j=6 -- after L2
        # normalisation that made kernels 6/7 (and 8/9) IDENTICAL. The
        # original Zernike stage never executed (see PORTING.md), so the
        # correct formula is used rather than the duplicated one.
        return (3.0 * r ** 3 - 2.0 * r) * torch.sin(t)
    if j == 8:  # horizontal coma (standard OSA form, same reasoning)
        return (3.0 * r ** 3 - 2.0 * r) * torch.cos(t)
    if j == 9:  # oblique trefoil
        return r ** 3 * torch.cos(3.0 * t)
    if j == 10:  # oblique quadrafoil
        return 2.0 * r ** 4 * torch.sin(4.0 * t)
    if j == 11:  # oblique secondary astigmatism
        return 2.0 * (4.0 * r ** 4 - 3.0 * r ** 2) * torch.sin(2.0 * t)
    if j == 12:  # primary spherical
        return 6.0 * r ** 4 - 6.0 * r ** 2 + torch.ones_like(r)
    if j == 13:  # vertical secondary astigmatism
        return 2.0 * (4.0 * r ** 4 - 3.0 * r ** 2) * torch.cos(2.0 * t)
    if j == 14:  # vertical quadrafoil
        return 2.0 * r ** 4 * torch.cos(4.0 * t)
    raise ValueError(f"Zernike index {j} out of range 0..14")


N_ZERNIKE = 15


def zernike_bank(kernel_size=11):
    """15 fixed Zernike kernels on the unit disk, shape (15, k, k).

    Pixels are tiled over [-1, 1]^2, values outside the unit disk are zeroed,
    and each kernel is L2-normalised so the 15 responses have comparable
    magnitude (the polynomials themselves have wildly different norms).
    """
    coords = torch.linspace(-1.0, 1.0, kernel_size, dtype=torch.float64)
    y, x = torch.meshgrid(coords, coords, indexing="ij")
    disk = (x ** 2 + y ** 2 <= 1.0).to(torch.float64)
    kernels = []
    for j in range(N_ZERNIKE):
        k = zernike_polynomial(j, x, y) * disk
        k = k / k.norm().clamp_min(1e-12)
        kernels.append(k.to(torch.float32))
    return torch.stack(kernels)


class MomentStem(nn.Module):
    """Fixed Gabor+Zernike filter stem. See module docstring.

    :param mode "sum" (3-channel output) or "concat" (identity+gabor+zernike)
    :param init "moments" for the real filter bank, "random" for the
        random-fixed control: same architecture, same per-kernel L2 norms,
        i.i.d. Gaussian filters (isolates moment STRUCTURE from fixedness).
    :param trainable if True, filters are nn.Parameters instead of buffers
        (the gabor-learn control: moment-initialised but free to train).
    :param seed seed for init="random" draws (the moment bank itself always
        uses the committed GABOR_SEED and is not affected).
    :param gabor_bank_type "random" (ported momentsnerf scheme) or "grid"
        (deterministic collision-free bank, see gabor_bank_grid).
    :param zernike_indices which Zernike polynomials (0..14) to keep, e.g.
        [1, 2, 3, 7, 11] for the top conv1-usage earners measured on the
        full-data ablation (tilts, oblique astigmatism, vertical coma,
        oblique secondary astigmatism). None = all 15.
    """

    def __init__(
        self,
        mode="concat",
        in_channels=3,
        kernel_size=11,
        use_gabor=True,
        use_zernike=True,
        include_identity=True,
        init="moments",
        trainable=False,
        seed=0,
        gabor_bank_type="random",
        zernike_indices=None,
    ):
        super().__init__()
        if mode not in ("sum", "concat"):
            raise ValueError(f"mode must be 'sum' or 'concat', got {mode!r}")
        if init not in ("moments", "random"):
            raise ValueError(f"init must be 'moments' or 'random', got {init!r}")
        if not (use_gabor or use_zernike):
            raise ValueError("at least one of use_gabor/use_zernike required")
        if kernel_size % 2 != 1:
            raise ValueError("kernel_size must be odd (same-padding contract)")
        if gabor_bank_type not in ("random", "grid"):
            raise ValueError(f"unknown gabor_bank_type {gabor_bank_type!r}")

        self.mode = mode
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.use_gabor = use_gabor
        self.use_zernike = use_zernike
        self.include_identity = include_identity and mode == "concat"
        self.trainable = trainable
        self.gabor_bank_type = gabor_bank_type
        self.padding = kernel_size // 2
        if zernike_indices is None:
            zernike_indices = list(range(N_ZERNIKE))
        if not all(0 <= j < N_ZERNIKE for j in zernike_indices) or len(
            set(zernike_indices)
        ) != len(zernike_indices):
            raise ValueError(f"invalid zernike_indices {zernike_indices!r}")
        self.zernike_indices = list(zernike_indices)
        self.n_zernike = len(self.zernike_indices) if use_zernike else 0
        self.n_gabor = (
            (in_channels if mode == "sum" else in_channels * in_channels)
            if use_gabor else 0
        )

        gabor_w = zernike_w = None
        if use_gabor:
            # K[o, i] = o-th Gabor response of input channel i.
            if gabor_bank_type == "grid":
                bank = gabor_bank_grid(in_channels, in_channels, kernel_size)
            else:
                bank = gabor_bank(in_channels, in_channels, kernel_size)
            if mode == "sum":
                # Dense conv: out[o] = sum_i K[o,i] * x_i -- the original
                # GaborConv2d(3, 3) "sum" variant.
                gabor_w = bank
            else:
                # Grouped conv (groups=in_channels): channel 3i+o holds input
                # i's o-th response -- the original 9-channel concat variant.
                gabor_w = torch.stack(
                    [bank[o, i] for i in range(in_channels) for o in range(in_channels)]
                ).unsqueeze(1)

        if use_zernike:
            zbank = zernike_bank(kernel_size)[self.zernike_indices]  # (n, k, k)
            if mode == "sum":
                # Depthwise conv with the mean Zernike kernel: each channel's
                # 15 responses averaged back into that channel.
                zernike_w = zbank.mean(dim=0, keepdim=True).repeat(in_channels, 1, 1)
                zernike_w = zernike_w.unsqueeze(1)  # (C, 1, k, k)
            else:
                # Each Z_j applied to the channel mean -> 15 output channels.
                zernike_w = (
                    zbank.unsqueeze(1).repeat(1, in_channels, 1, 1) / in_channels
                )

        if init == "random":
            gen = torch.Generator().manual_seed(seed)
            gabor_w = self._randomize(gabor_w, gen)
            zernike_w = self._randomize(zernike_w, gen)

        self._register("gabor_weight", gabor_w)
        self._register("zernike_weight", zernike_w)

        if mode == "sum":
            self.out_channels = in_channels
        else:
            self.out_channels = (
                (in_channels if self.include_identity else 0)
                + self.n_gabor
                + self.n_zernike
            )

    @staticmethod
    def _randomize(weight, gen):
        """Gaussian filters with the same shape and per-kernel L2 norm."""
        if weight is None:
            return None
        rand = torch.randn(weight.shape, generator=gen)
        norms = weight.flatten(2).norm(dim=2, keepdim=True).unsqueeze(-1)
        rand_norms = rand.flatten(2).norm(dim=2, keepdim=True).unsqueeze(-1)
        return rand / rand_norms.clamp_min(1e-12) * norms

    def _register(self, name, tensor):
        if tensor is None:
            setattr(self, name, None)
        elif self.trainable:
            setattr(self, name, nn.Parameter(tensor.clone()))
        else:
            self.register_buffer(name, tensor)

    @torch.no_grad()
    def calibrate_zca(self, x, eps=1e-5):
        """Calibration v2: fixed ZCA whitening of the stem output, folded
        into the kernels (concat mode only).

        Motivation (measured on CIFAR-100, gabor-only stem after std
        calibration): the 12-channel output has effective rank 7.7 -- gabor
        responses correlate with RGB up to |r|=0.66 and the covariance
        spectrum spans 60:1. Collinear stem channels are exactly what
        SGD+weight-decay re-allocates between SLOWLY, which is the measured
        failure mode at 10%% data (conv1 cannot shed the prior within the
        step budget; at 100%% it can). ZCA removes the collinearity while
        preserving all information (invertible linear map).

        Implementation: the whole stem (identity passthrough + moment
        convs) is materialised as ONE dense conv T (C_out, C_in, k, k);
        the whitening matrix W and mean mu of the stem response on the
        calibration batch are folded as fused_weight = W @ T,
        fused_bias = -W @ mu. Deterministic given the batch; the fused
        tensors are buffers and travel with checkpoints. Applies equally
        to init="random" (fair control)."""
        if self.mode != "concat":
            raise ValueError("calibrate_zca supports concat mode only")
        if self.trainable:
            raise ValueError("calibrate_zca is for fixed stems only")
        self.calibrate(x)

        k, cin, cout = self.kernel_size, self.in_channels, self.out_channels
        T = torch.zeros(cout, cin, k, k, dtype=x.dtype, device=x.device)
        row = 0
        if self.include_identity:
            for c in range(cin):
                T[c, c, k // 2, k // 2] = 1.0
            row = cin
        if self.gabor_weight is not None:
            # grouped conv channel (cin*i + o) filters input i only
            for i in range(cin):
                for o in range(cin):
                    T[row + cin * i + o, i] = self.gabor_weight[cin * i + o, 0]
            row += self.n_gabor
        if self.zernike_weight is not None:
            T[row:row + self.n_zernike] = self.zernike_weight

        out = self.forward(x)
        f = out.permute(1, 0, 2, 3).flatten(1)
        mu = f.mean(dim=1)
        fc = f - mu.unsqueeze(1)
        cov = (fc @ fc.T) / fc.shape[1]
        U, S, _ = torch.linalg.svd(cov)
        W = U @ torch.diag(1.0 / torch.sqrt(S + eps)) @ U.T

        self.register_buffer("fused_weight", torch.einsum("cd,dikl->cikl", W, T))
        self.register_buffer("fused_bias", -W @ mu)
        return self

    def _ensure_fused_buffers(self):
        """Create empty fused buffers so a ZCA checkpoint can be loaded into
        a freshly built stem without re-running calibration."""
        if getattr(self, "fused_weight", None) is None:
            k = self.kernel_size
            self.register_buffer(
                "fused_weight",
                torch.zeros(self.out_channels, self.in_channels, k, k),
            )
            self.register_buffer("fused_bias", torch.zeros(self.out_channels))

    @torch.no_grad()
    def calibrate(self, x):
        """Response calibration: rescale each fixed filter so its output
        channel has unit standard deviation on the calibration batch.

        Motivation (measured on CIFAR-100): uncalibrated concat output
        channels span a 145x std range (identity ~1.1, Gabor 0.04-0.09,
        Zernike 0.5-5.7), so at init conv1 barely sees the Gabor channels
        and is dominated by the low-pass Zernike ones. Calibration is a
        deterministic one-shot transform of the committed bank -- given the
        same batch it always produces the same scales; they live in the
        weight tensors and therefore in checkpoints. Identity channels are
        untouched. Applies equally to init="random" (fair control).
        """
        eps = 1e-8
        if self.gabor_weight is not None:
            if self.mode == "sum":
                out = F.conv2d(x, self.gabor_weight, padding=self.padding)
            else:
                out = F.conv2d(
                    x, self.gabor_weight, padding=self.padding, groups=self.in_channels
                )
            std = out.std(dim=(0, 2, 3)).clamp_min(eps)
            self.gabor_weight /= std.view(-1, 1, 1, 1)
        if self.zernike_weight is not None:
            if self.mode == "sum":
                # sequential: zernike sees the (calibrated) gabor output
                base = x
                if self.gabor_weight is not None:
                    base = F.conv2d(base, self.gabor_weight, padding=self.padding)
                out = F.conv2d(
                    base, self.zernike_weight, padding=self.padding,
                    groups=self.in_channels,
                )
            else:
                out = F.conv2d(x, self.zernike_weight, padding=self.padding)
            std = out.std(dim=(0, 2, 3)).clamp_min(eps)
            self.zernike_weight /= std.view(-1, 1, 1, 1)
        return self

    def filter_numel(self):
        """Total elements in the moment filters (the 'effective parameter'
        overhead a learned stem must match; buffers are invisible to
        parameter counters)."""
        total = 0
        for w in (self.gabor_weight, self.zernike_weight):
            if w is not None:
                total += w.numel()
        return total

    def forward(self, x):
        if x.dim() != 4 or x.shape[1] != self.in_channels:
            raise ValueError(
                f"expected (B, {self.in_channels}, H, W) input, got {tuple(x.shape)}"
            )
        if getattr(self, "fused_weight", None) is not None:
            return F.conv2d(
                x, self.fused_weight, bias=self.fused_bias, padding=self.padding
            )
        if self.mode == "sum":
            if self.gabor_weight is not None:
                x = F.conv2d(x, self.gabor_weight, padding=self.padding)
            if self.zernike_weight is not None:
                x = F.conv2d(
                    x, self.zernike_weight, padding=self.padding, groups=self.in_channels
                )
            return x
        parts = []
        if self.include_identity:
            parts.append(x)
        if self.gabor_weight is not None:
            parts.append(
                F.conv2d(x, self.gabor_weight, padding=self.padding, groups=self.in_channels)
            )
        if self.zernike_weight is not None:
            parts.append(F.conv2d(x, self.zernike_weight, padding=self.padding))
        return torch.cat(parts, dim=1)

    def extra_repr(self):
        return (
            f"mode={self.mode}, out_channels={self.out_channels}, "
            f"kernel_size={self.kernel_size}, gabor={self.use_gabor}, "
            f"zernike={self.use_zernike}, identity={self.include_identity}, "
            f"trainable={self.trainable}"
        )
