"""Auxiliary-prior training: use a fixed target WITHOUT putting it in the
deployed forward path, so the benefit scales with data instead of imposing the
mid-data penalty band.

Rationale (2026-07-14). Every forward-path placement of a fixed moment channel
(stem input, readout mask) helps only at <=5% data and NECESSARILY costs
accuracy once data can estimate features itself: conv1 commits to the fixed
channel during the high-LR phase and cannot back out. The moments occupy input
bandwidth abundant data wants for better features.

MomentAuxModel breaks that. The deployed network is a VANILLA ResNet -- RGB in,
logits out, no extra channels, zero inference overhead. During training only, a
small head taps an intermediate layer and is regressed onto a FIXED target map
(MSE or cosine, weight lambda, added to cross-entropy). The target SHAPES the
representation instead of OCCUPYING the input: a soft prior that abundant data
can override -- so it scales with data.

Target producers (all expose ``.out_channels`` and ``forward(x) -> (B,C,h,w)``,
optional ``.calibrate(x)``):
* ``MomentTarget``  -- the fixed moment/energy maps (the method).
* ``TeacherTarget`` -- a frozen pretrained backbone's intermediate features
  (the FitNets/distillation control: is a LEARNED target better than the fixed
  hand-crafted one, at the cost of training a whole teacher?).
* ``HOGTarget``     -- Histogram-of-Oriented-Gradients cells (the MaskFeat
  control: does the specific MOMENT descriptor matter vs. generic HOG?).
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

from .controls import IdentityStem
from .stem import LUMA_WEIGHTS


class MomentTarget(nn.Module):
    """Wraps a fixed stem; produces only its moment/energy maps (drops the
    identity passthrough)."""

    def __init__(self, stem):
        super().__init__()
        self.stem = stem
        self._n_identity = stem.in_channels
        self.out_channels = stem.out_channels - stem.in_channels

    def calibrate(self, x):
        if hasattr(self.stem, "calibrate"):
            self.stem.calibrate(x)
        return self

    def forward(self, x):
        return self.stem(x)[:, self._n_identity:]


class TeacherTarget(nn.Module):
    """Frozen pretrained backbone; produces its features at ``tap`` (the
    FitNets-style learned target). Standardized per-channel on the calibration
    batch so the MSE is comparably scaled to the moment target."""

    def __init__(self, ckpt_path, tap="layer3", backbone="resnet18", num_classes=100):
        super().__init__()
        from .backbones import build_model

        model = build_model(backbone, "none", num_classes=num_classes)
        sd = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(sd)
        self.net = model.net
        for p in self.net.parameters():
            p.requires_grad = False
        self.net.eval()
        self.tap = tap
        self._feat = None
        dict(self.net.named_modules())[tap].register_forward_hook(self._cap)
        with torch.no_grad():
            self.net(torch.zeros(1, 3, 32, 32))
        c = self._feat.shape[1]
        self.out_channels = c
        self.register_buffer("mean", torch.zeros(c))
        self.register_buffer("std", torch.ones(c))

    def _cap(self, module, inp, out):
        self._feat = out

    def train(self, mode=True):  # teacher stays in eval (frozen BN)
        super().train(mode)
        self.net.eval()
        return self

    @torch.no_grad()
    def calibrate(self, x):
        self._feat = None
        self.net(x)
        f = self._feat
        self.mean.copy_(f.mean(dim=(0, 2, 3)))
        self.std.copy_(f.std(dim=(0, 2, 3)).clamp_min(1e-6))
        return self

    def forward(self, x):
        self._feat = None
        self.net(x)
        return (self._feat - self.mean.view(1, -1, 1, 1)) / self.std.view(1, -1, 1, 1)


class HOGTarget(nn.Module):
    """Fixed Histogram-of-Oriented-Gradients cell descriptor (unsigned
    orientation in [0, pi)), soft-binned and gradient-magnitude weighted. The
    MaskFeat-style hand-crafted control."""

    def __init__(self, n_bins=9):
        super().__init__()
        self.n_bins = n_bins
        self.out_channels = n_bins
        self.register_buffer("luma_w", torch.tensor(LUMA_WEIGHTS).view(1, 3, 1, 1))
        kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]])
        self.register_buffer("kx", kx.view(1, 1, 3, 3))
        self.register_buffer("ky", kx.t().contiguous().view(1, 1, 3, 3))
        self.register_buffer("centers", torch.arange(n_bins).float() * math.pi / n_bins)
        self.register_buffer("calib_scale", torch.ones(n_bins))

    def _hog(self, x):
        luma = (x * self.luma_w).sum(dim=1, keepdim=True)
        gx = F.conv2d(luma, self.kx, padding=1)
        gy = F.conv2d(luma, self.ky, padding=1)
        mag = torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)          # (B,1,H,W)
        theta = torch.atan2(gy, gx) % math.pi               # (B,1,H,W) in [0,pi)
        bw = math.pi / self.n_bins
        d = (theta - self.centers.view(1, -1, 1, 1)).abs()  # (B,n_bins,H,W)
        d = torch.minimum(d, math.pi - d)                   # circular distance
        w = (1.0 - d / bw).clamp_min(0.0)                   # triangular soft-bin
        return w * mag

    @torch.no_grad()
    def calibrate(self, x):
        h = self._hog(x)
        self.calib_scale.copy_(1.0 / h.std(dim=(0, 2, 3)).clamp_min(1e-8))
        return self

    def forward(self, x):
        return self._hog(x) * self.calib_scale.view(1, -1, 1, 1)


def _to_spatial(feat):
    """Transformer taps yield (B, N, C) token tensors; the aux head and the
    pooled target need (B, C, H, W). Drop the cls token when present and fold
    the rest back onto their grid. Conv features pass through untouched."""
    if feat.dim() != 3:
        return feat
    b, n, c = feat.shape
    side = int(math.isqrt(n - 1)) if int(math.isqrt(n - 1)) ** 2 == n - 1 else int(math.isqrt(n))
    if side * side == n - 1:          # leading cls token
        feat = feat[:, 1:]
    elif side * side != n:
        raise ValueError(f"token count {n} is not a grid (+cls)")
    return feat.transpose(1, 2).reshape(b, c, side, side)


def _head_key(tap):
    """ModuleDict keys cannot contain '.', but tap names can (ConvNeXt's are
    'stages.2'). Flat ResNet names like 'layer3' are unaffected, so checkpoints
    and tests written against them keep working."""
    return tap.replace(".", "__")


class MomentAuxModel(nn.Module):
    """Vanilla backbone + training-only target-prediction auxiliary head(s).

    :param net a backbone already built for 3-channel input (stem "none").
    :param target a fixed target producer (MomentTarget / TeacherTarget /
        HOGTarget): ``forward(x) -> (B,C,h,w)``, ``.out_channels``.
    :param tap name (or list) of backbone submodule(s) whose output is regressed
        onto the (spatially pooled) target maps; multiple taps are averaged.
    :param aux_weight lambda; raw (tap-averaged) loss stored on ``last_aux``.
    :param loss_form "mse" or "cosine" (scale-free per-location).
    """

    def __init__(self, net, target, tap="layer3", aux_weight=0.1, loss_form="mse",
                 head_norm=False, stem=None):
        """:param head_norm project each aux head back to its initial weight norm
        after every optimizer step. The aux objective ||W.f - t||^2 is INVARIANT
        under (f -> f/c, W -> c.W), so SGD can minimise it by COLLAPSING the
        tapped features and inflating the head -- measured on ResNet-50 at
        lambda=1.0: layer3 std 0.596 -> 0.051 (12x) while ||W|| 1.64 -> 11.2
        (7x), aux falls, and CE STALLS AT CHANCE. Fixing ||W|| removes that
        degenerate direction: the only way to reduce the aux is to make the
        features genuinely predictive.

        :param stem normally None -> IdentityStem, i.e. the deployed path is a
        VANILLA backbone, which is the method's whole selling point (+0 inference
        params). Passing a real stem puts moments BOTH in the forward path and in
        the aux loss; that FORFEITS the vanilla-deploy property and exists only
        for the 1-2% combination experiment (forward-path magnitude still beats
        aux there: +2.55/+3.53 vs +1.91/+3.14, and the two act through different
        mechanisms -- a hard input constraint vs a soft feature prior -- so they
        may be additive). It must be opted into explicitly; see build_model."""
        super().__init__()
        if loss_form not in ("mse", "cosine"):
            raise ValueError(f"loss_form must be 'mse' or 'cosine', got {loss_form!r}")
        self.net = net
        # deployed path: identity by default (vanilla backbone), see :param stem
        self.stem = stem if stem is not None else IdentityStem(3)
        self.target = target
        self.taps = [tap] if isinstance(tap, str) else list(tap)
        self.aux_weight = aux_weight
        self.loss_form = loss_form
        self.n_moment = target.out_channels
        self.last_aux = None

        modules = dict(net.named_modules())
        self._feats = {}
        for t in self.taps:
            if t not in modules:
                raise ValueError(
                    f"tap {t!r} not in backbone; have e.g. layer1..layer4 "
                    f"(ResNet) or stages.0..stages.3 (ConvNeXt)"
                )
            modules[t].register_forward_hook(self._capture_factory(t))

        was_training = net.training
        net.eval()
        with torch.no_grad():  # via self.stem: the net's in_chans follows it
            net(self.stem(torch.zeros(1, 3, 32, 32)))
        net.train(was_training)
        self.aux_heads = nn.ModuleDict({
            _head_key(t): nn.Conv2d(
                _to_spatial(self._feats[t]).shape[1], self.n_moment, kernel_size=1
            )
            for t in self.taps
        })
        self.head_norm = head_norm
        # committed init norms; project_heads() restores these after each step
        self._head_norm0 = {
            t: float(self.aux_heads[_head_key(t)].weight.norm()) for t in self.taps
        }

    @torch.no_grad()
    def project_heads(self):
        """Restore each aux head's weight norm (call after optimizer.step()).
        No-op unless head_norm=True. See __init__ for why this matters."""
        if not self.head_norm:
            return
        for t in self.taps:
            w = self.aux_heads[_head_key(t)].weight
            w.mul_(self._head_norm0[t] / w.norm().clamp_min(1e-8))

    def _capture_factory(self, name):
        def hook(module, inp, out):
            self._feats[name] = out
        return hook

    def calibrate(self, x):
        if hasattr(self.target, "calibrate"):
            self.target.calibrate(x)
        return self

    def forward(self, x):
        self._feats = {}
        logits = self.net(self.stem(x))
        if self.training:
            # target reads the RAW image, never the stemmed tensor
            with torch.no_grad():
                tgt = self.target(x)
            losses = []
            for t in self.taps:
                feat = _to_spatial(self._feats[t])
                target = F.adaptive_avg_pool2d(tgt, feat.shape[-2:])
                pred = self.aux_heads[_head_key(t)](feat)
                if self.loss_form == "cosine":
                    losses.append((1.0 - F.cosine_similarity(pred, target, dim=1)).mean())
                else:
                    losses.append(F.mse_loss(pred, target))
            self.last_aux = torch.stack(losses).mean()
        else:
            self.last_aux = None
        return logits

    def extra_repr(self):
        return (
            f"taps={self.taps}, target={type(self.target).__name__}, "
            f"n_moment={self.n_moment}, aux_weight={self.aux_weight}, deployed=vanilla(3ch)"
        )
