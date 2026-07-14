"""Moment auxiliary prior: use the fixed moments WITHOUT putting them in the
deployed forward path, so the benefit scales with data instead of imposing the
mid-data penalty band.

Rationale (2026-07-14). Every forward-path placement of a fixed moment channel
(stem input, readout mask) helps only at <=5% data and NECESSARILY costs
accuracy once data can estimate features itself: conv1 commits to the fixed
channel during the high-LR phase and cannot back out (pruning doesn't recover,
warmup/unfreeze do nothing, learnable-from-init loses the gain). The moments
occupy input bandwidth abundant data wants for better features.

MomentAuxModel breaks that. The deployed network is a VANILLA ResNet -- RGB in,
logits out, no moment channels, zero inference overhead. During training only,
a small head taps an intermediate layer and is regressed onto the fixed moment
feature maps of the input (MSE, weight lambda, added to cross-entropy). The
moments SHAPE the representation instead of OCCUPYING the input: a soft prior
that abundant data can override. Expected to help at low data and go slack
(harmless) at high data -- i.e. scale with data.
"""

import torch
import torch.nn.functional as F
from torch import nn

from .controls import IdentityStem


class MomentAuxModel(nn.Module):
    """Vanilla backbone + training-only moment-prediction auxiliary head.

    :param net a backbone already built for 3-channel input (stem "none").
    :param moment_stem a fixed stem whose output is [identity | moment maps];
        only the moment maps (channels past ``in_channels``) are the target.
    :param tap name (or list of names) of backbone submodule(s) whose output is
        regressed onto the (spatially pooled) moment maps. Multiple taps = the
        prior shapes features at several depths; losses are averaged.
    :param aux_weight lambda; the raw (tap-averaged) MSE is stored on
        ``last_aux`` and the training loop adds ``aux_weight * last_aux`` to CE.
    """

    def __init__(self, net, moment_stem, tap="layer3", aux_weight=0.1):
        super().__init__()
        self.net = net
        self.stem = IdentityStem(moment_stem.in_channels)  # deployed path is identity
        self.moment_stem = moment_stem
        self.taps = [tap] if isinstance(tap, str) else list(tap)
        self.aux_weight = aux_weight
        self.n_identity = moment_stem.in_channels
        self.n_moment = moment_stem.out_channels - self.n_identity
        self.last_aux = None

        modules = dict(net.named_modules())
        self._feats = {}
        for t in self.taps:
            if t not in modules:
                raise ValueError(f"tap {t!r} not in backbone; have e.g. layer1..layer4")
            modules[t].register_forward_hook(self._capture_factory(t))

        # probe tap channel counts (size-independent) to size one head per tap
        was_training = net.training
        net.eval()
        with torch.no_grad():
            net(torch.zeros(1, self.n_identity, 32, 32))
        net.train(was_training)
        self.aux_heads = nn.ModuleDict({
            t: nn.Conv2d(self._feats[t].shape[1], self.n_moment, kernel_size=1)
            for t in self.taps
        })

    def _capture_factory(self, name):
        def hook(module, inp, out):
            self._feats[name] = out
        return hook

    def calibrate(self, x):
        """Calibrate the fixed moment target so its channels are unit-std on the
        committed batch -- keeps the MSE well-scaled across moment channels."""
        if hasattr(self.moment_stem, "calibrate"):
            self.moment_stem.calibrate(x)
        return self

    def forward(self, x):
        self._feats = {}
        logits = self.net(x)
        if self.training:
            with torch.no_grad():
                moments = self.moment_stem(x)[:, self.n_identity:]
            losses = []
            for t in self.taps:
                feat = self._feats[t]
                target = F.adaptive_avg_pool2d(moments, feat.shape[-2:])
                losses.append(F.mse_loss(self.aux_heads[t](feat), target))
            self.last_aux = torch.stack(losses).mean()
        else:
            self.last_aux = None
        return logits

    def extra_repr(self):
        return (
            f"taps={self.taps}, n_moment={self.n_moment}, aux_weight={self.aux_weight}, "
            f"deployed=vanilla({self.n_identity}ch)"
        )
