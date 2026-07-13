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
    :param tap name of the backbone submodule whose output is regressed onto
        the (spatially pooled) moment maps.
    :param aux_weight lambda; the raw MSE is stored on ``last_aux`` and the
        training loop adds ``aux_weight * last_aux`` to cross-entropy.
    """

    def __init__(self, net, moment_stem, tap="layer3", aux_weight=0.1):
        super().__init__()
        self.net = net
        self.stem = IdentityStem(moment_stem.in_channels)  # deployed path is identity
        self.moment_stem = moment_stem
        self.tap = tap
        self.aux_weight = aux_weight
        self.n_identity = moment_stem.in_channels
        self.n_moment = moment_stem.out_channels - self.n_identity
        self.last_aux = None

        modules = dict(net.named_modules())
        if tap not in modules:
            raise ValueError(f"tap {tap!r} not in backbone; have e.g. layer1..layer4")
        self._feat = None
        modules[tap].register_forward_hook(self._capture)

        # probe tap channel count (size-independent) to size the aux head
        was_training = net.training
        net.eval()
        with torch.no_grad():
            net(torch.zeros(1, self.n_identity, 32, 32))
        net.train(was_training)
        self.aux_head = nn.Conv2d(self._feat.shape[1], self.n_moment, kernel_size=1)

    def _capture(self, module, inp, out):
        self._feat = out

    def calibrate(self, x):
        """Calibrate the fixed moment target so its channels are unit-std on the
        committed batch -- keeps the MSE well-scaled across moment channels."""
        if hasattr(self.moment_stem, "calibrate"):
            self.moment_stem.calibrate(x)
        return self

    def forward(self, x):
        self._feat = None
        logits = self.net(x)
        if self.training:
            with torch.no_grad():
                target = self.moment_stem(x)[:, self.n_identity:]
                target = F.adaptive_avg_pool2d(target, self._feat.shape[-2:])
            pred = self.aux_head(self._feat)
            self.last_aux = F.mse_loss(pred, target)
        else:
            self.last_aux = None
        return logits

    def extra_repr(self):
        return (
            f"tap={self.tap}, n_moment={self.n_moment}, aux_weight={self.aux_weight}, "
            f"deployed=vanilla({self.n_identity}ch)"
        )
