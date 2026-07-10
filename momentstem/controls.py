"""Control stems. The controls ARE the contribution -- see README.

Stem registry (one name per experimental cell):

* ``none``         -- identity; vanilla backbone control.
* ``moments-sum``  -- MomentStem sum variant (3-channel output).
* ``moments-cat``  -- MomentStem concat variant (27-channel output).
* ``learned``      -- LearnedStem: ONE plain trainable conv with parameter and
  FLOP count matched (within 2%, asserted in tests) to the moments-cat
  overhead. Like the moment stem it is purely linear (no norm, no activation)
  so it adds the same depth and compute, differing only in whether the
  filters are learned or a fixed moment prior. Proves/disproves: it's the
  PRIOR, not the extra layer.
* ``random-fixed`` -- identical architecture to moments-cat, frozen i.i.d.
  Gaussian filters with matched per-kernel L2 norms. Proves/disproves: it's
  the moment STRUCTURE, not fixedness/scale.
* ``gabor-learn``  -- moment-initialised but trainable filters. Is fixing the
  filters a feature or a bug?
"""

import torch.nn.functional as F
from torch import nn

from .stem import MomentStem


class LearnedStem(nn.Module):
    """Plain trainable conv, parameter/FLOP-matched to moments-cat.

    moments-cat overhead (kernel_size=11): gabor grouped conv 9x1x11x11 =
    1089 + zernike conv 15x3x11x11 = 5445 -> 6534 filter elements and
    6534 MACs/pixel. A 9x9 dense conv 3->24 learned channels gives
    24*3*81 = 5832... we instead match the FULL 27-channel output the
    backbone sees: Conv2d(3, 27, 9) = 27*3*81 = 6561 params and MACs/pixel,
    a +0.41% mismatch (asserted < 2% in test_overhead_match).

    The identity passthrough of moments-cat costs 0 params/FLOPs, so all 27
    learned channels count toward the match.
    """

    def __init__(self, in_channels=3, out_channels=27, kernel_size=9):
        super().__init__()
        if kernel_size % 2 != 1:
            raise ValueError("kernel_size must be odd")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            padding=kernel_size // 2,
            bias=False,
        )

    def forward(self, x):
        return self.conv(x)


class IdentityStem(nn.Module):
    """Vanilla-backbone control."""

    def __init__(self, in_channels=3):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels

    def forward(self, x):
        return x


STEM_NAMES = (
    "none",
    "moments-sum",
    "moments-cat",
    "learned",
    "random-fixed",
    "gabor-learn",
)


def build_stem(name, in_channels=3, kernel_size=11, seed=0):
    """Build a stem by registry name.

    :param seed only used by ``random-fixed`` (a fresh filter draw per
        training seed, so results average over draws). The moment banks
        themselves are constants (committed seed) regardless.
    """
    if name == "none":
        return IdentityStem(in_channels)
    if name == "moments-sum":
        return MomentStem(mode="sum", in_channels=in_channels, kernel_size=kernel_size)
    if name == "moments-cat":
        return MomentStem(mode="concat", in_channels=in_channels, kernel_size=kernel_size)
    if name == "learned":
        reference = MomentStem(
            mode="concat", in_channels=in_channels, kernel_size=kernel_size
        )
        return LearnedStem(in_channels, out_channels=reference.out_channels)
    if name == "random-fixed":
        return MomentStem(
            mode="concat",
            in_channels=in_channels,
            kernel_size=kernel_size,
            init="random",
            seed=seed,
        )
    if name == "gabor-learn":
        return MomentStem(
            mode="concat",
            in_channels=in_channels,
            kernel_size=kernel_size,
            trainable=True,
        )
    raise ValueError(f"unknown stem {name!r}; choose from {STEM_NAMES}")
