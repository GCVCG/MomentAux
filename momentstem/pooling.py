"""Fixed multi-mask readout pooling (the 'richer fixed readout' line).

GAP is the zeroth spatial moment of the final feature map; everything above
it (layout, spread, elongation of activation mass) is discarded before the
classifier. Linear probes on frozen trained features showed +0.4..+1.5
points of linearly decodable class information in 8-mask pooling over GAP
at every data scale -- and that ZERNIKE mask structure specifically is no
better than random fixed masks at 4x4/8x8 resolutions (the disk has ~12
pixels; any spanning set works). This module tests that end-to-end.

MultiMaskPool replaces the backbone's global pool + flatten: features
(B, C, h, w) -> (B, C*J) via J spatial masks. Mask 0 is always the uniform
(GAP) mask so the baseline readout is nested in every variant.

Mask types:
  zernike -- Z_0..Z_{J-1} on the unit disk (fixed buffers)
  random  -- seeded unit-norm Gaussian masks (fixed buffers; the structure
             control)
  learned -- same init as random but trainable (the fixedness control;
             matched head parameters)
"""

import torch
from torch import nn

from .stem import zernike_polynomial


def make_masks(mask_type, hw, J=8, seed=0):
    if mask_type == "zernike":
        c = torch.linspace(-1.0, 1.0, hw)
        y, x = torch.meshgrid(c, c, indexing="ij")
        disk = ((x ** 2 + y ** 2) <= 1.0).float()
        ms = [zernike_polynomial(j, x, y) * disk for j in range(J)]
        masks = torch.stack(ms)
    elif mask_type in ("random", "learned"):
        g = torch.Generator().manual_seed(seed)
        masks = torch.randn(J, hw, hw, generator=g)
    else:
        raise ValueError(f"unknown mask_type {mask_type!r}")
    masks = masks / masks.flatten(1).norm(dim=1, keepdim=True).clamp_min(1e-8).view(-1, 1, 1)
    # mask 0 is always uniform: every variant nests the GAP readout
    masks[0] = 1.0 / hw
    return masks


class MultiMaskPool(nn.Module):
    def __init__(self, mask_type="random", hw=4, J=8, seed=0):
        super().__init__()
        self.mask_type = mask_type
        self.hw = hw
        self.J = J
        masks = make_masks(mask_type, hw, J, seed)
        if mask_type == "learned":
            self.masks = nn.Parameter(masks)
        else:
            self.register_buffer("masks", masks)

    def forward(self, x):
        if x.shape[-2:] != (self.hw, self.hw):
            raise ValueError(f"expected {self.hw}x{self.hw} feature maps, got {tuple(x.shape[-2:])}")
        return torch.einsum("nchw,jhw->ncj", x, self.masks).flatten(1)

    def extra_repr(self):
        return f"mask_type={self.mask_type}, hw={self.hw}, J={self.J}"
