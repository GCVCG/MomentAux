"""timm backbones behind a stem, plus param/FLOP accounting.

``build_model`` is the single constructor every experiment uses: it builds a
stem by name (see controls.py), a timm backbone whose input layer is widened
to the stem's output channel count, and (for ResNets on small inputs) applies
the standard CIFAR stem surgery -- conv1 becomes 3x3/stride 1 and the first
maxpool is removed -- identically for every stem variant.
"""

import timm
import torch
from torch import nn

from .controls import build_stem
from .stem import MomentStem

RESNETS = ("resnet18", "resnet34", "resnet50")
BACKBONES = RESNETS + ("convnext_tiny",)


class StemmedModel(nn.Module):
    def __init__(self, stem, net):
        super().__init__()
        self.stem = stem
        self.net = net

    def forward(self, x):
        return self.net(self.stem(x))


def build_model(
    backbone,
    stem_name,
    num_classes,
    small_input=True,
    pretrained=False,
    stem_kernel_size=11,
    stem_seed=0,
    stem_kwargs=None,
    head_pool=None,
    moment_aux=None,
):
    """Build stem + backbone for one experimental cell.

    :param small_input apply CIFAR-style ResNet surgery (3x3 stride-1 conv1,
        no first maxpool). Required for 32x32; we use it for 96x96 STL-10 too
        so the recipe is identical. Only implemented for ResNets.
    :param stem_kwargs extra MomentStem ablation options (see build_stem).
    :param head_pool replace GAP with MultiMaskPool, e.g.
        {"type": "zernike"|"random"|"learned", "J": 8, "hw": 4}; the linear
        head widens to features*J accordingly (see pooling.py).
    :param moment_aux use the moments as a training-only auxiliary prior on a
        VANILLA backbone (deployed path has no moment channels). Dict e.g.
        {"stem": "energy-magnitude", "tap": "layer3", "weight": 0.1,
         "kernel_size": 11, "stem_kwargs": {...}}. Requires stem_name "none".
        See aux.py. Mutually exclusive with head_pool.
    """
    stem = build_stem(
        stem_name, kernel_size=stem_kernel_size, seed=stem_seed, **(stem_kwargs or {})
    )
    net = timm.create_model(
        backbone,
        pretrained=pretrained,
        num_classes=num_classes,
        in_chans=stem.out_channels,
    )
    if small_input:
        if backbone not in RESNETS:
            raise ValueError(
                f"small_input surgery only implemented for {RESNETS}, got {backbone}"
            )
        conv1 = net.conv1
        net.conv1 = nn.Conv2d(
            stem.out_channels, conv1.out_channels, kernel_size=3, stride=1,
            padding=1, bias=False,
        )
        net.maxpool = nn.Identity()
    if head_pool:
        from .pooling import MultiMaskPool

        if backbone not in RESNETS:
            raise ValueError("head_pool only implemented for ResNets")
        pool = MultiMaskPool(
            mask_type=head_pool.get("type", "random"),
            hw=head_pool.get("hw", 4),
            J=head_pool.get("J", 8),
            seed=head_pool.get("seed", 0),
        )
        in_feats = net.fc.in_features
        net.global_pool = pool
        net.fc = nn.Linear(in_feats * pool.J, num_classes)
    if moment_aux:
        from .aux import MomentAuxModel

        if head_pool:
            raise ValueError("moment_aux and head_pool are mutually exclusive")
        if stem_name != "none":
            raise ValueError("moment_aux requires a vanilla backbone (stem 'none')")
        target = build_stem(
            moment_aux["stem"],
            kernel_size=moment_aux.get("kernel_size", 11),
            seed=stem_seed,
            **(moment_aux.get("stem_kwargs") or {}),
        )
        return MomentAuxModel(
            net,
            target,
            tap=moment_aux.get("tap", "layer3"),
            aux_weight=moment_aux.get("weight", 0.1),
            loss_form=moment_aux.get("loss", "mse"),
        )
    return StemmedModel(stem, net)


def count_params_flops(model, image_size=32, batch_size=1):
    """Params (trainable + fixed-filter buffers, reported separately) and
    fvcore-traced FLOPs (MACs) for the full model and the stem alone."""
    from fvcore.nn import FlopCountAnalysis

    model = model.eval()
    device = next(model.parameters()).device
    x = torch.zeros(
        batch_size, model.stem.in_channels, image_size, image_size, device=device
    )
    def _analyse(module):
        fca = FlopCountAnalysis(module, x)
        fca.unsupported_ops_warnings(False)
        fca.uncalled_modules_warnings(False)
        return int(fca.total())

    total_flops = _analyse(model)
    stem_flops = _analyse(model.stem)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    stem_trainable = sum(
        p.numel() for p in model.stem.parameters() if p.requires_grad
    )
    fixed_filters = (
        model.stem.filter_numel()
        if hasattr(model.stem, "filter_numel")
        and not getattr(model.stem, "trainable", False)
        else 0
    )
    return {
        "params_trainable": trainable,
        "params_stem_trainable": stem_trainable,
        "params_stem_fixed_filters": fixed_filters,
        "flops_total": total_flops,
        "flops_stem": stem_flops,
        "flops_image_size": image_size,
    }
