"""timm backbones behind a stem, plus param/FLOP accounting.

``build_model`` is the single constructor every experiment uses: it builds a
stem by name (see controls.py), a timm backbone whose input layer is widened
to the stem's output channel count, and (for ResNets on small inputs) applies
the standard CIFAR stem surgery -- conv1 becomes 3x3/stride 1 and the first
maxpool is removed -- identically for every stem variant.
"""

import math

import timm
import torch
from torch import nn
from torch.nn import functional as F

from .controls import build_stem
from .stem import MomentStem

RESNETS = ("resnet18", "resnet34", "resnet50")
BACKBONES = RESNETS + ("convnext_tiny", "vit_tiny")


class CosineClassifier(nn.Module):
    """Cosine-similarity classifier (Gidaris & Komodakis 2018): logits =
    s * cos(theta) between L2-normalized features and class weights; drop-in
    replacement for the final nn.Linear. This is the left-flank READOUT
    experiment: at 5 img/class a linear head must estimate direction AND
    magnitude+bias of every class vector from 5 examples; cosine removes the
    magnitude/bias degrees of freedom, which is the standard few-shot remedy.
    s is learnable (init 16, the common value for ~100-way)."""

    def __init__(self, in_features, num_classes, scale=16.0):
        super().__init__()
        self.in_features = in_features
        self.weight = nn.Parameter(torch.empty(num_classes, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.scale = nn.Parameter(torch.tensor(float(scale)))

    def forward(self, x):
        return self.scale * F.linear(
            F.normalize(x, dim=-1), F.normalize(self.weight, dim=-1)
        )


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
    head=None,
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
    if backbone == "vit_tiny":
        # ViT's small-input "surgery" happens at construction: 32x32 with
        # patch 4 gives an 8x8 token grid, so blocks.8 (of 12) mirrors the
        # ResNet layer3 tap in both spatial size and depth fraction. Aux taps
        # on token tensors are reshaped to (B,C,8,8) by aux._to_spatial.
        if not small_input:
            raise ValueError("vit_tiny is only wired for small inputs (32x32)")
        net = timm.create_model(
            "vit_tiny_patch16_224",
            pretrained=pretrained,
            num_classes=num_classes,
            in_chans=stem.out_channels,
            img_size=32,
            patch_size=4,
        )
    else:
        net = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=num_classes,
            in_chans=stem.out_channels,
        )
    if small_input and backbone != "vit_tiny":
        if backbone in RESNETS:
            conv1 = net.conv1
            net.conv1 = nn.Conv2d(
                stem.out_channels, conv1.out_channels, kernel_size=3, stride=1,
                padding=1, bias=False,
            )
            net.maxpool = nn.Identity()
        elif backbone.startswith("convnext"):
            # ConvNeXt's patchify stem (4x4 stride 4) would take 32x32 -> 8x8
            # BEFORE any block runs, leaving the stages at 8/4/2/1. Replacing it
            # with 3x3 stride 1 makes the stage resolutions mirror the ResNet
            # CIFAR surgery (32/16/8/4), so `stages.2` is the direct analogue of
            # `layer3` -- same spatial size, same depth fraction. The stem's
            # LayerNorm2d is kept.
            old = net.stem[0]
            net.stem[0] = nn.Conv2d(
                stem.out_channels, old.out_channels, kernel_size=3, stride=1,
                padding=1,
            )
        else:
            raise ValueError(
                f"small_input surgery implemented for {RESNETS} and convnext_*, "
                f"got {backbone}"
            )
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
    if head:
        # Replace the linear classifier. Happens BEFORE the moment_aux wrap so
        # aux cells get the same head as their baselines; MomentAuxModel never
        # touches net.fc, only its own aux heads.
        if head != "cosine":
            raise ValueError(f"unknown head '{head}' (only 'cosine' exists)")
        if backbone not in RESNETS:
            raise ValueError("head: cosine only implemented for ResNets")
        if head_pool:
            raise ValueError("head and head_pool are mutually exclusive")
        net.fc = CosineClassifier(net.fc.in_features, num_classes)
    if moment_aux:
        from .aux import HOGTarget, MomentAuxModel, MomentTarget, TeacherTarget

        if head_pool:
            raise ValueError("moment_aux and head_pool are mutually exclusive")
        if stem_name != "none" and not moment_aux.get("allow_forward_stem"):
            raise ValueError(
                "moment_aux requires a vanilla backbone (stem 'none'); the +0-"
                "inference-param deploy is the method's point. To deliberately "
                "put moments in BOTH the forward path and the aux loss (the 1-2% "
                "combination experiment), set moment_aux.allow_forward_stem: true "
                "-- and note the deployed model is then NOT vanilla."
            )
        # target kind: a teacher checkpoint (FitNets control), HOG (MaskFeat
        # control), or the fixed moment/energy stem (the method, default).
        if moment_aux.get("teacher"):
            target = TeacherTarget(
                moment_aux["teacher"], tap=moment_aux.get("tap", "layer3"),
                backbone=backbone, num_classes=num_classes,
            )
        elif moment_aux.get("hog"):
            target = HOGTarget(n_bins=moment_aux.get("hog_bins", 9))
        else:
            target = MomentTarget(build_stem(
                moment_aux["stem"],
                kernel_size=moment_aux.get("kernel_size", 11),
                seed=stem_seed,
                **(moment_aux.get("stem_kwargs") or {}),
            ))
        return MomentAuxModel(
            net,
            target,
            tap=moment_aux.get("tap", "layer3"),
            aux_weight=moment_aux.get("weight", 0.1),
            loss_form=moment_aux.get("loss", "mse"),
            head_norm=moment_aux.get("head_norm", False),
            stem=stem if moment_aux.get("allow_forward_stem") else None,
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
