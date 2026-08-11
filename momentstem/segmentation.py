"""Dense-prediction backbones, so the fusion rule can be tested on a task
whose metric is not top-1 accuracy.

WHY THIS IS A SEPARATE FILE, and why the trainer is a separate entry point:
the classification path carries 9,000 finished runs and a frozen recipe that
every one of them was measured under. Nothing here imports into that path or
changes a byte of it. A segmentation cell cannot affect a classification cell
even if this file is wrong.

WHAT TRANSFERS UNCHANGED, which is the point of running this at all. The
auxiliary target is already dense: MomentTarget produces
m(x) in R^{N x H x W}, a per-location oriented-energy map, and
MomentAuxModel pools it to whatever spatial size the tapped stage has
(F.adaptive_avg_pool2d in its forward). On a classification cell that pooling
is aggressive -- down to 8x8 or 4x4. On a segmentation cell at output stride
8 it is barely pooling at all. So the SAME bank, the SAME target, the SAME
tap and the SAME schedule apply here with no re-design, no re-pinning and no
new hyper-parameter.

STATE THE BIAS UP FRONT, because it is the first thing a referee should ask:
a dense spatial target on a dense spatial task is the most favourable venue
this prior could be given. A positive result here is therefore WEAKER evidence
of generality than a positive result on a task whose structure has nothing to
do with oriented energy. Detection is the harder test and is deliberately not
what this file starts with.

The head is an FCN rather than DeepLab's ASPP or an FPN: this study's claims
come from differences between two arms under one recipe, and the simplest
head that can express the task is the one least able to launder a difference
through its own capacity. Both arms of every pair get the identical head.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

SEG_BACKBONES = ("resnet18", "resnet34", "resnet50")


class FCNHead(nn.Sequential):
    """3x3 conv, BN, ReLU, dropout, 1x1 classifier -- the torchvision FCN head.

    Deliberately small. The question is what the BACKBONE learned, and a head
    with its own capacity can compensate for a worse backbone, which is
    exactly the confound the frozen-feature probe exists to avoid.
    """

    def __init__(self, in_ch, n_classes, mid=256):
        super().__init__(
            nn.Conv2d(in_ch, mid, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Conv2d(mid, n_classes, 1),
        )


class SegResNet(nn.Module):
    """Dilated ResNet encoder + FCN head, at output stride 8.

    Dilation rather than more upsampling: replacing the stride in the last two
    stages with dilation keeps the tapped stage at input/8 instead of
    input/16, so the auxiliary target is compared against features that still
    have real spatial extent. At stride 32 the tap would be 16x16 for a 512px
    crop and the dense target would be pooled almost to a classification
    target, which would make this experiment a worse test of the thing it is
    meant to test.

    Submodules are named `backbone.layer1..layer4` so a tap string reads the
    same as everywhere else in the study (`backbone.layer3` here against
    `layer3` there), and MomentAuxModel's hook mechanism attaches unchanged.
    """

    def __init__(self, backbone="resnet18", n_classes=21, output_stride=8,
                 pretrained=False, in_channels=3):
        super().__init__()
        if backbone not in SEG_BACKBONES:
            raise ValueError(f"segmentation backbone must be one of "
                             f"{SEG_BACKBONES}, got {backbone!r}")
        # timm rather than torchvision: torchvision's BasicBlock refuses
        # dilation ("Dilation > 1 not supported in BasicBlock"), so its
        # resnet18/34 cannot run at output stride 8 at all. timm implements it,
        # and it is also the library every classification cell in this study
        # was built with, so the encoder here is the same code path as the
        # backbone whose behaviour we are comparing against.
        import timm
        if output_stride not in (8, 16):
            raise ValueError("output_stride must be 8 or 16")
        net = timm.create_model(backbone, pretrained=pretrained, num_classes=0,
                                in_chans=in_channels, output_stride=output_stride)
        self.backbone = net
        self.head = FCNHead(net.num_features, n_classes)

    def features(self, x):
        # timm's own feature path, so the encoder is byte-identical to the
        # classification backbone up to the dilation setting.
        return self.backbone.forward_features(x)

    def forward(self, x):
        size = x.shape[-2:]
        y = self.head(self.features(x))
        # Predict at stride 8 and upsample: the standard FCN protocol. Both
        # arms are scored identically, so the interpolation is not a variable.
        return F.interpolate(y, size=size, mode="bilinear", align_corners=False)


class SegSwin(nn.Module):
    """Swin-Tiny encoder + FCN head on its stride-8 stage.

    WHY AN ATTENTION BACKBONE HERE AT ALL. The study's strongest classification
    claim is the attention regime -- ViT-B gains +26.01 at ImageNet-100, the
    largest Delta measured, and every conv is neutral at full data. The dense
    transplant so far uses ResNet-18 ONLY, so that claim has no dense evidence
    whatsoever. A dense attention pair asks whether the attention deficit is a
    property of the backbone or of the classification task.

    Swin rather than ViT, for a structural reason rather than a preference:
    plain ViT emits a single stride-16 token grid with no hierarchy, so a dense
    head on it needs its own upsampling decoder, and the decoder's capacity
    would then be a variable between the conv and attention arms. Swin is
    hierarchical, so its stride-8 stage is a drop-in analog of the dilated
    ResNet's stride-8 output and BOTH backbones can wear the identical FCN
    head. Swin is also already a classification population in this study
    (91 law cells), so its dense behaviour can be compared to its own
    classification behaviour rather than only to ResNet-18's.

    DEVIATION, stated: Swin reaches stride 8 by patch-merging rather than by
    dilation, so its stride-8 features are 192-channel where the dilated
    ResNet-18's are 256. Nothing is tuned to compensate; both arms of the pair
    are identical, which is what a Delta requires.

    CARRIES THE BISTABILITY CAVEAT. On classification, swin-none is seed-
    bistable or at chance on most datasets while swin-aux trains tightly. If
    that recurs here, these cells are reported with the same flag and are not
    headline rows.
    """

    def __init__(self, backbone="swin_tiny_patch4_window7_224", n_classes=21,
                 output_stride=8, pretrained=False, in_channels=3,
                 image_size=512):
        super().__init__()
        import timm
        if output_stride != 8:
            raise ValueError("SegSwin taps the stride-8 stage; output_stride "
                             "must be 8")
        # features_only with out_indices=(1,) selects the stride-8 stage.
        # img_size must be passed explicitly: Swin's window partitioning is
        # resolution-dependent, and the default 224 would silently mis-shape a
        # 512 crop.
        net = timm.create_model(backbone, pretrained=pretrained,
                                in_chans=in_channels, img_size=image_size,
                                features_only=True, out_indices=(1,))
        self.backbone = net
        self._img_size = image_size
        ch = net.feature_info.channels()[0]
        self._stride = net.feature_info.reduction()[0]
        self.head = FCNHead(ch, n_classes)

    def features(self, x):
        # THE PADDING LIVES HERE, not in forward(), and that placement is the
        # whole point. Swin asserts its input matches the img_size it was
        # built with, and dense evaluation is native-resolution with variable
        # sizes. The first version of this fix padded inside forward() -- but
        # analysis/dense_probe.py calls features() DIRECTLY, so every Swin
        # probe died on "Input height (366) doesn't match model (512)" while
        # training was fine. Any path that reaches the backbone must get the
        # same treatment, so it belongs on the shared one.
        H, W = x.shape[-2:]
        S = self._img_size
        if (H, W) != (S, S):
            if H > S or W > S:
                raise ValueError(
                    f"SegSwin was built for {S}x{S} and cannot pad a {H}x{W} "
                    "input down; rebuild with a larger image_size")
            x = F.pad(x, (0, S - W, 0, S - H))     # right/bottom, zeros = mean
        f = self.backbone(x)[0]
        # timm returns Swin features as NHWC. The FCN head, the auxiliary
        # target and the probe all assume NCHW, and a wrong permutation here
        # would train on transposed features while still producing a plausible
        # loss curve -- so it is decided by the CHANNEL COUNT the head expects
        # rather than by assuming a layout.
        if f.shape[1] != self.head[0].in_channels and \
                f.shape[-1] == self.head[0].in_channels:
            f = f.permute(0, 3, 1, 2).contiguous()
        if (H, W) != (S, S):
            # Crop the feature map back to the REAL image's extent. Without
            # this the caller upsamples a padded map over the native size and
            # every prediction is spatially stretched -- a silent
            # misalignment that still produces a plausible mIoU.
            r = self._stride
            f = f[..., :-(-H // r), :-(-W // r)]
        return f

    def forward(self, x):
        # Padding is handled in features(), which crops the feature map back
        # to the real extent, so this is the same two lines as SegResNet.
        size = x.shape[-2:]
        y = self.head(self.features(x))
        return F.interpolate(y, size=size, mode="bilinear", align_corners=False)


SEG_ATTN_BACKBONES = ("swin_tiny_patch4_window7_224",)


def build_seg_model(backbone="resnet18", n_classes=21, output_stride=8,
                    pretrained=False, moment_aux=None, image_size=512,
                    in_channels=3):
    """Build the segmentation net, optionally wrapped in the SAME auxiliary
    head the classification cells use.

    The wrapper is imported here rather than reimplemented precisely so that a
    difference between a segmentation result and a classification result cannot
    be an artifact of a second implementation of the prior.
    """
    if backbone in SEG_ATTN_BACKBONES:
        net = SegSwin(backbone, n_classes, output_stride, pretrained,
                      in_channels, image_size)
        default_tap = "backbone.layers_1"
    else:
        net = SegResNet(backbone, n_classes, output_stride, pretrained,
                        in_channels)
        default_tap = "backbone.layer3"
    if not moment_aux:
        return net
    from .aux import MomentAuxModel, MomentTarget
    from .controls import build_stem
    # Constructed exactly as build_model does it, so the bank, its calibration
    # and its channel count are identical to the classification cells'.
    target = MomentTarget(build_stem(
        moment_aux["stem"],
        in_channels=in_channels,
        kernel_size=moment_aux.get("kernel_size", 11),
        seed=moment_aux.get("stem_seed", 0),
        **(moment_aux.get("stem_kwargs") or {}),
    ))
    kw = dict(
        tap=moment_aux.get("tap", default_tap),
        aux_weight=moment_aux.get("weight", 1.0),
        loss_form=moment_aux.get("loss", "mse"),
        head_norm=moment_aux.get("head_norm", True),
        image_size=image_size,
    )
    # `in_channels` was added to MomentAuxModel for the multispectral cells and
    # is absent from older deployed copies of aux.py. Dense cells are RGB, so
    # passing it is unnecessary there; pass it ONLY when it would change
    # anything, and only if the installed signature accepts it. Without this
    # the aux arm fails to build on a cluster whose aux.py predates the
    # parameter -- which is exactly what happened, while the baseline arm
    # built fine and would have produced a one-armed "envelope".
    if in_channels != 3:
        import inspect
        if "in_channels" not in inspect.signature(MomentAuxModel).parameters:
            raise RuntimeError(
                "this MomentAuxModel does not support in_channels; the "
                "deployed momentstem/aux.py is older than this file")
        kw["in_channels"] = in_channels
    return MomentAuxModel(net, target, **kw)
