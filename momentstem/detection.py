"""Anchor-free detection head on the study's dilated ResNet-18.

WHAT THIS IS FOR. Classification pools the auxiliary target to 8x8; segmentation
compares it against features of the same spatial extent as the output. Detection
is neither: its output is sparse and localized, and it is the only task in the
study with a coordinate REGRESSION head. So it asks something the other two
cannot -- whether oriented-energy structure helps a network that must localize
rather than label.

THE HEAD IS DELIBERATELY MINIMAL, for the same reason FCNHead was chosen over
ASPP or an FPN: the claims come from a difference between two arms under one
recipe, and the least expressive head that can do the task is the one least able
to launder that difference through its own capacity. So: FCOS-style, anchor
free, SINGLE LEVEL at stride 8 -- the same stride, the same backbone and the
same tap (`backbone.layer3`) the segmentation cells use, so detection differs
from segmentation in the head and the loss and in nothing else.

THE COST OF SINGLE-LEVEL, STATED UP FRONT rather than discovered later: real
FCOS spreads objects over five pyramid levels by size, which is most of how it
handles scale. One level at stride 8 must regress every size from one feature
map, so absolute AP will be well below published VOC numbers. That is
acceptable here and would not be in a detection paper: both arms wear the
identical head, and only their difference is claimed.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

INF = 1e8


class FCOSHead(nn.Module):
    """Two 3x3 conv towers -> {class logits, centerness, ltrb regression}.

    The classification bias is initialized so that the initial foreground
    probability is ~0.01. Without it the focal loss starts with an enormous
    gradient from ~26k background locations per image and the first epochs are
    spent undoing that rather than learning.
    """

    def __init__(self, in_ch, n_classes, mid=128, n_conv=2, stride=8):
        super().__init__()
        self.stride = stride
        self.n_classes = n_classes

        def tower():
            layers = []
            c = in_ch
            for _ in range(n_conv):
                layers += [nn.Conv2d(c, mid, 3, padding=1, bias=False),
                           nn.GroupNorm(32, mid), nn.ReLU(inplace=True)]
                c = mid
            return nn.Sequential(*layers)

        self.cls_tower = tower()
        self.reg_tower = tower()
        self.cls_logits = nn.Conv2d(mid, n_classes, 3, padding=1)
        self.centerness = nn.Conv2d(mid, 1, 3, padding=1)
        self.bbox_pred = nn.Conv2d(mid, 4, 3, padding=1)
        # learnable per-level scale on the regression output, as in FCOS
        self.scale = nn.Parameter(torch.tensor(1.0))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.constant_(self.cls_logits.bias, -math.log((1 - 0.01) / 0.01))

    def forward(self, f):
        c = self.cls_tower(f)
        r = self.reg_tower(f)
        # exp() keeps ltrb positive; scale lets the net choose the range rather
        # than inheriting whatever the stride implies.
        return (self.cls_logits(c), self.centerness(c),
                torch.exp(self.scale * self.bbox_pred(r)))


def locations(h, w, stride, device):
    ys = torch.arange(h, device=device, dtype=torch.float32) * stride + stride // 2
    xs = torch.arange(w, device=device, dtype=torch.float32) * stride + stride // 2
    y, x = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([x.reshape(-1), y.reshape(-1)], dim=1)


def assign_targets(locs, boxes, labels, n_classes, radius=1.5, stride=8):
    """FCOS assignment: a location is positive if it lies inside a box AND
    within `radius`*stride of that box's centre; ties go to the smaller box.

    The centre-sampling radius matters more at one level than at five: without
    it, every location inside a large box becomes positive and the regression
    targets at box edges dominate. Returned targets are (cls, ltrb, centerness).
    """
    n_loc = locs.shape[0]
    dev = locs.device
    cls_t = torch.full((n_loc,), n_classes, dtype=torch.int64, device=dev)  # bg
    reg_t = torch.zeros((n_loc, 4), dtype=torch.float32, device=dev)
    ctr_t = torch.zeros((n_loc,), dtype=torch.float32, device=dev)
    if boxes.numel() == 0:
        return cls_t, reg_t, ctr_t

    xs, ys = locs[:, 0:1], locs[:, 1:2]
    l = xs - boxes[:, 0].unsqueeze(0)
    t = ys - boxes[:, 1].unsqueeze(0)
    r = boxes[:, 2].unsqueeze(0) - xs
    b = boxes[:, 3].unsqueeze(0) - ys
    ltrb = torch.stack([l, t, r, b], dim=2)               # (n_loc, n_box, 4)
    inside = ltrb.min(dim=2).values > 0

    cx = (boxes[:, 0] + boxes[:, 2]) / 2
    cy = (boxes[:, 1] + boxes[:, 3]) / 2
    near = ((xs - cx.unsqueeze(0)).abs() < radius * stride) & \
           ((ys - cy.unsqueeze(0)).abs() < radius * stride)
    pos = inside & near

    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    area = area.unsqueeze(0).expand(n_loc, -1).clone()
    area[~pos] = INF
    min_area, min_idx = area.min(dim=1)
    has = min_area < INF
    if has.any():
        sel = min_idx[has]
        cls_t[has] = labels[sel]
        reg_t[has] = ltrb[has, sel]
        lt = reg_t[has]
        lr_ = torch.stack([lt[:, 0], lt[:, 2]], 1)
        tb = torch.stack([lt[:, 1], lt[:, 3]], 1)
        ctr_t[has] = torch.sqrt(
            (lr_.min(1).values / lr_.max(1).values.clamp(min=1e-6)) *
            (tb.min(1).values / tb.max(1).values.clamp(min=1e-6)))
    return cls_t, reg_t, ctr_t


def giou_loss(pred, target, weight):
    """GIoU on ltrb, weighted by centerness -- FCOS's own formulation."""
    pl, pt, pr, pb = pred.unbind(1)
    tl, tt, tr, tb = target.unbind(1)
    pa = (pl + pr) * (pt + pb)
    ta = (tl + tr) * (tt + tb)
    iw = torch.min(pl, tl) + torch.min(pr, tr)
    ih = torch.min(pt, tt) + torch.min(pb, tb)
    inter = (iw.clamp(min=0) * ih.clamp(min=0))
    union = pa + ta - inter
    iou = inter / union.clamp(min=1e-6)
    ew = torch.max(pl, tl) + torch.max(pr, tr)
    eh = torch.max(pt, tt) + torch.max(pb, tb)
    enc = (ew * eh).clamp(min=1e-6)
    giou = iou - (enc - union) / enc
    return ((1 - giou) * weight).sum() / weight.sum().clamp(min=1e-6)


class DetResNet(nn.Module):
    """Dilated ResNet-18 at output stride 8 + FCOSHead.

    `backbone.layerN` naming matches SegResNet and the classification models, so
    a tap string reads the same here as everywhere else and MomentAuxModel's
    hook attaches unchanged.
    """

    def __init__(self, backbone="resnet18", n_classes=20, output_stride=8,
                 pretrained=False, in_channels=3):
        super().__init__()
        import timm
        net = timm.create_model(backbone, pretrained=pretrained, num_classes=0,
                                in_chans=in_channels, output_stride=output_stride)
        self.backbone = net
        self.stride = output_stride
        self.head = FCOSHead(net.num_features, n_classes, stride=output_stride)
        self.n_classes = n_classes

    def features(self, x):
        return self.backbone.forward_features(x)

    def forward(self, x):
        return self.head(self.features(x))


def build_det_model(backbone="resnet18", n_classes=20, output_stride=8,
                    pretrained=False, moment_aux=None, image_size=512,
                    in_channels=3):
    """Same wrapper the other two tasks use, imported rather than reimplemented
    so a detection-vs-segmentation difference cannot be an artifact of a second
    implementation of the prior."""
    net = DetResNet(backbone, n_classes, output_stride, pretrained, in_channels)
    if not moment_aux:
        return net
    from .aux import MomentAuxModel, MomentTarget
    from .controls import build_stem
    target = MomentTarget(build_stem(
        moment_aux["stem"],
        in_channels=in_channels,
        kernel_size=moment_aux.get("kernel_size", 11),
        seed=moment_aux.get("stem_seed", 0),
        **(moment_aux.get("stem_kwargs") or {}),
    ))
    return MomentAuxModel(
        net, target,
        tap=moment_aux.get("tap", "backbone.layer3"),
        aux_weight=moment_aux.get("weight", 1.0),
        loss_form=moment_aux.get("loss", "mse"),
        head_norm=moment_aux.get("head_norm", True),
        image_size=image_size,
    )
