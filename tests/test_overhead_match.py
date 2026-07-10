"""The 'learned' control is DEFINED by matching the moments-cat overhead.
Assert params and fvcore FLOPs within 2%."""

import torch
from fvcore.nn import FlopCountAnalysis

from momentstem import LearnedStem, MomentStem, build_model, build_stem


def _flops(module, size=32):
    return FlopCountAnalysis(module.eval(), torch.zeros(1, 3, size, size)) \
        .unsupported_ops_warnings(False).total()


def test_learned_params_match_moments_cat_filters():
    moments = build_stem("moments-cat")
    learned = build_stem("learned")
    assert isinstance(learned, LearnedStem)
    target = moments.filter_numel()
    actual = sum(p.numel() for p in learned.parameters())
    ratio = actual / target
    assert abs(ratio - 1) < 0.02, f"param mismatch {ratio:.4f} (target {target}, got {actual})"


def test_learned_flops_match_moments_cat():
    for size in (32, 96):
        f_m = _flops(build_stem("moments-cat"), size)
        f_l = _flops(build_stem("learned"), size)
        ratio = f_l / f_m
        assert abs(ratio - 1) < 0.02, f"FLOP mismatch at {size}px: {ratio:.4f}"


def test_full_model_flops_close():
    """Downstream of the stem the models are identical (same in_chans conv1),
    so full-model FLOPs must also agree within 2%."""
    f_m = _flops(build_model("resnet18", "moments-cat", num_classes=100))
    f_l = _flops(build_model("resnet18", "learned", num_classes=100))
    assert abs(f_l / f_m - 1) < 0.02


def test_sum_stem_adds_zero_trainable_params_to_backbone():
    vanilla = build_model("resnet18", "none", num_classes=100)
    summed = build_model("resnet18", "moments-sum", num_classes=100)
    n = lambda m: sum(p.numel() for p in m.parameters() if p.requires_grad)
    assert n(vanilla) == n(summed), "sum stem must not change trainable params"
