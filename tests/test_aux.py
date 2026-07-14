"""MomentAuxModel contracts: deployed path is vanilla, the moment target is
fixed, and the auxiliary loss is present in train / absent in eval."""

import torch

from momentstem import build_model
from momentstem.aux import MomentAuxModel


def _model(target="energy-magnitude", tap="layer3", weight=0.1):
    return build_model(
        "resnet18", "none", num_classes=100,
        moment_aux={"stem": target, "tap": tap, "weight": weight},
    )


def test_deployed_path_is_vanilla():
    m = _model()
    assert isinstance(m, MomentAuxModel)
    # deployed forward == the bare backbone on RGB (no moment channels)
    assert m.stem.in_channels == 3 and m.stem.out_channels == 3
    x = torch.randn(2, 3, 32, 32)
    m.eval()
    with torch.no_grad():
        assert torch.allclose(m(x), m.net(x), atol=1e-6)
    assert m.last_aux is None  # eval never computes the aux loss


def test_aux_loss_present_in_train_and_trains_features_only():
    m = _model()
    m.calibrate(torch.randn(8, 3, 32, 32))
    m.train()
    x = torch.randn(4, 3, 32, 32)
    logits = m(x)
    assert m.last_aux is not None and m.last_aux.item() >= 0
    (torch.nn.functional.cross_entropy(logits, torch.randint(0, 100, (4,)))
     + m.aux_weight * m.last_aux).backward()
    # the fixed moment target receives no gradient
    assert all(p.grad is None for p in m.moment_stem.parameters())
    # the aux head does (it shapes the backbone features)
    assert m.aux_heads["layer3"].weight.grad is not None


def test_target_channel_count_matches_stem():
    for target, n in (("energy-magnitude", 8), ("moments-cat", 9)):
        kw = {"stem_kwargs": {"use_zernike": False}} if target == "moments-cat" else {}
        m = build_model("resnet18", "none", num_classes=10,
                        moment_aux={"stem": target, **kw})
        assert m.n_moment == n
        assert m.aux_heads["layer3"].out_channels == n


def test_cosine_loss_form():
    m = build_model("resnet18", "none", num_classes=10,
                    moment_aux={"stem": "energy-magnitude", "loss": "cosine", "weight": 0.3})
    assert m.loss_form == "cosine"
    m.calibrate(torch.randn(8, 3, 32, 32))
    m.train()
    m(torch.randn(2, 3, 32, 32))
    assert m.last_aux is not None and 0.0 <= m.last_aux.item() <= 2.0


def test_invariance_features_as_aux_targets():
    for t in ("energy-rotinv", "energy-structure", "energy-steerable", "energy-invariants"):
        m = build_model("resnet18", "none", num_classes=10, moment_aux={"stem": t})
        assert m.n_moment == m.aux_heads["layer3"].out_channels > 0


def test_moment_aux_rejects_forward_path_combos():
    import pytest
    with pytest.raises(ValueError):
        build_model("resnet18", "moments-cat", num_classes=10,
                    moment_aux={"stem": "energy-magnitude"})  # needs stem 'none'


def test_multi_layer_aux_taps():
    """A list of taps builds one head per layer and averages their losses."""
    m = build_model("resnet18", "none", num_classes=10,
                    moment_aux={"stem": "energy-magnitude",
                                "tap": ["layer2", "layer3", "layer4"], "weight": 0.3})
    assert set(m.aux_heads.keys()) == {"layer2", "layer3", "layer4"}
    m.calibrate(torch.randn(8, 3, 32, 32))
    m.train()
    m(torch.randn(2, 3, 32, 32))
    assert m.last_aux is not None and m.last_aux.item() >= 0
    # heads are sized to each tap's channel width (256/512 for r18 layer3/4)
    assert m.aux_heads["layer3"].in_channels == 256
    assert m.aux_heads["layer4"].in_channels == 512
