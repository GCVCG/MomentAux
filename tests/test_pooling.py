"""MultiMaskPool contracts: GAP nesting, fixedness per mask type, shapes,
and head accounting through build_model."""

import pytest
import torch

from momentstem import build_model
from momentstem.pooling import MultiMaskPool, make_masks


def test_mask_zero_is_uniform_gap_for_every_type():
    for t in ("zernike", "random", "learned"):
        m = make_masks(t, hw=4)
        assert torch.allclose(m[0], torch.full((4, 4), 0.25)), f"{t}: mask 0 must be GAP"


def test_pool_output_shape_and_gap_nesting():
    pool = MultiMaskPool("random", hw=4, J=8)
    x = torch.randn(2, 512, 4, 4)
    out = pool(x)
    assert out.shape == (2, 512 * 8)
    # channel c's first slot is GAP of channel c up to the unit-L2 scale
    # (uniform mask has value 1/hw, so slot0 = hw * mean -- same information)
    gap = x.mean(dim=(2, 3))
    assert torch.allclose(out.view(2, 512, 8)[:, :, 0], 4.0 * gap, atol=1e-4)
    with pytest.raises(ValueError):
        pool(torch.randn(2, 512, 8, 8))  # wrong spatial size must be loud


def test_fixed_types_are_buffers_learned_is_parameter():
    for t in ("zernike", "random"):
        pool = MultiMaskPool(t, hw=4)
        assert sum(p.numel() for p in pool.parameters()) == 0
    learned = MultiMaskPool("learned", hw=4)
    assert sum(p.numel() for p in learned.parameters()) == 8 * 16


def test_build_model_head_pool_wiring():
    ref = build_model("resnet18", "none", num_classes=100)
    for t in ("zernike", "random", "learned"):
        m = build_model("resnet18", "none", num_classes=100,
                        head_pool={"type": t, "J": 8, "hw": 4})
        out = m(torch.randn(2, 3, 32, 32))
        assert out.shape == (2, 100)
        assert m.net.fc.in_features == 512 * 8
    # head widening is identical across types; learned adds only mask params
    n = lambda mm: sum(p.numel() for p in mm.parameters() if p.requires_grad)
    nz = n(build_model("resnet18", "none", num_classes=100, head_pool={"type": "zernike"}))
    nr = n(build_model("resnet18", "none", num_classes=100, head_pool={"type": "random"}))
    nl = n(build_model("resnet18", "none", num_classes=100, head_pool={"type": "learned"}))
    assert nz == nr and nl == nr + 8 * 16
    assert nr - n(ref) == 512 * 7 * 100  # widened fc only


def test_fixed_masks_survive_training_step():
    model = build_model("resnet18", "none", num_classes=10,
                        head_pool={"type": "zernike"})
    before = model.net.global_pool.masks.clone()
    opt = torch.optim.SGD(model.parameters(), lr=0.5, momentum=0.9)
    loss = torch.nn.functional.cross_entropy(
        model(torch.randn(4, 3, 32, 32)), torch.randint(0, 10, (4,))
    )
    loss.backward()
    opt.step()
    assert torch.equal(before, model.net.global_pool.masks)
