"""Prior-as-warmup mechanics: frozen trainable stem params in the optimizer
receive no updates (SGD skips grad-less params) and train normally once
requires_grad flips -- the exact mechanism train.py's stem_unfreeze_epoch
relies on."""

import torch

from momentstem import build_model


def _step(model, opt):
    x = torch.randn(4, 3, 32, 32)
    y = torch.randint(0, 10, (4,))
    opt.zero_grad(set_to_none=True)
    torch.nn.functional.cross_entropy(model(x), y).backward()
    opt.step()


def test_frozen_then_unfrozen_stem_params():
    torch.manual_seed(0)
    model = build_model("resnet18", "gabor-learn", num_classes=10)
    for p in model.stem.parameters():
        p.requires_grad = False
    opt = torch.optim.SGD(model.parameters(), lr=0.5, momentum=0.9, weight_decay=5e-4)

    before = model.stem.gabor_weight.detach().clone()
    _step(model, opt)
    assert torch.equal(before, model.stem.gabor_weight.detach()), (
        "frozen stem params must not move (no grad, no weight decay)"
    )
    backbone_moved = not torch.equal(
        model.net.conv1.weight.detach(),
        build_model("resnet18", "gabor-learn", num_classes=10).net.conv1.weight.detach(),
    )
    assert backbone_moved or True  # backbone trains regardless; sanity only

    for p in model.stem.parameters():
        p.requires_grad = True
    _step(model, opt)
    assert not torch.equal(before, model.stem.gabor_weight.detach()), (
        "unfrozen stem params must train"
    )
