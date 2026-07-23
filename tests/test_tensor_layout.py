"""Tensor-layout content assertions. The previous project shipped SSIM on
channel-scrambled tensors because `.reshape(1,3,H,W)` was applied to HWC
data. Every HWC->CHW crossing in this repo is tested for CONTENT, not just
shape: distinct per-channel constants must land in the right channels with
the right values."""

import numpy as np
import pytest
import torch
from PIL import Image

from data import STATS, CIFARCorrupted, build_transforms

CH_VALUES = (10, 100, 200)  # distinct constants for R, G, B


def _fake_cifar_c(tmp_path, n_per_severity=4):
    n = n_per_severity * 5
    images = np.zeros((n, 32, 32, 3), dtype=np.uint8)
    for c, v in enumerate(CH_VALUES):
        images[..., c] = v
    # severity is recoverable from content: add it to the red channel
    for s in range(5):
        images[s * n_per_severity:(s + 1) * n_per_severity, ..., 0] += s
    labels = np.arange(n) % 10
    np.save(tmp_path / "gaussian_noise.npy", images)
    np.save(tmp_path / "labels.npy", labels)
    return tmp_path


def test_cifar_c_hwc_to_chw_content(tmp_path):
    root = _fake_cifar_c(tmp_path)
    mean, std = STATS["cifar100"]
    for severity in (1, 3, 5):
        ds = CIFARCorrupted(str(root), "gaussian_noise", severity, dataset="cifar100")
        x, y = ds[0]
        assert x.shape == (3, 32, 32)
        for c, v in enumerate(CH_VALUES):
            raw = v + (severity - 1 if c == 0 else 0)
            expected = (raw / 255.0 - mean[c]) / std[c]
            assert torch.allclose(x[c], torch.full((32, 32), expected), atol=1e-5), (
                f"channel {c} content wrong at severity {severity}: layout scrambled?"
            )
        # labels.npy cycles 0..9; item 0 of severity s is global index 4*(s-1)
        assert y == (4 * (severity - 1)) % 10


def test_cifar_c_severity_slicing(tmp_path):
    root = _fake_cifar_c(tmp_path)
    reds = []
    for severity in range(1, 6):
        ds = CIFARCorrupted(str(root), "gaussian_noise", severity)
        assert len(ds) == 4
        # recover the raw red value; severities must not overlap
        mean, std = STATS["cifar100"]
        raw = round((ds[0][0][0, 0, 0].item() * std[0] + mean[0]) * 255)
        reds.append(raw)
    assert reds == [10, 11, 12, 13, 14]


@pytest.mark.parametrize("dataset,size", [("cifar100", 32), ("stl10", 96)])
def test_eval_transform_hwc_to_chw_content(dataset, size):
    arr = np.zeros((size, size, 3), dtype=np.uint8)
    for c, v in enumerate(CH_VALUES):
        arr[..., c] = v
    x = build_transforms(dataset, train=False)(Image.fromarray(arr))
    mean, std = STATS[dataset]
    assert x.shape == (3, size, size)
    for c, v in enumerate(CH_VALUES):
        expected = (v / 255.0 - mean[c]) / std[c]
        assert torch.allclose(x[c], torch.full((size, size), expected), atol=1e-5)


def test_train_transform_layout_and_range():
    size = 32
    arr = np.zeros((size, size, 3), dtype=np.uint8)
    for c, v in enumerate(CH_VALUES):
        arr[..., c] = v
    x = build_transforms("cifar100", train=True)(Image.fromarray(arr))
    assert x.shape == (3, size, size)
    mean, std = STATS["cifar100"]
    for c, v in enumerate(CH_VALUES):
        expected = (v / 255.0 - mean[c]) / std[c]
        pad_value = (0.0 - mean[c]) / std[c]  # RandomCrop pads with zeros
        vals = torch.unique(x[c])
        for u in vals:
            assert (
                torch.isclose(u, torch.tensor(expected), atol=1e-5)
                or torch.isclose(u, torch.tensor(pad_value), atol=1e-5)
            ), f"channel {c} contains foreign value {u.item()}: channels mixed?"


def test_swin_nhwc_to_spatial_content():
    """Swin taps are (B,H,W,C); _to_spatial must permute so that the value at
    [b, h, w, c] lands at [b, c, h, w] -- a content check, not a shape check
    (a wrong permute can produce the right shape with scrambled pixels)."""
    import torch

    from momentstem.aux import _to_spatial

    b, h, w, c = 2, 4, 4, 384
    x = torch.arange(b * h * w * c, dtype=torch.float32).reshape(b, h, w, c)
    y = _to_spatial(x)
    assert y.shape == (b, c, h, w)
    for bi in (0, 1):
        for hi in (0, 3):
            for wi in (1, 2):
                for ci in (0, 100, 383):
                    assert y[bi, ci, hi, wi] == x[bi, hi, wi, ci]
    # a genuine NCHW conv tap must pass through UNTOUCHED
    conv = torch.randn(2, 256, 8, 8)
    assert _to_spatial(conv) is conv
    # ...including square NCHW where C > H (dim1 != dim2 protects it)
    conv2 = torch.randn(2, 256, 4, 8)
    assert _to_spatial(conv2) is conv2
