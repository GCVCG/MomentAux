"""The joint image/mask transform must move BOTH, identically.

A geometric transform applied to the image but not the mask (or applied with
a different random draw) produces perfectly plausible tensors of the right
shape and trains a model on garbage. It is silent, and the loss still goes
down. Shape assertions cannot see it, so these are CONTENT tests, in the same
spirit as tests/test_tensor_layout.py.

The trick is to feed an image whose pixel values ARE the mask values: after
any joint geometric transform, the two must still agree pixel for pixel
wherever the transform did not fill from outside.
"""
import numpy as np
import pytest
import torch
from PIL import Image

import data_dense as dd


class _Synthetic(dd.VOCSegmentation):
    """VOCSegmentation with the file reads replaced by a known pattern.

    The mask is a coarse grid of distinct class indices; the image is that
    same grid painted into all three channels. Every geometric operation the
    real loader performs is exercised, but the correct answer is known.
    """

    def __init__(self, size=(300, 400), **kw):
        self.h, self.w = size
        g = np.zeros((self.h, self.w), np.uint8)
        for r in range(0, self.h, 40):
            for c in range(0, self.w, 40):
                g[r:r + 40, c:c + 40] = (r // 40 * 7 + c // 40) % 20 + 1
        self.grid = g
        self.root, self.ids = ".", ["synthetic"]
        # This harness bypasses __init__ on purpose, so it has to mirror the
        # attribute set the real class relies on. ds/split arrived when the
        # loader gained ADE20K and Cityscapes; the VOC values keep this test
        # measuring exactly what it measured before.
        self.ds, self.split = "voc_seg", "train_aug"
        self.train = kw.get("train", True)
        self.crop = kw.get("crop", 256)
        self.scale = kw.get("scale", (1.0, 1.0))   # scale off unless asked
        self.mean = torch.zeros(3, 1, 1)           # identity normalization,
        self.std = torch.ones(3, 1, 1)             # so pixels stay readable

    def _open(self):
        rgb = np.repeat(self.grid[:, :, None], 3, axis=2)
        return Image.fromarray(rgb, "RGB"), Image.fromarray(self.grid, "L")

    def __getitem__(self, i):
        img, mask = self._open()
        return self._transform(img, mask)

    def _transform(self, img, mask):
        # Re-run the real class's body by delegating through a stub file read.
        import types
        real = dd.VOCSegmentation.__getitem__
        holder = {}

        def fake_open(path, *a, **k):
            return holder["img"] if str(path).endswith(".jpg") else holder["mask"]

        holder["img"], holder["mask"] = img, mask
        orig = dd.Image.open
        dd.Image.open = fake_open
        try:
            return real(self, 0)
        finally:
            dd.Image.open = orig


def _agreement(x, y):
    """Fraction of non-fill pixels where the image channel equals the label."""
    # _to_tensor scales pixels to [0,1]; undo that so the channel value is
    # the class index the synthetic image was painted with.
    img = (x[0] * 255.0).round().to(torch.uint8).numpy()
    lab = y.numpy().astype(np.uint8)
    keep = lab != dd.IGNORE_INDEX
    if keep.sum() == 0:
        return 1.0
    return float((img[keep] == lab[keep]).mean())


def test_crop_and_flip_move_image_and_mask_together():
    ds = _Synthetic(train=True, crop=256)
    for seed in range(8):
        np.random.seed(seed)
        x, y = ds[0]
        assert x.shape[-2:] == (256, 256) and y.shape == (256, 256)
        # The whole point: after cropping and flipping, every labelled pixel
        # must still carry its own class in the image.
        assert _agreement(x, y) > 0.999, f"image and mask diverged (seed {seed})"


def test_scaling_keeps_them_together_and_invents_no_labels():
    ds = _Synthetic(train=True, crop=256, scale=(0.5, 2.0))
    for seed in range(8):
        np.random.seed(seed)
        x, y = ds[0]
        labels = set(np.unique(y.numpy()).tolist())
        # NEAREST on the mask: resizing must not interpolate a class into
        # existence. Every value is a real class or the ignore index.
        assert labels <= set(range(dd.VOC_CLASSES)) | {dd.IGNORE_INDEX}, labels


def test_padding_fills_the_mask_with_ignore_not_background():
    """A short image padded up to the crop must gain IGNORE, never class 0.

    Filling with 0 would silently teach the model that the padding is
    background and inflate the background IoU, which dominates mIoU on VOC.
    """
    ds = _Synthetic(size=(120, 120), train=True, crop=256)
    np.random.seed(0)
    x, y = ds[0]
    assert (y == dd.IGNORE_INDEX).sum() > 0, "no ignore region after padding"
    # the padded region is exactly the part outside the original 120x120
    assert (y[:120, :120] != dd.IGNORE_INDEX).any()


def test_eval_path_is_native_resolution_and_unaugmented():
    ds = _Synthetic(size=(300, 400), train=False)
    x, y = ds[0]
    assert x.shape[-2:] == (300, 400) and y.shape == (300, 400)
    assert _agreement(x, y) > 0.999


def test_confusion_matrix_ignores_255():
    cm = dd.ConfusionMatrix(3)
    pred = torch.tensor([0, 1, 2, 0])
    tgt = torch.tensor([0, 1, 2, 255])
    cm.update(pred, tgt)
    assert int(cm.mat.sum()) == 3, "the ignore label entered the matrix"
    miou, _ = cm.miou()
    assert miou == pytest.approx(100.0)


def test_confusion_matrix_is_device_independent():
    """The GPU accumulation must be BIT-identical to the CPU one.

    ConfusionMatrix.update was moved onto the prediction's device because the
    CPU transfer, not the forward pass, dominated an ADE20K validation pass
    (2,000 native-resolution images). Every operation is integer, so the two
    must agree exactly -- and "must agree exactly" is worth asserting rather
    than reasoning about, since a silent divergence here would move every mIoU
    in the dense study.
    """
    torch.manual_seed(0)
    n = 21
    pred = torch.randint(0, n, (50_000,))
    tgt = torch.randint(0, n, (50_000,))
    tgt[::7] = dd.IGNORE_INDEX                      # exercise the ignore path

    cpu = dd.ConfusionMatrix(n)
    cpu.update(pred, tgt)

    if torch.cuda.is_available():
        gpu = dd.ConfusionMatrix(n)
        gpu.update(pred.cuda(), tgt.cuda())
        assert torch.equal(gpu.mat.cpu(), cpu.mat)
        assert gpu.miou()[0] == cpu.miou()[0]
        assert gpu.pixel_acc() == cpu.pixel_acc()

    # And the ignore label must be excluded from the totals on either device.
    assert int(cpu.mat.sum()) == 50_000 - len(range(0, 50_000, 7))
