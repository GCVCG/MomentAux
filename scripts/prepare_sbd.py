"""Convert SBD's MATLAB segmentation masks to indexed PNGs, once.

The augmented VOC training set (10,582 images) is VOC's own 1,464 plus SBD's
annotations. SBD ships them as .mat files, which is a format the loader has no
business knowing about, and decoding one per __getitem__ would put a scipy
call and a MATLAB struct traversal inside the training loop.

WHY THIS IS A SEPARATE PREPARE STEP rather than a fallback in the loader:
9,118 of the 10,582 ids are SBD-only. A loader that quietly tried .png and
then .mat would work, and would also hide the case where NEITHER exists --
which is exactly how a run ends up training on a silently smaller dataset.
Converting up front means a missing mask is a missing file at prepare time,
loudly, before any GPU is booked.

    python scripts/prepare_sbd.py --root data/sbd

Writes data/sbd/cls_png/<id>.png as mode-P images with the VOC palette, so
they are byte-compatible with VOC's own SegmentationClass masks: same class
indices, same 255 ignore, readable by the same np.array() call.
"""
import argparse
import os

import numpy as np
from PIL import Image


def voc_palette():
    """The standard VOC colour map, so the PNGs look right in an image viewer.

    Only the INDICES matter to training -- the loader reads them with
    np.array() and never touches the palette -- but a mask that renders as
    black-on-black is much harder to eyeball when something goes wrong.
    """
    palette = np.zeros((256, 3), np.uint8)
    for i in range(256):
        c, r, g, b = i, 0, 0, 0
        for j in range(8):
            r |= ((c >> 0) & 1) << (7 - j)
            g |= ((c >> 1) & 1) << (7 - j)
            b |= ((c >> 2) & 1) << (7 - j)
            c >>= 3
        palette[i] = (r, g, b)
    return palette.flatten().tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/sbd")
    args = ap.parse_args()
    import scipy.io as sio

    src = os.path.join(args.root, "cls")
    dst = os.path.join(args.root, "cls_png")
    if not os.path.isdir(src):
        raise SystemExit(f"{src} missing -- fetch SBD first (see data_dense.py)")
    os.makedirs(dst, exist_ok=True)
    pal = voc_palette()

    mats = sorted(f for f in os.listdir(src) if f.endswith(".mat"))
    written = skipped = 0
    for f in mats:
        out = os.path.join(dst, f[:-4] + ".png")
        if os.path.exists(out):
            skipped += 1
            continue
        m = sio.loadmat(os.path.join(src, f), struct_as_record=False,
                        squeeze_me=True)
        seg = np.asarray(m["GTcls"].Segmentation, dtype=np.uint8)
        img = Image.fromarray(seg, mode="P")
        img.putpalette(pal)
        img.save(out)
        written += 1
    print(f"{len(mats)} SBD masks: {written} written, {skipped} already present")

    # Verify rather than assume: re-read a few and check the indices survive.
    for f in mats[:5]:
        a = np.array(Image.open(os.path.join(dst, f[:-4] + ".png")))
        m = sio.loadmat(os.path.join(src, f), struct_as_record=False,
                        squeeze_me=True)
        b = np.asarray(m["GTcls"].Segmentation, dtype=np.uint8)
        assert (a == b).all(), f"{f}: png does not round-trip the .mat indices"
    print("round-trip verified on 5 masks")


if __name__ == "__main__":
    main()
