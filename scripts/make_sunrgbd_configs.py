"""Generate the SUN RGB-D sensor-fusion configs (limitations campaign block H,
2026-08-23): configs/sensorfusion/sf_sunrgbd_{rgb,depth,all}_{none,aux}_{pct}pct.yaml
copied from sf_so2sat_<src>_<arm>_<pct>pct.yaml VERBATIM except `name`,
`dataset` and the header comment, plus the 100% cells (subset_pct: 100, the
same form build_dataset treats as "no subset") and the ViT-tiny pairs at 10%
derived from diagsfvit_eurosatms_* (AdamW diag namespace, tap blocks.8).

FRACTIONS ARE 3/5/10/25/100, NOT THE PRE-REGISTERED 1/2/5/10/25/100: SUN
RGB-D has 4,845 training frames over 19 classes, so 1% leaves ZERO samples
for the smallest classes (make_subset_indices refuses) and 2% is ~97 images,
below one batch of 128 under drop_last -- the same floor that makes stl10/cub
@1-2% impossible. 3% (~145 images, one batch/epoch) is the scarce end that
can train. Recorded as a deviation in CLAUDE.md.

Run: python scripts/make_sunrgbd_configs.py   (idempotent; prints a diff count)
"""
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
CFG = os.path.join(HERE, "..", "configs", "sensorfusion")
SRC_HEAD = {
    "rgb": "SOURCE A: RGB appearance, 3 channels.",
    "depth": "SOURCE B: raw depth (geometry), 1 channel -- toolbox decode, clip 10 m, [0,1], 0 = missing.",
    "all": "SENSOR-FUSED: RGB + depth, 4 channels.",
}


def r18(src, arm, pct):
    tmpl_pct = pct if pct in (1, 2, 5, 10, 25) else 25
    t = open(os.path.join(CFG, f"sf_so2sat_sar_{arm}_{tmpl_pct}pct.yaml")).read()
    body = t.split("\n", 4)[4]   # drop the 4 header comment lines
    body = body.replace(f"name: sf_so2sat_sar_{arm}_{tmpl_pct}pct",
                        f"name: sf_sunrgbd_{src}_{arm}_{pct}pct")
    body = body.replace("dataset: so2sat_sar", f"dataset: sunrgbd_{src}")
    body = body.replace(f"subset_pct: {tmpl_pct}", f"subset_pct: {pct}")
    head = (f"# SUN RGB-D scene classification (19 classes, official split): the\n"
            f"# third multi-source population -- appearance vs geometry. {SRC_HEAD[src]}\n"
            "# ResNet-18 under the FROZEN recipe (SGD 0.1, cosine, 200 ep, batch 128,\n"
            "# crop+flip): the only deviation from a headline cell is the input source,\n"
            "# so these are directly comparable to each other.\n")
    return head + body


def vit(src, arm):
    t = open(os.path.join(CFG, f"diagsfvit_eurosatms_nir_{arm}_10pct.yaml")).read()
    body = t.split("\n", 3)[3]   # drop the 3 header comment lines
    body = body.replace(f"name: diagsfvit_eurosatms_nir_{arm}_10pct",
                        f"name: diagsf_sunrgbd_{src}_vit_{arm}_10pct")
    body = body.replace("dataset: eurosatms_nir", f"dataset: sunrgbd_{src}")
    head = ("# SUN RGB-D sensor fusion on ViT-tiny (64px => patch 8, 8x8 tokens, tap\n"
            "# blocks.8) -- is the source-level result specific to ResNet-18? AdamW =>\n"
            f"# diagnostic namespace. {SRC_HEAD[src]}\n")
    return head + body


def main():
    written = changed = 0
    for src in ("rgb", "depth", "all"):
        for arm in ("none", "aux"):
            for pct in (3, 5, 10, 25, 100):
                p = os.path.join(CFG, f"sf_sunrgbd_{src}_{arm}_{pct}pct.yaml")
                txt = r18(src, arm, pct)
                assert f"subset_pct: {pct}\n" in txt and f"dataset: sunrgbd_{src}\n" in txt
                old = open(p).read() if os.path.exists(p) else None
                open(p, "w").write(txt)
                written += 1
                changed += old != txt
            p = os.path.join(CFG, f"diagsf_sunrgbd_{src}_vit_{arm}_10pct.yaml")
            txt = vit(src, arm)
            assert f"dataset: sunrgbd_{src}\n" in txt and "subset_pct: 10\n" in txt
            old = open(p).read() if os.path.exists(p) else None
            open(p, "w").write(txt)
            written += 1
            changed += old != txt
    print(f"wrote {written} configs ({changed} changed)")


if __name__ == "__main__":
    main()
