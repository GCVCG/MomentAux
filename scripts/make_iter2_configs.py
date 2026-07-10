"""Iteration-2 configs, driven by the full-data ablation findings:

- champion confirmation: gabor-only calibrated needs more seeds at full data
- zernike pruned, not dropped: keep the top-5 conv1-usage earners measured
  on ablf_cat_calib (Z1/Z2 tilts, Z3 oblique astig, Z7 vertical coma,
  Z11 oblique secondary astig) -> 17-channel stem
- generality: champion vs baseline on resnet50, full data
- low-data: champion at 10% and 1% (plus the missing 1% baseline), where
  the prior's early-lead phenomenon lives

    python scripts/make_iter2_configs.py
"""

import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RECIPE = dict(
    dataset="cifar100",
    epochs=200,
    batch_size=128,
    lr=0.1,
    weight_decay=5.0e-4,
    momentum=0.9,
    num_workers=8,
    small_input=True,
    pretrained=False,
    stem_kernel_size=11,
)

GABOR_ONLY = dict(stem_calibrate=True, stem_kwargs=dict(use_zernike=False))
PRUNED = dict(
    stem_calibrate=True, stem_kwargs=dict(zernike_indices=[1, 2, 3, 7, 11])
)

CELLS = {
    # zernike pruned to top-5 earners (17ch), full data
    "ablf_catpruned_calib": dict(
        subset_pct=100, backbone="resnet18", stem="moments-cat", **PRUNED
    ),
    # resnet50 generality, full data
    "ablf50_none": dict(subset_pct=100, backbone="resnet50", stem="none"),
    "ablf50_gaboronly_calib": dict(
        subset_pct=100, backbone="resnet50", stem="moments-cat", **GABOR_ONLY
    ),
    # low-data: champion + missing baselines
    "abl10_gaboronly_calib": dict(
        subset_pct=10, backbone="resnet18", stem="moments-cat", **GABOR_ONLY
    ),
    "abl1_gaboronly_calib": dict(
        subset_pct=1, backbone="resnet18", stem="moments-cat", **GABOR_ONLY
    ),
    "abl1_none": dict(subset_pct=1, backbone="resnet18", stem="none"),
    "abl1_cat_calib": dict(
        subset_pct=1, backbone="resnet18", stem="moments-cat", stem_calibrate=True
    ),
}

CONFIG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "configs", "ablations_full",
)


def main():
    os.makedirs(CONFIG_DIR, exist_ok=True)
    for name, overrides in CELLS.items():
        cfg = {"name": name, **RECIPE, **overrides}
        with open(os.path.join(CONFIG_DIR, f"{name}.yaml"), "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
    print(f"wrote {len(CELLS)} iter-2 configs to {CONFIG_DIR}")


if __name__ == "__main__":
    main()
