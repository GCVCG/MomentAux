"""FULL-data ablations: CIFAR-100 100%, full frozen 200-epoch recipe,
resnet18. This is the 'go deeper into the moments' battery -- no small-data
shortcuts; every cell trains on all 50k images.

Cells:
  ablf_none              vanilla baseline
  ablf_cat_calib         calibrated moment stem (current best variant)
  ablf_learned           param/FLOP-matched plain conv control
  ablf_randomfixed_calib frozen random filters, calibrated (structure test)
  ablf_gaboronly_calib   which family carries the value?
  ablf_zernikeonly_calib
  ablf_noidentity_calib  does the RGB passthrough matter?
  ablf_gaborlearn_calib  is fixing a feature or a bug at full data?

    python scripts/make_fulldata_ablation_configs.py
"""

import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RECIPE = dict(
    dataset="cifar100",
    subset_pct=100,
    backbone="resnet18",
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

CELLS = {
    "ablf_none": dict(stem="none"),
    "ablf_cat_calib": dict(stem="moments-cat", stem_calibrate=True),
    "ablf_learned": dict(stem="learned"),
    "ablf_randomfixed_calib": dict(stem="random-fixed", stem_calibrate=True),
    "ablf_gaboronly_calib": dict(
        stem="moments-cat", stem_calibrate=True, stem_kwargs=dict(use_zernike=False)
    ),
    "ablf_zernikeonly_calib": dict(
        stem="moments-cat", stem_calibrate=True, stem_kwargs=dict(use_gabor=False)
    ),
    "ablf_noidentity_calib": dict(
        stem="moments-cat", stem_calibrate=True,
        stem_kwargs=dict(include_identity=False),
    ),
    "ablf_gaborlearn_calib": dict(stem="gabor-learn", stem_calibrate=True),
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
    print(f"wrote {len(CELLS)} full-data ablation configs to {CONFIG_DIR}")


if __name__ == "__main__":
    main()
