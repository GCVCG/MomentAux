"""Ablation configs for diagnosing/fixing the moment stem (resnet18,
CIFAR-100 10%, full frozen recipe, seed 0). Two waves:

Wave 1 (diagnosis -- why did moments-cat trail the baseline at smoke scale?)
  isolates schedule length, response calibration, and the gabor bank choice.
Wave 2 (dissection -- generated after wave 1 picks a variant) splits the
  contribution by filter family, identity passthrough, and kernel size.

    python scripts/make_ablation_configs.py
"""

import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RECIPE = dict(
    dataset="cifar100",
    subset_pct=10,
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

WAVE1 = {
    # name -> overrides
    "abl_none": dict(stem="none"),
    "abl_cat_v1": dict(stem="moments-cat"),  # as smoke, but full schedule
    "abl_cat_calib": dict(stem="moments-cat", stem_calibrate=True),
    "abl_cat_calib_grid": dict(
        stem="moments-cat", stem_calibrate=True,
        stem_kwargs=dict(gabor_bank_type="grid"),
    ),
    "abl_learned": dict(stem="learned"),
    "abl_sum_calib": dict(stem="moments-sum", stem_calibrate=True),
}

WAVE2 = {
    # dissection of the calibrated concat stem
    "abl_gaboronly_calib_grid": dict(
        stem="moments-cat", stem_calibrate=True,
        stem_kwargs=dict(gabor_bank_type="grid", use_zernike=False),
    ),
    "abl_zernikeonly_calib": dict(
        stem="moments-cat", stem_calibrate=True,
        stem_kwargs=dict(use_gabor=False),
    ),
    "abl_noidentity_calib_grid": dict(
        stem="moments-cat", stem_calibrate=True,
        stem_kwargs=dict(gabor_bank_type="grid", include_identity=False),
    ),
    "abl_randomfixed_calib": dict(stem="random-fixed", stem_calibrate=True),
    "abl_gaborlearn_calib_grid": dict(
        stem="gabor-learn", stem_calibrate=True,
        stem_kwargs=dict(gabor_bank_type="grid"),
    ),
    "abl_cat_calib_grid_k5": dict(
        stem="moments-cat", stem_calibrate=True, stem_kernel_size=5,
        stem_kwargs=dict(gabor_bank_type="grid"),
    ),
    "abl_cat_calib_grid_k7": dict(
        stem="moments-cat", stem_calibrate=True, stem_kernel_size=7,
        stem_kwargs=dict(gabor_bank_type="grid"),
    ),
}

CONFIG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "ablations"
)


def main():
    os.makedirs(CONFIG_DIR, exist_ok=True)
    for name, overrides in {**WAVE1, **WAVE2}.items():
        cfg = {"name": name, **RECIPE, **overrides}
        with open(os.path.join(CONFIG_DIR, f"{name}.yaml"), "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
    print(f"wrote {len(WAVE1) + len(WAVE2)} ablation configs to {CONFIG_DIR}")


if __name__ == "__main__":
    main()
