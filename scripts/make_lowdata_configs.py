"""Low-data hardening cells: the +1.4pt @1% result is the headline claim and
currently rests on 1 seed. This grid gives {none, gabor-only, full-cat,
learned} x {1%, 5%, 10%} x 3 seeds, ALL trained on one device (local 3090)
so hardware is not a confound. Existing cell names are reused where the cell
already exists (extra seeds extend them); 5% and the missing controls are new.

    python scripts/make_lowdata_configs.py
"""

import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RECIPE = dict(
    dataset="cifar100",
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

GABOR_ONLY = dict(stem_calibrate=True, stem_kwargs=dict(use_zernike=False))

CELLS = {
    # 1% -- extends abl1_none / abl1_gaboronly_calib / abl1_cat_calib
    "abl1_learned": dict(subset_pct=1, stem="learned"),
    # 5% -- all new
    "abl5_none": dict(subset_pct=5, stem="none"),
    "abl5_gaboronly_calib": dict(subset_pct=5, stem="moments-cat", **GABOR_ONLY),
    "abl5_cat_calib": dict(subset_pct=5, stem="moments-cat", stem_calibrate=True),
    "abl5_learned": dict(subset_pct=5, stem="learned"),
    # 10% -- same-hardware baseline (wave-1 abl_none ran on the H100)
    "abl10_none": dict(subset_pct=10, stem="none"),
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
    print(f"wrote {len(CELLS)} low-data configs to {CONFIG_DIR}")


if __name__ == "__main__":
    main()
