"""Calibration-v2 (ZCA) validation cells.

The D1 forensics located the 10%-deficit mechanism: the std-calibrated stem
output is collinear (effective rank 7.7/12, gabor-vs-RGB |r| up to 0.66),
and re-allocating conv1 weight between collinear inputs is the slow mode of
SGD+weight-decay -- feasible in the 78k steps of full data, not in the 7.8k
steps of 10%. calibrate_zca folds a fixed whitening (rank 12.0, corr ~ 0)
into the stem kernels: deterministic, zero trainable params, invertible.

Prediction to falsify: ZCA removes (most of) the -1.4 deficit at 10% while
keeping the +1.9/+1.4 gains at 5%/1% and staying free at 100%.

    python scripts/make_zca_configs.py
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

CELLS = {}
for pct in (1, 5, 10, 100):
    CELLS[f"zca{pct}_gaboronly"] = dict(
        subset_pct=pct,
        stem="moments-cat",
        stem_calibrate="zca",
        stem_kwargs=dict(use_zernike=False),
    )

CONFIG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "configs", "diagnostics",
)


def main():
    os.makedirs(CONFIG_DIR, exist_ok=True)
    for name, overrides in CELLS.items():
        cfg = {"name": name, **RECIPE, **overrides}
        with open(os.path.join(CONFIG_DIR, f"{name}.yaml"), "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
    print(f"wrote {len(CELLS)} zca configs to {CONFIG_DIR}")


if __name__ == "__main__":
    main()
