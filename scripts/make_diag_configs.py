"""Diagnostic configs for the 10%-deficit investigation (option 3).

Evidence from existing runs (see analysis in git history / final report):
- at 10% the champion LEADS until ~ep40 then collapses ep40-80 as both nets
  reach memorization, recovering only partially (-1.4 final)
- conv1's gabor/identity usage ratio ends at 0.59 at 1/5/10% but 0.27 at
  100%: the net sheds the prior when it has enough STEPS (78k at full data
  vs 7.8k at 10%), and 10% is where it needs to shed but cannot.

Two testable consequences:
1. crossover map: Delta(data) should cross zero between 5%% and 10%% and
   shrink again toward 25%% as steps grow -> cells at 2/3/7/15%%.
2. steps hypothesis: at 10%%, a 400-epoch cosine (2x steps) should shrink
   the deficit; a 100-epoch cosine (0.5x steps) shifts the balance toward
   the prior. Diagnostic-only cells; the frozen 200-ep recipe remains the
   headline protocol.

    python scripts/make_diag_configs.py
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

GABOR_ONLY = dict(
    stem="moments-cat", stem_calibrate=True, stem_kwargs=dict(use_zernike=False)
)

CELLS = {}
# crossover map
for pct in (2, 3, 7, 15):
    CELLS[f"abl{pct}_none"] = dict(subset_pct=pct, stem="none")
    CELLS[f"abl{pct}_gaboronly_calib"] = dict(subset_pct=pct, **GABOR_ONLY)
# steps hypothesis at 10%
for ep in (100, 400):
    CELLS[f"diag10e{ep}_none"] = dict(subset_pct=10, stem="none", epochs=ep)
    CELLS[f"diag10e{ep}_gaboronly_calib"] = dict(subset_pct=10, epochs=ep, **GABOR_ONLY)

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
    print(f"wrote {len(CELLS)} diagnostic configs to {CONFIG_DIR}")


if __name__ == "__main__":
    main()
