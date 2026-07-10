"""Generate one YAML config per experimental cell (run after any grid change,
then commit configs/). The recipe fields are identical in every file by
construction -- the config only selects the cell.

    python scripts/make_configs.py
"""

import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

STEMS = ("none", "moments-sum", "moments-cat", "learned", "random-fixed", "gabor-learn")
BACKBONES = ("resnet18", "resnet34", "resnet50")
CIFAR_SUBSETS = (1, 5, 10, 25, 100)

RECIPE = dict(
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

CONFIG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs")


def write(name, **cell):
    cfg = {"name": name, **cell, **RECIPE}
    path = os.path.join(CONFIG_DIR, f"{name}.yaml")
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    return path


def main():
    os.makedirs(CONFIG_DIR, exist_ok=True)
    n = 0
    for backbone in BACKBONES:
        for stem in STEMS:
            for pct in CIFAR_SUBSETS:
                name = f"cifar100_{backbone}_{stem}_{pct}pct"
                write(name, dataset="cifar100", subset_pct=pct, backbone=backbone, stem=stem)
                n += 1
            name = f"stl10_{backbone}_{stem}"
            write(name, dataset="stl10", subset_pct=None, backbone=backbone, stem=stem)
            n += 1
    # Milestone-2 smoke cell (epochs overridden on the command line).
    for stem in ("none", "moments-cat"):
        write(f"smoke_cifar100_resnet18_{stem}_10pct", dataset="cifar100",
              subset_pct=10, backbone="resnet18", stem=stem)
        n += 1
    print(f"wrote {n} configs to {CONFIG_DIR}")


if __name__ == "__main__":
    main()
