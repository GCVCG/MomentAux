"""Generate a config for EVERY missing (configuration x data-portion) cell.

    python scripts/make_grid_configs.py [--out configs/grid] [--dry-run]

User decision 2026-07-22: "no missing data for any configuration". Every one of
the study's configurations is expanded to the full fraction grid. This script
is the reproducible record of how those cells were created -- the configs it
writes are committed, exactly like hand-written ones.

Naming: `grid_<slug>_<pct>pct`, or `diaggrid_<slug>_<pct>pct` when the
configuration deviates from the frozen recipe (AdamW / custom head / custom
init / extra augmentation / non-200 epochs), because train.py refuses to run
those without a `diag` prefix. Existing cells keep their historical names; the
exporters group by config FIELDS, not by name, so old and new cells land on the
same row of the pivot.

IMPOSSIBLE cells are skipped, not emitted: with batch 128 and drop_last=True a
subset below 128 images yields ZERO batches and the loader is empty
(stl10/cub @1-2%, tin20/tin20b @1%). Those stay blank in the tables by
necessity, and the exporters mark them "n/a (<1 batch)".
"""

import argparse
import hashlib
import json
import os
import re
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "analysis"))

import data as data_mod
from export_results_csv import cfgget, config_key, family_key, is_baseline, load_cells

GRID = (1, 2, 3, 5, 7, 10, 15, 20, 25, 50, 100)
BATCH = 128
DS_SHORT = {"cifar100": "c100", "cifar100super": "c100s", "cifar10": "c10",
            "stl10": "stl", "tin": "tin", "tinsuper": "tsuper",
            "tinsem": "tsem", "tin20": "tin20", "tin20b": "tin20b", "cub": "cub"}
BB_SHORT = {"resnet18": "r18", "resnet34": "r34", "resnet50": "r50",
            "convnext_tiny": "cnx", "vit_tiny": "vit"}


def n_images(ds, pct):
    total = {"cifar100": 50000, "cifar100super": 50000, "cifar10": 50000,
             "stl10": 5000, "tin": 100000, "tinsuper": 100000,
             "tinsem": 100000, "tin20": 10000, "tin20b": 10000,
             "cub": 5994}.get(ds, 0)
    return int(round(total * pct / 100.0))


def slug(cfg):
    aux = cfgget(cfg, "moment_aux") or {}
    bits = [DS_SHORT.get(cfgget(cfg, "dataset"), str(cfgget(cfg, "dataset"))),
            BB_SHORT.get(cfgget(cfg, "backbone"), str(cfgget(cfg, "backbone")))]
    opt = (cfgget(cfg, "optimizer", "sgd") or "sgd").lower()
    if opt != "sgd":
        bits.append(opt)
    ep = cfgget(cfg, "epochs", 200)
    if ep != 200:
        bits.append(f"e{ep}")
    stem = cfgget(cfg, "stem", "none") or "none"
    if stem != "none":
        bits.append(f"st{stem.replace('-', '')}k{cfgget(cfg, 'stem_kernel_size', 11)}")
        kw = cfgget(cfg, "stem_kwargs") or {}
        if kw:
            # deterministic hash of the FULL kwargs: abbreviating the values
            # collided (e.g. two variants both truncating to "useFals")
            h = hashlib.sha1(json.dumps(kw, sort_keys=True).encode()).hexdigest()[:6]
            bits.append("kw" + h)
    if aux:
        if aux.get("teacher"):
            bits.append("axteach")
        elif aux.get("hog"):
            bits.append("axhog")
        else:
            t = str(aux.get("stem", "")).replace("energy-", "").replace("-", "")
            tap = str(aux.get("tap", "")).replace("layer", "L").replace(
                "blocks.", "b").replace("stages.", "s").replace(
                "'", "").replace(" ", "").replace("[", "").replace("]", "").replace(",", "")
            bits.append(f"ax{t}{tap}")
        w0, wf = aux.get("weight", ""), aux.get("weight_final", None)
        bits.append(f"l{w0}" + (f"to{wf}" if wf is not None and wf != w0 else ""))
        if aux.get("loss"):
            bits.append(aux["loss"])
        if aux.get("head_norm"):
            bits.append("hn")
        if aux.get("allow_forward_stem"):
            bits.append("fwd")
    if cfgget(cfg, "init_from"):
        bits.append("ssl")
    if cfgget(cfg, "augment"):
        bits.append(str(cfgget(cfg, "augment")))
    if cfgget(cfg, "head"):
        bits.append(str(cfgget(cfg, "head")) + "head")
    if cfgget(cfg, "head_pool"):
        hp = cfgget(cfg, "head_pool")
        bits.append("pool" + str(hp.get("type", "")))
    # Readable prefix + a short hash of the FULL config_key: guarantees
    # uniqueness no matter which distinguishing field the prefix omits
    # (calibration mode, head_pool, head_norm, unfreeze epoch, aux kind,
    # which SimCLR pretrain, ...).
    h = hashlib.sha1(config_key(cfg).encode()).hexdigest()[:6]
    return "_".join(str(b).replace(".", "") for b in bits) + "_" + h


def deviates(cfg):
    return bool((cfgget(cfg, "optimizer", "sgd") or "sgd").lower() != "sgd"
                or cfgget(cfg, "epochs", 200) != 200 or cfgget(cfg, "head")
                or cfgget(cfg, "init_from") or cfgget(cfg, "augment"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="configs/grid")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cells = load_cells(["runs", "runs_turing"])
    fams, have = {}, {}
    for cell, rec in cells.items():
        cfg = rec["cfg"]
        k = config_key(cfg)
        fams.setdefault(k, cfg)
        have.setdefault(k, {})[cfgget(cfg, "subset_pct") or 100] = cell
    # Also treat any (configuration, portion) that ALREADY HAS A CONFIG FILE as
    # covered, even with no run yet -- otherwise in-flight waves get duplicated.
    import glob as _glob
    for f in _glob.glob("configs/**/*.yaml", recursive=True):
        try:
            c = yaml.safe_load(open(f))
        except Exception:
            continue
        if not isinstance(c, dict) or "dataset" not in c or "backbone" not in c:
            continue
        k = config_key(c)
        if k in fams:
            have.setdefault(k, {}).setdefault(c.get("subset_pct") or 100,
                                              c.get("name", f))

    # baseline cell name per (family_key, pct) -- needed by teacher configs
    # family_key is a JSON string now, so index the CONFIG directly.
    base_at = {}
    for cell, rec in cells.items():
        cfg = rec["cfg"]
        if is_baseline(cfg):
            k = (cfgget(cfg, "dataset"), cfgget(cfg, "backbone"),
                 (cfgget(cfg, "optimizer", "sgd") or "sgd").lower(),
                 cfgget(cfg, "epochs", 200), cfgget(cfg, "subset_pct") or 100)
            prev = base_at.get(k)
            if prev is None or len(cells[prev]["seeds"]) < len(rec["seeds"]):
                base_at[k] = cell

    slugs, written, skipped_imposs, plan = {}, 0, 0, []
    for k, cfg in fams.items():
        s = slug(cfg)
        if s in slugs and slugs[s] != k:
            raise SystemExit(f"slug collision: {s}")
        slugs[s] = k
        ds = cfgget(cfg, "dataset")
        pre = "diaggrid" if deviates(cfg) else "grid"
        for pct in GRID:
            if pct in have[k]:
                continue
            if n_images(ds, pct) < BATCH:
                skipped_imposs += 1
                continue
            name = f"{pre}_{s}_{pct}pct"
            new = {kk: vv for kk, vv in cfg.items()
                   if kk not in ("name", "subset_pct", "epochs_overridden")}
            new["name"] = name
            new["subset_pct"] = pct
            if new.get("init_from"):
                # Preserve the pretrain VARIANT tag (simclr_pre vs simclr_pre50
                # = 200- vs 50-epoch pretrain). Rewriting every family to one
                # path made two distinct configurations generate identical
                # configs. Tag = the original run dir minus its _<pct>pct tail.
                orig = str(cfgget(cfg, "init_from"))
                d = orig.split("/")[1] if "/" in orig else "simclr_pre"
                tag = re.sub(r"_\d+pct$", "", d)
                bb = BB_SHORT.get(cfgget(cfg, "backbone"), "x")
                new["init_from"] = (f"runs/{tag}__{DS_SHORT.get(ds, ds)}_"
                                    f"{bb}_{pct}pct/seed{{seed}}/pretrain.pt")
            aux = new.get("moment_aux") or {}
            if aux.get("teacher"):
                bcell = base_at.get((ds, cfgget(cfg, "backbone"),
                                     (cfgget(cfg, "optimizer", "sgd") or "sgd").lower(),
                                     cfgget(cfg, "epochs", 200), pct))
                if bcell is None:      # baseline itself is a new grid cell
                    bslug = None
                    for k2, c2 in fams.items():
                        if is_baseline(c2) and family_key(c2) == (
                                ds, pct, cfgget(cfg, "backbone"),
                                (cfgget(cfg, "optimizer", "sgd") or "sgd").lower(),
                                cfgget(cfg, "epochs", 200),
                                cfgget(cfg, "head") or "linear"):
                            bslug = slug(c2)
                            break
                    if bslug is None:
                        for k2, c2 in fams.items():
                            if is_baseline(c2) and cfgget(c2, "dataset") == ds \
                                    and cfgget(c2, "backbone") == cfgget(cfg, "backbone") \
                                    and (cfgget(c2, "optimizer", "sgd") or "sgd").lower() == "sgd" \
                                    and cfgget(c2, "epochs", 200) == 200:
                                bslug = slug(c2)
                                break
                    bcell = f"grid_{bslug}_{pct}pct"
                new["moment_aux"] = dict(aux)
                new["moment_aux"]["teacher"] = f"runs/{bcell}/seed0/last.pt"
            plan.append((name, ds, pct, bool(new.get("init_from")),
                         bool(aux.get("teacher"))))
            if not args.dry_run:
                os.makedirs(args.out, exist_ok=True)
                with open(os.path.join(args.out, f"{name}.yaml"), "w") as f:
                    f.write("# AUTO-GENERATED by scripts/make_grid_configs.py "
                            "(full-grid completion, 2026-07-22).\n")
                    yaml.safe_dump(new, f, sort_keys=False)
            written += 1

    ssl_n = sum(1 for p in plan if p[3])
    tea_n = sum(1 for p in plan if p[4])
    print(f"configurations: {len(fams)}   configs written: {written}"
          f"   (of which need a SimCLR pretrain: {ssl_n}; need a teacher ckpt: {tea_n})")
    print(f"impossible cells skipped (<1 batch of {BATCH}): {skipped_imposs}")
    print(f"total runs at 3 seeds: {written * 3} (+{ssl_n * 3} pretrains)")


if __name__ == "__main__":
    main()
