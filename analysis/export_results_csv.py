"""Export EVERY run in the study to one flat CSV for inspection.

    python analysis/export_results_csv.py [--runs-root runs [--runs-root runs_turing]]
                                          [--out results/all_results.csv]

One row per CELL (config name), merging seeds across every --runs-root given
(later roots win on a seed collision, so a re-run supersedes an older one).
Each row carries the cell's own accuracy, its linear-probe accuracy when
present, and -- when a matching baseline exists -- the paired comparison the
study actually reports: Delta (e2e gain), G (probe gap = feature gain), and
readout (Delta - G).

Baseline matching is automatic and conservative: a cell's baseline is the
cell with the SAME (dataset, subset_pct, backbone, optimizer, epochs, head)
that has no moment_aux, no init_from, and stem 'none'. That keeps every
comparison within its own recipe family -- a ViT cell never pairs with a
ResNet one, a cosine-head cell never with a linear-head one, a 400-epoch
control never with a 200-epoch cell.

`is_headline` marks cells that obey the frozen recipe (SGD, 200 epochs, no
custom head, no custom init, name not prefixed `diag`); everything else is a
diagnostic and must never enter a headline table.
"""

import argparse
import csv
import glob
import json
import math
import os
import re
import statistics as st

TRAIN_SIZE = {"cifar100": 50000, "cifar100super": 50000, "cifar10": 50000,
              "stl10": 5000, "tin": 100000, "tinsuper": 100000,
              "tinsem": 100000, "tin20": 10000, "tin20b": 10000, "cub": 5994,
              "eurosat": 21600, "dtd": 3760, "pathmnist": 89996,
              "food101": 75750}


def load_cells(roots):
    """{cell: {"seeds": {seed: acc}, "cfg": config, "probes": {name: [acc]}}}"""
    cells = {}
    for root in roots:
        for path in sorted(glob.glob(os.path.join(root, "*", "seed*", "final.json"))):
            cell = path.split(os.sep)[-3]
            with open(path) as f:
                final = json.load(f)
            rec = cells.setdefault(cell, {"seeds": {}, "cfg": {}, "probes": {}})
            rec["seeds"][final.get("seed", path)] = 100.0 * final["final_test_acc"]
            rec["cfg"] = final.get("config", final)
        for path in sorted(glob.glob(os.path.join(root, "*", "linear_probe*.json"))):
            cell = path.split(os.sep)[-2]
            if cell not in cells:
                continue
            with open(path) as f:
                payload = json.load(f)
            name = os.path.basename(path)
            cells[cell]["probes"][name] = [100.0 * r["probe_test"]
                                           for r in payload["results"]]
    return cells


def cfgget(cfg, key, default=None):
    return cfg.get(key, default) if isinstance(cfg, dict) else default


def init_label(raw):
    """Human-readable tag for a pretrained init, from its checkpoint path.

    The pretrain FAMILY is encoded in the path (simclr_pre / simsiam_pre /
    dino_pre), exactly as the worklist generators route it. This used to
    hardcode "simclr-init" for ANY init_from, so every SimSiam and DINO row
    in the exported tables was LABELLED AS SimCLR (2026-08-04) -- the rows
    were distinct (config_key keeps the variant), but three different SSL
    methods rendered under one name, e.g. DINO's C100@100% 62.30 appeared as
    a "simclr-init" number. Grouping was never wrong; the NAME was.
    """
    if "simsiam_pre" in raw:
        return "simsiam-init"
    if "dino_pre" in raw:
        return "dino-init"
    if "pre50" in raw:
        return "simclr50-init"
    if "_deit" in raw:                      # DeiT-strength contrastive views
        return "simclr-init(deit-views)"
    return "simclr-init"


def sem_of_diff(a, b):
    if len(a) < 2 or len(b) < 2:
        return ""
    return math.sqrt(st.stdev(a) ** 2 / len(a) + st.stdev(b) ** 2 / len(b))


def fmt(x, nd=2):
    return "" if x == "" or x is None else f"{x:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-root", action="append", default=None,
                    help="repeatable; later roots win on seed collisions")
    ap.add_argument("--out", default="results/all_results.csv")
    args = ap.parse_args()
    roots = args.runs_root or ["runs", "runs_turing"]
    roots = [r for r in roots if os.path.isdir(r)]

    cells = load_cells(roots)

    # baseline lookup: family key -> baseline cell name. Preference order:
    # (1) more seeds; (2) an ORIGINAL named cell over a grid_* re-run (equal
    # power: the ledger's recorded numbers live in the named cells, and a
    # different-num_workers re-run is a different augmentation stream); (3)
    # name, as a deterministic last resort. Before (2), equal-seed ties fell
    # to load order -- c10_none_7pct vs its grid twin differ by 0.77.
    def base_rank(cell):
        return (len(cells[cell]["seeds"]),
                0 if cell.startswith("grid_") else 1,
                cell)
    baselines = {}
    for cell, rec in cells.items():
        if is_baseline(rec["cfg"]):
            k = family_key(rec["cfg"])
            cur = baselines.get(k)
            if cur is None or base_rank(cur) < base_rank(cell):
                baselines[k] = cell

    CHANCE = {"cifar100": 1.0, "cifar100super": 5.0, "cifar10": 10.0,
              "stl10": 10.0, "tin": 0.5, "tin20": 5.0, "tin20b": 5.0,
              "tinsuper": 5.0, "tinsem": 5.0, "cub": 0.5,
              "eurosat": 10.0, "dtd": 100.0 / 47, "pathmnist": 100.0 / 9,
              "food101": 100.0 / 101}

    rows = []
    for cell, rec in sorted(cells.items()):
        cfg, accs = rec["cfg"], list(rec["seeds"].values())
        aux = cfgget(cfg, "moment_aux") or {}
        ds = cfgget(cfg, "dataset")
        pct = cfgget(cfg, "subset_pct") or 100
        n_img = (int(round(TRAIN_SIZE.get(ds, 0) * pct / 100.0))
                 if ds in TRAIN_SIZE else "")
        probe = rec["probes"].get("linear_probe.json")
        # BISTABLE cell: >=1 seed collapsed to ~chance while the cell as a
        # whole trains. A mean over a bimodal set misrepresents both modes
        # (ConvNeXt-SGD grid re-runs: seeds {0.84, 42.25, 19.54}); flag it so
        # no downstream reader mistakes the mean for a typical run.
        ch = CHANCE.get(ds, 1.0)
        collapsed = [v for v in accs if v <= ch * 1.5]
        bistable = bool(collapsed) and st.mean(accs) > ch * 3
        # headline requires the FROZEN recipe *and* >=3 seeds -- the study's
        # own repeated lesson is that 1-2 seed numbers support nothing.
        headline = (not cell.startswith("diag")
                    and (cfgget(cfg, "optimizer", "sgd") or "sgd").lower() == "sgd"
                    and cfgget(cfg, "epochs", 200) == 200
                    and not cfgget(cfg, "head") and not cfgget(cfg, "init_from")
                    and not cfgget(cfg, "augment") and len(accs) >= 3
                    and not bistable)

        row = {
            "cell": cell,
            "dataset": ds,
            "backbone": cfgget(cfg, "backbone"),
            "optimizer": (cfgget(cfg, "optimizer", "sgd") or "sgd").lower(),
            "epochs": cfgget(cfg, "epochs", 200),
            "head": cfgget(cfg, "head") or "linear",
            "stem": cfgget(cfg, "stem", "none"),
            "init_from": "yes" if cfgget(cfg, "init_from") else "",
            "aux_target": aux.get("stem", ""),
            "aux_tap": aux.get("tap", ""),
            "aux_lambda0": aux.get("weight", ""),
            "aux_lambda_final": aux.get("weight_final", ""),
            "subset_pct": pct,
            "n_images": n_img,
            "n_seeds": len(accs),
            "acc_mean": fmt(st.mean(accs)),
            "acc_std": fmt(st.stdev(accs) if len(accs) > 1 else 0.0),
            "probe_mean": fmt(st.mean(probe)) if probe else "",
            "probe_std": fmt(st.stdev(probe)) if probe and len(probe) > 1 else "",
            "n_probe_seeds": len(probe) if probe else "",
            "is_headline": "yes" if headline else "no",
            "bistable": ("%d/%d seeds at chance" % (len(collapsed), len(accs))
                         if bistable else ""),
            "baseline_cell": "", "base_acc": "", "delta": "", "delta_sem": "",
            "G": "", "G_sem": "", "readout": "",
        }

        bcell = baselines.get(family_key(cfg))
        if bcell and bcell != cell:
            baccs = list(cells[bcell]["seeds"].values())
            row["baseline_cell"] = bcell
            row["base_acc"] = fmt(st.mean(baccs))
            row["delta"] = fmt(st.mean(accs) - st.mean(baccs))
            row["delta_sem"] = fmt(sem_of_diff(accs, baccs))
            bprobe = cells[bcell]["probes"].get("linear_probe.json")
            if probe and bprobe:
                g = st.mean(probe) - st.mean(bprobe)
                row["G"] = fmt(g)
                row["G_sem"] = fmt(sem_of_diff(probe, bprobe))
                row["readout"] = fmt(st.mean(accs) - st.mean(baccs) - g)
        rows.append(row)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    write_pivot(cells, baselines,
                os.path.join(os.path.dirname(args.out) or ".",
                             "results_by_portion.csv"))
    paired = sum(1 for r in rows if r["delta"] != "")
    withG = sum(1 for r in rows if r["G"] != "")
    print(f"wrote {args.out}: {len(rows)} cells "
          f"({paired} with a paired Delta, {withG} with G/readout) "
          f"from roots {roots}")



# --------------------------------------------------------------------------
# Pivot view: one row per CONFIGURATION, one column block per DATA PORTION.
# The configuration key is every config field EXCEPT subset_pct, so a cell and
# its 1/5/10/25/100% siblings collapse onto one row and the envelope reads
# left-to-right. Baselines sort to the top of each (dataset, backbone) group
# so a variant can be compared against its own baseline down the column.
# --------------------------------------------------------------------------

def config_key(cfg):
    """Everything that defines the experimental condition EXCEPT the data
    fraction, so a cell and its siblings at other fractions collapse onto one
    pivot row.

    2026-07-22 FIX: the earlier key omitted head_pool, stem_calibrate,
    stem_unfreeze_epoch, aux head_norm, the aux TARGET KIND (moment / hog /
    teacher) and the init_from path. That silently merged 15 distinct
    (config, portion) slots -- e.g. the HOG control with the FitNets teacher
    control, and the MultiMaskPool readouts with the plain baseline -- dropping
    22 cells from the pivot. num_workers is deliberately NOT part of the key:
    it affects the augmentation stream (see CLAUDE.md) but not which
    experiment a cell IS.
    """
    return json.dumps(config_fields(cfg), sort_keys=True)


def config_fields(cfg):
    """Every field that defines the experimental condition except the data
    fraction. INTERVENTION_FIELDS below are the things an experiment TESTS;
    everything else must be matched between a cell and its baseline."""
    aux = cfgget(cfg, "moment_aux") or {}
    kind = ("teacher" if aux.get("teacher") else
            "hog" if aux.get("hog") else "moment" if aux else "")
    stem = cfgget(cfg, "stem", "none") or "none"
    # init_from embeds the data fraction (runs/simclr_pre_5pct/...), which would
    # split ONE configuration into a separate family per fraction. Keep only the
    # pretrain VARIANT tag, so simclr_pre / simclr_pre50 / simclr_pre_vit stay
    # distinct while 1% and 5% of the same variant land on one row.
    raw = str(cfgget(cfg, "init_from") or "")
    init = ""
    if raw:
        d = raw.split("/")[1] if "/" in raw else raw
        init = re.sub(r"_+\d+pct$", "", re.sub(r"__[a-z0-9]+_[a-z0-9]+_\d+pct$", "", d))
    return {
        "dataset": cfgget(cfg, "dataset"), "backbone": cfgget(cfg, "backbone"),
        "optimizer": (cfgget(cfg, "optimizer", "sgd") or "sgd").lower(),
        # train.py fills these from RECIPE when a config omits them, so a raw
        # YAML and its stored final.json copy must normalise to the same key.
        "epochs": cfgget(cfg, "epochs", 200),
        "lr": cfgget(cfg, "lr", 0.1),
        "weight_decay": cfgget(cfg, "weight_decay", 5e-4),
        "momentum": cfgget(cfg, "momentum", 0.9),
        "batch_size": cfgget(cfg, "batch_size", 128),
        "small_input": cfgget(cfg, "small_input", True),
        "pretrained": cfgget(cfg, "pretrained", False),
        "stem": stem,
        "stem_kernel_size": cfgget(cfg, "stem_kernel_size", 11) if stem != "none" else None,
        "stem_kwargs": cfgget(cfg, "stem_kwargs") or {},
        "stem_calibrate": cfgget(cfg, "stem_calibrate", False),
        "stem_unfreeze_epoch": cfgget(cfg, "stem_unfreeze_epoch"),
        "head": cfgget(cfg, "head") or "linear",
        "head_pool": cfgget(cfg, "head_pool") or None,
        "augment": cfgget(cfg, "augment") or None,
        "init_from": init,
        "aux_kind": kind,
        "aux": {k: (str(v) if k in ("tap",) else v)
                for k, v in sorted(aux.items()) if k != "teacher"},
        "aux_teacher": bool(aux.get("teacher")),
    }


# What an experiment TESTS (so a baseline is the same cell with these cleared).
# Everything NOT listed here -- optimizer, epochs, lr, head, augment, ... -- is
# matched, so a DeiT-augmented aux cell pairs with the DeiT-augmented baseline
# and never with the plain one.
INTERVENTION_FIELDS = ("stem", "stem_kernel_size", "stem_kwargs",
                       "stem_calibrate", "stem_unfreeze_epoch", "head_pool",
                       "init_from", "aux_kind", "aux", "aux_teacher")


def family_key(cfg):
    f = config_fields(cfg)
    for k in INTERVENTION_FIELDS:
        f.pop(k, None)
    f["subset_pct"] = cfgget(cfg, "subset_pct") or 100
    return json.dumps(f, sort_keys=True)


def is_baseline(cfg):
    f = config_fields(cfg)
    return (not f["aux_kind"] and not f["init_from"] and not f["head_pool"]
            and f["stem"] == "none")


def config_label(cfg):
    """Compact, unambiguous description of the configuration (fraction-free).
    Must distinguish every family config_key distinguishes."""
    aux = cfgget(cfg, "moment_aux") or {}
    stem = cfgget(cfg, "stem", "none") or "none"
    bits = []
    if aux:
        if aux.get("teacher"):
            head = "aux-TEACHER(fitnets)"
        elif aux.get("hog"):
            head = "aux-HOG"
        else:
            tgt = str(aux.get("stem", "?")).replace("energy-", "")
            tap = str(aux.get("tap", "?")).replace("layer", "L").replace("blocks.", "b")
            head = f"aux({tgt}@{tap}"
        lam0, lamf = aux.get("weight", "?"), aux.get("weight_final", None)
        lam = f"L{lam0}>{lamf}" if lamf is not None and lamf != lam0 else f"L{lam0}"
        extra = f",{aux['loss']}" if aux.get("loss") else ""
        hn = ",hn" if aux.get("head_norm") else ""
        fwd = ",fwd-stem" if aux.get("allow_forward_stem") else ""
        bits.append(head + ("," if head.startswith("aux(") else "(") + lam + extra + hn + fwd + ")")
    if stem != "none":
        cal = cfgget(cfg, "stem_calibrate", False)
        caltag = ",zca" if cal == "zca" else ("" if cal else ",nocal")
        uf = cfgget(cfg, "stem_unfreeze_epoch")
        bits.append(f"stem({stem},k{cfgget(cfg, 'stem_kernel_size', 11)}{caltag}"
                    + (f",unfreeze@{uf}" if uf is not None else "") + ")")
        kw = cfgget(cfg, "stem_kwargs") or {}
        if kw:
            bits.append("kw(" + ",".join(f"{k}={v}" for k, v in sorted(kw.items())) + ")")
    if cfgget(cfg, "head_pool"):
        hp = cfgget(cfg, "head_pool")
        bits.append(f"pool({hp.get('type','?')},J{hp.get('J','?')})")
    if cfgget(cfg, "init_from"):
        bits.append(init_label(str(cfgget(cfg, "init_from"))))
    if cfgget(cfg, "augment"):
        bits.append(f"{cfgget(cfg,'augment')}-aug")
    if cfgget(cfg, "head"):
        bits.append(f"{cfgget(cfg,'head')}-head")
    if not bits:
        bits.append("BASELINE")
    tail = []
    if (cfgget(cfg, "optimizer", "sgd") or "sgd").lower() != "sgd":
        tail.append((cfgget(cfg, "optimizer") or "").lower())
    if cfgget(cfg, "epochs", 200) != 200:
        tail.append(f"{cfgget(cfg, 'epochs')}ep")
    return " + ".join(bits) + (f"  [{','.join(tail)}]" if tail else "")


def write_pivot(cells, baselines, out_path):
    fams = {}
    for cell, rec in cells.items():
        cfg = rec["cfg"]
        pct = cfgget(cfg, "subset_pct") or 100
        accs = list(rec["seeds"].values())
        k = config_key(cfg)
        f = fams.setdefault(k, {"label": config_label(cfg),
                                "dataset": cfgget(cfg, "dataset"),
                                "backbone": cfgget(cfg, "backbone"),
                                "is_baseline": is_baseline(cfg),
                                "cells": {}, "by_pct": {}})
        bcell = baselines.get(family_key(cfg))
        delta = ""
        if bcell and bcell != cell:
            baccs = list(cells[bcell]["seeds"].values())
            delta = st.mean(accs) - st.mean(baccs)
        # keep the better-powered cell if two share a (config, pct)
        prev = f["by_pct"].get(pct)
        if prev is None or prev["n"] < len(accs):
            f["by_pct"][pct] = {"acc": st.mean(accs),
                                "std": st.stdev(accs) if len(accs) > 1 else 0.0,
                                "n": len(accs), "delta": delta}
            f["cells"][pct] = cell

    pcts = sorted({p for f in fams.values() for p in f["by_pct"]})
    # TWO-ROW header: the metric name spans its block once (row 1) and the
    # data portions repeat beneath it (row 2), instead of "acc@1%, acc@2%..."
    lead = ["configuration", "dataset", "backbone", "role"]
    groups = [("accuracy (test top-1 %)", "acc"),
              ("delta vs own baseline (pts)", "delta"),
              ("seeds", "n")]
    row1 = [""] * len(lead)
    row2 = list(lead)
    for title, _ in groups:
        row1 += [title] + [""] * (len(pcts) - 1)
        row2 += [f"{p}%" for p in pcts]
    row1 += [""]
    row2 += ["cells"]

    def impossible(dataset, pct):
        """A cell that CANNOT exist under the frozen recipe: fewer than one
        batch of 128 (drop_last=True -> empty loader). These are 'n/a', not
        'missing' -- the distinction the full-grid goal needs visible."""
        n = TRAIN_SIZE.get(dataset)
        return n is not None and int(n * pct / 100) < 128

    body = []
    for f in fams.values():
        r = [f["label"], f["dataset"], f["backbone"],
             "baseline" if f["is_baseline"] else "variant"]
        for _, kind in groups:
            for p in pcts:
                d = f["by_pct"].get(p)
                if not d:
                    r.append("n/a (<1 batch)" if kind == "acc"
                             and impossible(f["dataset"], p) else "")
                elif kind == "acc":
                    r.append(fmt(d["acc"]))
                elif kind == "delta":
                    r.append(fmt(d["delta"]) if d["delta"] != "" else "")
                else:
                    r.append(d["n"])
        r.append(" ".join(f["cells"][p] for p in sorted(f["cells"])))
        body.append(r)
    body.sort(key=lambda r: (str(r[1]), str(r[2]), 0 if r[3] == "baseline" else 1, r[0]))

    with open(out_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(row1)
        w.writerow(row2)
        w.writerows(body)
    rows = body
    print(f"wrote {out_path}: {len(rows)} configurations x {len(pcts)} data portions "
          f"({', '.join(str(p) + '%' for p in pcts)})")

if __name__ == "__main__":
    main()
