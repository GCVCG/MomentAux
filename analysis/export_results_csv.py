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
import statistics as st

TRAIN_SIZE = {"cifar100": 50000, "cifar100super": 50000, "cifar10": 50000,
              "stl10": 5000, "tin": 100000, "tinsuper": 100000,
              "tinsem": 100000, "tin20": 10000, "tin20b": 10000, "cub": 5994}


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


def family_key(cfg):
    return (cfgget(cfg, "dataset"), cfgget(cfg, "subset_pct") or 100,
            cfgget(cfg, "backbone"), (cfgget(cfg, "optimizer", "sgd") or "sgd").lower(),
            cfgget(cfg, "epochs", 200), cfgget(cfg, "head") or "linear")


def is_baseline(cfg):
    return (not cfgget(cfg, "moment_aux") and not cfgget(cfg, "init_from")
            and (cfgget(cfg, "stem", "none") or "none") == "none")


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

    # baseline lookup: family key -> baseline cell name (prefer more seeds)
    baselines = {}
    for cell, rec in cells.items():
        if is_baseline(rec["cfg"]):
            k = family_key(rec["cfg"])
            cur = baselines.get(k)
            if cur is None or len(cells[cur]["seeds"]) < len(rec["seeds"]):
                baselines[k] = cell

    rows = []
    for cell, rec in sorted(cells.items()):
        cfg, accs = rec["cfg"], list(rec["seeds"].values())
        aux = cfgget(cfg, "moment_aux") or {}
        ds = cfgget(cfg, "dataset")
        pct = cfgget(cfg, "subset_pct") or 100
        n_img = (int(round(TRAIN_SIZE.get(ds, 0) * pct / 100.0))
                 if ds in TRAIN_SIZE else "")
        probe = rec["probes"].get("linear_probe.json")
        headline = (not cell.startswith("diag")
                    and (cfgget(cfg, "optimizer", "sgd") or "sgd").lower() == "sgd"
                    and cfgget(cfg, "epochs", 200) == 200
                    and not cfgget(cfg, "head") and not cfgget(cfg, "init_from"))

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
    aux = cfgget(cfg, "moment_aux") or {}
    return (
        cfgget(cfg, "dataset"), cfgget(cfg, "backbone"),
        (cfgget(cfg, "optimizer", "sgd") or "sgd").lower(),
        cfgget(cfg, "epochs", 200), cfgget(cfg, "head") or "linear",
        cfgget(cfg, "stem", "none") or "none",
        cfgget(cfg, "stem_kernel_size", 11) if (cfgget(cfg, "stem") or "none") != "none" else "",
        json.dumps(cfgget(cfg, "stem_kwargs") or {}, sort_keys=True),
        aux.get("stem", ""), str(aux.get("tap", "")), str(aux.get("weight", "")),
        str(aux.get("weight_final", "")), aux.get("loss", ""),
        "init" if cfgget(cfg, "init_from") else "",
        cfgget(cfg, "augment") or "",
    )


def config_label(cfg):
    """Compact, unambiguous description of the configuration (fraction-free)."""
    aux = cfgget(cfg, "moment_aux") or {}
    stem = cfgget(cfg, "stem", "none") or "none"
    bits = []
    if aux:
        tgt = str(aux.get("stem", "?")).replace("energy-", "")
        tap = str(aux.get("tap", "?")).replace("layer", "L").replace("blocks.", "b")
        lam0, lamf = aux.get("weight", "?"), aux.get("weight_final", None)
        lam = f"L{lam0}>{lamf}" if lamf is not None and lamf != lam0 else f"L{lam0}"
        extra = f",{aux['loss']}" if aux.get("loss") else ""
        bits.append(f"aux({tgt}@{tap},{lam}{extra})")
    if stem != "none":
        bits.append(f"stem({stem},k{cfgget(cfg, 'stem_kernel_size', 11)})")
    if cfgget(cfg, "init_from"):
        bits.append("simclr-init")
    if cfgget(cfg, "augment"):
        bits.append(f"{cfgget(cfg,'augment')}-aug")
    if cfgget(cfg, "head"):
        bits.append(f"{cfgget(cfg,'head')}-head")
    if not bits:
        bits.append("BASELINE")
    ep = cfgget(cfg, "epochs", 200)
    tail = []
    if (cfgget(cfg, "optimizer", "sgd") or "sgd").lower() != "sgd":
        tail.append((cfgget(cfg, "optimizer") or "").lower())
    if ep != 200:
        tail.append(f"{ep}ep")
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
    header = (["config", "dataset", "backbone", "role"]
              + [f"acc@{p}%" for p in pcts]
              + [f"delta@{p}%" for p in pcts]
              + [f"n@{p}%" for p in pcts] + ["cells"])
    rows = []
    for k, f in fams.items():
        r = {"config": f["label"], "dataset": f["dataset"],
             "backbone": f["backbone"],
             "role": "baseline" if f["is_baseline"] else "variant"}
        for p in pcts:
            d = f["by_pct"].get(p)
            r[f"acc@{p}%"] = fmt(d["acc"]) if d else ""
            r[f"delta@{p}%"] = (fmt(d["delta"]) if d and d["delta"] != "" else "")
            r[f"n@{p}%"] = d["n"] if d else ""
        r["cells"] = " ".join(f["cells"][p] for p in sorted(f["cells"]))
        rows.append(r)

    rows.sort(key=lambda r: (str(r["dataset"]), str(r["backbone"]),
                             0 if r["role"] == "baseline" else 1, r["config"]))
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=header)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out_path}: {len(rows)} configurations x {len(pcts)} data portions "
          f"({', '.join(str(p) + '%' for p in pcts)})")

if __name__ == "__main__":
    main()
