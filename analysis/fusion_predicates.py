"""Block B-1 (2026-08-23): single-arm fusion predicates, derived on every
MEASURED two-source combination and scored out-of-family.

THE QUESTION. Wave 1 (2026-08-20) falsified Algorithm 1: the published
"G-magnitude agreement" predicate called 1 of 9 held-out pairs, the fix-set
overlap S(A,B) was at chance (AUC 0.50), and the strength-asymmetry rule
derived in the post-mortem scored 1/6 then 0/2 one family over. This module
asks, on EVERY combination the study has trained, whether ANY quantity
computable from the two SINGLE arms and their shared baseline ranks the
measured STACK / SUBSTITUTE / INTERFERE outcome -- and whether it still does
when the family it is tested on was never seen (leave-one-family-out). The
pre-registered bar is AUC >= 0.80 out-of-family.

WHAT IS A PAIR. baseline B0, single arms A and B (each a training-time
source on top of B0), combination C (both sources on top of B0). Every
quantity the predicates consume comes from B0, A and B; the combination
enters ONLY through the outcome label. outcome() is prospective_currency's:
combo minus the better single arm, against 2 SEM -> STACK / SUBSTITUTE /
SUBSTITUTE(cost) (the last is the paper's "interference").

PHASES (each a flag; all read/write under results/fusion_predicates/):
  --build    resolve the derivation set from results/all_results.csv
             (cells, seeds, Delta, G) -> pairs.json; lists unresolved pairs
  --extract  penultimate features (fixed 2,000-image test subset), per-image
             correctness and per-class accuracy for every LOCAL checkpoint
             of every baseline / single arm -> cache/<cell>_<seed>.npz.
             Uses analysis.linear_probe.extract, i.e. the exact feature path
             every recorded G was measured under.
  --shots25  25-shot probe gap for cells whose full-train G exists but whose
             25-shot value does not (writes linear_probe_shots.json ONLY if
             the file is absent; never overwrites; never a full probe)
  --score    predicates, AUC + bootstrap CI, leave-one-family-out, 2-rule
             combinations -> results/fusion_predicates.json / .md
  --calls    wave-3 calls under the best rule and the published rule

CHECKPOINT POLICY for --extract: last.pt where it exists, else best.pt, and
the choice is recorded per cell. Recorded G values were measured on best.pt
and the prospective passes on last.pt; for the predicates here (similarity,
per-class deltas, fix sets) the two differ by seed-noise on every cell
checked (best ~ final to <=0.1 on these diag cells).

NOTHING HERE TRAINS. The derivation set is what exists; the test set is
wave 3, whose combinations have never been trained and whose calls are
committed in the ledger before they are.
"""

import argparse
import csv
import glob
import json
import math
import os
import sys
from itertools import combinations

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

OUT_DIR = "results/fusion_predicates"
CACHE_DIR = os.path.join(OUT_DIR, "cache")
PAIRS_JSON = os.path.join(OUT_DIR, "pairs.json")
RUN_TREES = ("runs", "runs_bscpull", "runs_turing")
N_CKA = 2000          # test images used for CKA (stratified, deterministic)
MAX_SEEDS = 3         # seeds per cell used for checkpoint-based predicates
AUC_BAR = 0.80        # pre-registered out-of-family bar


# --------------------------------------------------------------------------
# 1. the derivation set


def champion(ds, pct):
    """The scratch reference-configuration prior cell (and its baseline) for
    (dataset, pct): lambda0=1.0 cosine->0, magnitude target, tap layer3,
    ResNet-18, frozen recipe. Names differ by campaign; this is the lookup
    the ledger uses."""
    c = {
        "cifar10": {p: (f"c10_aux_{p}pct", f"c10_none_{p}pct")
                    for p in (1, 2, 3, 5, 7, 10, 15, 25, 100)},
        "cifar100": {p: (f"auxmag_{p}pct_sched0", f"abl{p}_none")
                     for p in (1, 2, 3, 5, 7, 10, 15, 25)},
        "eurosat": {p: (f"grid_esat_r18_axmagnitudeL3_l10to00_hn_dd367d_{p}pct",
                        f"grid_esat_r18_e99fb4_{p}pct")
                    for p in (1, 2, 3, 5, 7, 10, 15, 20, 25, 50, 100)},
        "food101": {p: (f"grid_food_r18_axmagnitudeL3_l10to00_hn_8bd74b_{p}pct",
                        f"grid_food_r18_9ee7da_{p}pct")
                    for p in (1, 2, 3, 5, 7, 10, 15, 20, 25, 50, 100)},
        "dtd": {p: (f"grid_dtd_r18_axmagnitudeL3_l10to00_hn_742cfa_{p}pct",
                    f"grid_dtd_r18_0ca6c1_{p}pct")
                for p in (5, 7, 10, 15, 20, 25, 50, 100)},
        "pathmnist": {p: (f"grid_path_r18_axmagnitudeL3_l10to00_hn_1daa80_{p}pct",
                          f"grid_path_r18_89e5af_{p}pct")
                      for p in (1, 2, 3, 5, 7, 10, 15, 20, 25, 50, 100)},
        "cub": {**{p: (f"grid_cub_r18_axmagnitudeL3_l10to00_ad3977_{p}pct",
                       f"grid_cub_r18_d85097_{p}pct")
                   for p in (3, 5, 7, 10, 15, 20, 50)},
                25: ("cub_aux_25pct", "cub_none_25pct"),
                100: ("cub_aux_100pct", "cub_none_100pct")},
        "stl10": {**{p: (f"grid_stl_r18_axmagnitudeL3_l10to00_3ebf19_{p}pct",
                         f"grid_stl_r18_9016e6_{p}pct")
                     for p in (3, 5, 7, 15, 25)},
                  10: ("stl_aux_10pct", "stl_none_10pct"),
                  20: ("stl_aux_20pct", "stl_none_20pct"),
                  50: ("stl_aux_50pct", "stl_none_50pct"),
                  100: ("grid_stl_r18_axmagnitudeL3_l10to00_3ebf19_100pct", "stl10_none")},
        "tin": {**{p: (f"tin_aux_{p}pct", f"tin_none_{p}pct")
                   for p in (1, 2, 3, 5, 7, 10, 15, 25, 100)},
                20: ("grid_tin_r18_axmagnitudeL3_l10to00_391ace_20pct", "grid_tin_r18_791598_20pct"),
                50: ("grid_tin_r18_axmagnitudeL3_l10to00_391ace_50pct", "grid_tin_r18_791598_50pct")},
    }
    c["cifar100"][100] = ("auxmag_100pct_sched0", "ablf_none")
    c["cifar100"][20] = ("grid_c100_r18_axmagnitudeL3_l10to00_b0bf76_20pct", "grid_c100_r18_3a0225_20pct")
    c["cifar100"][50] = ("grid_c100_r18_axmagnitudeL3_l10to00_b0bf76_50pct", "grid_c100_r18_3a0225_50pct")
    c["cifar10"][20] = ("grid_c10_r18_axmagnitudeL3_l10to00_8490d9_20pct", "grid_c10_r18_ebb591_20pct")
    c["cifar10"][50] = ("grid_c10_r18_axmagnitudeL3_l10to00_8490d9_50pct", "grid_c10_r18_ebb591_50pct")
    return c.get(ds, {}).get(pct)


def pair_spec():
    """Every measured two-source combination the ledger records, as
    (family, dataset, backbone, pct, baseline, A, B, combo) with source
    labels. Families are named by SOURCE TYPE, which is what leave-one-
    family-out holds out."""
    P = []

    def add(family, ds, bb, pct, base, A, Al, B, Bl, C, note=""):
        P.append({"name": f"{family}:{ds}@{pct}:{bb}" + (f":{note}" if note else ""),
                  "family": family, "dataset": ds, "backbone": bb, "pct": pct,
                  "baseline": base, "A": {"cell": A, "label": Al},
                  "B": {"cell": B, "label": Bl}, "combo": C, "note": note})

    # ---- prior|simclr (conv, 1x vs 2x; wave 1 + B1 + the original combo)
    add("prior|simclr", "cifar100", "resnet18", 5, "abl5_none",
        "auxmag_5pct_sched0", "prior", "diaggrid_simclr_c100_5pct", "simclr",
        "diagssl_simclraux_5pct")
    for p in (10, 25):
        add("prior|simclr", "cifar100", "resnet18", p, f"abl{p}_none",
            f"auxmag_{p}pct_sched0", "prior", f"diaggrid_simclr_c100_{p}pct", "simclr",
            f"diagprosp_simclraux_c100_{p}pct")
    add("prior|simclr", "cifar10", "resnet18", 2, "c10_none_2pct",
        "c10_aux_2pct", "prior", "diaggrid_simclr_c10_2pct", "simclr",
        "diagprosp_simclraux_c10_2pct")
    add("prior|simclr", "stl10", "resnet18", 10, "stl_none_10pct",
        "stl_aux_10pct", "prior", "diaggrid_simclr_stl_10pct", "simclr",
        "diagprosp_simclraux_stl_10pct")
    for ds, tag in (("eurosat", "esat"), ("food101", "food")):
        for p in (5, 10, 25):
            pr, base = champion(ds, p)
            add("prior|simclr", ds, "resnet18", p, base, pr, "prior",
                f"diaggrid_simclr_{tag}_{p}pct", "simclr", f"diagfuse_{tag}_priorssl_{p}pct")
    # ViT, plain recipe and DeiT recipe
    add("prior|simclr", "cifar100", "vit_tiny", 10, "diagvit_none_10pct",
        "diagvit_aux_10pct", "prior", "diagsslvit_simclr_10pct", "simclr",
        "diagsslvitaux_10pct")
    add("prior|simclr", "cifar100", "vit_tiny", 10, "diagdeit_none_10pct",
        "diagdeit_aux_10pct", "prior", "diagdeitssl_simclr_10pct", "simclr",
        "diagdeitsslaux_10pct", note="deit")

    # ---- prior|simsiam
    add("prior|simsiam", "tin", "resnet18", 10, "tin_none_10pct", "tin_aux_10pct", "prior",
        "diaggrid_simsiam_tin_10pct", "simsiam", "diagprosp_simsiamaux_tin_10pct")
    add("prior|simsiam", "cifar10", "resnet18", 5, "c10_none_5pct", "c10_aux_5pct", "prior",
        "diaggrid_simsiam_c10_5pct", "simsiam", "diagprosp_simsiamaux_c10_5pct")
    for p in (5, 10):
        add("prior|simsiam", "cifar100", "resnet18", p, f"abl{p}_none",
            f"auxmag_{p}pct_sched0", "prior", f"diaggrid_simsiam_c100_{p}pct", "simsiam",
            f"diaggrid_simsiamaux_c100_{p}pct")
    add("prior|simsiam", "tin", "resnet18", 5, "tin_none_5pct", "tin_aux_5pct", "prior",
        "diaggrid_simsiam_tin_5pct", "simsiam", "diaggrid_simsiamaux_tin_5pct")

    # ---- prior|dino (ViT)
    add("prior|dino", "cifar100", "vit_tiny", 25, "diagvit_none_25pct", "diagvit_aux_25pct",
        "prior", "diaggrid_dino_c100_25pct", "dino", "diagprosp_dinoaux_c100_25pct")
    add("prior|dino", "cifar10", "vit_tiny", 10, "diaggrid_vit_c10_none_10pct",
        "diaggrid_vit_c10_aux_10pct", "prior", "diaggrid_dino_c10_10pct", "dino",
        "diagprosp_dinoaux_c10_10pct")
    add("prior|dino", "food101", "vit_tiny", 50, "diaggrid_vit_food_none_50pct",
        "diaggrid_vit_food_aux_50pct", "prior", "diaggrid_dino_food_50pct", "dino",
        "diagprosp_dinoaux_food_50pct")
    add("prior|dino", "dtd", "vit_tiny", 50, "diaggrid_vit_dtd_none_50pct",
        "diaggrid_vit_dtd_aux_50pct", "prior", "diaggrid_dino_dtd_50pct", "dino",
        "diagprosp_dinoaux_dtd_50pct")
    for p in (5, 10):
        add("prior|dino", "cifar100", "vit_tiny", p, f"diagvit_none_{p}pct",
            f"diagvit_aux_{p}pct", "prior", f"diaggrid_dino_vit_{p}pct", "dino",
            f"diaggrid_dinoaux_vit_{p}pct")

    # ---- prior|aug (DeiT augmentation as the second source)
    for p in (1, 2, 3, 5, 7, 10, 15, 25, 100):
        add("prior|aug", "cifar100", "vit_tiny", p, f"diagvit_none_{p}pct",
            f"diagvit_aux_{p}pct", "prior", f"diagdeit_none_{p}pct", "aug",
            f"diagdeit_aux_{p}pct")
    for p in (1, 2, 5, 10, 15, 25, 100):
        add("prior|aug", "tin", "vit_tiny", p, f"diagvit_none_tin_{p}pct",
            f"diagvit_aux_tin_{p}pct", "prior", f"diagdeit_none_tin_{p}pct", "aug",
            f"diagdeit_aux_tin_{p}pct")
    for ds, tag in (("eurosat", "esat"), ("food101", "food")):
        for p in (5, 10, 25):
            pr, base = champion(ds, p)
            add("prior|aug", ds, "resnet18", p, base, pr, "prior",
                f"diagfuse_{tag}_aug_{p}pct", "aug", f"diagfuse_{tag}_prioraug_{p}pct")

    # ---- aug|simclr
    for ds, tag, ps in (("food101", "food", (5, 10)), ("eurosat", "esat", (5,))):
        for p in ps:
            pr, base = champion(ds, p)
            add("aug|simclr", ds, "resnet18", p, base, f"diagfuse_{tag}_aug_{p}pct", "aug",
                f"diaggrid_simclr_{tag}_{p}pct", "simclr", f"diagprosp_augssl_{tag}_{p}pct")
    for p in (1, 2, 3, 5, 7, 10, 15, 25, 100):
        add("aug|simclr", "cifar100", "vit_tiny", p, f"diagvit_none_{p}pct",
            f"diagdeit_none_{p}pct", "aug", f"diagsslvit_simclr_{p}pct", "simclr",
            f"diagdeitssl_simclr_{p}pct")
    add("aug|simclr", "tin", "vit_tiny", 10, "diagvit_none_tin_10pct", "diagdeit_none_tin_10pct",
        "aug", "diagsslvit_tin_simclr_10pct", "simclr", "diagdeitssl_tin_simclr_10pct")

    # ---- aug|mae
    for p in (1, 2, 5, 10, 25):
        add("aug|mae", "cifar100", "vit_tiny", p, f"diagvit_none_{p}pct",
            f"diagdeit_none_{p}pct", "aug", f"diagmae_vit_{p}pct", "mae",
            f"diagdeitmae_vit_{p}pct")

    # ---- stem|prior (two hand-crafted priors: forward-path energy stem +
    # aux at lambda0=2.0, the 2026-07-17 fwd-combo; combo uses the lambda0=2
    # aux, so the prior arm is auxmag_Npct_sched2)
    for p in (1, 2):
        add("stem|prior", "cifar100", "resnet18", p, f"abl{p}_none", f"enmag_{p}pct", "stem",
            f"auxmag_{p}pct_sched2", "prior", f"combo_{p}pct")

    # ---- prior|transfer (the interference family): baseline = scratch none,
    # A = ImageNet-pretrained init (diagtransfer2_*_none), B = scratch prior,
    # combo = diagtransfer2_*_aux
    for tag, ds in (("c10", "cifar10"), ("c100", "cifar100"), ("esat", "eurosat"),
                    ("food", "food101"), ("dtd", "dtd"), ("path", "pathmnist"),
                    ("cub", "cub"), ("stl", "stl10"), ("tin", "tin")):
        for p in (1, 2, 3, 5, 7, 10, 15, 20, 25, 50, 100):
            ch = champion(ds, p)
            if ch is None:
                continue
            pr, base = ch
            add("prior|transfer", ds, "resnet18", p, base, f"diagtransfer2_{tag}_none_{p}pct",
                "transfer", pr, "prior", f"diagtransfer2_{tag}_aux_{p}pct")
    return P


def load_results():
    rows = {}
    with open("results/all_results.csv") as f:
        for r in csv.DictReader(f):
            rows[r["cell"]] = r
    return rows


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def cell_stats(rows, cell):
    r = rows.get(cell)
    if r is None:
        return None
    n = int(r["n_seeds"] or 0)
    acc = _f(r["acc_mean"])
    sd = _f(r["acc_std"])
    sem = sd / math.sqrt(n) if n > 1 and not math.isnan(sd) else float("nan")
    pn = int(r["n_probe_seeds"] or 0)
    pm = _f(r["probe_mean"])
    psd = _f(r["probe_std"])
    psem = psd / math.sqrt(pn) if pn > 1 and not math.isnan(psd) else float("nan")
    return {"cell": cell, "n": n, "acc": acc, "acc_sem": sem, "probe": pm,
            "probe_sem": psem, "n_probe": pn, "dataset": r["dataset"],
            "backbone": r["backbone"], "pct": int(r["subset_pct"]),
            "pretrained": r["pretrained"] in ("yes", "True", "true", "1"), "init_from": r["init_from"]}


def outcome(d_combo, dA, dB, sem):
    best = max(dA, dB)
    if d_combo - best > 2 * sem:
        return "STACK"
    if d_combo - best < -2 * sem:
        return "SUBSTITUTE(cost)"
    return "SUBSTITUTE"


def find_ckpts(cell):
    """(tree, ckpt_name, [seed dirs]) for the first tree holding checkpoints."""
    for tree in RUN_TREES:
        for ck in ("last.pt", "best.pt"):
            seeds = sorted(d for d in glob.glob(f"{tree}/{cell}/seed*/{ck}"))
            if seeds:
                return tree, ck, [os.path.basename(os.path.dirname(s)) for s in seeds]
    return None, None, []


def build_pairs():
    rows = load_results()
    spec = pair_spec()
    pairs, unresolved = [], []
    for p in spec:
        st = {k: cell_stats(rows, c) for k, c in
              (("base", p["baseline"]), ("A", p["A"]["cell"]), ("B", p["B"]["cell"]),
               ("C", p["combo"]))}
        missing = [k for k, v in st.items() if v is None or v["n"] < 3]
        if missing:
            unresolved.append({"name": p["name"], "why": f"missing/under-seeded {missing}"})
            continue
        if any(math.isnan(st[k]["probe"]) for k in ("base", "A", "B")):
            unresolved.append({"name": p["name"], "why": "no recorded G on "
                               + ",".join(k for k in ("base", "A", "B") if math.isnan(st[k]["probe"]))})
            continue
        b = st["base"]
        r = dict(p)
        for k in ("A", "B", "C"):
            s = st[k]
            r[f"d{k}"] = s["acc"] - b["acc"]
            r[f"d{k}_sem"] = math.hypot(s["acc_sem"], b["acc_sem"])
            r[f"g{k}"] = s["probe"] - b["probe"]
            r[f"g{k}_sem"] = math.hypot(s["probe_sem"], b["probe_sem"])
            r[f"acc{k}"] = s["acc"]
            r[f"n{k}"] = s["n"]
        r["acc_base"] = b["acc"]
        r["n_base"] = b["n"]
        r["pretrained_A"] = st["A"]["pretrained"]
        r["pretrained_B"] = st["B"]["pretrained"]
        best = "A" if r["dA"] >= r["dB"] else "B"
        sem = math.hypot(st["C"]["acc_sem"], st[best]["acc_sem"])
        r["combo_minus_best"] = st["C"]["acc"] - st[best]["acc"]
        r["combo_minus_best_sem"] = sem
        r["sigma"] = r["combo_minus_best"] / sem if sem > 0 else float("nan")
        r["outcome"] = outcome(r["dC"], r["dA"], r["dB"], sem)
        r["resolved"] = abs(r["combo_minus_best"]) > 2 * sem
        # local checkpoints (for the feature-side predicates)
        ck = {}
        for k, c in (("base", p["baseline"]), ("A", p["A"]["cell"]), ("B", p["B"]["cell"])):
            tree, name, seeds = find_ckpts(c)
            ck[k] = {"tree": tree, "ckpt": name, "seeds": seeds[:MAX_SEEDS]}
        r["ckpts"] = ck
        r["ckpt_complete"] = all(len(v["seeds"]) >= 2 for v in ck.values())
        pairs.append(r)
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(PAIRS_JSON, "w") as f:
        json.dump({"pairs": pairs, "unresolved": unresolved}, f, indent=1)
    print(f"derivation set: {len(pairs)} pairs resolved, {len(unresolved)} unresolved")
    fam = {}
    for r in pairs:
        fam.setdefault(r["family"], []).append(r["outcome"])
    for k, v in sorted(fam.items()):
        print(f"  {k:16s} n={len(v):3d}  STACK {v.count('STACK'):2d}  SUB {v.count('SUBSTITUTE'):2d}"
              f"  INTERFERE {v.count('SUBSTITUTE(cost)'):2d}"
              f"  ckpt-complete {sum(r['ckpt_complete'] for r in pairs if r['family']==k)}")
    for u in unresolved:
        print("  UNRESOLVED", u["name"], "--", u["why"])
    return pairs


# --------------------------------------------------------------------------
# 2. checkpoint-based extraction


def find_config(cell):
    hits = glob.glob(f"configs/**/{cell}.yaml", recursive=True)
    if not hits:
        raise FileNotFoundError(cell)
    # prefer the shallowest match
    return sorted(hits, key=lambda h: (h.count("/"), h))[0]


def cka_subset(y, n_total, seed=0):
    """Stratified deterministic subset of test indices for CKA."""
    y = np.asarray(y)
    classes = np.unique(y)
    per = max(1, n_total // len(classes))
    out = []
    for c in classes:
        idx = np.flatnonzero(y == c)
        rs = np.random.RandomState(seed * 7919 + int(c))
        out.extend(idx[rs.permutation(len(idx))][:per].tolist())
    return np.array(sorted(out))


def _test_loader(ds, data_root, workers):
    import torch
    from torch.utils.data import DataLoader
    import data as data_mod
    if ds == "food101":
        # 25k full-resolution JPEGs per pass; reuse B1's byte-verified cache
        from analysis.prospective_currency import build_cache
        te = build_cache(ds, data_root, False, "test")
    else:
        te = data_mod.build_dataset(ds, data_root, train=False, download=False)
    return DataLoader(te, batch_size=256, num_workers=workers, shuffle=False)


def extract_all(pairs, data_root, workers, only_dataset=None):
    import torch
    import yaml
    import data as data_mod
    from momentstem import build_model
    from analysis.linear_probe import extract
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(CACHE_DIR, exist_ok=True)
    # cells needed, grouped by dataset
    need = {}
    for r in pairs:
        if not r["ckpt_complete"]:
            continue
        for k, c in (("base", r["baseline"]), ("A", r["A"]["cell"]), ("B", r["B"]["cell"])):
            ck = r["ckpts"][k]
            need.setdefault(r["dataset"], {})[c] = ck
    for ds, cells in sorted(need.items()):
        if only_dataset and ds != only_dataset:
            continue
        todo = {c: ck for c, ck in cells.items()
                if any(not os.path.exists(os.path.join(CACHE_DIR, f"{c}_{s}.npz"))
                       for s in ck["seeds"])}
        if not todo:
            print(f"[{ds}] all {len(cells)} cells cached")
            continue
        print(f"[{ds}] extracting {len(todo)} cells", flush=True)
        loader = _test_loader(ds, data_root, workers)
        ys = np.concatenate([y.numpy() for _, y in loader])
        sub = cka_subset(ys, N_CKA)
        n_classes = data_mod.NUM_CLASSES[ds]
        for cell, ck in sorted(todo.items()):
            cfg = yaml.safe_load(open(find_config(cell)))
            for seed in ck["seeds"]:
                out = os.path.join(CACHE_DIR, f"{cell}_{seed}.npz")
                if os.path.exists(out):
                    continue
                path = os.path.join(ck["tree"], cell, seed, ck["ckpt"])
                model = build_model(
                    cfg["backbone"], cfg.get("stem", "none"),
                    num_classes=data_mod.NUM_CLASSES[cfg["dataset"]],
                    small_input=cfg.get("small_input", True),
                    stem_kernel_size=cfg.get("stem_kernel_size", 11),
                    stem_kwargs=cfg.get("stem_kwargs"),
                    head_pool=cfg.get("head_pool"), head=cfg.get("head"),
                    moment_aux=cfg.get("moment_aux"),
                    image_size=data_mod.IMAGE_SIZE[cfg["dataset"]],
                    in_channels=data_mod.INPUT_CHANNELS.get(cfg["dataset"], 3),
                ).to(device)
                sd = torch.load(path, map_location=device)
                if isinstance(sd, dict) and "state_dict" in sd:
                    sd = sd["state_dict"]
                # legacy key naming of the fixed-lambda aux family
                if any(k.startswith("moment_stem.") for k in sd) and \
                        not any(k.startswith("moment_stem.") for k in model.state_dict()):
                    sd = {k.replace("moment_stem.", "target.stem.", 1): v for k, v in sd.items()}
                model.load_state_dict(sd)
                model.eval()
                feats, y = extract(model, loader, device)
                # e2e predictions through the trained classifier
                preds = []
                with torch.no_grad():
                    for x, _ in loader:
                        x = x.to(device, non_blocking=True)
                        with torch.autocast("cuda", enabled=device.type == "cuda"):
                            o = model(x)
                        if isinstance(o, (tuple, list)):
                            o = o[0]
                        preds.append(o.float().argmax(1).cpu())
                preds = torch.cat(preds).numpy()
                y = y.numpy()
                ok = preds == y
                pc = np.array([ok[y == c].mean() if (y == c).any() else np.nan
                               for c in range(n_classes)])
                np.savez_compressed(out, feats=feats[sub].numpy().astype(np.float16),
                                    sub=sub, ok=ok, per_class=pc, acc=ok.mean(),
                                    ckpt=ck["ckpt"], tree=ck["tree"])
                print(f"  {cell} {seed} [{ck['ckpt']}]: acc {ok.mean()*100:.2f}", flush=True)
                del model
                torch.cuda.empty_cache()


# --------------------------------------------------------------------------
# 3. predicates


def linear_cka(X, Y):
    X = X - X.mean(0, keepdims=True)
    Y = Y - Y.mean(0, keepdims=True)
    num = np.linalg.norm(Y.T @ X, "fro") ** 2
    den = np.linalg.norm(X.T @ X, "fro") * np.linalg.norm(Y.T @ Y, "fro")
    return float(num / den) if den > 0 else float("nan")


def load_cache(cell, seeds):
    out = []
    for s in seeds:
        p = os.path.join(CACHE_DIR, f"{cell}_{s}.npz")
        if os.path.exists(p):
            z = np.load(p, allow_pickle=True)
            out.append({"feats": z["feats"].astype(np.float32), "ok": z["ok"],
                        "pc": z["per_class"], "acc": float(z["acc"])})
    return out


def fixed_similarity(base_masks, a_masks, b_masks):
    """prospective_currency's normalized fixed-set overlap, verbatim."""
    def fixed(base_ok, arm_ok):
        return (~base_ok) & arm_ok

    def jac(x, y):
        u = (x | y).sum()
        return float((x & y).sum() / u) if u else 0.0

    def within(masks, bases):
        v = [jac(fixed(bases[i], masks[i]), fixed(bases[j], masks[j]))
             for i in range(len(masks)) for j in range(i + 1, len(masks))]
        return float(np.mean(v)) if v else float("nan")

    n = min(len(base_masks), len(a_masks), len(b_masks))
    base_masks, a_masks, b_masks = base_masks[:n], a_masks[:n], b_masks[:n]
    wa, wb = within(a_masks, base_masks), within(b_masks, base_masks)
    across = [jac(fixed(base_masks[i], a_masks[i]), fixed(base_masks[j], b_masks[j]))
              for i in range(n) for j in range(n)]
    a = float(np.mean(across))
    return a / np.sqrt(wa * wb) if wa > 0 and wb > 0 else float("nan")


SHOTS_DIR = os.path.join(OUT_DIR, "shots25")


def shots25(cell):
    """25-shot probe test accuracy (mean over seeds) if recorded -- from the
    cell's linear_probe_shots.json, or from this module's own 25-shot pass."""
    cands = [f"{tree}/{cell}/linear_probe_shots.json" for tree in RUN_TREES]
    cands.append(os.path.join(SHOTS_DIR, f"{cell}.json"))
    for p in cands:
        if os.path.exists(p):
            d = json.load(open(p))
            v = [r["shots"].get("25") or r["shots"].get(25) for r in d["results"]
                 if "shots" in r]
            v = [x for x in v if x is not None]
            if v:
                return float(np.mean(v)) * 100
    return float("nan")


def feature_predicates(r):
    """Everything that needs the checkpoints. NaN where they are absent."""
    out = {k: float("nan") for k in
           ("cka_AB", "cka_Abase", "cka_Bbase", "cka_AA", "cka_BB", "ncka_AB",
            "cka_base_min", "cka_base_absdiff",
            "pc_corr", "pc_corr_within", "S_AB", "ok_acc_base", "ok_acc_A", "ok_acc_B")}
    if not r["ckpt_complete"]:
        return out
    ck = r["ckpts"]
    base = load_cache(r["baseline"], ck["base"]["seeds"])
    A = load_cache(r["A"]["cell"], ck["A"]["seeds"])
    B = load_cache(r["B"]["cell"], ck["B"]["seeds"])
    if min(len(base), len(A), len(B)) < 2:
        return out
    n = min(len(base), len(A), len(B))
    base, A, B = base[:n], A[:n], B[:n]
    # CKA across seeds (all i,j pairs for cross-arm; i<j within-arm)
    out["cka_AB"] = float(np.mean([linear_cka(A[i]["feats"], B[j]["feats"])
                                   for i in range(n) for j in range(n)]))
    out["cka_Abase"] = float(np.mean([linear_cka(A[i]["feats"], base[j]["feats"])
                                      for i in range(n) for j in range(n)]))
    out["cka_Bbase"] = float(np.mean([linear_cka(B[i]["feats"], base[j]["feats"])
                                      for i in range(n) for j in range(n)]))
    out["cka_AA"] = float(np.mean([linear_cka(A[i]["feats"], A[j]["feats"])
                                   for i, j in combinations(range(n), 2)]))
    out["cka_BB"] = float(np.mean([linear_cka(B[i]["feats"], B[j]["feats"])
                                   for i, j in combinations(range(n), 2)]))
    out["ncka_AB"] = out["cka_AB"] / math.sqrt(out["cka_AA"] * out["cka_BB"])
    out["cka_base_min"] = min(out["cka_Abase"], out["cka_Bbase"])
    out["cka_base_absdiff"] = abs(out["cka_Abase"] - out["cka_Bbase"])
    # per-class delta correlation (mean over seeds of each arm)
    pcb = np.mean([x["pc"] for x in base], 0)
    pca = np.mean([x["pc"] for x in A], 0) - pcb
    pcbb = np.mean([x["pc"] for x in B], 0) - pcb
    m = ~(np.isnan(pca) | np.isnan(pcbb))
    if m.sum() >= 3 and pca[m].std() > 0 and pcbb[m].std() > 0:
        out["pc_corr"] = float(np.corrcoef(pca[m], pcbb[m])[0, 1])
    # within-arm per-class-delta reproducibility (the noise floor of pc_corr)
    wa = [np.corrcoef(A[i]["pc"] - pcb, A[j]["pc"] - pcb)[0, 1]
          for i, j in combinations(range(n), 2)]
    wb = [np.corrcoef(B[i]["pc"] - pcb, B[j]["pc"] - pcb)[0, 1]
          for i, j in combinations(range(n), 2)]
    out["pc_corr_within"] = float(np.nanmean(wa + wb))
    out["S_AB"] = fixed_similarity([x["ok"] for x in base], [x["ok"] for x in A],
                                   [x["ok"] for x in B])
    out["ok_acc_base"] = float(np.mean([x["acc"] for x in base]) * 100)
    out["ok_acc_A"] = float(np.mean([x["acc"] for x in A]) * 100)
    out["ok_acc_B"] = float(np.mean([x["acc"] for x in B]) * 100)
    return out


def summary_predicates(r):
    dA, dB, gA, gB = r["dA"], r["dB"], r["gA"], r["gB"]
    mx = max(abs(gA), abs(gB))
    mn = min(abs(gA), abs(gB))
    out = {}
    out["REL"] = abs(gA - gB) / mx if mx > 0 else float("nan")          # (a) published
    out["G_min"] = min(gA, gB)
    out["G_max"] = max(gA, gB)
    out["G_sum"] = gA + gB
    out["D_min"] = min(dA, dB)
    out["D_max"] = max(dA, dB)
    out["readout_A"] = dA - gA
    out["readout_B"] = dB - gB
    out["readout_absmax"] = max(abs(dA - gA), abs(dB - gB))
    out["readout_absdiff"] = abs((dA - gA) - (dB - gB))
    out["G_ratio"] = (mn / mx) if mx > 0 else float("nan")
    out["G_absdiff"] = abs(gA - gB)
    out["D_absdiff"] = abs(dA - dB)
    out["D_sum"] = dA + dB
    # (f) amplifier flag as pre-registered: one source with ~no own G beside
    # a strong one, and that weak source moving accuracy >= 2x its own G
    weak, strong = (("A", "B") if abs(gA) <= abs(gB) else ("B", "A"))
    gw, dw = r[f"g{weak}"], r[f"d{weak}"]
    out["amp_flag"] = float(mn <= 1.0 and mx >= 3.0 and dw >= 2 * gw)
    # softer version: the weak source's accuracy-per-feature excess
    out["amp_soft"] = (dw - gw)
    # (g) baseline height, backbone family, source-type indicators
    out["base_acc"] = r["acc_base"]
    out["is_vit"] = float(r["backbone"].startswith("vit"))
    out["has_aug"] = float("aug" in (r["A"]["label"], r["B"]["label"]))
    out["has_pretrained"] = float(r["pretrained_A"] or r["pretrained_B"])
    out["has_ssl_init"] = float(any(l in ("simclr", "simsiam", "dino", "mae")
                                    for l in (r["A"]["label"], r["B"]["label"])))
    out["pct"] = float(r["pct"])
    # (e) label efficiency: 25-shot probe gap relative to full-train G
    s_base, s_A, s_B = shots25(r["baseline"]), shots25(r["A"]["cell"]), shots25(r["B"]["cell"])
    g25A, g25B = s_A - s_base, s_B - s_base
    out["G25_A"], out["G25_B"] = g25A, g25B
    out["leff_A"] = g25A / gA if abs(gA) > 0.5 else float("nan")
    out["leff_B"] = g25B / gB if abs(gB) > 0.5 else float("nan")
    out["leff_absdiff"] = abs(out["leff_A"] - out["leff_B"])
    out["leff_min"] = min(out["leff_A"], out["leff_B"])
    return out



def shots25_pass(cells, data_root, workers):
    """25-shot linear probes on the frozen features of LOCAL checkpoints,
    same extract()/probe()/shots_subset() as analysis/linear_probe.py.
    Writes results/fusion_predicates/shots25/<cell>.json, and ALSO the
    cell's linear_probe_shots.json when -- and only when -- that file does
    not exist (never overwrites a recorded probe file)."""
    import torch
    import yaml
    import data as data_mod
    from torch.utils.data import DataLoader
    from momentstem import build_model
    from analysis.linear_probe import extract, probe, shots_subset, set_eval_transform
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(SHOTS_DIR, exist_ok=True)
    by_ds = {}
    for cell in cells:
        if not math.isnan(shots25(cell)):
            continue
        tree, ck, seeds = find_ckpts(cell)
        if not seeds:
            continue
        cfg = yaml.safe_load(open(find_config(cell)))
        by_ds.setdefault(cfg["dataset"], []).append((cell, cfg, tree, ck, seeds[:MAX_SEEDS]))
    for ds, items in sorted(by_ds.items()):
        print(f"[{ds}] 25-shot probes for {len(items)} cells", flush=True)
        if ds == "food101":
            from analysis.prospective_currency import build_cache
            tr = build_cache(ds, data_root, True, "train")
            te = build_cache(ds, data_root, False, "test")
        else:
            tr = set_eval_transform(
                data_mod.build_dataset(ds, data_root, train=True, download=False), ds)
            te = data_mod.build_dataset(ds, data_root, train=False, download=False)
        tr_ld = DataLoader(tr, batch_size=512, num_workers=workers, shuffle=False)
        te_ld = DataLoader(te, batch_size=512, num_workers=workers, shuffle=False)
        n_classes = data_mod.NUM_CLASSES[ds]
        for cell, cfg, tree, ck, seeds in items:
            res = []
            for seed in seeds:
                path = os.path.join(tree, cell, seed, ck)
                model = build_model(
                    cfg["backbone"], cfg.get("stem", "none"),
                    num_classes=data_mod.NUM_CLASSES[cfg["dataset"]],
                    small_input=cfg.get("small_input", True),
                    stem_kernel_size=cfg.get("stem_kernel_size", 11),
                    stem_kwargs=cfg.get("stem_kwargs"),
                    head_pool=cfg.get("head_pool"), head=cfg.get("head"),
                    moment_aux=cfg.get("moment_aux"),
                    image_size=data_mod.IMAGE_SIZE[cfg["dataset"]],
                    in_channels=data_mod.INPUT_CHANNELS.get(cfg["dataset"], 3),
                ).to(device)
                sd = torch.load(path, map_location=device)
                if any(k.startswith("moment_stem.") for k in sd) and \
                        not any(k.startswith("moment_stem.") for k in model.state_dict()):
                    sd = {k.replace("moment_stem.", "target.stem.", 1): v for k, v in sd.items()}
                model.load_state_dict(sd)
                model.eval()
                trf, trY = extract(model, tr_ld, device)
                tef, teY = extract(model, te_ld, device)
                sel = shots_subset(trY, 25)
                torch.manual_seed(0)
                _, acc = probe(trf[sel], trY[sel], tef, teY, n_classes, device)
                res.append({"seed": seed, "shots_only": True, "shots": {"25": acc}})
                print(f"  {cell} {seed} [{ck}]: 25-shot test {acc*100:.2f}", flush=True)
                del model
                torch.cuda.empty_cache()
            rec = {"config": find_config(cell), "ckpt": ck, "tree": tree,
                   "note": "25-shot-only probe written by analysis/fusion_predicates.py "
                           "(block B-1); identical extract/probe/shots_subset to linear_probe.py",
                   "results": res}
            with open(os.path.join(SHOTS_DIR, f"{cell}.json"), "w") as f:
                json.dump(rec, f, indent=1)
            target = os.path.join(tree, cell, "linear_probe_shots.json")
            if not os.path.exists(target):
                with open(target, "w") as f:
                    json.dump(rec, f, indent=1)


# --------------------------------------------------------------------------
# wave-3 calls


def wave3_spec():
    """The block-B phase-2 pairs (CLAUDE.md '### B.'): combinations that have
    never been trained. Same resolution as pair_spec(), combo = None."""
    W = []

    def add(family, ds, bb, pct, base, A, Al, B, Bl, note=""):
        W.append({"name": f"{family}:{ds}@{pct}:{bb}" + (f":{note}" if note else ""),
                  "family": family, "dataset": ds, "backbone": bb, "pct": pct,
                  "baseline": base, "A": {"cell": A, "label": Al},
                  "B": {"cell": B, "label": Bl}, "combo": None, "note": note})
    for ds, tag, p in (("dtd", "dtd", 15), ("eurosat", "esat", 10), ("stl10", "stl", 20),
                       ("cifar10", "c10", 5), ("tin", "tin", 5)):
        pr, base = champion(ds, p)
        simclr = "diagssl_tin_simclr_5pct" if ds == "tin" else f"diaggrid_simclr_{tag}_{p}pct"
        add("prior|simclr", ds, "resnet18", p, base, pr, "prior", simclr, "simclr")
    add("prior|dino", "stl10", "vit_tiny", 50, "diaggrid_vit_stl_none_50pct",
        "diaggrid_vit_stl_aux_50pct", "prior", "diaggrid_dino_stl_50pct", "dino")
    add("prior|dino", "food101", "vit_tiny", 25, "diaggrid_vit_food101_none_25pct",
        "diaggrid_vit_food101_aux_25pct", "prior", "diaggrid_dino_food_25pct", "dino")
    add("prior|simsiam", "cifar100", "resnet18", 25, "abl25_none", "auxmag_25pct_sched0",
        "prior", "diaggrid_simsiam_c100_25pct", "simsiam")
    for ds, tag, p in (("food101", "food", 25), ("eurosat", "esat", 10)):
        pr, base = champion(ds, p)
        add("aug|simclr", ds, "resnet18", p, base, f"diagfuse_{tag}_aug_{p}pct", "aug",
            f"diaggrid_simclr_{tag}_{p}pct", "simclr")
    for p in (5, 10, 25):
        add("prior|mae", "cifar100", "vit_tiny", p, f"diagvit_none_{p}pct",
            f"diagvit_aux_{p}pct", "prior", f"diagmae_vit_{p}pct", "mae")
    return W


def resolve_singles(spec):
    """Resolve a pair spec WITHOUT a combo (wave 3): deltas, G, checkpoints."""
    rows = load_results()
    out = []
    for p in spec:
        st = {k: cell_stats(rows, c) for k, c in
              (("base", p["baseline"]), ("A", p["A"]["cell"]), ("B", p["B"]["cell"]))}
        r = dict(p)
        miss = [k for k, v in st.items() if v is None or v["n"] < 3]
        r["missing"] = miss
        if miss:
            out.append(r)
            continue
        b = st["base"]
        for k in ("A", "B"):
            s = st[k]
            r[f"d{k}"] = s["acc"] - b["acc"]
            r[f"d{k}_sem"] = math.hypot(s["acc_sem"], b["acc_sem"])
            r[f"g{k}"] = s["probe"] - b["probe"]
            r[f"g{k}_sem"] = math.hypot(s["probe_sem"], b["probe_sem"])
            r[f"acc{k}"] = s["acc"]
        r["acc_base"] = b["acc"]
        r["pretrained_A"] = st["A"]["pretrained"]
        r["pretrained_B"] = st["B"]["pretrained"]
        ck = {}
        for k, c in (("base", p["baseline"]), ("A", p["A"]["cell"]), ("B", p["B"]["cell"])):
            tree, name, seeds = find_ckpts(c)
            ck[k] = {"tree": tree, "ckpt": name, "seeds": seeds[:MAX_SEEDS]}
        r["ckpts"] = ck
        r["ckpt_complete"] = all(len(v["seeds"]) >= 2 for v in ck.values())
        r["carveout"] = os.path.exists(f"data/valcarve/{p['dataset']}.json")
        out.append(r)
    return out


def published_call(r):
    from analysis.prospective_currency import decide
    return decide(r["dA"], r["dB"], r["gA"], r["gB"], r["pretrained_A"], r["pretrained_B"])[0]


# --------------------------------------------------------------------------
# 4. scoring


def auc(scores, labels):
    """Rank AUC of `scores` for labels==1 (higher score -> more positive)."""
    s = np.asarray(scores, float)
    y = np.asarray(labels, int)
    m = ~np.isnan(s)
    s, y = s[m], y[m]
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan"), 0
    gt = (pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()
    return float(gt / (len(pos) * len(neg))), int(m.sum())


def boot_auc(scores, labels, n=2000, seed=0):
    s = np.asarray(scores, float)
    y = np.asarray(labels, int)
    m = ~np.isnan(s)
    s, y = s[m], y[m]
    rs = np.random.RandomState(seed)
    vals = []
    for _ in range(n):
        idx = rs.randint(0, len(s), len(s))
        a, _ = auc(s[idx], y[idx])
        if not math.isnan(a):
            vals.append(a)
    if not vals:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def best_threshold(scores, labels, direction):
    """Threshold maximizing accuracy of (score >= t) == positive when
    direction=+1, or (score <= t) when direction=-1. Returns (t, acc)."""
    s = np.asarray(scores, float)
    y = np.asarray(labels, int)
    m = ~np.isnan(s)
    s, y = s[m], y[m]
    if len(s) == 0:
        return float("nan"), float("nan")
    cand = np.unique(s)
    mids = np.concatenate([[cand[0] - 1e-9], (cand[:-1] + cand[1:]) / 2, [cand[-1] + 1e-9]])
    best = (float("nan"), -1)
    for t in mids:
        pred = (s >= t) if direction > 0 else (s <= t)
        acc = (pred.astype(int) == y).mean()
        if acc > best[1]:
            best = (float(t), float(acc))
    return best


def apply_rule(score, t, direction):
    if np.isnan(score):
        return None
    return bool(score >= t) if direction > 0 else bool(score <= t)


PREDICATES = [
    # name, description, computed-from
    ("REL", "|GA-GB|/max(|GA|,|GB|)  (published G-agreement)", "summary"),
    ("G_min", "min(GA,GB)", "summary"),
    ("G_max", "max(GA,GB)", "summary"),
    ("G_sum", "GA+GB", "summary"),
    ("D_min", "min(dA,dB)", "summary"),
    ("D_max", "max(dA,dB)", "summary"),
    ("readout_absmax", "max |d-G| over the two arms", "summary"),
    ("readout_absdiff", "|(dA-GA) - (dB-GB)|", "summary"),
    ("G_ratio", "min|G|/max|G|  (= 1-REL)", "summary"),
    ("G_absdiff", "|GA-GB| (absolute strength asymmetry)", "summary"),
    ("D_absdiff", "|dA-dB|", "summary"),
    ("D_sum", "dA+dB", "summary"),
    ("amp_soft", "weak source's d-G (accuracy beyond its own features)", "summary"),
    ("amp_flag", "amplifier flag (pre-registered form)", "summary"),
    ("base_acc", "baseline accuracy", "summary"),
    ("pct", "data fraction", "summary"),
    ("is_vit", "attention backbone", "summary"),
    ("has_aug", "one source is DeiT augmentation", "summary"),
    ("has_pretrained", "one source is an ImageNet init", "summary"),
    ("has_ssl_init", "one source is an SSL init", "summary"),
    ("cka_AB", "linear CKA(A,B) penultimate features, 2k test imgs", "ckpt"),
    ("ncka_AB", "CKA(A,B)/sqrt(CKA(A,A')CKA(B,B'))", "ckpt"),
    ("cka_Abase", "CKA(A,base)", "ckpt"),
    ("cka_Bbase", "CKA(B,base)", "ckpt"),
    ("cka_base_min", "min(CKA(A,base),CKA(B,base))  (the stronger reshaping)", "ckpt"),
    ("cka_base_absdiff", "|CKA(A,base)-CKA(B,base)|", "ckpt"),
    ("pc_corr", "Pearson r of per-class deltas (A-base) vs (B-base)", "ckpt"),
    ("S_AB", "normalized fix-set overlap S(A,B)", "ckpt"),
    ("leff_absdiff", "|G25/G| label-efficiency asymmetry", "shots"),
    ("leff_min", "min label-efficiency ratio", "shots"),
]


def lofo(rows, key, positives):
    """Leave-one-family-out: fit direction+threshold on the other families,
    test on the held-out one. Returns per-family accuracy and pooled."""
    fams = sorted(set(r["family"] for r in rows))
    per = {}
    hits = tot = 0
    for f in fams:
        tr = [r for r in rows if r["family"] != f and not np.isnan(r["pred"][key])]
        te = [r for r in rows if r["family"] == f and not np.isnan(r["pred"][key])]
        if len(te) == 0 or len(tr) < 5:
            continue
        s = [r["pred"][key] for r in tr]
        y = [int(r["outcome"] in positives) for r in tr]
        a, _ = auc(s, y)
        direction = 1 if (not math.isnan(a) and a >= 0.5) else -1
        t, _ = best_threshold(s, y, direction)
        h = 0
        for r in te:
            call = apply_rule(r["pred"][key], t, direction)
            h += int(call == (r["outcome"] in positives))
        per[f] = {"n": len(te), "acc": h / len(te), "thr": t, "dir": direction,
                  "n_pos": sum(int(r["outcome"] in positives) for r in te)}
        hits += h
        tot += len(te)
    return per, (hits / tot if tot else float("nan")), tot


def lofo_auc(rows, key, positives):
    """Out-of-family AUC: for each held-out family, rank its pairs using the
    direction fit on the others, then pool the held-out scores (signed by
    the fitted direction) and compute one AUC. Also returns the per-family
    AUC where the held-out family has both classes."""
    fams = sorted(set(r["family"] for r in rows))
    pooled_s, pooled_y, per = [], [], {}
    for f in fams:
        tr = [r for r in rows if r["family"] != f and not np.isnan(r["pred"][key])]
        te = [r for r in rows if r["family"] == f and not np.isnan(r["pred"][key])]
        if not te or len(tr) < 5:
            continue
        a, _ = auc([r["pred"][key] for r in tr], [int(r["outcome"] in positives) for r in tr])
        d = 1 if (not math.isnan(a) and a >= 0.5) else -1
        s = [d * r["pred"][key] for r in te]
        y = [int(r["outcome"] in positives) for r in te]
        pooled_s += s
        pooled_y += y
        fa, _ = auc(s, y)
        per[f] = fa
    a, n = auc(pooled_s, pooled_y)
    return a, n, per


def two_rule(rows, gate_key, gate_thr, gate_dir, gate_call, key, positives):
    """A 2-predicate rule: if gate fires -> gate_call; else threshold `key`
    (fit on the non-gated training rows). Scored LOFO like the singles."""
    fams = sorted(set(r["family"] for r in rows))
    hits = tot = 0
    per = {}
    for f in fams:
        tr = [r for r in rows if r["family"] != f]
        te = [r for r in rows if r["family"] == f]
        tr2 = [r for r in tr if not apply_rule(r["pred"][gate_key], gate_thr, gate_dir)
               and not np.isnan(r["pred"][key])]
        if len(te) == 0 or len(tr2) < 5:
            continue
        s = [r["pred"][key] for r in tr2]
        y = [int(r["outcome"] in positives) for r in tr2]
        a, _ = auc(s, y)
        d = 1 if (not math.isnan(a) and a >= 0.5) else -1
        t, _ = best_threshold(s, y, d)
        h = n = 0
        for r in te:
            g = apply_rule(r["pred"][gate_key], gate_thr, gate_dir)
            if g:
                call = gate_call
            else:
                if np.isnan(r["pred"][key]):
                    continue
                call = apply_rule(r["pred"][key], t, d)
            n += 1
            h += int(call == (r["outcome"] in positives))
        if n:
            per[f] = {"n": n, "acc": h / n, "thr": t, "dir": d}
            hits += h
            tot += n
    return per, (hits / tot if tot else float("nan")), tot


def score(pairs, positives=("STACK",), tag="all"):
    for r in pairs:
        r["pred"] = {**summary_predicates(r), **feature_predicates(r)}
    rows = pairs
    res = {"n_pairs": len(rows), "positives": list(positives),
           "n_pos": sum(int(r["outcome"] in positives) for r in rows), "predicates": {}}
    for key, desc, src in PREDICATES:
        s = [r["pred"][key] for r in rows]
        y = [int(r["outcome"] in positives) for r in rows]
        a, n = auc(s, y)
        lo, hi = boot_auc(s, y)
        # orient so that AUC >= 0.5 (direction recorded)
        d = 1 if (not math.isnan(a) and a >= 0.5) else -1
        a_or = a if d > 0 else 1 - a
        lo_or, hi_or = (lo, hi) if d > 0 else (1 - hi, 1 - lo)
        t, acc_in = best_threshold(s, y, d)
        per, lofo_acc, lofo_n = lofo(rows, key, positives)
        la, ln, lper = lofo_auc(rows, key, positives)
        res["predicates"][key] = {
            "desc": desc, "source": src, "n": n, "auc": a_or, "ci": [lo_or, hi_or],
            "direction": d, "threshold_in_sample": t, "acc_in_sample": acc_in,
            "lofo_acc": lofo_acc, "lofo_n": lofo_n, "lofo_per_family": per,
            "lofo_auc": la, "lofo_auc_n": ln, "lofo_auc_per_family": lper}
    return res


# --------------------------------------------------------------------------


def write_md(pairs, scores, path):
    L = []
    L.append("# Fusion predicates, derived on every measured combination (block B-1)\n")
    L.append("Regenerate: `python analysis/fusion_predicates.py --build --score` "
             "(plus `--extract` for the checkpoint-based predicates and "
             "`--shots25` for the label-efficiency ones).\n")
    L.append(
        "Context (see the CLAUDE.md wave-1/wave-2 entries, 2026-08-19/20): the published\n"
        "Algorithm-1 predicate (REL, G-magnitude agreement) called 1 of 9 held-out pairs;\n"
        "the strength-asymmetry rule derived from wave 1 scored 1/6 then 0/2 one family\n"
        "over; fix-set overlap S(A,B) was at chance. This pass asks the question on every\n"
        "measured combination at once, with leave-one-FAMILY-out as the honesty criterion\n"
        "(family = source-type pair). OUTCOME LABELS use the recorded test-split exporter\n"
        "values (results/all_results.csv), 2-SEM rule; 'INTERFERE' = SUBSTITUTE(cost).\n"
        "Checkpoint predicates: last.pt (best.pt fallback), max 3 seeds, penultimate\n"
        "features on a fixed stratified 2,000-image test subset.\n")
    n = len(pairs)
    fam = {}
    for r in pairs:
        fam.setdefault(r["family"], []).append(r)
    L.append(f"## Derivation set: {n} pairs, {len(fam)} families\n")
    L.append("| family | n | STACK | SUBSTITUTE | INTERFERE | resolved | ckpt-complete |")
    L.append("|---|---|---|---|---|---|---|")
    for f, rs in sorted(fam.items()):
        o = [r["outcome"] for r in rs]
        L.append(f"| {f} | {len(rs)} | {o.count('STACK')} | {o.count('SUBSTITUTE')} | "
                 f"{o.count('SUBSTITUTE(cost)')} | {sum(r['resolved'] for r in rs)} | "
                 f"{sum(r['ckpt_complete'] for r in rs)} |")
    L.append("")
    L.append("### Pairs\n")
    L.append("| pair | dA | dB | GA | GB | dC | GC | combo-best | sigma | outcome | REL | CKA(A,B) | nCKA | pc_r | S | leff |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in sorted(pairs, key=lambda r: (r["family"], r["dataset"], r["backbone"], r["pct"], r["note"])):
        p = r["pred"]
        fmt = lambda v: "" if (v is None or (isinstance(v, float) and math.isnan(v))) else f"{v:.2f}"
        L.append(f"| {r['name']} | {r['dA']:.2f} | {r['dB']:.2f} | {r['gA']:.2f} | {r['gB']:.2f} | "
                 f"{r['dC']:.2f} | {fmt(r['gC'])} | {r['combo_minus_best']:+.2f} | {r['sigma']:+.1f} | "
                 f"{r['outcome']} | {fmt(p['REL'])} | {fmt(p['cka_AB'])} | {fmt(p['ncka_AB'])} | "
                 f"{fmt(p['pc_corr'])} | {fmt(p['S_AB'])} | {fmt(p['leff_absdiff'])} |")
    L.append("")
    for tag, sc in scores.items():
        L.append(f"## Predicate scores: {tag}  (n={sc['n_pairs']}, positives={sc['positives']}, "
                 f"n_pos={sc['n_pos']})\n")
        L.append("| predicate | n | AUC (in-sample, oriented) | 95% CI | dir | thr (in-sample) | acc (in) | LOFO acc | LOFO n | LOFO AUC (pooled OOF) | per-family LOFO acc |")
        L.append("|---|---|---|---|---|---|---|---|---|---|---|")
        for key, d in sorted(sc["predicates"].items(), key=lambda kv: -(kv[1]["lofo_auc"] if not math.isnan(kv[1]["lofo_auc"]) else -1)):
            pf = " ".join(f"{f}:{v['acc']:.2f}({v['n']})" for f, v in sorted(d["lofo_per_family"].items()))
            L.append(f"| {key} | {d['n']} | {d['auc']:.3f} | [{d['ci'][0]:.2f},{d['ci'][1]:.2f}] | "
                     f"{'>=' if d['direction']>0 else '<='} | {d['threshold_in_sample']:.3f} | "
                     f"{d['acc_in_sample']:.2f} | {d['lofo_acc']:.2f} | {d['lofo_n']} | "
                     f"{d['lofo_auc']:.3f} | {pf} |")
        L.append("")
        if "two_rules" in sc:
            L.append("### Two-predicate rules (gate, then threshold), LOFO\n")
            L.append("| rule | LOFO acc | n | per-family |")
            L.append("|---|---|---|---|")
            for name, d in sc["two_rules"].items():
                pf = " ".join(f"{f}:{v['acc']:.2f}({v['n']})" for f, v in sorted(d["per"].items()))
                L.append(f"| {name} | {d['lofo_acc']:.2f} | {d['n']} | {pf} |")
            L.append("")
        if "verdict" in sc:
            L.append("### Verdict\n")
            L.append(sc["verdict"])
            L.append("")
    with open(path, "w") as f:
        f.write("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--calls", action="store_true")
    ap.add_argument("--shots25", action="store_true")
    ap.add_argument("--shots25-wave3", action="store_true",
                    help="also run 25-shot probes on the wave-3 single arms")
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--only-dataset", default=None)
    args = ap.parse_args()

    if args.build:
        pairs = build_pairs()
    else:
        pairs = json.load(open(PAIRS_JSON))["pairs"]

    if args.extract:
        extract_all(pairs, args.data_root, args.workers, args.only_dataset)
        if args.calls or args.shots25_wave3:
            w3 = [r for r in resolve_singles(wave3_spec()) if not r["missing"]]
            extract_all(w3, args.data_root, args.workers, args.only_dataset)

    if args.shots25:
        cells = set()
        for r in pairs:
            if r["family"] == "prior|transfer":
                continue  # the interference family is scored without (e)
            cells.update([r["baseline"], r["A"]["cell"], r["B"]["cell"]])
        if args.shots25_wave3:
            for r in resolve_singles(wave3_spec()):
                if not r["missing"]:
                    cells.update([r["baseline"], r["A"]["cell"], r["B"]["cell"]])
        shots25_pass(sorted(cells), args.data_root, args.workers)

    if args.score:
        scores = {}
        core = [r for r in pairs if r["family"] != "prior|transfer"]
        scores["core (no transfer): STACK vs rest"] = score(core, ("STACK",))
        scores["core, resolved only: STACK vs INTERFERE"] = score(
            [r for r in core if r["resolved"]], ("STACK",))
        scores["all families: STACK vs rest"] = score(pairs, ("STACK",))
        scores["core: INTERFERE vs rest"] = score(core, ("SUBSTITUTE(cost)",))
        # 2-predicate rules on the core set
        for tag in ("core (no transfer): STACK vs rest",):
            rows = core
            for r in rows:
                r["pred"] = {**summary_predicates(r), **feature_predicates(r)}
            two = {}
            for gate, gthr, gdir, gcall in (("has_aug", 0.5, 1, True), ("amp_flag", 0.5, 1, True),
                                           ("has_pretrained", 0.5, 1, False)):
                for key in ("REL", "cka_AB", "ncka_AB", "pc_corr", "G_min", "readout_absmax"):
                    per, acc, nn = two_rule(rows, gate, gthr, gdir, gcall, key, ("STACK",))
                    two[f"{gate}->{'STACK' if gcall else 'not'} else {key}"] = {
                        "lofo_acc": acc, "n": nn, "per": per}
            scores[tag]["two_rules"] = two
            # a predicate computable on only a corner of the set cannot be
            # "best": require coverage of >= 60% of the pairs it is judged on
            min_n = 0.6 * scores[tag]["n_pairs"]
            best = max(((k, v) for k, v in scores[tag]["predicates"].items()
                        if v["lofo_auc_n"] >= min_n),
                       key=lambda kv: kv[1]["lofo_auc"] if not math.isnan(kv[1]["lofo_auc"]) else -1)
            k, d = best
            reach = d["lofo_auc"] >= AUC_BAR
            scores[tag]["best_single"] = {"predicate": k, **d}
            scores[tag]["verdict"] = (
                f"Best single-arm predicate out-of-family: **{k}** (LOFO pooled AUC "
                f"{d['lofo_auc']:.3f}, in-sample AUC {d['auc']:.3f} "
                f"[{d['ci'][0]:.2f},{d['ci'][1]:.2f}], LOFO accuracy {d['lofo_acc']:.2f} "
                f"on {d['lofo_n']} pairs). The pre-registered bar is LOFO AUC >= {AUC_BAR}: "
                + ("**REACHED**." if reach else "**NOT REACHED**. No single-arm predicate "
                   "separates STACK from SUBSTITUTE/INTERFERE out of family."))
        with open(os.path.join("results", "fusion_predicates.json"), "w") as f:
            json.dump({"pairs": [{k: v for k, v in r.items() if k != "ckpts"} for r in pairs],
                       "scores": scores}, f, indent=1, default=float)
        write_md(pairs, scores, os.path.join("results", "fusion_predicates.md"))
        for tag, sc in scores.items():
            print(f"\n== {tag}  n={sc['n_pairs']} pos={sc['n_pos']}")
            for key, d in sorted(sc["predicates"].items(), key=lambda kv: -(kv[1]["lofo_auc"] if not math.isnan(kv[1]["lofo_auc"]) else -1)):
                print(f"  {key:16s} n={d['n']:3d} AUC {d['auc']:.3f} [{d['ci'][0]:.2f},{d['ci'][1]:.2f}] "
                      f"dir {d['direction']:+d} thr {d['threshold_in_sample']:.3f} acc_in {d['acc_in_sample']:.2f} "
                      f"LOFO acc {d['lofo_acc']:.2f}/{d['lofo_n']}  LOFO-AUC {d['lofo_auc']:.3f}")
            if "verdict" in sc:
                print(" ", sc["verdict"])

    if args.calls:
        res = json.load(open("results/fusion_predicates.json"))
        sc = res["scores"]["core (no transfer): STACK vs rest"]
        bs = sc["best_single"]
        key, d = bs["predicate"], bs["direction"]
        thr = bs["threshold_in_sample"]
        w3 = resolve_singles(wave3_spec())
        print(f"\nbest rule: {key} {'>=' if d>0 else '<='} {thr:.3f} -> STACK else SUBSTITUTE")
        print(f"{'pair':34s} {'dA':>6} {'dB':>6} {'gA':>6} {'gB':>6}  {key:>9} "
              f"{'best-rule':>10} {'published':>11}  ckpt carve missing")
        out = []
        for r in w3:
            if r["missing"]:
                print(f"{r['name']:34s}  -- missing {r['missing']}")
                out.append({"name": r["name"], "missing": r["missing"]})
                continue
            r["pred"] = {**summary_predicates(r), **feature_predicates(r)}
            v = r["pred"][key]
            call = apply_rule(v, thr, d)
            call = "n/m" if call is None else ("STACK" if call else "SUBSTITUTE")
            pub = published_call(r)
            print(f"{r['name']:34s} {r['dA']:6.2f} {r['dB']:6.2f} {r['gA']:6.2f} {r['gB']:6.2f}  "
                  f"{(f'{v:.3f}' if not np.isnan(v) else 'n/a'):>9} {call:>10} {pub:>11}  "
                  f"{r['ckpt_complete']} {r['carveout']}")
            out.append({"name": r["name"], "dA": r["dA"], "dB": r["dB"], "gA": r["gA"],
                        "gB": r["gB"], "predicate": key, "value": v, "best_rule_call": call,
                        "published_call": pub, "ckpt_complete": r["ckpt_complete"],
                        "carveout": r["carveout"],
                        "pred": {k: v2 for k, v2 in r["pred"].items()}})
        with open(os.path.join("results", "fusion_predicates_wave3_calls.json"), "w") as f:
            json.dump({"rule": {"predicate": key, "direction": d, "threshold": thr,
                                "note": "threshold fit in-sample on the full core "
                                        "derivation set; direction from its AUC"},
                       "calls": out}, f, indent=1, default=float)


if __name__ == "__main__":
    main()
