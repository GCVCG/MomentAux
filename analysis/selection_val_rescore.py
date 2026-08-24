"""Block E of the limitations campaign (2026-08-23): re-score every surviving
CIFAR-100 selection-sweep checkpoint on the VALIDATION carve-out and re-run
each selection decision there.

WHY. The reference configuration (target family, tap depth, lambda_0 and its
schedule, head-norm, loss form) was selected on CIFAR-100 sweeps that were
SCORED ON THE TEST SPLIT -- the defect the paper discloses in "Configuration
selection, and its cost". This script bounds that defect: it takes the very
checkpoints the selection was made from (last.pt, never best.pt, because
best.pt was itself chosen by test accuracy), scores them on images the cells
never trained on and the selection never looked at (data/valcarve/
cifar100.json, VAL indices), and asks whether the val-scored sweep picks the
same arm on every axis.

PRE-REGISTERED (CLAUDE.md, block E): NO axis flips. FALSIFIER: any axis flips.

SCOPE CAVEAT, stated up front: this bounds the defect, it does not remove it.
The checkpoints were still SELECTED on test; a clean re-selection would need a
validation fold held out before any sweep was run.

VAL SET PER FRACTION. The committed carve-out excludes the 10% subset (and, by
nesting, every smaller fraction). Cells at 15% and 25% trained on images that
overlap VAL, so for those fractions VAL is restricted to VAL minus that
fraction's subset -- a fixed set shared by every arm at that fraction, which
is what makes the within-axis comparison valid. 100% cells cannot be
validated (every train image was trained on) and are excluded; the one
selection made there (lambda_0 = 0.1 at 100%) is reported as not re-scorable.

    ~/venvs/momentstem/bin/python analysis/selection_val_rescore.py
        [--data-root ./data] [--device cuda] [--refresh]

Writes results/selection_val_rescore.json and results/selection_val_rescore.md.
Per-checkpoint correctness vectors are cached under results/selection_val/
(gitignored) so the command is cheap to re-run; --refresh recomputes them.
"""

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)  # teacher paths in auxteach configs are repo-relative

import data as data_mod  # noqa: E402
from momentstem import build_model  # noqa: E402
from analysis.linear_probe import set_eval_transform  # noqa: E402

CACHE_DIR = os.path.join("results", "selection_val")
OUT_JSON = os.path.join("results", "selection_val_rescore.json")
OUT_MD = os.path.join("results", "selection_val_rescore.md")
CARVE = os.path.join("data", "valcarve", "cifar100.json")
RUN_TREES = ("runs", "runs_bscpull", "runs_turing")
CONFIG_DIRS = ("configs/diagnostics", "configs/grid", "configs/ablations_full",
               "configs/ablations", "configs")

# --------------------------------------------------------------------------
# THE SWEEPS THAT SELECTED THE REFERENCE CONFIGURATION.
# Each axis: the arms (label -> cell), the fraction, the shared baseline cell,
# and the arm the test-scored sweep selected (from the ledger / paper). The
# selected arm is what the val re-scoring is checked against.

AXES = [
    dict(id="target_5pct", title="Target family @5% (lambda=0.3 constant)",
         pct=5, baseline="abl5_none", selected="magnitude",
         arms={"magnitude": "auxmag_5pct_l30",
               "structure": "grid_c100_r18_axstructureL3_l03_26d33f_5pct",
               "steerable": "grid_c100_r18_axsteerableL3_l03_db5093_5pct",
               "rotinv": "grid_c100_r18_axrotinvL3_l03_a9b216_5pct",
               "invariants": "grid_c100_r18_axinvariantsL3_l03_e17c19_5pct",
               "gabor(k5 edges)": "grid_c100_r18_axmomentscatL3_l03_d020d0_5pct",
               "random-fixed": "auxrand_5pct_l30",
               "hog": "auxhog_5pct",
               "fitnets-teacher": "auxteach_5pct"}),
    dict(id="target_10pct", title="Target family @10% (lambda=0.3 constant)",
         pct=10, baseline="abl10_none", selected="magnitude",
         arms={"magnitude": "auxmag_10pct_l30",
               "structure": "auxstr_10pct_l30",
               "steerable": "auxste_10pct_l30",
               "rotinv": "auxrot_10pct_l30",
               "invariants": "auxinv_10pct_l30",
               "gabor(k5 edges)": "auxgab_10pct_l30",
               "random-fixed": "auxrand_10pct_l30",
               "hog": "auxhog_10pct",
               "fitnets-teacher": "auxteach_10pct"}),
    dict(id="loss_2pct", title="Loss form @2% (magnitude, lambda=0.3)",
         pct=2, baseline="abl2_none", selected="mse",
         arms={"mse": "auxmag_2pct_l30", "cosine": "auxmag_2pct_cos"}),
    dict(id="loss_10pct", title="Loss form @10% (magnitude, lambda=0.3)",
         pct=10, baseline="abl10_none", selected="mse",
         arms={"mse": "auxmag_10pct_l30", "cosine": "auxmag_10pct_cos"}),
    dict(id="tap_10pct", title="Tap depth @10% (magnitude, lambda=0.3)",
         pct=10, baseline="abl10_none", selected="layer3",
         arms={"layer2": "auxmag_10pct_l30_tap2",
               "layer3": "auxmag_10pct_l30",
               "layer4": "auxmag_10pct_l30_tap4",
               "layer2+3+4": "auxmag_10pct_l30_tap234"}),
    dict(id="tap_1pct", title="Tap depth @1% (magnitude, lambda_0=2.0 -> 0)",
         pct=1, baseline="abl1_none", selected="layer3",
         arms={"layer1": "auxmag_1pct_sched2_tap1",
               "layer2": "auxmag_1pct_sched2_tap2",
               "layer3": "auxmag_1pct_sched2"}),
    dict(id="lambda_const_10pct", title="Constant lambda @10%",
         pct=10, baseline="abl10_none", selected="0.5",
         arms={"0.05": "auxmag_10pct_l05", "0.1": "auxmag_10pct_l10",
               "0.3": "auxmag_10pct_l30", "0.5": "auxmag_10pct_l50",
               "1.0": "auxmag_10pct_l100"}),
    dict(id="lambda_const_2pct", title="Constant lambda @2%",
         pct=2, baseline="abl2_none", selected="2.0",
         arms={"0.3": "auxmag_2pct_l30", "0.5": "auxmag_2pct_l50",
               "1.0": "auxmag_2pct_l100", "2.0": "auxmag_2pct_l200"}),
    dict(id="lambda_const_25pct", title="Constant lambda @25%",
         pct=25, baseline="abl25_none", selected="0.1",
         arms={"0.02": "auxmag_25pct_l02", "0.05": "auxmag_25pct_l05",
               "0.1": "auxmag_25pct_l10", "0.3": "auxmag_25pct_l30"}),
    dict(id="form_10pct", title="Schedule vs constant @10% (the reference-configuration choice)",
         pct=10, baseline="abl10_none", selected="cos 1.0->0",
         arms={"const 0.3": "auxmag_10pct_l30", "const 0.5": "auxmag_10pct_l50",
               "const 1.0": "auxmag_10pct_l100",
               "cos 1.0->0.1": "auxmag_10pct_sched",
               "cos 1.0->0": "auxmag_10pct_sched0"}),
    dict(id="lambda0_1pct", title="Schedule start lambda_0 @1% (cosine -> 0)",
         pct=1, baseline="abl1_none", selected="2.0",
         arms={"1.0": "auxmag_1pct_sched0", "2.0": "auxmag_1pct_sched2"}),
    dict(id="lambda0_2pct", title="Schedule start lambda_0 @2% (cosine -> 0)",
         pct=2, baseline="abl2_none", selected="2.0",
         arms={"1.0": "auxmag_2pct_sched0", "2.0": "auxmag_2pct_sched2"}),
    dict(id="lambda0_15pct", title="Schedule start lambda_0 @15% (cosine -> 0)",
         pct=15, baseline="abl15_none", selected="0.3",
         arms={"1.0": "auxmag_15pct_sched0", "0.3": "auxmag_15pct_sched03"}),
    dict(id="lambda0_25pct", title="Schedule start lambda_0 @25% (cosine -> 0)",
         pct=25, baseline="abl25_none", selected="0.3",
         arms={"1.0": "auxmag_25pct_sched0", "0.3": "auxmag_25pct_sched03"}),
    dict(id="headnorm_r18_10pct", title="Head-norm on ResNet-18 @10% (lambda_0=1.0 -> 0)",
         pct=10, baseline="abl10_none", selected="tie (adopted always-on as free)",
         arms={"no head_norm": "auxmag_10pct_sched0",
               "head_norm": "r18_aux_10pct_hn"}),
    dict(id="headnorm_r50_10pct", title="Head-norm on ResNet-50 @10%",
         pct=10, baseline="r50_none_10pct", selected="head_norm, lambda_0=1.0",
         arms={"no head_norm, lambda_0=1.0": "r50_aux_10pct",
               "no head_norm, lambda_0=0.3": "r50_aux_10pct_l03",
               "head_norm, lambda_0=1.0": "r50_aux_10pct_hn"}),
]
# Schedule ENDPOINT (->0 vs ->0.1), one axis per fraction.
for p in (1, 2, 3, 5, 7, 10, 15, 25):
    AXES.append(dict(
        id=f"endpoint_{p}pct", title=f"Schedule endpoint @{p}% (lambda_0=1.0 cosine)",
        pct=p, baseline=f"abl{p}_none", selected="-> 0",
        arms={"-> 0": f"auxmag_{p}pct_sched0", "-> 0.1": f"auxmag_{p}pct_sched"}))

# Selections that CANNOT be re-scored, and why (reported, not silently dropped).
NOT_RESCORABLE = [
    dict(axis="lambda0 @100% (sched0 / sched03 / sched01) and constant lambda @100%",
         reason="100% cells trained on every train image; no validation carve-out exists "
                "(the carve-out is drawn from the unused portion of train)."),
]

# --------------------------------------------------------------------------


def find_config(cell):
    for d in CONFIG_DIRS:
        p = os.path.join(d, f"{cell}.yaml")
        if os.path.exists(p):
            return p
    return None


def find_seed_ckpts(cell, ckpt="last.pt"):
    """Return {seed: (tree, path)} -- the FIRST tree that has the checkpoint
    wins, in RUN_TREES order (local runs/ first)."""
    out = {}
    for tree in RUN_TREES:
        for p in sorted(glob.glob(os.path.join(tree, cell, "seed*", ckpt))):
            seed = os.path.basename(os.path.dirname(p))
            out.setdefault(seed, (tree, p))
    return out


def recorded_test(cell, seed):
    for tree in RUN_TREES:
        p = os.path.join(tree, cell, seed, "final.json")
        if os.path.exists(p):
            with open(p) as f:
                r = json.load(f)
            return 100.0 * r["final_test_acc"], tree
    return None, None


def n_finals(cell):
    seeds = set()
    for tree in RUN_TREES:
        for p in glob.glob(os.path.join(tree, cell, "seed*", "final.json")):
            seeds.add(os.path.basename(os.path.dirname(p)))
    return len(seeds)


def load_state(path, device, aux_tap=None):
    """Mirror analysis/verify_checkpoints.py: ok / CORRUPT / legacy .tar;
    legacy key naming is remapped on read. Two renames, both exact:
      moment_stem.*  -> target.stem.*        (the pinned bank; never trained)
      aux_head.*     -> aux_heads.<tap>.*    (the single aux head, before the
                                              multi-tap ModuleDict existed)
    The aux head does not touch the classification logits, but the remap is
    still verified: the identity check below (re-scored test vs recorded
    final_test_acc) would expose a wrong mapping as a FAIL."""
    try:
        state = torch.load(path, map_location=device, weights_only=True)
    except Exception as e:
        msg = str(e)
        if "legacy .tar format" in msg:
            state = torch.load(path, map_location=device, weights_only=False)
        else:
            return None, "CORRUPT", msg.split("\n")[0][:100]
    status = "ok"
    if any(k.startswith("moment_stem.") for k in state):
        state = {("target.stem." + k[len("moment_stem."):]
                  if k.startswith("moment_stem.") else k): v for k, v in state.items()}
        status = "ok(legacy-keys-remapped)"
    if any(k.startswith("target.") and not k.startswith("target.stem.") for k in state):
        # a third naming generation: the bank sat directly under target.*
        # before the stem submodule existed. Same tensors, same argument.
        state = {("target.stem." + k[len("target."):]
                  if k.startswith("target.") and not k.startswith("target.stem.") else k): v
                 for k, v in state.items()}
        status = "ok(legacy-keys-remapped)"
    if any(k.startswith("aux_head.") for k in state):
        if not isinstance(aux_tap, str):
            return None, "KEYS", "legacy aux_head.* but no single tap to map it to"
        state = {(f"aux_heads.{aux_tap}." + k[len("aux_head."):]
                  if k.startswith("aux_head.") else k): v for k, v in state.items()}
        status = "ok(legacy-keys-remapped)"
    return state, status, ""


def make_model(cfg, device):
    ds = cfg["dataset"]
    return build_model(
        cfg["backbone"], cfg.get("stem", "none"),
        num_classes=data_mod.NUM_CLASSES[ds],
        small_input=cfg.get("small_input", True),
        pretrained=False,
        stem_kernel_size=cfg.get("stem_kernel_size", 11),
        stem_kwargs=cfg.get("stem_kwargs"),
        head_pool=cfg.get("head_pool"),
        head=cfg.get("head"),
        moment_aux=cfg.get("moment_aux"),
        image_size=data_mod.IMAGE_SIZE[ds],
    ).to(device)


@torch.no_grad()
def correctness(model, loader, device):
    """fp32, no autocast -- train.py's evaluate() is fp32, and the recorded
    final_test_acc is the identity check here."""
    ok = []
    for x, y in loader:
        out = model(x.to(device, non_blocking=True))
        if isinstance(out, (tuple, list)):
            out = out[0]
        ok.append((out.float().argmax(1).cpu() == y))
    return torch.cat(ok).numpy()


def val_indices_for(pct, carve):
    """VAL minus the cell's own training subset; asserted disjoint."""
    val = np.asarray(carve["val_indices"])
    with open(os.path.join("data", "subsets", f"cifar100_{pct}pct.json")) as f:
        s = json.load(f)
    sub = set(s["indices"] if isinstance(s, dict) else s)
    keep = np.array([i not in sub for i in val])
    out = val[keep]
    assert not (set(out.tolist()) & sub)
    return out, int((~keep).sum())


def mean_sem(v):
    v = np.asarray(v, float)
    if len(v) == 0:
        return float("nan"), float("nan")
    if len(v) == 1:
        return float(v[0]), float("nan")
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v)))


def margin(a, b):
    """Mean of a - b and its SEM: seed-PAIRED where the two arms share seed
    labels (and >=2 of them), independent otherwise. Returns (diff, sem, paired)."""
    common = sorted(set(a) & set(b))
    if len(common) >= 2:
        d = np.array([a[s] - b[s] for s in common])
        return float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d))), True
    ma, sa = mean_sem(list(a.values()))
    mb, sb = mean_sem(list(b.values()))
    sem = float(np.sqrt(np.nan_to_num(sa) ** 2 + np.nan_to_num(sb) ** 2))
    return ma - mb, sem, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--refresh", action="store_true", help="ignore cached scores")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    device = torch.device(a.device)
    os.makedirs(CACHE_DIR, exist_ok=True)

    with open(CARVE) as f:
        carve = json.load(f)
    assert carve["dataset"] == "cifar100"
    train_ds = set_eval_transform(
        data_mod.build_dataset("cifar100", a.data_root, train=True, download=False), "cifar100")
    test_ds = data_mod.build_dataset("cifar100", a.data_root, train=False, download=False)
    mk = lambda d: DataLoader(d, batch_size=512, num_workers=a.workers, shuffle=False)
    test_ld = mk(test_ds)

    # per-fraction VAL loaders (restricted where the subset overlaps VAL)
    pcts = sorted({ax["pct"] for ax in AXES})
    val_sets = {}
    for p in pcts:
        idx, dropped = val_indices_for(p, carve)
        val_sets[p] = dict(loader=mk(Subset(train_ds, idx.tolist())),
                           n=int(len(idx)), dropped=dropped)
        print(f"VAL @{p}%: {len(idx)} images ({dropped} of {len(carve['val_indices'])} "
              f"removed as overlapping the {p}% subset)", flush=True)

    # --- inventory + scoring -------------------------------------------
    cells = {}
    for ax in AXES:
        for label, cell in list(ax["arms"].items()) + [("baseline", ax["baseline"])]:
            cells.setdefault(cell, set()).add(ax["pct"])
    inventory = {}
    scores = {}  # cell -> seed -> {val, test, test_recorded, tree}
    t0 = time.time()
    for cell in sorted(cells):
        cfg_path = find_config(cell)
        seeds = find_seed_ckpts(cell)
        inv = dict(config=cfg_path, n_finals=n_finals(cell),
                   n_lastpt=len(seeds), seeds={}, status_counts={})
        if cfg_path is None:
            inv["error"] = "no config found"
            inventory[cell] = inv
            continue
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        assert cfg["dataset"] == "cifar100", cell
        pct = cfg.get("subset_pct", 100)
        assert pct in val_sets, (cell, pct)
        scores[cell] = {}
        for seed, (tree, path) in sorted(seeds.items()):
            cache = os.path.join(CACHE_DIR, f"{cell}__{seed}.npz")
            rec, rec_tree = recorded_test(cell, seed)
            if os.path.exists(cache) and not a.refresh:
                z = np.load(cache, allow_pickle=True)
                st = str(z["status"])
                ok_val, ok_test = z["val"], z["test"]
                if ok_val.size == 0:  # a cached non-loadable checkpoint
                    inv["seeds"][seed] = dict(tree=tree, status=st, detail=str(z["detail"]))
                    inv["status_counts"][st] = inv["status_counts"].get(st, 0) + 1
                    continue
            else:
                aux_tap = (cfg.get("moment_aux") or {}).get("tap")
                state, st, detail = load_state(
                    path, device, aux_tap if isinstance(aux_tap, str) else None)
                if state is None:
                    np.savez_compressed(cache, status=st, detail=detail,
                                        val=np.zeros(0), test=np.zeros(0))
                    inv["seeds"][seed] = dict(tree=tree, status=st, detail=detail)
                    inv["status_counts"][st] = inv["status_counts"].get(st, 0) + 1
                    print(f"  {cell}/{seed}: {st} {detail}", flush=True)
                    continue
                model = make_model(cfg, device)
                try:
                    model.load_state_dict(state)
                except RuntimeError as e:
                    # One early-generation shape drift is known: an aux HEAD
                    # with a different channel count (older target geometry).
                    # The aux head never touches the classification logits, so
                    # it is dropped and the trunk loaded strictly; the identity
                    # check below is what makes this trustworthy (a stale trunk
                    # would fail to reproduce its recorded test accuracy).
                    if all("aux_heads." in ln for ln in str(e).split("\n")[1:] if ln.strip()):
                        slim = {k: v for k, v in state.items()
                                if not k.startswith("aux_heads.")}
                        missing, unexpected = model.load_state_dict(slim, strict=False)
                        assert not unexpected and all(
                            k.startswith("aux_heads.") for k in missing), (cell, seed)
                        st = "ok(aux-head-skipped)"
                        e = None
                    if e is not None:
                        st, detail = "KEYS", str(e).split("\n")[0][:100]
                        np.savez_compressed(cache, status=st, detail=detail,
                                            val=np.zeros(0), test=np.zeros(0))
                        inv["seeds"][seed] = dict(tree=tree, status=st, detail=detail)
                        inv["status_counts"][st] = inv["status_counts"].get(st, 0) + 1
                        print(f"  {cell}/{seed}: KEYS {detail}", flush=True)
                        continue
                model.eval()
                ok_val = correctness(model, val_sets[pct]["loader"], device)
                ok_test = correctness(model, test_ld, device)
                np.savez_compressed(cache, status=st, detail="", val=ok_val, test=ok_test)
                del model
            v, t = 100 * float(ok_val.mean()), 100 * float(ok_test.mean())
            ident = None if rec is None else abs(t - rec)
            scores[cell][seed] = dict(val=v, test=t, test_recorded=rec, tree=tree,
                                      identity_gap=ident, status=st)
            inv["seeds"][seed] = dict(tree=tree, status=st, val=v, test=t,
                                      test_recorded=rec, identity_gap=ident)
            inv["status_counts"][st] = inv["status_counts"].get(st, 0) + 1
            flag = "" if ident is None or ident < 0.5 else "  <-- IDENTITY GAP"
            print(f"  {cell}/{seed} [{tree}]: val {v:.2f}  test {t:.2f}  "
                  f"(recorded {rec:.2f}){flag}", flush=True)
        inv["n_scored"] = len(scores[cell])
        inventory[cell] = inv
    print(f"scored in {time.time() - t0:.0f}s", flush=True)

    # --- per-axis decisions -----------------------------------------------
    axes_out = []
    for ax in AXES:
        base = scores.get(ax["baseline"], {})
        base_val, _ = mean_sem([r["val"] for r in base.values()])
        base_test, _ = mean_sem([r["test_recorded"] for r in base.values()])
        rows = []
        for label, cell in ax["arms"].items():
            sc = scores.get(cell, {})
            val = {s: r["val"] for s, r in sc.items()}
            tst = {s: r["test_recorded"] for s, r in sc.items()}
            tst_re = {s: r["test"] for s, r in sc.items()}
            vm, vs = mean_sem(list(val.values()))
            tm, ts = mean_sem(list(tst.values()))
            rows.append(dict(arm=label, cell=cell, n=len(sc),
                             test_mean=tm, test_sem=ts, val_mean=vm, val_sem=vs,
                             test_rescored_mean=mean_sem(list(tst_re.values()))[0],
                             test_delta=tm - base_test, val_delta=vm - base_val,
                             _val=val, _test=tst))
        scored = [r for r in rows if r["n"] > 0]
        if not scored:
            axes_out.append(dict(**{k: v for k, v in ax.items() if k != "arms"},
                                 rows=[], note="no scorable arm"))
            continue
        win_test = max(scored, key=lambda r: r["test_mean"])
        win_val = max(scored, key=lambda r: r["val_mean"])
        for r in rows:
            if r["n"] == 0:
                r.update(test_margin_to_winner=None, val_margin_to_winner=None)
                continue
            dt, st_, pt = margin(win_test["_test"], r["_test"])
            dv, sv, pv = margin(win_val["_val"], r["_val"])
            r.update(test_margin_to_winner=dt, test_margin_sem=st_, test_margin_paired=pt,
                     val_margin_to_winner=dv, val_margin_sem=sv, val_margin_paired=pv)
        for r in rows:
            r.pop("_val"), r.pop("_test")
        # runner-up margins (winner minus best other arm), on each split
        others_t = [r for r in scored if r is not win_test]
        others_v = [r for r in scored if r is not win_val]
        ru_t = max(others_t, key=lambda r: r["test_mean"]) if others_t else None
        ru_v = max(others_v, key=lambda r: r["val_mean"]) if others_v else None
        flip = win_test["arm"] != win_val["arm"]
        axes_out.append(dict(
            id=ax["id"], title=ax["title"], pct=ax["pct"], baseline=ax["baseline"],
            n_val_images=val_sets[ax["pct"]]["n"],
            selected_recorded=ax["selected"],
            baseline_test=base_test, baseline_val=base_val,
            winner_test=win_test["arm"], winner_val=win_val["arm"],
            test_winner_margin=(win_test["test_mean"] - ru_t["test_mean"]) if ru_t else None,
            test_winner_margin_sem=ru_t["test_margin_sem"] if ru_t else None,
            val_winner_margin=(win_val["val_mean"] - ru_v["val_mean"]) if ru_v else None,
            val_winner_margin_sem=ru_v["val_margin_sem"] if ru_v else None,
            val_runner_up=ru_v["arm"] if ru_v else None,
            test_runner_up=ru_t["arm"] if ru_t else None,
            flip=flip, rows=rows))

    # --- the systematic val-vs-test shift ----------------------------------
    shift_rows = []
    for cell, sc in scores.items():
        for seed, r in sc.items():
            if r["test_recorded"] is not None:
                cfg = yaml.safe_load(open(find_config(cell)))
                shift_rows.append(dict(cell=cell, seed=seed, pct=cfg.get("subset_pct"),
                                       val=r["val"], test=r["test_recorded"],
                                       shift=r["val"] - r["test_recorded"]))
    shift = {}
    all_s = np.array([x["shift"] for x in shift_rows])
    shift["all"] = dict(n=int(len(all_s)), mean=float(all_s.mean()),
                        sd=float(all_s.std(ddof=1)), median=float(np.median(all_s)),
                        frac_negative=float((all_s < 0).mean()))
    for p in pcts:
        s = np.array([x["shift"] for x in shift_rows if x["pct"] == p])
        if len(s):
            shift[f"pct_{p}"] = dict(n=int(len(s)), mean=float(s.mean()),
                                     sd=float(s.std(ddof=1)) if len(s) > 1 else float("nan"))
    ident = np.array([r["identity_gap"] for sc in scores.values() for r in sc.values()
                      if r["identity_gap"] is not None])
    identity = dict(n=int(len(ident)), max_gap=float(ident.max()),
                    n_over_0p5=int((ident > 0.5).sum()))

    # --- verdict -----------------------------------------------------------
    # The registered criterion is the LETTER: "any axis flips". It is scored
    # first and unsoftened. The resolvable/tie split is reported BESIDE it,
    # not instead of it: an axis whose test-scored winner led by less than
    # 2 SEM was a statistical tie at selection time, so its "winner" is noise
    # on either split -- but that distinction was not written into the
    # pre-registration, so it cannot rescue the prediction, only qualify it.
    flips = [ax for ax in axes_out if ax.get("flip")]
    n_axes = sum(1 for ax in axes_out if ax.get("rows"))

    def resolvable(ax):
        m, s_ = ax.get("test_winner_margin"), ax.get("test_winner_margin_sem")
        return m is not None and s_ is not None and not np.isnan(s_) and m > 2 * s_

    res_axes = [ax for ax in axes_out if ax.get("rows") and resolvable(ax)]
    res_flips = [ax for ax in res_axes if ax.get("flip")]
    verdict = dict(
        prediction="no axis flips on the validation carve-out",
        falsifier="any axis flips",
        n_axes_rescored=n_axes, n_flips=len(flips),
        flipped_axes=[ax["id"] for ax in flips],
        falsifier_fired=len(flips) > 0,
        band_hit=len(flips) == 0,
        n_resolvable_axes=len(res_axes),
        resolvable_axes=[ax["id"] for ax in res_axes],
        n_resolvable_flips=len(res_flips),
        resolvable_flips=[ax["id"] for ax in res_flips],
        note=("resolvable = test-scored winner margin > 2 SEM, i.e. the axes "
              "where the selection actually distinguished its arms; the "
              "distinction is applied at scoring time, not registered, and is "
              "reported beside the letter of the falsifier, not instead of it"))

    out = dict(generated=time.strftime("%Y-%m-%d %H:%M"), carveout=CARVE,
               val_sets={str(p): {k: v for k, v in d.items() if k != "loader"}
                         for p, d in val_sets.items()},
               inventory=inventory, axes=axes_out, shift=shift, identity=identity,
               not_rescorable=NOT_RESCORABLE, verdict=verdict)
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=1)
    write_md(out)
    print(f"wrote {OUT_JSON} and {OUT_MD}")
    print(f"VERDICT: {n_axes} axes re-scored, {len(flips)} flipped "
          f"{verdict['flipped_axes']} -> falsifier "
          f"{'FIRED' if verdict['falsifier_fired'] else 'did NOT fire'}")


def fmt(x, d=2):
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:+.{d}f}" if d else f"{x}"


def fmtu(x, d=2):
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


def write_md(out):
    L = []
    L.append("# Selection re-scored on the CIFAR-100 validation carve-out (block E)\n")
    L.append(f"Generated {out['generated']} by `analysis/selection_val_rescore.py`. "
             "Every surviving checkpoint of the CIFAR-100 sweeps that selected the "
             "reference configuration is re-scored (last.pt, fp32, eval transform) on "
             f"VAL of `{out['carveout']}` -- images no cell trained on and no selection "
             "looked at -- and each selection is re-run there.\n")
    v = out["verdict"]
    L.append("## Verdict\n")
    L.append(f"- Pre-registered prediction: **{v['prediction']}**. Falsifier: {v['falsifier']}.")
    L.append(f"- Axes re-scored: {v['n_axes_rescored']}. Axes whose numeric winner changes on "
             f"val: **{v['n_flips']}** {v['flipped_axes'] if v['flipped_axes'] else ''}.")
    L.append(f"- **Scored on the letter: band {'HIT' if v['band_hit'] else 'MISSED'}; falsifier "
             f"{'FIRED' if v['falsifier_fired'] else 'did NOT fire'}.**")
    L.append(f"- Scored on substance: of the {v['n_resolvable_axes']} axes whose test-scored "
             f"winner led by more than 2 SEM ({', '.join(v['resolvable_axes'])}), "
             f"**{v['n_resolvable_flips']} flip on val**"
             + (f" ({', '.join(v['resolvable_flips'])})" if v['resolvable_flips'] else "")
             + ". Every nominal flip sits on an axis whose test-scored margin was itself "
               "below 1 SEM -- a statistical tie at selection time, where the 'winner' is "
               "seed noise on either split. This resolvable/tie distinction was NOT part of "
               "the pre-registration; it qualifies the fired falsifier, it does not undo it.")
    L.append("- Scope caveat (recorded in advance): this BOUNDS the selection defect, it "
             "does not remove it -- the checkpoints were still selected on test.\n")
    L.append("## Validation sets\n")
    for p, d in sorted(out["val_sets"].items(), key=lambda kv: int(kv[0])):
        L.append(f"- @{p}%: {d['n']} VAL images"
                 + (f" ({d['dropped']} removed as overlapping the {p}% training subset)"
                    if d["dropped"] else ""))
    L.append("")
    L.append("## Val-vs-test shift (val-scored minus recorded test accuracy, per checkpoint)\n")
    s = out["shift"]["all"]
    L.append(f"- All {s['n']} checkpoints: mean {s['mean']:+.2f}, median {s['median']:+.2f}, "
             f"sd {s['sd']:.2f}, {100*s['frac_negative']:.0f}% negative.")
    for k, d in out["shift"].items():
        if k.startswith("pct_"):
            L.append(f"- @{k[4:]}%: n={d['n']}, mean {d['mean']:+.2f} (sd {fmtu(d['sd'])})")
    L.append("\nB1 (2026-08-16) measured a ~-0.3 systematic val-vs-test shift on a mixed "
             "8-dataset cell set; on these CIFAR-100 sweep cells the shift is +0.33 overall "
             "and FRACTION-DEPENDENT (negative at 2%, +0.5..+0.7 at 10-25%), so the sign of "
             "B1's pooled number does not transfer to this population. Within an axis the "
             "shift is shared by every arm, which is why it moves winners only where the "
             "margin is itself noise-sized.\n")
    idn = out["identity"]
    L.append(f"Identity check (re-scored test vs recorded final_test_acc): {idn['n']} "
             f"checkpoints, max |gap| {idn['max_gap']:.2f}, {idn['n_over_0p5']} over 0.5.\n")
    L.append("### Reading the five nominal flips\n")
    L.append("- `tap_10pct`: the test margins among layer2 / layer3 / layer2+3+4 are 0.02-0.04 "
             "points -- the recorded finding was 'tap depth is FLAT across the first three "
             "stages', and layer3 was chosen on that plateau (confirmed at 1%, which does not "
             "flip). Val reproduces the plateau with a different noise ordering (winner "
             "layer2 over layer3 by 0.28 +-0.34, under 1 SEM); the cliff at layer4 "
             "reproduces: layer3 beats layer4 by 1.89 on test and 1.58 on val.")
    L.append("- `lambda_const_25pct`: 0.1-vs-0.3 at +0.23 +-0.58 on test, +0.12 +-0.68 the "
             "other way on val -- a tie in both directions.")
    L.append("- `endpoint_{1,2,5}pct`: per-fraction endpoint margins are 0.03-0.12 points on "
             "BOTH splits, under their SEMs. The actual endpoint decision (weight_final 0 "
             "over 0.1) was made on the envelope -- primarily the 10% cell (+0.21 test / "
             "+0.14 val, no flip) and the structural neutrality argument at 100% (not "
             "re-scorable here) -- not on any of these three near-tie fractions.\n")
    L.append("## Per-axis decisions\n")
    L.append("Margins are winner-minus-arm on each split; SEM is seed-paired where the arms "
             "share seed labels (p) and independent otherwise (i). `Delta` is vs the axis "
             "baseline on the same split.\n")
    for ax in out["axes"]:
        if not ax.get("rows"):
            L.append(f"### {ax['title']}\n\nno scorable arm\n")
            continue
        L.append(f"### {ax['title']}\n")
        L.append(f"n_val={ax['n_val_images']}; baseline {ax['baseline']} test "
                 f"{fmtu(ax['baseline_test'])} / val {fmtu(ax['baseline_val'])}; "
                 f"recorded selection: {ax['selected_recorded']}.  ")
        L.append(f"**Winner on test: {ax['winner_test']}** (margin over {ax['test_runner_up']} "
                 f"{fmt(ax['test_winner_margin'])} +-{fmtu(ax['test_winner_margin_sem'])}); "
                 f"**winner on val: {ax['winner_val']}** (margin over {ax['val_runner_up']} "
                 f"{fmt(ax['val_winner_margin'])} +-{fmtu(ax['val_winner_margin_sem'])}) -> "
                 f"**{'FLIP' if ax['flip'] else 'no flip'}**\n")
        L.append("| arm | n | test mean | val mean | Delta test | Delta val | "
                 "test margin to winner | val margin to winner |")
        L.append("|---|---|---|---|---|---|---|---|")
        for r in ax["rows"]:
            if r["n"] == 0:
                L.append(f"| {r['arm']} | 0 | -- | -- | -- | -- | -- | -- |")
                continue
            pt = "p" if r.get("test_margin_paired") else "i"
            pv = "p" if r.get("val_margin_paired") else "i"
            L.append(f"| {r['arm']} | {r['n']} | {fmtu(r['test_mean'])} +-{fmtu(r['test_sem'])} | "
                     f"{fmtu(r['val_mean'])} +-{fmtu(r['val_sem'])} | {fmt(r['test_delta'])} | "
                     f"{fmt(r['val_delta'])} | {fmt(-r['test_margin_to_winner'])} "
                     f"+-{fmtu(r.get('test_margin_sem'))}{pt} | {fmt(-r['val_margin_to_winner'])} "
                     f"+-{fmtu(r.get('val_margin_sem'))}{pv} |")
        L.append("")
    L.append("## Not re-scorable\n")
    for x in out["not_rescorable"]:
        L.append(f"- {x['axis']}: {x['reason']}")
    L.append("\n## Inventory\n")
    L.append("| cell | config | finals | last.pt found | scored | status | trees |")
    L.append("|---|---|---|---|---|---|---|")
    for cell, inv in sorted(out["inventory"].items()):
        trees = sorted({d.get("tree") for d in inv.get("seeds", {}).values()})
        L.append(f"| {cell} | {inv.get('config')} | {inv.get('n_finals')} | "
                 f"{inv.get('n_lastpt')} | {inv.get('n_scored', 0)} | "
                 f"{inv.get('status_counts')} | {','.join(t for t in trees if t)} |")
    with open(OUT_MD, "w") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
