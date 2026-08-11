"""Verify stored checkpoints by IDENTITY, not by loading.

    python analysis/verify_checkpoints.py --sample 8
    python analysis/verify_checkpoints.py --cells abl5_none auxmag_5pct_sched0

A checkpoint can load perfectly and be the wrong network. That is not
hypothetical here: on 2026-08-06 a duplicate-lane race overwrote 21 finished
best.pt files with mid-run weights, every one of which passed torch.load, and
the damage was only visible by EVALUATING them against their own recorded
final.json. The rule that came out of it -- verify by identity, cheap, one
val pass -- is what this script automates.

It is also the check to run after ANY change to the training or model code,
which is the other reason it exists. If a refactor silently changes how a
model is built, the symptom is a checkpoint that no longer reproduces the
accuracy it was recorded with, and nothing else in the pipeline notices.

PASS means the re-evaluated accuracy is within --tol of the recorded value.
The tolerance exists for nondeterministic kernels, not for real drift: on
these cells the agreement is normally to two decimal places.
"""
import argparse
import glob
import json
import os
import random
import sys

import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import data as data_mod
from momentstem import build_model


def find_config(cell):
    for d in ("configs/grid", "configs/diagnostics", "configs"):
        p = os.path.join(d, f"{cell}.yaml")
        if os.path.exists(p):
            return p
    return None


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.autocast("cuda", enabled=device.type == "cuda"):
            out = model(x)
        correct += int((out.argmax(1) == y).sum())
        total += y.numel()
    return 100.0 * correct / max(total, 1)


def verify_one(cell, seed, runs, data_root, device, ckpt_name):
    cfg_path = find_config(cell)
    seed_dir = os.path.join(runs, cell, f"seed{seed}")
    final = os.path.join(seed_dir, "final.json")
    ckpt = os.path.join(seed_dir, ckpt_name)
    if not (cfg_path and os.path.exists(final) and os.path.exists(ckpt)):
        return None
    cfg = yaml.safe_load(open(cfg_path))
    with open(final) as f:
        rec = json.load(f)
    # PAIR THE CHECKPOINT WITH THE RIGHT FIELD. best.pt holds the best epoch's
    # weights and last.pt the final epoch's; comparing best.pt against
    # final_test_acc measures the gap between two different epochs and reports
    # it as checkpoint damage. My first run of this script did exactly that
    # and produced one spurious FAIL.
    key = "best_test_acc" if ckpt_name.startswith("best") else "final_test_acc"
    recorded = rec.get(key)
    if recorded is None:                      # older records may lack best_
        recorded = rec.get("final_test_acc")
        if recorded is None:
            return None
    recorded *= 100.0

    ds = cfg["dataset"]
    try:
        state = torch.load(ckpt, map_location=device, weights_only=True)
    except Exception as e:
        msg = str(e)
        if "legacy .tar format" in msg:
            # torch.save's pre-zip format, not damage: these load fine, they
            # just predate weights_only. Retry rather than report a failure.
            state = torch.load(ckpt, map_location=device, weights_only=False)
        else:
            # PytorchStreamReader failures are the silent-corruption signature
            # recorded 2026-08-05 -- byte-identical size, unreadable content,
            # invisible to rsync's size+mtime check. Report, do not abort.
            return {"cell": cell, "seed": seed, "status": "CORRUPT",
                    "detail": msg.split("\n")[0][:80]}

    model = build_model(
        cfg["backbone"], cfg["stem"],
        num_classes=data_mod.NUM_CLASSES[ds],
        small_input=cfg.get("small_input", True),
        pretrained=cfg.get("pretrained", False),
        stem_kernel_size=cfg.get("stem_kernel_size", 11),
        stem_kwargs=cfg.get("stem_kwargs"),
        head_pool=cfg.get("head_pool"),
        head=cfg.get("head"),
        moment_aux=cfg.get("moment_aux"),
        image_size=data_mod.IMAGE_SIZE[ds],
    ).to(device)
    if any(k.startswith("moment_stem.") for k in state):
        # The early fixed-lambda aux cells (auxmag_*, auxgab_*, auxrand_*) were
        # written when the auxiliary TARGET module was a top-level attribute
        # called moment_stem; it later moved to target.stem. The tensors under
        # it are the pinned bank and its calibration -- fixed, never trained --
        # so only the path changed and a prefix remap is exact.
        #
        # This is a READ-side shim in a diagnostic, deliberately: aux.py is not
        # touched, so no cell's training behaviour can move. The remap is only
        # trustworthy because the evaluation below CHECKS it -- if a remapped
        # checkpoint failed to reproduce its recorded accuracy, that would show
        # up as a FAIL rather than being waved through.
        state = {("target.stem." + k[len("moment_stem."):] if
                  k.startswith("moment_stem.") else k): v
                 for k, v in state.items()}
    try:
        model.load_state_dict(state)
    except RuntimeError as e:
        # A key mismatch is NOT damage either: the early fixed-lambda aux cells
        # were written when the auxiliary target module was called moment_stem,
        # before it moved under target. The tensors are the pinned bank and are
        # unchanged; only the path to them differs. Classify it as such instead
        # of letting one old cell abort the whole sweep.
        return {"cell": cell, "seed": seed, "status": "KEYS",
                "detail": str(e).split("\n")[1].strip()[:80] if "\n" in str(e)
                          else str(e)[:80]}

    test = data_mod.build_dataset(ds, data_root, train=False)
    loader = torch.utils.data.DataLoader(test, batch_size=256, shuffle=False,
                                         num_workers=4)
    got = evaluate(model, loader, device)
    return {"cell": cell, "seed": seed, "recorded": recorded, "evaluated": got,
            "diff": got - recorded, "field": key}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="*", default=None)
    ap.add_argument("--sample", type=int, default=8,
                    help="if --cells is absent, verify this many random cells")
    ap.add_argument("--seed-pick", type=int, default=0)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--tol", type=float, default=0.5,
                    help="allowed |evaluated - recorded| in points")
    ap.add_argument("--device",
                    default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    device = torch.device(a.device)

    if a.cells:
        cells = a.cells
    else:
        # Sample across the tree rather than taking the first N: the first N
        # alphabetically are all one family and would not exercise the
        # backbones, stems and aux paths that a code change could break.
        have = sorted({p.split(os.sep)[1] for p in
                       glob.glob(os.path.join(a.runs, "*", "seed*", "final.json"))})
        random.Random(0).shuffle(have)
        cells = have[:a.sample * 3]     # oversample; many lack a checkpoint

    checked, bad = [], []
    for cell in cells:
        r = verify_one(cell, a.seed_pick, a.runs, a.data_root, device, a.ckpt)
        if r is None:
            continue
        checked.append(r)
        if "status" in r:
            # CORRUPT is damage and must be re-pulled; KEYS is an old naming
            # and the recorded number still stands. Counting them together
            # would hide the one that matters.
            bad.append(r)
            print(f"  {r['status']:<4s} {r['cell']:<38s} {r['detail']}",
                  flush=True)
        else:
            flag = "ok  " if abs(r["diff"]) <= a.tol else "FAIL"
            if flag == "FAIL":
                bad.append(r)
            print(f"  {flag} {r['cell']:<38s} recorded {r['recorded']:6.2f}  "
                  f"evaluated {r['evaluated']:6.2f}  diff {r['diff']:+.2f}",
                  flush=True)
        if not a.cells and len(checked) >= a.sample:
            break

    n_corrupt = sum(r.get("status") == "CORRUPT" for r in checked)
    n_keys = sum(r.get("status") == "KEYS" for r in checked)
    n_drift = len(bad) - n_corrupt - n_keys
    print(f"\n{len(checked)} checkpoints examined: "
          f"{len(checked) - len(bad)} verified, {n_drift} outside +-{a.tol}, "
          f"{n_corrupt} CORRUPT, {n_keys} legacy key naming")
    if bad:
        sys.exit(1)


if __name__ == "__main__":
    main()
