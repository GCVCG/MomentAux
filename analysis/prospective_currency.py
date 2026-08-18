"""Re-score the Section 9.3 fusion procedure on a VALIDATION carve-out (B1).

The procedure decides STACK / SUBSTITUTE / TAX for a pair of training-time
sources from four numbers measured on the sources SEPARATELY: Delta_A, Delta_B
(end-to-end gain over a shared baseline) and G_A, G_B (linear-probe gain over
the same baseline). In the paper all four are measured on the held-out TEST
split. This script recomputes all four on images the cells never trained on and
that no reported number is read off, then re-runs the identical decision rule.

WHAT IS HELD FIXED, so that only the split changes: the decision rule
(decide() below), the shared baseline of each pair, the checkpoint policy, the
probe hyper-parameters (analysis.linear_probe.probe is imported, not
reimplemented) and the feature-extraction path (analysis.linear_probe.extract,
likewise).

CHECKPOINT POLICY: last.pt, for every arm and every measurement. best.pt is
selected BY TEST ACCURACY, so scoring it on a validation split would carry the
very contamination this experiment removes back in through the checkpoint
choice. final.json's final_test_acc is the last epoch, so the test-scored
numbers here reproduce the paper's Delta exactly (asserted in the report).

SPLITS (see analysis/make_val_carveout.py for the disjointness argument):
    FIT -- probe head fitted here, nothing evaluated here
    VAL -- everything evaluated here, nothing fitted here
    TEST -- the paper's split, kept only for the side-by-side comparison

Phase 1 (--measure) caches one record per (cell, seed). Phase 2 (--decide)
reads the cache and runs the rule. They are separate so the rule can be
re-examined without re-running a GPU pass.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset, Subset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import data as data_mod
from momentstem import build_model
from analysis.linear_probe import extract, probe, set_eval_transform

CACHE_DIR = "results/prospective"


# --------------------------------------------------------------------------
# data


class Uint8Cache(Dataset):
    """Images pre-decoded to uint8 exactly as they enter the eval transform.

    Only an I/O shortcut: Food-101 decodes 75,750 full-resolution JPEGs per
    pass and this script makes ~54 passes over it. The cached array holds the
    image at the point the real loader hands it to the transform (i.e. after
    Squash64's 64px resize), and the SAME build_transforms(train=False) is
    applied on top, so it is byte-identical to the live loader -- which
    --verify-cache checks against random indices rather than assuming.
    """

    def __init__(self, arr, targets, transform):
        from PIL import Image
        self.arr, self.targets, self.transform, self._Image = (
            arr, targets, transform, Image)

    def __len__(self):
        return len(self.arr)

    def __getitem__(self, i):
        return self.transform(self._Image.fromarray(self.arr[i])), int(self.targets[i])


def _strip_transform(ds):
    """Set .transform to None on the node that owns it, returning the old one."""
    node = ds
    for _ in range(4):
        if hasattr(node, "transform"):
            old, node.transform = node.transform, None
            return node, old
        node = getattr(node, "dataset", None) or getattr(node, "base", None)
        if node is None:
            break
    raise RuntimeError("no .transform to strip")


def build_cache(dataset, data_root, train, tag, verify=True):
    path = os.path.join(CACHE_DIR, f"imgcache_{dataset}_{tag}.npy")
    tf = data_mod.build_transforms(dataset, train=False)
    live = set_eval_transform(
        data_mod.build_dataset(dataset, data_root, train=train, download=False), dataset)
    targets = np.asarray(list(_targets_of(live)))
    if os.path.exists(path):
        arr = np.load(path, mmap_mode="r")
    else:
        raw = data_mod.build_dataset(dataset, data_root, train=train, download=False)
        node, _ = _strip_transform(raw)
        first = np.asarray(raw[0][0])
        arr = np.zeros((len(raw),) + first.shape, dtype=np.uint8)
        t0 = time.time()
        for i in range(len(raw)):
            arr[i] = np.asarray(raw[i][0])
        os.makedirs(CACHE_DIR, exist_ok=True)
        np.save(path, arr)
        print(f"  cached {dataset}/{tag}: {arr.shape} in {time.time()-t0:.0f}s", flush=True)
        arr = np.load(path, mmap_mode="r")
    cached = Uint8Cache(arr, targets, tf)
    if verify:
        rs = np.random.RandomState(0)
        for i in rs.randint(0, len(cached), size=64):
            xc, yc = cached[int(i)]
            xl, yl = live[int(i)]
            assert yc == yl and torch.equal(xc, xl), f"cache mismatch at {i}"
    return cached


def _targets_of(ds):
    node = ds
    for _ in range(4):
        t = getattr(node, "targets", None)
        if t is not None:
            return t
        node = getattr(node, "dataset", None) or getattr(node, "base", None)
    raise RuntimeError("no .targets")


def make_loaders(dataset, data_root, carve, batch=256, workers=8, cache=False):
    if cache:
        tr = build_cache(dataset, data_root, True, "train")
        te = build_cache(dataset, data_root, False, "test")
    else:
        tr = set_eval_transform(
            data_mod.build_dataset(dataset, data_root, train=True, download=False), dataset)
        te = data_mod.build_dataset(dataset, data_root, train=False, download=False)
    mk = lambda d: DataLoader(d, batch_size=batch, num_workers=workers, shuffle=False)
    return (mk(Subset(tr, carve["fit_indices"])),
            mk(Subset(tr, carve["val_indices"])),
            mk(te))


# --------------------------------------------------------------------------
# measurement


@torch.no_grad()
def logits_correct(model, loader, device):
    """Per-image correctness of the trained classifier (the e2e score)."""
    ok, ys = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast("cuda", enabled=device.type == "cuda"):
            out = model(x)
        if isinstance(out, (tuple, list)):
            out = out[0]
        ok.append((out.float().argmax(1).cpu() == y))
        ys.append(y)
    return torch.cat(ok).numpy(), torch.cat(ys).numpy()


def measure_cell(cell, cfg_path, carve, loaders, device, ckpt="last.pt"):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    ds = cfg["dataset"]
    fit_ld, val_ld, test_ld = loaders
    n_classes = data_mod.NUM_CLASSES[ds]
    out = []
    run_dir = os.path.join("runs", cell)
    for seed_dir in sorted(d for d in os.listdir(run_dir) if d.startswith("seed")):
        path = os.path.join(run_dir, seed_dir, ckpt)
        if not os.path.exists(path):
            continue
        model = build_model(
            cfg["backbone"], cfg.get("stem", "none"),
            num_classes=n_classes,
            small_input=cfg.get("small_input", True),
            stem_kernel_size=cfg.get("stem_kernel_size", 11),
            stem_kwargs=cfg.get("stem_kwargs"),
            head_pool=cfg.get("head_pool"),
            head=cfg.get("head"),
            moment_aux=cfg.get("moment_aux"),
            image_size=data_mod.IMAGE_SIZE[ds],
            in_channels=data_mod.INPUT_CHANNELS.get(ds, 3),
        ).to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()

        fitf, fitY = extract(model, fit_ld, device)
        valf, valY = extract(model, val_ld, device)
        tef, teY = extract(model, test_ld, device)
        # ONE fitted head, evaluated on both splits. probe() couples the fit to
        # its evaluation set, so it is called twice -- with the global RNG reset
        # to the same state first, since torch.nn.Linear draws its init from it.
        # Identical init + identical fit data => identical head; the returned
        # FIT accuracies are asserted equal, which is what proves it rather than
        # assuming LBFGS reproduced itself.
        torch.manual_seed(0)
        f_val, p_val = probe(fitf, fitY, valf, valY, n_classes, device)
        torch.manual_seed(0)
        f_test, p_test = probe(fitf, fitY, tef, teY, n_classes, device)
        assert abs(f_val - f_test) < 1e-6, (
            f"the two probe fits diverged ({f_val} vs {f_test}); val and test "
            "would not be scored by the same head")
        ok_val, _ = logits_correct(model, val_ld, device)
        ok_test, _ = logits_correct(model, test_ld, device)

        with open(os.path.join(run_dir, seed_dir, "final.json")) as f:
            rec_test = json.load(f)["final_test_acc"] * 100
        r = {"seed": seed_dir, "ckpt": ckpt,
             "e2e_val": float(ok_val.mean() * 100),
             "e2e_test": float(ok_test.mean() * 100),
             "e2e_test_recorded": rec_test,
             "probe_val": p_val * 100, "probe_test": p_test * 100}
        np.savez_compressed(os.path.join(CACHE_DIR, f"mask_{cell}_{seed_dir}.npz"),
                            val=ok_val, test=ok_test)
        out.append(r)

        print(f"  {cell} {seed_dir}: e2e val {r['e2e_val']:.2f} test {r['e2e_test']:.2f} "

              f"(recorded {rec_test:.2f})  probe val {p_val*100:.2f} test {p_test*100:.2f}",
              flush=True)
    if not out:
        # A cell with no local checkpoint would otherwise cache an empty record
        # and drop its pair from the comparison SILENTLY -- exactly how a
        # partial result gets reported as a complete one. Six diaggrid_ssl_*
        # cells are in this state locally (probed on the cluster, checkpoints
        # never pulled, and there is no network from here to pull them).
        raise SystemExit(f"{cell}: no {ckpt} under runs/{cell}/seed*/ -- refusing "
                         "to record an empty cell")
    return out


# --------------------------------------------------------------------------
# the decision rule (identical for test-scored and val-scored inputs)

G_ZERO = 1.0        # |G| <= this is "supplies nothing"
G_EQ_REL = 0.25     # |G_A - G_B| <= this * max(|G_A|,|G_B|) is "G_A ~ G_B"
ASYM_REL = 0.5      # |D_A - D_B| > this * max(|D_A|,|D_B|) is "large"


def decide(dA, dB, gA, gB, pretrained_A=False, pretrained_B=False):
    """Algorithm 1 of the paper, with its qualitative predicates pinned to
    numbers ONCE. The thresholds are read off the paper's own worked example
    (14.83 vs 13.46, a 9% relative gap, is called agreement; 14.83 vs 3.17,
    79%, is not) and are applied identically to test-scored and val-scored
    inputs, so any change of call is a change of the inputs, never of the rule.
    """
    notes = []
    if pretrained_A or pretrained_B:
        return "TAX", ["a mature pre-trained initialization is present"]
    if abs(gA) <= G_ZERO or abs(gB) <= G_ZERO:
        return "N/A", [f"a source supplies nothing (G {gA:+.2f} / {gB:+.2f})"]
    if abs(dA - dB) > ASYM_REL * max(abs(dA), abs(dB)):
        notes.append(f"asymmetric arms (|dA-dB| = {abs(dA-dB):.2f}); deprioritize")
    if abs(gA - gB) <= G_EQ_REL * max(abs(gA), abs(gB)):
        return "SUBSTITUTE", notes + [f"G agree to {abs(gA-gB):.2f} on {max(gA,gB):.2f}"]
    return "STACK", notes + [f"G differ by {abs(gA-gB):.2f} on {max(gA,gB):.2f}"]


def outcome(d_combo, dA, dB, sem):
    """What training actually showed: does the combination exceed the better
    single arm by more than 2 SEM of the difference?"""
    best = max(dA, dB)
    if d_combo - best > 2 * sem:
        return "STACK"
    if d_combo - best < -2 * sem:
        return "SUBSTITUTE(cost)"
    return "SUBSTITUTE"


# --------------------------------------------------------------------------


def sensitivity(rows):
    """The rule's predicates had to be pinned to numbers, and a conclusion that
    holds only at one threshold is not a conclusion. The same thresholds are
    applied to both splits and swept together."""
    global G_ZERO, G_EQ_REL
    keep = (G_ZERO, G_EQ_REL)
    n = len(rows)
    print("\nthreshold sensitivity (calls unchanged rec->val / test->val):")
    for gz in (0.5, 1.0, 2.0):
        for ge in (0.15, 0.25, 0.35):
            G_ZERO, G_EQ_REL = gz, ge
            a = b = 0
            for r in rows:
                c = {s: decide(r[f"dA_{s}"], r[f"dB_{s}"], r[f"gA_{s}"],
                               r[f"gB_{s}"])[0] for s in ("rec", "test", "val")}
                a += c["rec"] == c["val"]
                b += c["test"] == c["val"]
            print(f"  G_ZERO={gz:<4} G_EQ_REL={ge:<5} -> {a}/{n}  {b}/{n}")
    G_ZERO, G_EQ_REL = keep


def recorded(cell):
    """The paper's own inputs for this cell: Delta from final.json's
    final_test_acc and G from the recorded full-train-fit linear probe. Kept
    separate from anything measured here so the comparison can distinguish
    'the eval split moved the call' from 'the probe's fit set moved it'."""
    import glob
    e = [json.load(open(f))["final_test_acc"] * 100
         for f in sorted(glob.glob(f"runs/{cell}/seed*/final.json"))]
    p = [r["probe_test"] * 100
         for r in json.load(open(f"runs/{cell}/linear_probe.json"))["results"]]
    return e, p


def mean_sem(v):
    v = np.asarray(v, dtype=float)
    m = v.mean()
    s = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
    return float(m), float(s)


def fixed_similarity(base_masks, a_masks, b_masks):
    """currency_evidence.py's normalized fixed-set overlap, on this split.

    S(A,B) = across(A,B) / sqrt(within(A) * within(B)) using the Jaccard of the
    sets of baseline errors each arm FIXES. The within-arm (across-seed) term
    is the control for "some images are simply easier".
    """
    def fixed(base_ok, arm_ok):
        return (~base_ok) & arm_ok

    def jac(x, y):
        u = (x | y).sum()
        return float((x & y).sum() / u) if u else 0.0

    def within(masks, bases):
        v = [jac(fixed(bases[i], masks[i]), fixed(bases[j], masks[j]))
             for i in range(len(masks)) for j in range(i + 1, len(masks))]
        return float(np.mean(v)) if v else float("nan")

    wa, wb = within(a_masks, base_masks), within(b_masks, base_masks)
    across = [jac(fixed(base_masks[i], a_masks[i]), fixed(base_masks[j], b_masks[j]))
              for i in range(len(a_masks)) for j in range(len(b_masks))]
    a = float(np.mean(across))
    return a / np.sqrt(wa * wb) if wa > 0 and wb > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="analysis/prospective_pairs.json")
    ap.add_argument("--measure", action="store_true")
    ap.add_argument("--decide", action="store_true")
    ap.add_argument("--only-dataset", default=None)
    ap.add_argument("--only-pct", default=None,
                    help="comma-separated fractions, so one dataset's cells can "
                         "be split across two processes on the same GPU")
    ap.add_argument("--tag", default=None,
                    help="cache-file suffix; two concurrent processes over the "
                         "same dataset MUST use different tags or the last "
                         "writer erases the other's cells")
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    with open(args.spec) as f:
        spec = json.load(f)
    os.makedirs(CACHE_DIR, exist_ok=True)
    # One cache file per dataset when a dataset is measured on its own, so two
    # measurement processes can share the GPU without the last writer erasing
    # the other's cells. --decide merges every cells*.json it finds.
    import glob as _glob
    key = args.tag or args.only_dataset
    cache_path = os.path.join(CACHE_DIR, f"cells_{key}.json" if key else "cells.json")
    cells = {}
    for f in sorted(_glob.glob(os.path.join(CACHE_DIR, "cells*.json"))):
        cells.update(json.load(open(f)))
    own = json.load(open(cache_path)) if os.path.exists(cache_path) else {}

    if args.measure:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        by_ds = {}
        for p in spec["pairs"]:
            if args.only_dataset and p["dataset"] != args.only_dataset:
                continue
            if args.only_pct and p["pct"] not in [int(x) for x in args.only_pct.split(",")]:
                continue
            for role in ("baseline", "A", "B", "combo"):
                by_ds.setdefault(p["dataset"], {})[p[role]["cell"]] = p[role]["config"]
        for ds, cellmap in by_ds.items():
            carve = json.load(open(f"data/valcarve/{ds}.json"))
            print(f"[{ds}] VAL {carve['n_val']} / FIT {carve['n_fit']} "
                  f"(excluded pcts {carve['excluded_pcts']})", flush=True)
            loaders = make_loaders(ds, args.data_root, carve, workers=args.workers,
                                   cache=spec.get("cache_images", {}).get(ds, False))
            for cell, cfg in sorted(cellmap.items()):
                if cell in cells:
                    print(f"  {cell}: cached", flush=True)
                    continue
                own[cell] = cells[cell] = measure_cell(cell, cfg, carve, loaders, device)
                with open(cache_path, "w") as f:
                    json.dump(own, f, indent=1)

    if args.decide:
        rows = []
        for p in spec["pairs"]:
            r = {"pair": p["name"], "dataset": p["dataset"], "pct": p["pct"],
                 "A": p["A"]["label"], "B": p["B"]["label"]}
            arms = {}
            ok = True
            for role in ("baseline", "A", "B", "combo"):
                c = p[role]["cell"]
                if c not in cells:
                    ok = False
                    break
                arms[role] = cells[c]
            if not ok:
                continue
            for split in ("rec", "test", "val"):
                if split == "rec":
                    base_e, base_p = recorded(p["baseline"]["cell"])
                else:
                    base_e = [x[f"e2e_{split}"] for x in arms["baseline"]]
                    base_p = [x[f"probe_{split}"] for x in arms["baseline"]]
                for role, tag in (("A", "A"), ("B", "B"), ("combo", "C")):
                    if split == "rec":
                        e, pr = recorded(p[role]["cell"])
                    else:
                        e = [x[f"e2e_{split}"] for x in arms[role]]
                        pr = [x[f"probe_{split}"] for x in arms[role]]
                    dm_, ds_ = mean_sem(e)
                    bm_, bs_ = mean_sem(base_e)
                    r[f"d{tag}_{split}"] = dm_ - bm_
                    r[f"d{tag}_{split}_sem"] = float(np.hypot(ds_, bs_))
                    gm_, gs_ = mean_sem(pr)
                    bpm, bps = mean_sem(base_p)
                    r[f"g{tag}_{split}"] = gm_ - bpm
                    r[f"g{tag}_{split}_sem"] = float(np.hypot(gs_, bps))
                    r[f"acc{tag}_{split}"], r[f"acc{tag}_{split}_sem"] = dm_, ds_
                call, why = decide(r[f"dA_{split}"], r[f"dB_{split}"],
                                   r[f"gA_{split}"], r[f"gB_{split}"],
                                   p["A"].get("pretrained", False),
                                   p["B"].get("pretrained", False))
                r[f"call_{split}"] = call
                r[f"why_{split}"] = why
                # Combo vs the better single arm, compared on the ARM
                # accuracies: the shared baseline cancels in the difference, so
                # folding its seed noise into the SEM would only inflate it.
                best = "A" if r[f"dA_{split}"] >= r[f"dB_{split}"] else "B"
                sem = float(np.hypot(r[f"accC_{split}_sem"], r[f"acc{best}_{split}_sem"]))
                r[f"combo_minus_best_{split}"] = r[f"accC_{split}"] - r[f"acc{best}_{split}"]
                r[f"combo_minus_best_{split}_sem"] = sem
                r[f"outcome_{split}"] = outcome(r[f"dC_{split}"], r[f"dA_{split}"],
                                                r[f"dB_{split}"], sem)
            # same-cells diagnostic, on both splits
            for split in ("test", "val"):
                try:
                    def masks(role):
                        return [np.load(os.path.join(
                            CACHE_DIR, f"mask_{p[role]['cell']}_{x['seed']}.npz"))[split]
                            for x in arms[role]]
                    r[f"S_{split}"] = fixed_similarity(masks("baseline"), masks("A"),
                                                       masks("B"))
                except Exception as e:  # masks are a diagnostic, not the rule
                    r[f"S_{split}"] = float("nan")
            rows.append(r)
        with open("results/prospective_currency.json", "w") as f:
            json.dump({"thresholds": {"G_ZERO": G_ZERO, "G_EQ_REL": G_EQ_REL,
                                      "ASYM_REL": ASYM_REL},
                       "pairs": rows}, f, indent=1)
        def block(tag):
            return (f"{'dA':>6} {'dB':>6} {'gA':>6} {'gB':>6} {'call':>11}")
        print("\n" + " " * 30
              + "----- PAPER PROTOCOL (test) ----- | "
                "------ SAME FIT, test eval ----- | "
                "--------- VAL-SCORED ----------- |")
        print(f"{'pair':30s} {block('rec')} | {block('test')} | {block('val')} | "
              f"outcome(val)  S_test S_val")
        for r in rows:
            line = f"{r['pair']:30s} "
            for s in ("rec", "test", "val"):
                line += (f"{r['dA_'+s]:6.2f} {r['dB_'+s]:6.2f} {r['gA_'+s]:6.2f} "
                         f"{r['gB_'+s]:6.2f} {r['call_'+s]:>11} | ")
            line += (f"{r['outcome_val']:<14s} {r['S_test']:5.2f} {r['S_val']:5.2f}"
                     f"  {'FLIP' if r['call_rec'] != r['call_val'] else ''}")
            print(line)
        n = len(rows)
        if not n:
            raise SystemExit("no pair has all four arms measured yet")
        print(f"\ncalls unchanged rec->val:  {sum(r['call_rec'] == r['call_val'] for r in rows)}/{n}")
        print(f"calls unchanged test->val: {sum(r['call_test'] == r['call_val'] for r in rows)}/{n}")
        # SENSITIVITY: the rule's predicates had to be pinned to numbers, and a
        # conclusion that only holds at one threshold is not a conclusion. Same
        # thresholds applied to both splits, swept together.
        # Does the procedure still WORK when run prospectively, not just agree
        # with itself? SUBSTITUTE is credited when training showed no gain over
        # the better single arm, whether that came out flat or with a cost.
        def hit(call, out):
            if call in ("N/A", "TAX"):
                return None
            return (call == "STACK") == (out == "STACK")
        for s in ("rec", "test", "val"):
            h = [hit(r[f"call_{s}"], r[f"outcome_{s}"]) for r in rows]
            h = [x for x in h if x is not None]
            print(f"call agrees with the trained outcome ({s}): {sum(h)}/{len(h)}")

        sensitivity(rows)
        print()
        for ref in ("test", "rec"):
            for q, lab in (("d", "Delta"), ("g", "G")):
                sh = [r[f"{q}{t}_val"] - r[f"{q}{t}_{ref}"] for r in rows for t in "AB"]
                print(f"{lab:5s} val-minus-{ref:4s}: mean {np.mean(sh):+.2f}  "
                      f"mean|.| {np.mean(np.abs(sh)):.2f}  max|.| {np.max(np.abs(sh)):.2f}  "
                      f"({sum(s > 0 for s in sh)}/{len(sh)} positive, "
                      f"{sum(abs(s) <= 1.0 for s in sh)}/{len(sh)} within 1.0)")


if __name__ == "__main__":
    main()
