#!/usr/bin/env python3
"""Is the spectral imprint SPECIFIC to the prior, or just what good features look like?

The paper's mechanistic figure reports one number on one cell: alignment between
the tapped features and the moment target is +0.525 for the prior arm against
+0.215 for the baseline. That cannot distinguish

    "the prior leaves an oriented-energy imprint"      (mechanism)
from
    "good features here happen to look like oriented energy"   (incidental)

because any intervention that improves features might show the same thing.

THE TEST. Measure the same alignment gap across families that reach a positive
G by DIFFERENT routes, all on the same dataset, backbone and probe protocol:

    moment prior   -- target IS oriented energy         expect a large gap
    SimCLR init    -- large G, no oriented-energy target expect a small gap
    random target  -- an auxiliary loss with no structure expect ~zero

The SimCLR arm is the decisive one. If it shows the prior's alignment gap at
matched G, the imprint is evidence only that features improved, and the figure
must be demoted from mechanism to illustration. Predictions I1-I3 and that
falsifier were recorded in CLAUDE.md before this was run.

The alignment statistic is the published one (Pearson r between the tapped
layer3 channel-mean energy map and the target's), but measured against ONE
pinned target for every family rather than against each model's own -- a
SimCLR-init cell has no auxiliary head and so no target of its own, and
per-family targets would make the numbers incomparable anyway.
"""
import argparse
import csv
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import visualize_features as VF  # noqa: E402

FAMILIES = [
    ("moment prior", lambda c: c.startswith("auxmag_")),
    ("SimCLR init", lambda c: c.startswith("diagssl_simclr") or
                              c.startswith("diagsslbudget_simclr")),
    ("random target", lambda c: c.startswith("auxrand")),
    ("HOG target", lambda c: "axhog" in c),
    ("learned teacher", lambda c: "axteach" in c),
]


def family_of(cell):
    for name, test in FAMILIES:
        if test(cell):
            return name
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/all_results.csv")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--data-root", default="./data")
    ap.add_argument("--n", type=int, default=512, help="test images per cell")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/imprint_specificity.json")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()

    rows = [r for r in csv.DictReader(open(a.results))
            if r.get("dataset") == "cifar100" and r.get("backbone") == "resnet18"
            and r.get("G") not in (None, "") and r.get("baseline_cell")]
    todo = []
    for r in rows:
        fam = family_of(r["cell"])
        if fam is None:
            continue
        ck_a = os.path.join(a.runs, r["cell"], f"seed{a.seed}", "best.pt")
        ck_b = os.path.join(a.runs, r["baseline_cell"], f"seed{a.seed}", "best.pt")
        if os.path.exists(ck_a) and os.path.exists(ck_b):
            todo.append((fam, r))
    print(f"{len(todo)} cells across {len({t[0] for t in todo})} families")

    dev = torch.device(a.device)

    # ONE pinned target for every family. target_alignment() in
    # visualize_features reads the target off the AUX MODEL, which works for
    # moment-prior cells and is impossible for the decisive family: a
    # SimCLR-init cell has no auxiliary head and therefore no .target at all.
    # Measuring each family against its own target would also make the numbers
    # incomparable. So the target is built once from the champion's pinned bank
    # and applied to every model, which is what "does this representation carry
    # oriented-energy structure?" actually asks.
    import yaml
    from momentstem.aux import MomentTarget
    from momentstem.controls import build_stem
    champ = yaml.safe_load(open("configs/diagnostics/auxmag_5pct_sched0.yaml"))["moment_aux"]
    target = MomentTarget(build_stem(champ["stem"], in_channels=3,
                                     kernel_size=champ.get("kernel_size", 11),
                                     seed=champ.get("stem_seed", 0),
                                     **(champ.get("stem_kwargs") or {}))).to(dev).eval()

    import torch.nn.functional as F
    import data as data_mod

    @torch.no_grad()
    def alignment(model, test_ds, n):
        """Mean Pearson r between the tapped layer3 energy map and the pinned
        target's map. Same statistic as the published figure, but against a
        fixed target rather than the model's own."""
        feats = {}
        h = dict(model.net.named_modules())["layer3"].register_forward_hook(
            lambda _m, _i, o: feats.__setitem__("f", o))
        rs = []
        for i0 in range(0, n, 128):
            xs = torch.stack([test_ds[i][0] for i in range(i0, min(i0 + 128, n))]).to(dev)
            model.net(model.stem(xs))
            tgt = F.adaptive_avg_pool2d(target(xs), feats["f"].shape[-2:]).mean(1)
            emap = feats["f"].abs().mean(1)
            rs += [VF._corr(emap[j].cpu(), tgt[j].cpu()) for j in range(len(xs))]
        h.remove()
        return sum(rs) / len(rs)

    test_ds = data_mod.build_dataset("cifar100", a.data_root, train=False)
    out = []
    for fam, r in sorted(todo, key=lambda t: (t[0], float(t[1]["subset_pct"]))):
        cell, base = r["cell"], r["baseline_cell"]
        try:
            models = VF.load_pair(base, cell, f"seed{a.seed}", dev)
            ab = alignment(models[base][0], test_ds, a.n)
            aa = alignment(models[cell][0], test_ds, a.n)
        except Exception as e:
            print(f"  SKIP {cell}: {type(e).__name__}: {e}")
            continue
        gap = aa - ab
        rec = {"family": fam, "cell": cell, "baseline": base,
               "pct": float(r["subset_pct"]), "G": float(r["G"]),
               "delta": float(r["delta"]) if r.get("delta") else None,
               "align_base": ab, "align_aux": aa, "align_gap": gap}
        out.append(rec)
        print(f"  {fam:15s} {r['subset_pct']:>4}%  G={rec['G']:+6.2f}  "
              f"align {ab:+.3f} -> {aa:+.3f}   gap {gap:+.3f}   {cell[:34]}")

    if out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        json.dump(out, open(a.out, "w"), indent=2)
        print(f"\nwrote {a.out} ({len(out)} cells)")
        import statistics as st
        print(f"\n{'family':16s} {'n':>3s} {'mean G':>8s} {'mean gap':>10s}  {'gap range':>16s}")
        for fam, _ in FAMILIES:
            f = [r for r in out if r["family"] == fam]
            if not f:
                continue
            gaps = [r["align_gap"] for r in f]
            print(f"{fam:16s} {len(f):3d} {st.mean(r['G'] for r in f):+8.2f} "
                  f"{st.mean(gaps):+10.3f}  [{min(gaps):+.3f}, {max(gaps):+.3f}]")


if __name__ == "__main__":
    main()
