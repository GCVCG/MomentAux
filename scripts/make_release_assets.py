#!/usr/bin/env python3
"""Build the tagged release's assets, reproducibly, and verify what went in.

The v1.0 assets were assembled by hand. That is the same hazard as every
other hand-maintained number in this project: nobody can tell later whether
an asset matches the tree it claims to come from, and `docs/ARTIFACTS.md`
promises specific contents that nothing checks. This script builds all six
assets from the working tree, prints the counts that the manifest claims, and
writes SHA256SUMS.

    python scripts/make_release_assets.py --out dist/
    python scripts/make_release_assets.py --out dist/ --verify   # counts only

WHAT EACH ASSET IS FOR is documented in docs/ARTIFACTS.md; the mapping from
that manifest to the globs below is the thing to keep in step.

REPRODUCIBILITY. Members are added in sorted order with a fixed mtime and
uid/gid, and the gzip wrapper is stamped mtime=0 too, so building the same
tree twice gives byte-identical tarballs. That is what lets a reader check a
download against SHA256SUMS from the repo alone. Verify it the way it was
verified here -- build twice into different directories and diff SHA256SUMS.
"""
import argparse
import fnmatch
import hashlib
import gzip
import os
import sys
import tarfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EPOCH = (2026, 1, 1, 0, 0, 0)  # fixed, so tarballs are byte-reproducible

# asset -> (list of (root, pattern-predicate), human description)
def _under(*dirs):
    return lambda p: any(p.startswith(d + os.sep) or p == d for d in dirs)


# Run trees that must NEVER enter an asset, even though their names start
# "runs" and the predicates below key on that prefix.
#
# runs_dense_50e holds the WITHDRAWN 50-epoch segmentation generation. The
# dense recipe was re-pinned to 200 epochs (see the dense re-pin decision in
# CLAUDE.md) and every released dense number comes from the 200-epoch tree in
# runs_dense. The two generations reached OPPOSITE conclusions on the dense
# attention result, so shipping them side by side under near-identical cell
# names invites a reader to aggregate across both and get a table that looks
# complete, is internally consistent, and is wrong. This is not hypothetical:
# on 2026-08-16 the local runs_dense tree was found to hold 50 cells of the
# 50-epoch generation, and a regeneration would have blended them silently.
# Keep the archive out of the release; it is preserved on the cluster and
# locally under this name for provenance, not for distribution.
EXCLUDED_RUN_TREES = ("runs_dense_50e",)


def _in_runs(p):
    """True for a path under a SHIPPABLE run tree."""
    top = p.split(os.sep)[0]
    return top.startswith("runs") and top not in EXCLUDED_RUN_TREES


ASSETS = {
    "result-tables.tar.gz": (
        lambda p: p.startswith("results" + os.sep) and p.endswith(
            (".csv", ".md", ".tex", ".xlsx", ".json")),
        "aggregated tables, the law audit, and the per-analysis JSON records",
    ),
    "run-records.tar.gz": (
        # 2026-08-16/17. This predicate used to list four exact basenames and
        # silently shipped less than it promised, in three separate ways:
        #
        #   * "linear_probe_SHOTS.json" matched NOTHING -- every fixed-shot
        #     probe on disk is lowercase linear_probe_shots.json (70 files),
        #     and docs/ARTIFACTS.md documented the file as shipped. README.md
        #     promised "every linear_probe*.json" while 101 of 2,664 were left
        #     out (the shots probes, the LASTPT probes behind the ImageNet G
        #     values measured after the wrong-epoch quarantine, and the
        #     cross-label-space tin/tin20/cifar100/cifar100super probes).
        #   * dense_probe.json was absent, so every dense G and all 30 rows of
        #     results/dense_law.csv rested on records the release omitted.
        #   * runs_det was not even a collector root, so the whole detection
        #     task -- 36 finals and 30 det_probe.json -- shipped nowhere, while
        #     results/det_results.csv was released and the paper has a
        #     detection section.
        #
        # A GLOB is used for the probe family deliberately: an exact-name list
        # is what let a case typo delete 70 files from the release without any
        # visible failure. Adding a probe variant must not require editing this
        # tuple. The trade is that anything named linear_probe*.json ships, so
        # do not park scratch files under that name.
        lambda p: (os.path.basename(p) in (
            "final.json", "dense_probe.json", "det_probe.json",
            "robustness.json", "cifair.json", "head_forms_5shot.json",
            "per_class_delta.json")
            or fnmatch.fnmatch(os.path.basename(p), "linear_probe*.json")
        ) and _in_runs(p),
        "every run's authoritative record: config as executed, accuracy, "
        "FLOPs, environment, and the probes behind G",
    ),
    "training-curves.tar.gz": (
        lambda p: os.path.basename(p) == "metrics.csv" and _in_runs(p),
        "per-epoch train/test accuracy, loss components, the lambda schedule",
    ),
    "configs-and-subsets.tar.gz": (
        lambda p: (p.startswith("configs" + os.sep) and p.endswith((".yaml", ".yml")))
        or (p.startswith("data" + os.sep + "subsets") and p.endswith(".json")),
        "every cell's config and the committed subset indices, so any cell "
        "can be re-run on byte-identical images",
    ),
    "logs.tar.gz": (
        lambda p: p.startswith("logs" + os.sep) and p.endswith((".log", ".txt")),
        "campaign logs: what was submitted when, and what failed",
    ),
}


def walk(root):
    for base, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        # Hidden FILES are skipped for the same reason hidden dirs are, which
        # the original only did for dirs: results/.bak_MomentStem_results.xlsx
        # (a stray 2026-08-04 backup) was being shipped inside
        # result-tables.tar.gz as if it were a released table -- a stale
        # duplicate of the workbook, i.e. exactly the two-generations hazard
        # EXCLUDED_RUN_TREES guards against, one directory over.
        for f in files:
            if f.startswith("."):
                continue
            full = os.path.join(base, f)
            yield os.path.relpath(full, ROOT)


def collect(pred):
    # runs_det added 2026-08-17: the detection task's records were unreachable
    # from here, so no predicate could ship them however it was written.
    # EXCLUDED_RUN_TREES still governs which "runs*" trees may ship.
    tops = ("results", "runs", "runs_turing", "runs_dense", "runs_det",
            "runs_ckptfix", "configs", "data", "logs")
    out = []
    for t in tops:
        d = os.path.join(ROOT, t)
        if os.path.isdir(d):
            out += [p for p in walk(d) if pred(p)]
    return sorted(set(out))


def build(name, pred, outdir, verify):
    members = collect(pred)
    total = sum(os.path.getsize(os.path.join(ROOT, p)) for p in members)
    print(f"  {name:30} {len(members):7,} files  {total/1048576:8.1f} MB raw")
    if verify:
        return None
    path = os.path.join(outdir, name)
    # mtime=0 on the GZIP WRAPPER as well as on the members. tarfile's "w:gz"
    # stamps the current time into the gzip header, so fixing only the member
    # mtimes still gives a different checksum on every build -- which the
    # reproducibility check below caught, after the docstring had already
    # claimed otherwise.
    with gzip.GzipFile(path, "wb", compresslevel=9, mtime=0) as gz, \
            tarfile.open(fileobj=gz, mode="w") as tar:
        for p in members:
            info = tar.gettarinfo(os.path.join(ROOT, p), arcname=p)
            info.mtime = int(time.mktime(EPOCH + (0, 0, -1)))
            info.uid = info.gid = 0
            info.uname = info.gname = "momentaux"
            with open(os.path.join(ROOT, p), "rb") as fh:
                tar.addfile(info, fh)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "dist"))
    ap.add_argument("--verify", action="store_true",
                    help="report what each asset would contain, build nothing")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    print(f"{'asset':30} {'files':>13}  {'size':>11}")
    built = []
    for name, (pred, _desc) in ASSETS.items():
        p = build(name, pred, args.out, args.verify)
        if p:
            built.append(p)
    if args.verify:
        return

    sums = []
    for p in sorted(built):
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        sums.append(f"{h.hexdigest()}  {os.path.basename(p)}")
        print(f"  {os.path.basename(p):30} -> {os.path.getsize(p)/1048576:8.1f} MB packed")
    with open(os.path.join(args.out, "SHA256SUMS"), "w") as fh:
        fh.write("\n".join(sums) + "\n")
    print(f"\nwrote {len(built)} assets + SHA256SUMS to {args.out}")


if __name__ == "__main__":
    main()
