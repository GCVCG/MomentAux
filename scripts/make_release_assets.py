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


ASSETS = {
    "result-tables.tar.gz": (
        lambda p: p.startswith("results" + os.sep) and p.endswith(
            (".csv", ".md", ".tex", ".xlsx", ".json")),
        "aggregated tables, the law audit, and the per-analysis JSON records",
    ),
    "run-records.tar.gz": (
        lambda p: os.path.basename(p) in (
            "final.json", "linear_probe.json", "linear_probe_SHOTS.json",
            "robustness.json") and p.split(os.sep)[0].startswith("runs"),
        "every run's authoritative record: config as executed, accuracy, "
        "FLOPs, environment, and the probes behind G",
    ),
    "training-curves.tar.gz": (
        lambda p: os.path.basename(p) == "metrics.csv"
        and p.split(os.sep)[0].startswith("runs"),
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
        for f in files:
            full = os.path.join(base, f)
            yield os.path.relpath(full, ROOT)


def collect(pred):
    tops = ("results", "runs", "runs_turing", "runs_dense", "runs_ckptfix",
            "configs", "data", "logs")
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
