#!/usr/bin/env python3
"""Assemble the exact bundle that gets uploaded, or refuse to.

WHY THIS EXISTS. Every packaging defect in this manuscript's review history
came from the same place: the bundle was assembled by hand. The compiled PDF
was absent from five consecutive submissions and the response letter from two,
and in both cases the file was present in the repository the whole time. The
manifest was corrected, then the checker was written, and the package still
shipped without them, because nothing connected the check to the act of
assembling. A referee eventually asked the obvious question: was the checker
run at all? This script makes that question unanswerable in the bad direction.

WHAT IT DOES. It runs check_submission.py --strict FIRST and stops on any
failure, so a bundle cannot be produced from a manuscript that fails its own
check. It then copies every file named in SUBMISSION_FILES.md, resolving the
one entry that deliberately lives outside paper/ (the response letter), and
refuses to finish if any named file is missing. Files are flattened into the
bundle root because that is what the submission system receives, and a name
collision after flattening is an error rather than a silent overwrite.

    python scripts/make_submission.py            # build submission/
    python scripts/make_submission.py --zip      # and a .zip beside it

The bundle is a directory of real files, not a report about them, so what you
upload is what was checked.
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER = os.path.join(ROOT, "paper")
MANIFEST = os.path.join(PAPER, "SUBMISSION_FILES.md")
OUT = os.path.join(ROOT, "submission")

# Named in the manifest but not uploaded: they describe the package rather than
# forming part of it. Listed explicitly so the reason is on the record.
SKIP = {"SUBMISSION_FILES.md", "check_submission.py"}


def manifest_entries():
    """Every backticked path in the manifest table, in order."""
    out, seen = [], set()
    for line in open(MANIFEST).read().splitlines():
        if not line.startswith("|"):
            continue
        for item in re.findall(r"`([^`]+)`", line):
            if not re.search(r"\.[a-z]+$", item) or item in seen:
                continue
            seen.add(item)
            out.append(item)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", action="store_true", help="also write submission.zip")
    args = ap.parse_args()

    # The check runs first and its failure is fatal. This is the whole point of
    # the script: there is no path from a failing manuscript to a bundle.
    print("Running check_submission.py --strict\n")
    rc = subprocess.call([sys.executable,
                          os.path.join(PAPER, "check_submission.py"), "--strict"])
    if rc != 0:
        print("\nREFUSING TO ASSEMBLE: the manuscript fails its own check.\n"
              "Fix the failures above, rebuild, and run this again.")
        return 2

    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    os.makedirs(OUT)

    copied, missing, clashes = [], [], []
    # dst basename -> source it came from. A file may legitimately be named by
    # two manifest rows (the graphical abstract is both a figure and its own
    # deliverable), which is a duplicate listing and not a clash. Only two
    # DIFFERENT sources flattening to one name is an error.
    origin = {}
    for item in manifest_entries():
        if os.path.basename(item) in SKIP:
            continue
        pattern = os.path.join(PAPER, item)
        matches = sorted(__import__("glob").glob(pattern))
        if not matches:
            missing.append(item)
            continue
        for src in matches:
            name = os.path.basename(src)
            dst = os.path.join(OUT, name)
            prev = origin.get(name)
            if prev is not None:
                if os.path.realpath(prev) != os.path.realpath(src):
                    clashes.append(name)
                continue
            origin[name] = src
            shutil.copy2(src, dst)
            copied.append((item, name))

    for item in missing:
        print(f"  MISSING  {item}")
    for name in clashes:
        print(f"  COLLISION {name} (two manifest entries flatten to one name)")
    if missing or clashes:
        shutil.rmtree(OUT)
        print("\nREFUSING TO ASSEMBLE: manifest names files that are absent or "
              "that collide once flattened.")
        return 2

    # The two files that have actually gone missing in practice. Checking them
    # by name here is redundant with the manifest walk above, and it stays
    # because redundancy is cheap and this specific failure has cost five
    # review rounds.
    for must in ("main.pdf", "RESPONSE_TO_REFEREE.md"):
        if not os.path.exists(os.path.join(OUT, must)):
            print(f"\nREFUSING TO ASSEMBLE: {must} is not in the bundle.")
            shutil.rmtree(OUT)
            return 2

    for item, name in copied:
        note = "" if item.endswith(name) else f"   (from {item})"
        print(f"  ok    {name}{note}")

    total = sum(os.path.getsize(os.path.join(OUT, f)) for f in os.listdir(OUT))
    print(f"\n{len(copied)} files, {total/1e6:.1f} MB in {os.path.relpath(OUT, ROOT)}/")

    if args.zip:
        zpath = OUT + ".zip"
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
            for f in sorted(os.listdir(OUT)):
                z.write(os.path.join(OUT, f), f)
        print(f"wrote {os.path.relpath(zpath, ROOT)}")

    print("\nThis directory is the upload. Nothing else needs assembling.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
