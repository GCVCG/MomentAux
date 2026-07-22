"""Split the grid work queue between the two clusters, DISJOINTLY.

    python scripts/split_worklist.py            # writes worklist.turing / worklist.bsc

User decision 2026-07-22: turing's 2 GPUs must not be spent on configurations
the study has already closed (forward-path stems in the penalty band, tap and
lambda variants that were settled, MultiMaskPool readouts, ConvNeXt-under-SGD
which sits at chance, the random-target and FitNets controls). BSC's 32 GPUs
absorb those instead.

The split is disjoint BY CONSTRUCTION -- turing takes the expensive tail of the
LIVE work (its GPUs have no queue wait and a 48h walltime, which suits long
runs), BSC takes literally everything else. No cross-site coordination needed.
"""

import glob
import os
import sys

import yaml

TURING_PORTIONS = (50, 100)      # the expensive live tail goes to turing


def is_dead(c):
    """Configurations the ledger has already closed. Filling them completes the
    table but re-measures known failures, so they are BSC-only."""
    aux = c.get("moment_aux") or {}
    stem = c.get("stem", "none") or "none"
    opt = (c.get("optimizer", "sgd") or "sgd").lower()
    if c.get("head_pool"):
        return True                                    # readout fails e2e
    if stem != "none":
        return True                                    # forward-path penalty band
    if aux.get("teacher") or aux.get("stem") == "random-fixed":
        return True                                    # ~0-gain controls
    if c.get("backbone") == "convnext_tiny" and opt == "sgd":
        return True                                    # chance-level (0.92%)
    if aux:
        if aux.get("hog"):
            return False                               # documented ~half: keep
        if aux.get("stem") != "energy-magnitude":
            return True                                # superseded target
        if str(aux.get("tap")) not in ("layer3", "blocks.8", "stages.2"):
            return True                                # settled tap variant
        if aux.get("weight_final") != 0.0:
            return True                                # superseded lambda
    return False


def main():
    lines = open("worklist.txt").read().splitlines()
    turing, bsc = [], []
    import re
    for line in lines:
        m = re.search(r"configs/grid/(\S+?)\.yaml", line)
        cfg = yaml.safe_load(open("configs/grid/%s.yaml" % m.group(1)))
        pct = cfg.get("subset_pct") or 100
        if not is_dead(cfg) and pct in TURING_PORTIONS:
            turing.append(line)
        else:
            bsc.append(line)
    # turing: most expensive first (the worklist is already cheap->expensive)
    turing.reverse()
    open("worklist.turing", "w").write("\n".join(turing) + "\n")
    open("worklist.bsc", "w").write("\n".join(bsc) + "\n")
    print("turing (live, %s%%): %d tasks" % ("/".join(map(str, TURING_PORTIONS)), len(turing)))
    print("bsc    (everything else): %d tasks" % len(bsc))
    print("total %d (was %d) -- disjoint: %s"
          % (len(turing) + len(bsc), len(lines),
             set(turing).isdisjoint(set(bsc))))


if __name__ == "__main__":
    main()
