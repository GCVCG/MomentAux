"""Generate data/subsets/tin_semantic_order.json: tin's 200 wnids sorted by
their WordNet HYPERNYM PATH, committed once (run locally; needs nltk+wordnet,
which the cluster never does -- data.py only reads the JSON).

This is the ONE-VARIABLE control for the semantic-vs-arbitrary coarse-
partition question (Q6.9j caveat): tinsuper's 20 groups are blocks of 10 in
LEXICOGRAPHIC wnid order (arbitrary -- offset order scatters semantics);
tinsem's are blocks of 10 in HYPERNYM-PATH order, which places semantically
related synsets adjacent (all fish before all birds before all vehicles...).
Same images, same block-of-10 construction, same 20x50 label structure at
tin@1%'s committed subset -- ONLY the sort key changes. No clustering
hyperparameters, fully deterministic, immune to cherry-picking.

The path string is root->leaf synset names joined with '/'; ties (identical
paths) break by wnid, also deterministic. For multi-inheritance synsets the
FIRST hypernym path (WordNet's canonical one) is used.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nltk.corpus import wordnet as wn

from data import SUBSET_DIR, tin_root

ORDER_PATH = os.path.join(SUBSET_DIR, "tin_semantic_order.json")


def hypernym_path(wnid):
    syn = wn.synset_from_pos_and_offset(wnid[0], int(wnid[1:]))
    path = syn.hypernym_paths()[0]  # canonical (first) path, root->leaf
    return "/".join(s.name() for s in path)


def main():
    root = tin_root("./data")
    wnids = sorted(
        d for d in os.listdir(os.path.join(root, "train"))
        if os.path.isdir(os.path.join(root, "train", d))
    )
    assert len(wnids) == 200, f"expected 200 tin wnids, got {len(wnids)}"
    paths = {w: hypernym_path(w) for w in wnids}
    ordered = sorted(wnids, key=lambda w: (paths[w], w))
    payload = {
        "order": ordered,
        "paths": paths,  # committed for auditability of the grouping
    }
    with open(ORDER_PATH, "w") as f:
        json.dump(payload, f, indent=1, sort_keys=True)
    print(f"wrote {ORDER_PATH}")
    for g in range(20):
        block = ordered[g * 10:(g + 1) * 10]
        leaves = [paths[w].rsplit("/", 1)[-1] for w in block]
        print(f"group {g:2d}: {', '.join(leaves)}")


if __name__ == "__main__":
    main()
