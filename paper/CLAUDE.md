- FOURTH INSTANCE OF THE STALE-SIBLING CLASS, caught by the final writer
  pass: tables/partitions.tex's GENERATED caption still said "twenty-one
  dataset identities ... twelve independent image sources" after the scope
  repair moved both counts (20/11). The generator, not the emitted file, was
  the stale site -- make_partition_table.py carried the literals -- so the
  fix went to the generator and the regenerated diff touches the caption
  line only. A generated artifact is only as current as the constants inside
  its generator; regeneration alone cannot catch a literal the generator
  itself hardcodes.
- FINAL GATE (2026-08-18): article exactly 35 pages, 49 references, abstract
  250 words, highlights 5 x <=85 chars, 0 LaTeX errors, 0 undefined refs,
  all 93 generated macros consumed, supplementary xr refs resolve. The
  consolidated pass applied ~60 audited number corrections, P1-P9, the
  frozen-parameter generalization in the data-vs-optimization section (three
  frozen constants, one failure mode, stated once), the two-repair audit
  narration with the breadth trade stated both ways, and the tab:scale
  ceiling caveat extended to every ImageNet-100 row.
