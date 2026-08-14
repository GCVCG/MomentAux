# Response to the referee

This letter is cumulative and is organized newest first: round 4, then round
3, round 2 and round 1.

---

## Round 4

Four items, two of them blocking. We accept all four. One correction to the
record first, because it bears on how the referee should read the rest: the
letter was not deleted. It was moved from `paper/` to `docs/` so that
`paper/` would hold only files LaTeX compiles, and `SUBMISSION_FILES.md`
lists only files under `paper/`. The assembled package therefore dropped it
silently. That is a worse failure than deleting it deliberately would have
been, because nobody involved knew it had happened, and it is the same shape
as the `.gitignore`/`main.pdf` fault: a manifest that cannot see a file
cannot report it missing. The manifest now names the letter with its real
path, and `check_submission.py` fails without it.

**B1. The headline moved from 79.1% to 85.7% and we did not say so.**
Accepted without qualification, and the referee's reconstruction is exactly
right. We have added a paragraph to Section 4.2 that states the previous
figure, the scope rule, why it is the correct one, that it was applied when
the law was first stated and lost when the audit was rewritten for
seed-paired uncertainty, and the split of the cells it removes. The numbers
are generated, not typed: `make_paper_numbers.py` now recomputes the audit a
second time under the *old* predicate and emits `auditPrevRate`,
`auditPrevResolvable`, `auditExclCells`, `auditExclResolvable`,
`auditExclWrong`, `auditExclCorrect`, `auditExclRate` and the two
below-crossing rates, so the disclosure cannot drift from the table it
discloses.

Every figure in the referee's report reproduces from our records: 1,100
cells to 1,009, 511 resolvable to 461, 79.1% to 85.7%, and of the 50
resolvable cells removed, 41 wrong against 9 correct. Below the crossing the
count of correct cells is unchanged at 293 while 23 disappear, all of them
wrong, moving that flank from 87.7% to 94.2%. Answering the referee's
question directly: taken on their own the excluded cells agree with the sign
law **18.0%** of the time. We report that in the manuscript rather than only
here, because it is a result — it is the tax of Section 6.3 seen from the
readout side — and because a rate that low is precisely why removing those
cells moves the headline as much as it does.

The specific defect was that the scope filter tested the exported
`pretrained` field against the strings `true` and `1` while the exporter
writes `yes`. We note it in the audit script's own docstring as well, since
that is where the next person will look.

**B2. Response letter.** This section is the answer. The letter now covers
rounds 1 to 4 and ships with the package.

**NB1. Figure 3's clipped legend.** Fixed, and the class of fault is fixed
with it. `pubstyle.save` holds every figure's width at the column measure so
that figures are not rescaled by different factors in LaTeX; when a label
overran that width it was *cropped, silently*, which is how
`cannot test it: unresolved (450` reached the referee. `save` now raises
rather than crops. Running it over the whole figure set immediately caught a
**second** clipped figure — Figure 8's panel title overran by 18.3 points —
which the parenthesis check could not have found because nothing in it was
unbalanced. Both are regenerated; all eleven figures now pass, and the text
layer of every one has balanced parentheses.

**NB2. The variance decomposition moved with the rescoping.** Section 4.1
now says so, gives the previous four values, and explains the direction: the
removed cells carry large negative Δ and large negative G and were the
widest-scattered points in the fit.

**NB3. Unused macros.** `vitBestDelta` is now used in Section 3.8, where it
was doing its consistency-check job as a comment.

**NB4. The 50-epoch ensemble checkpoint, raised for the third time.** Our
apologies for needing three rounds. The reason is that the regression carries
an asymmetry term, so a pair whose members differ greatly in accuracy reports
that gap rather than diversity; 50 epochs is where SimCLR sits within a point
of the prior (30.81 against 30.53) where 200 and 800 epochs put it 3.9 and
9.2 points ahead. Section 7 now states this, states that the alternative
would confound diversity with the budget effect Section 6 isolates, and
states the cost — we have not measured how a longer-pre-trained arm
ensembles.

**NB5.** We agree and have not folded the dense law cells into the headline
audit.

**NB6. Pascal-Context.** Reordered so the limitation precedes the result, and
sharpened: the obstruction is not only that the baseline sits below the
bracket but that Δ runs +0.10 to +0.17 mIoU there, so readout is a difference
between two small numbers whatever its sign.

**Question 2: was `check_submission.py` run?** No. It was written during the
round-3 revision and never added to the packaging step, so it existed and sat
unused — which is the referee's "fixes what is named and does not always
generalize the naming" with an unusually clean illustration. It is now run,
and it now also checks the response letter and the highlights-to-manuscript
consistency. It passes.

---

The referee was right in round 3 that our previous letter answered only the
first review and carried numbers the manuscript had superseded; every figure
below is now the manuscript's current value, and where a round 1 table
predates a change in the audit's scope, both values are shown rather than the
old one quietly overwritten.

Across four rounds we have not disputed a point, because we have not yet
found one that was wrong. Two of the referee's objections cost us the
paper's headline number and one of its claims: the sign-law agreement fell
from 96.4% to **79.1%** when we accepted that our uncertainty formula
assumed an independence that does not hold, and the correlation between
source asymmetry and fusion gain was **withdrawn from the abstract** rather
than defended. Both changes are in the paper.

**What changed in rounds 1-3.** The two blocking issues, both clerical: the
letter is rewritten (this document), and the ViT baseline in Section 3.5 is
corrected from 62.7% to **61.4%**, which is what the records say and what
makes Table 8's +13.9 consistent. Beyond those, we took the referee's
closing observation — that our corrections land item by item rather than by
class — as the substantive criticism of this round, and answered it with
two build steps: the recurring numbers are now generated from the run
records as LaTeX macros, and citation hygiene is now a script. Both are
described under round 3 below, and both caught defects that had not been
reported.

---

# Round 3

Two blocking issues, both clerical, both ours. We also treat the referee's
closing observation — that our corrections have landed item by item rather
than by class — as the most useful criticism in this round, and have
answered it with two build steps rather than two more fixes.

### R3-1. The response letter was not updated for round 2

**Accepted without qualification.** The referee is right that the letter
answered a review it had outgrown, and right that it contradicted the
manuscript it accompanied. This document is now cumulative: round 3 first,
then round 2, then round 1, each with its own headings. Every number in it
has been brought to the manuscript's current values, and where a round 1
table was written before the audit's scope expanded, both figures are shown
rather than the old one silently overwritten.

We note the specific contradiction the referee found and confirm the
direction: the letter said 498 / 393 / 78.9% and held-out 273 at 79.9%; the
manuscript says **511 / 404 / 79.1%** and held-out **272 at 80.1%**. The
manuscript was right. The letter was stale.

### R3-2. The 62.7% in Section 3.5

**Correct, and the mechanism is worth stating because it is diagnostic.**
The manuscript's own records give the DeiT-augmented ViT-tiny pair at full
CIFAR-100 as **75.25 → 61.39**, a gain of **+13.86**, which is Table 8's
+13.9. The correct sentence is 75.3% against **61.4%**, as in the previous
two versions. The 62.7% we typed is the *above-crossing sign-agreement rate*
from Table 3 — a number from a different table on a different quantity that
happened to be in the same working buffer. It is now fixed, and both numbers
in that sentence are generated from the records rather than typed (below).

### The class-level answer

The referee observes that "the habit of checking every claim against its
table has not fully generalized". That is fair, and the pattern is visible
across rounds: six table contradictions in round 1, three lost citations in
round 2, one lost citation and one table contradiction in round 3. Each was
fixed properly and individually; nothing stopped the next one. We have
therefore made both classes into build steps.

**Numbers.** `scripts/make_paper_numbers.py` emits every recurring figure —
the audit partition, the compute accounting, the headline ViT cell — as
LaTeX macros derived from the run records and the audit. The prose now uses
`\vitBest`, `\computeGpuHours`, `\auditResidSD` and so on by name, so a
prose number and its table read one definition and cannot disagree. This
retires R3-2, NB1 and NB2 at the source rather than one at a time. It
immediately caught a fourth instance nobody had reported: Python rounds
half to **even**, so 75.25 became 75.2 and 2.15 became 2.1 in some places
and 2.2 in others. The generator rounds half up throughout.

**Citations.** `paper/check_citations.py` reports any bibliography entry no
`\cite` reaches, any citation without an entry, and the count against the
journal's hard 50-reference limit. Run on the round 3 sources it reproduces
exactly what the referee found by hand: `alain2016understanding` and
`ulyanov2018deep` uncited, 50 of 52 entries printing.

### Non-blocking items

**NB1, the GPU-hour split.** Correct: the components summed to 3,339
against 3,341. They are now recomputed per device class from the retained
run records, so the split sums to the total by construction. The device
figures move slightly (2,903 / 82 / 356) because the energy is now
integrated per device rather than at one average draw.

**NB2, 2.2 against 2.1.** Both were roundings of the audit's 2.15. The
manuscript now uses one generated macro in all three places; it reads 2.2.

**NB3, the linear-probe attribution.** Restored. Alain and Bengio are cited
where the protocol is defined, which is the right place, since that
protocol produces G.

**NB4, small-data ViT remedies.** Fixed, and the referee's sharper point —
that the closest competitors to our headline claim were unfindable — is
taken. The paragraph now cites Liu et al. (NeurIPS 2021), which adds a
dense relative-localization auxiliary task for exactly this deficit, and
says how it differs from ours: their target is computed from the images,
ours is pinned in advance. Swin is no longer offered as a small-data
remedy; it is described as an architecture that reintroduces the
hierarchical bias attention lacks, which is why we include it as a backbone.

*Both restorations required removing something, and we would rather say so
than have it noticed.* The journal caps research articles at 50 references
and states that non-compliance may lead to desk rejection; we were at 50.
We removed the historical attribution for Gabor filters as models of early
visual cortex, deleting the clause with it rather than leaving it
unsupported, and the corroborating citation for the ViT deficit being
representational rather than optimizational — a claim this paper measures
directly on 909 cells, so it now rests on our own evidence. Method and
competitor attributions outrank corroborating ones under a hard cap. We
would restore both if the editor can grant even two references.

**NB5, the 50-epoch ensemble checkpoint.** This one we believe is already
addressed, and we point rather than argue. Section 8.2 reads: "The
self-supervised arm here is a 50-epoch pre-training checkpoint rather than
the 200- or 800-epoch ones of Section 3.7, because this analysis needs two
arms of *matched accuracy* so that the asymmetry term does not dominate,
and that is the budget at which SimCLR lands within a point of the prior.
The budget lesson of Section 3.7 applies here too: a longer-pre-trained arm
would ensemble differently, and we have not measured that." That text was
added in the round 2 revision. If the referee read it and found it
insufficient we will expand it; if it was simply buried mid-paragraph, we
are happy to move it to the head of the subsection where it will be seen.

**NB6, highlight 4.** Accepted. Naming the budget at which we win is still
choosing the budget. It now reads: "Free prior beats 2×-compute SSL on
small ViTs; at 5× the ordering flips with data" (81 characters).

**NB7, the compiled PDF.** Supplied with this submission. Our apology for
three rounds without one.

### On what the referee could not verify

The referee notes, correctly, that whether the numbers in the tables come
from the records they claim is the one thing a referee cannot establish
from the paper alone. We cannot fix that within the manuscript, but we can
narrow it: every table and figure is regenerated by one command from the
released records, the audit is a committed script rather than a query, and
the two new checkers above are in the same repository. We would welcome the
editor appointing a reproducibility reviewer with access to the artifact.

---

# Round 2

Six blocking issues. All were accepted; five were fixed by re-running an
analysis, and the sixth by deleting a claim.

### R2-1. Figure 3 was drawn from the retracted audit

**Confirmed and fixed.** `paper/figs/make_figs.py` still computed the
readout SEM as `hypot(delta_sem, G_sem)` — the independent form that round 1
retracted — so the figure drew the old classification beneath a caption
already corrected to the new one. It now calls
`analysis/audit_law_paired.py`, the same source of truth as the graphical
abstract, which had been fixed while this generator was missed.

Regenerating it exposed that the tables were stale in the *other*
direction, because cells had landed since they were written. Every audit
number in the paper is now re-derived from one run of the audit: **1,100**
in scope, **511** resolvable, **404** correct = **79.1%**, **107** wrong,
**487** unresolved, **102** in the crossing bracket, held out **218 / 272 =
80.1%**, clustered **200 / 258**. The caption's stale 989 is now 1,100.

### R2-2. The r = −0.80 is a two-cluster artifact

**Accepted, and withdrawn rather than softened.** We recomputed it: pooled
r = −0.801, but **within** EuroSAT-MS it is **+0.08** and within So2Sat
**−0.05**, and the two clusters are disjoint on both axes. Its effective
sample size is two populations, not ten points, and those populations
differ in modality as well as in asymmetry, so nothing in our design
separates "too unequal" from "cross-modal".

It is removed from the abstract, restated in Section 8.1 as a contrast with
the confound named and the third population that would break it specified,
and demoted in the decision procedure to a ranking heuristic explicitly
flagged as the weakest-supported step. Highlight 2 was advertising the same
withdrawn claim and was rewritten.

### R2-3. The held-out count was unexplained

**Fixed, with the bias declared.** 272 of 511, and the shortfall is not a
sampling choice: re-fitting the bracket without a dataset makes it much
wider (for CIFAR-100, [13.7, 86.8]), and a cell inside a bracket makes no
prediction, so **193** are swallowed; **46** more sit in datasets too small
to receive a fold. 272 + 193 + 46 = 511. We also state what this does to
the comparison — it keeps the cells furthest from the crossing, where the
account is most confident — so the held-out rate reads as an upper bound
rather than as like-for-like.

### R2-4. Compute accounting not regenerated

**Rebuilt from the run records:** 3,341 GPU-hours over 8,827 records, with
the device split re-derived, 2,639 kWh and 51 g CO₂eq per run. The
8,827-against-8,887 discrepancy resolves by deletion: every retained record
carries a wall-clock field, so there is no second population of timed runs.

### R2-5. Three uncited claims

**Restored:** the scattering transformer described in technical detail now
cites Patro et al.; the carbon methodology cites Lacoste et al.; the
small-data ViT paragraph cites Swin for hierarchical bias. Staying inside
the 50-reference cap meant trading two attributions out rather than
conceding priority to Smirnov, which we judged the more valuable citation.

### R2-6. The letter contradicted the manuscript on the ViT 5% cell

**Fixed, and the manuscript was right.** That cell was deepened from 3 to 6
seeds after the letter was drafted, and the margin **strengthened** from
−1.08 (2.0 SEM) to **−1.57 (3.9 SEM)**. The letter now carries both and
says which is current. We note the irony that our fix for a stale letter in
round 2 was itself item-level, which is how the whole letter came to be
stale in round 3.

---

# Round 1

## Blocking issues

### 1. The law is validated in sample

**Accepted in principle, and tested.** We agree that a hit rate computed
with the crossing bracket estimated from the audited cells is a
goodness-of-fit statistic. We have run the check the referee asks for, in
the strongest form we could: **leave-one-dataset-out**. For each dataset,
the crossing bracket is re-estimated from the other datasets only, and that
dataset's cells are then audited against it.

| | cells | correct | rate | 95% CI |
|---|---:|---:|---:|---|
| in sample | 511 | 404 | 79.1% | [75.3, 82.4] |
| **held out** | **272** | **218** | **80.1%** | **[75.0, 84.5]** |

*(These are the current manuscript's values. When this response was first
written the same query returned 498 / 393 / 78.9% in sample and 273 / 218 /
79.9% held out; the multi-source cells entered the law's scope in round 2,
which moved the audit to the figures above. Nothing else about the analysis
changed.)*

The two are statistically indistinguishable. We therefore report, and now
state in Section 4.2, that the account **generalizes across datasets at the
rate it fits within them**. This does not rescue the 96% figure, which
falls for the separate reason in issue 2, but it does mean the fall is not
attributable to in-sample validation. Both figures are in Table 3.

### 2. Equation 4 assumes an independence that does not hold

**Accepted in full. This is the most consequential correction in the
revision, and the referee's reasoning was exactly right.**

Δ and G are computed from the same checkpoints, so across seeds they are
positively correlated and `Var(Δ − G)` is strictly smaller than the sum of
variances. We now form the readout per seed and take its standard error
directly:

```
readout_s = (aux_acc_s − aux_eval_s) − (base_acc_s − base_eval_s)
```

The independent formula **overstated the readout's uncertainty by a median
factor of 1.8**, in 83% of cells. Correcting it admits 1.8 times as many
cells to the audit, and the additional cells are precisely the borderline
ones the old bar excluded. As the referee predicted, the rate falls:

| | cells | correct | rate |
|---|---:|---:|---:|
| as published | 278 | 268 | 96.4% |
| **corrected** | **511** | **404** | **79.1%** |

The readout *values* are unchanged (median difference 0.003 points); this
is entirely the uncertainty formula. Every dependent number has been
re-derived: Tables 2 and 3, the abstract, the highlights, the introduction,
Sections 4 and 10, the conclusion and the graphical abstract. The audit is
now a committed script, `analysis/audit_law_paired.py`, which prints every
law-related number in the paper from one command.

We also report the disaggregation we had not previously made visible:
**below the crossing the sign is right 87.7% of the time, above it 61.4%**.
We now describe the account as informative on the scarce-data flank and
close to uninformative once the term it predicts has decayed to nothing.

### 3. The claim that readout depends on baseline accuracy alone

**Accepted; the claim is withdrawn.** We ran the decomposition:

| | R² |
|---|---:|
| baseline accuracy alone (5-point bins) | **0.18** |
| dataset, on the residual | 0.10 |
| backbone, on the residual | 0.004 |
| data fraction, on the residual | 0.015 |
| residual SD at fixed baseline | **2.2 points** |

Section 4.1 now states these numbers directly and says that baseline
accuracy is the largest single determinant and the only one whose effect is
consistent in sign, but that the account predicts a **sign and a trend, not
a value**. The referee's related observation about band widths is fair: the
pre-registered bands were set by reading the measured curve at matched
baseline heights on comparable cells, which is nearer to a nearest-neighbour
estimate than to a global curve, and we no longer imply otherwise.

### 4. Claims contradicted by our own tables

**All accepted; all corrected.** The referee is right that the pattern was
one-directional, which is the part we find most instructive.

- "Every convolutional backbone is neutral (+0.04)" at 1.28M images: false,
  MobileNetV3 gains +1.95 in the same block. Narrowed to the ResNet
  baseline, in the introduction, Section 3 and the graphical abstract.
- "The ten exceptions share one signature": five do. Corrected in both
  places.
- Transfer advantage "+13 to +24": Table 12 tops out at +18; scoped to the
  cells shown. The decision guide's tax range "−10 to −24" likewise, now
  "up to −17".
- Amplification ratio at 1%: the table said 2.2, its own columns give
  3.20/1.35 = 2.4. The text's 1.4–2.4× range was right and the table was
  the outlier; the table is corrected.

### 5. Configuration selection appears to have used test accuracy

**Confirmed, and disclosed.** There is no validation split anywhere in the
training or data code. Target family, tap depth, λ₀, head-norm and loss
form were swept on CIFAR-100 and scored on the same test split that
supplies this paper's CIFAR-100 numbers.

A new Section 3.6, *Configuration selection, and what it costs us*, states
this, marks the CIFAR-100 cells a **selection set** whose absolute values
carry a selection bias of unknown size, and re-bases the general claims:
the chosen setting was fixed once and applied verbatim to eleven further
datasets, nine backbones and two ImageNet-scale stages that played no part
in selection, and those populations are not contaminated. We think this is
in fact the stronger claim, and we now say so explicitly rather than
leaving the selection question unaddressed.

We did not re-select on a validation split, which would require re-running
the sweeps. We are willing to do so if the editor considers disclosure
insufficient.

### 6. Tuning parity is asymmetric

**Accepted as disclosure.** A new Section 3.7, *Comparator parity*, states
that comparators run at published defaults inside the frozen recipe at a
declared cost multiple while the prior received the sweep above; that at 1%
of CIFAR-100 self-supervised pre-training means 200 epochs over 500 images,
far short of published small-data practice; that our view-strength check
varies the view distribution and not the budget, so it does not address
this; and therefore that **undertraining and method weakness are not
separable under a cost-normalized protocol, by construction**. We now write
that "SimSiam learns almost nothing at this scale" is a statement about
SimSiam *at this budget*, and that readers optimizing for attainable
accuracy rather than accuracy per unit compute should treat our
self-supervised numbers as lower bounds.

**The pre-training epoch sweep the referee marks optional is now in the
revision, run across the whole data envelope, and it went against us in a
way two fractions would have hidden.** Both self-supervised comparators
were re-run at four times their pre-training budget at every fraction from
1 to 25% (new Table 4, Section 3.7). Gains over the baseline:

| CIFAR-100 | cost | 1% | 2% | 3% | 5% | 7% | 10% | 15% | 25% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **Prior (ours)** | **1.02×** | +1.40 | +2.50 | +3.68 | +5.15 | +4.87 | +3.75 | +2.55 | +0.16 |
| SimSiam, 200 ep | 2.00× | −0.04 | −0.11 | +0.11 | +0.13 | +0.87 | +0.61 | +0.17 | −0.03 |
| SimCLR, 200 ep | 2.00× | +2.25 | +4.88 | +5.99 | +9.05 | +9.86 | +8.76 | +6.64 | +2.81 |
| SimSiam, 800 ep | 5.00× | −0.11 | +0.19 | +0.54 | +1.90 | +3.39 | **+4.33** | +3.65 | +0.79 |
| SimCLR, 800 ep | 5.00× | **+4.52** | **+8.33** | **+10.82** | **+14.38** | **+13.88** | **+10.92** | **+8.36** | **+2.77** |

Three consequences, and we accept all of them.

1. **The referee was right about the budget.** SimSiam gains nothing at its
   published budget anywhere on the envelope, and up to +4.33 at four times
   it. We withdraw "SimSiam learns almost nothing at this scale."

2. **Against the stronger comparator the prior now loses everywhere,
   including at 1%.** SimCLR at 5× beats it by 3.12 to 9.24 points, at 8 to
   37 standard errors. Our earlier claim of near-parity with
   self-supervision at 1% was a statement at 2× and does not survive 5×. We
   have rewritten the convolutional positioning accordingly: **on conv
   backbones the prior's case is cost, not accuracy**. The attention
   results are measured at 2× and are unaffected; whether 5× SimCLR would
   also overturn the prior on ViT under a modern recipe is a question this
   experiment does not answer, and we do not claim it either way.

3. **Two fractions would have misled us.** At 5% and 10% alone — the points
   we would naturally have chosen, being where the referee's objection bit
   — the story reads "prior beats 5× SimSiam at the scarcer fraction, ties
   at the other." The envelope instead shows a crossover: the prior wins
   from 1 to 7% and loses at 10 to 25%, most clearly at 15% (−1.10, 5.2
   standard errors). We flag this because it is the same sparse-grid error
   we identify in our own earlier readings elsewhere in the paper.

4. **We ran the same test on the attention regime, unprompted.** Leaving
   that comparison at 2× while raising the convolutional one to 5× would
   have reproduced the very asymmetry the referee identified. Against
   ViT-tiny under the DeiT recipe, 5× SimCLR gives 30.45 / 40.57 / 55.85 at
   5 / 10 / 25%, against the prior's 28.87 / 41.29 / 57.32. The comparator
   **wins at 5%** (−1.57, 3.9 SEM), the two are level at 10%, and the prior
   wins at 25% (+1.47, 7.2 SEM).

   *Correction to an earlier draft of this letter, which quoted 29.97
   against 28.89 for a margin of −1.08 at 2.0 SEM.* That 5% cell was the
   one marginal result in the comparison, so we deepened both of its arms
   from three seeds to six. The margin **strengthened** rather than
   regressing, from −1.08 (2.0 SEM) to −1.57 (3.9 SEM), and the manuscript
   reports the six-seed values throughout. We flag it because the earlier
   number made this cell look marginal when it is not: the comparator
   genuinely wins there, and we would rather the record be unambiguous
   about a result that goes against us. The attention claim therefore survives, but
   we now state it with its budget attached: the prior leads at every
   fraction at 2×, and only in the higher-data band at 5×.

   Repeating it on Tiny-ImageNet at 5, 10 and 25% gives the same ordering
   at every matched fraction as CIFAR-100: the comparator wins at 5% (3.9
   and 3.7 SEM), the two are level at 10%, and the prior wins at 25% (7.2
   and 5.1 SEM). We report that we first measured only 5 and 10% there,
   concluded the populations told different stories, predicted the 25% cell
   against our own method, and were wrong by 5.1 SEM — the same two-point
   error the budget envelope itself was run to avoid.

   We also record that this contradicted our own pre-registered prediction —
   we expected 25% to be the cell that flipped and it was 5% — and why: the
   extra budget buys the comparator +8.32 / +4.51 / +3.53 at 5 / 10 / 25%,
   decreasing with data, because 200 epochs over 2,500 images is fewer than
   four thousand steps. We had assumed the comparator was data-starved where
   it was in fact step-starved.

---

## The scope question

We take this seriously and have acted on the concrete part of it. The
referee is right that the submitted bibliography engaged no fusion
literature, and that a framing asserted rather than grounded is
indistinguishable from a decorative one.

A new **Section 2.1, *Relation to fusion theory***, does the grounding, and
its first move is to concede priority. Multi-sensor fusion has
distinguished **complementary** from **redundant** and **cooperative**
sources since Durrant-Whyte (1988); our *stack* and *substitute* are that
distinction, reached empirically from the opposite direction. We now say so.
What we claim instead is operational: the classical taxonomy states what
sources *are*, given known sensing geometry, and offers no way to determine
which case holds when the sources are a hand-crafted prior and a learned
representation, whose relationship is not evident a priori and, as our
Section 6 shows, is not predictable from their family names either, since
the same prior is redundant with effective self-supervision and
complementary with augmentation. A cheap frozen-feature measurement decides
it in advance.

Our third outcome, *tax*, has no classical counterpart; the nearest account
is negative transfer, which that literature treats as source–target
mismatch rather than as a fusion outcome. We name partial information
decomposition as the formal version of "currency" and explicitly disclaim
computing it: our measurement is an ordinal proxy. And we place the method
in von Rueden et al.'s informed-machine-learning taxonomy, which catalogues
*where* knowledge can be injected but does not address *when* injecting it
is worth anything given what the data and available pre-trained artifacts
already supply. That is the question this benchmark answers.

Whether this makes the paper in scope is the editor's to decide, and we
would rather have a ruling now than after a further round. If the ruling
goes the other way we are grateful for the referee's explicit statement
that the work is strong and the venue may simply be wrong.

## On the source comments

The referee is correct, and we are embarrassed. The LaTeX source carried
working comments recording our venue-framing decision and an unresolved
instruction to ourselves about the generative-AI declaration. They have
been removed. We note that the referee chose to record this as an
observation rather than an allegation, and we appreciate the fairness of
that.
