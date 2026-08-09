# Response to the referee

We thank the referee for a review of unusual care. Several points were not
stylistic disagreements but errors on our part, and one of them changes the
paper's central number. We have made every change we could make from
existing records, and we say plainly where a criticism stands and we have
only disclosed rather than removed the problem.

**Summary of what changed.** The headline sign-law agreement falls from
96.4% to **78.9%**, because the referee is right that our uncertainty
formula was wrong. The claim that the readout term depends on baseline
accuracy alone is **withdrawn**. Configuration selection used the test
split and this is now disclosed, with the general claims re-based on the
populations that played no part in selection. Six factual overstatements,
all of which the referee caught by checking our own tables, are corrected.

---

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
| in sample | 498 | 393 | 78.9% | [75.1, 82.3] |
| **held out** | **273** | **218** | **79.9%** | **[74.7, 84.2]** |

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
| **corrected** | **498** | **393** | **78.9%** |

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
   ViT-tiny under the DeiT recipe, 5× SimCLR gives 29.97 / 40.57 / 55.85 at
   5 / 10 / 25%, against the prior's 28.89 / 41.29 / 57.32. The comparator
   **wins at 5%** (−1.08, 2.0 SEM), the two are level at 10%, and the prior
   wins at 25% (+1.47, 7.2 SEM). The attention claim therefore survives, but
   we now state it with its budget attached: the prior leads at every
   fraction at 2×, and from about 10% upward at 5%.

   Repeating it on Tiny-ImageNet narrows the claim further: at 5× the
   comparator is ahead at 5% (3.7 SEM) and level at 10%, so on that dataset
   the prior leads nowhere we measured, while at 2× it leads on both
   populations. The manuscript now states the modern-recipe claim with both
   its budget and its population attached.

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
