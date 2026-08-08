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
revision, and it went partly against us.** We re-ran SimSiam, our weakest
comparator, at four times its pre-training budget (new Table 4,
Section 3.7):

| CIFAR-100 | cost | 5% | Δ | 10% | Δ |
|---|---:|---:|---:|---:|---:|
| Baseline | 1.00× | 25.36 | — | 40.28 | — |
| **Prior (ours)** | **1.02×** | **30.51** | **+5.15** | **44.03** | **+3.75** |
| SimSiam, 200 ep | 2.00× | 25.49 | +0.13 | 40.89 | +0.61 |
| SimCLR, 200 ep | 2.00× | 34.41 | +9.05 | 49.04 | +8.76 |
| SimSiam, 800 ep | 5.00× | 27.26 | +1.90 | 44.61 | +4.33 |

The referee was right. SimSiam's near-null result is a property of its
budget, not of the method: four times the pre-training multiplies its gain
by four to seven. **We withdraw "SimSiam learns almost nothing at this
scale"** and now say it is budget-starved at the cost we normalized to.

The ordering survives, but only partly, and we state which part. At 5% the
prior still leads five-fold-cost SimSiam by 3.25 ± 0.38 at a fiftieth of
the overhead. At 10% the two are level (+0.58 ± 0.42, unresolvable), and we
record a tie rather than claim a win. We also say plainly that this tests
one method at one multiple and cannot bound the question in general.

---

## Non-blocking issues

1. **Dataset and backbone counts.** Corrected throughout. The grid holds
   **12 datasets** (one evaluation-only) plus **five relabelled controls**,
   giving 16 dataset identities, 15 in the law's scope; and **9 backbones
   across 5 architecture families**. "14 datasets" and "7 backbone
   families" were both unsupported.
2. **Backbone families.** As above; the audit table now reads 9 (5), and
   the appendix explains why the identity count exceeds the image-source
   count.
3. **GPU-hour components.** Corrected to 3,281, the sum of its parts.
4. **Table 15 versus Table 5.** The table now names its backbone
   (ResNet-18) and states that these cells predate the champion's final
   weight schedule, so they are internally comparable to each other, which
   is what an ablation requires, but not to the envelope.
5. **Section 6.3's unnamed cell.** Now named: CIFAR-100 at 10%, the
   head-norm cells of the backbone sweep.
6. **Equation 2.** Rewritten as a pointwise complex modulus.
7. **Appendix B's readout branch provenance.** Addressed by the new
   variance decomposition, which states the residual spread directly.
8. **CIFAR-10 unimodality.** Softened to a plateau across 1–2% whose
   difference (0.29) the seed uncertainty cannot resolve.
9. **Uncited references.** ResNet, Swin, ConvNeXt, MobileNetV3, AdamW,
   Raghu et al. and Ulyanov et al. are now cited in the text.
10. **Anonymization contradiction.** Removed.
11. **PathMNIST ethics.** A sentence now confirms it is a public,
    de-identified benchmark requiring no ethics approval.

**Minor.** The duplicated dataset table is deleted. The 123.6 pt overfull
box is reproduced by the *unmodified* `cas-dc-template.tex` shipped with
the Elsevier template; it is a defect in the class file, marks an empty
box, and no visible text overflows. We have left it rather than patch the
publisher's class.

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
