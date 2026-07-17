# MomentStem — research questions and answers

Every question this study has actually posed, with the answer, the evidence, and
its status. Negatives are recorded as prominently as positives: several cost a
full ablation to establish and are easy to re-litigate by accident.

`CLAUDE.md` remains the authoritative working ledger (denser, chronological).
This file is the navigable index into it. Last updated 2026-07-16.

**Status legend**
| tag | meaning |
|---|---|
| **SETTLED** | answered, multi-seed, survived at least one attempt to break it |
| **OPEN** | running or not yet run |
| **FALSIFIED** | we predicted X, measured not-X. Keep it dead. |
| **RETRACTED** | *we* claimed it, then found our own claim was wrong |

All cells: CIFAR-100, ResNet-18, frozen recipe (SGD m=0.9, lr 0.1, wd 5e-4,
cosine, 200 epochs, batch 128, crop+flip only), 3 seeds, unless stated. Δ is
top-1 vs the identical baseline cell.

---

## 0. The original hypotheses (from README)

**Q0.1 — H1: does a moment prior beat a vanilla backbone, with the gap growing as
data shrinks?**
**A: Qualified yes — but only after abandoning the forward path.** True for
forward-path stems below 5%, *false* at 10–25% where they actively hurt. The
honest version of H1 only holds for the aux formulation (§3). — **SETTLED**

**Q0.2 — H2: capacity substitution — does R18+moments approach vanilla R34?**
**A: No.** — **FALSIFIED**

**Q0.3 — H3: is the gain larger under corruption shift (CIFAR-C)?**
**A: No.** Null at 100%; at 5% it just tracks the clean gain rather than
exceeding it. The prior buys accuracy, not robustness. — **FALSIFIED**

---

## 1. Forward-path moment stems

**Q1.1 — Do fixed Gabor moment filters in the stem help?**
**A: Only at ≤5%, and they cost accuracy at 10–25% ("the penalty band").**
k11 Δ: +1.3@1%, +1.9@5%, **−1.4@10%, −2.4@15%**, ~0@100%. — **SETTLED**

**Q1.2 — Is there one best kernel size?**
**A: No — kernel size is a DATA-REGIME KNOB.** k11 wins 1–3% (coarser prior);
k5 wins 5–15% and nearly erases the band (+2.4@5%, −0.1@10%). k7 is not the
midpoint — it is worse than both at 10%. — **SETTLED**

**Q1.3 — Is the band caused by channel-scale imbalance?**
**A: No.** Calibration fixed a real v1 bug; the band survived it. — **FALSIFIED**

**Q1.4 — By channel collinearity (ZCA)?** **A: No.** — **FALSIFIED**

**Q1.5 — By an insufficient step budget?**
**A: No.** The deficit persists at 800 epochs. — **FALSIFIED**

**Q1.6 — Is the prior useful as an INITIALISATION (then learnable)?**
**A: No — worse than both alternatives.** `gabor-learn` loses the low-data gain
entirely *and* deepens the mid-data deficit. — **FALSIFIED**

**Q1.7 — As a WARMUP (freeze, then unfreeze at the overtake point)?**
**A: No — changes nothing** (−1.18/−2.38 vs fixed −1.40/−2.37 at 10/15%).
The benefit is constitutively tied to fixedness + low data. — **FALSIFIED**

**Q1.8 — Can a NONLINEAR / INVARIANT fixed feature escape the band?**
This was the main hypothesis for "make moments work after 5%": a prior encoding
what a mid-data net cannot self-learn (phase/rotation invariance, 2nd-order
statistics) might survive.
**A: No. Comprehensively falsified.** At 10%: magnitude −1.74, rotinv −5.64,
structure −5.20 — all *worse* than the linear k11 stem (−1.40), the invariant
ones catastrophically. **The band is agnostic to linear-vs-nonlinear and to
which invariance.** Any fixed pre-committed extra channel costs accuracy at
10%+. Strongly supports the low-data-statistics-estimation account. — **FALSIFIED**

**Q1.9 — Is phase-invariant energy a better LOW-data prior than oriented edges?**
**A: Yes — energy-magnitude is the low-data champion.** Δ +2.55@1%, +3.53@2%,
+3.34@5%, beating both Gabor stems. But it does NOT wash out at 100% (−3.89) the
way linear stems do, and its band penalty grows faster (−7.34@25%): a sharp ≤5%
specialist, harmful everywhere above. — **SETTLED**

**Q1.10 — Is rotation-invariance a genuine low-data prior?**
**A: Yes, but strictly worse than phase-invariance.** steerable (principled
angular-harmonic) +1.87@1% beats its crude rotinv form (+0.97) and beats the
Gabor stems at 1%, but loses to magnitude at every point and goes negative
earlier. — **SETTLED**

**Q1.11 — Zernike moments, anywhere?**
**A: Dead at every placement tried** (stem, MultiMaskPool readout). — **SETTLED**

**Q1.12 — Does a MultiMaskPool (Zernike/random/learned) readout work?**
**A: No.** Fails end-to-end under the frozen recipe despite +0.4–1.5 in linear
probes — a standing warning that probe gains need not survive end-to-end. — **FALSIFIED**

---

## 2. The pivot

**Q2.1 — Can the moments be made to scale WITH data instead of specialising to
low data?**
The deployment objection: a method that only helps below 5% is not deployable.
**A: Yes — MomentAux.** Move the moments OFF the forward path and make them a
training-only *soft prior*. Deployed model is a plain ResNet (RGB→logits,
identical FLOPs, **+0 inference params**); during training only, a 1×1-conv head
taps layer3 and is regressed onto fixed moment maps (MSE·λ + CE). — **SETTLED**

**KEY INSIGHT.** Forward-path moments are a **hard constraint** that occupies
input bandwidth abundant data wants back — hence the penalty band. Aux-loss
moments are a **soft prior** that shapes features without occupying the input,
so abundant data overrides them instead of paying for them.

---

## 3. MomentAux — the headline result

**Q3.1 — Does it actually scale?**
**A: Yes — positive at EVERY scale.** Champion (λ0 cosine→0, magnitude target,
tap layer3, MSE), best-per-regime λ0, Δ vs baseline:

| data | 1% | 2% | 3% | 5% | 7% | 10% | 15% | 25% | 100% |
|---|---|---|---|---|---|---|---|---|---|
| **Δ** | +1.91 | +3.14 | +3.68 | **+5.30** | +4.87 | +4.14 | +2.94 | +0.97 | +0.15 |
| λ0 | 2.0 | 2.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.3 | 0.3 | 0.1 |

Beats the best forward-path stem decisively at 3–15% (@10%: +4.14 vs −0.09) and
ties at 100%. — **SETTLED**

**CORRECTION (2026-07-16) — "POSITIVE AT EVERY SCALE" OVERSTATED THE 100% CELL.**
The ledger claimed **+0.24 (±0.10, ~2.4σ — small but real)** at 100%. That rested
on a **2-seed** baseline (78.43±0.07). With the third seed the baseline is
**78.52±0.16**, aux is 78.67±0.12, so:
**Δ = +0.15, combined σ ≈ 0.20 → 0.77σ — INDISTINGUISHABLE FROM ZERO.**
This does NOT damage the method: **neutrality at full data is exactly what the
structural λ→0 argument (Q3.2) predicts**, and +0.15±0.20 is a clean neutral.
What dies is the specific claim of a REAL POSITIVE at 100%. Say "positive at
every scale up to 25%, and neutral at 100% by construction" — not "positive
everywhere".
*Second time in one day a σ from 2-3 seeds nearly became a finding (cf. Q6.4).
Treat any 2-seed cell as provisional, especially when it anchors a claim.*

**Q3.2 — Must λ decay, and to what?**
**A: Always cosine-decay λ to EXACTLY 0.** That is what makes high-data
neutrality *structural* rather than tuned: late training becomes pure CE, so
100% neutrality is guaranteed by construction. →0.0 dominates →0.1 (@10% +4.14
vs +3.64; @100% +0.08 vs −0.19). — **SETTLED**

**Q3.3 — Is λ_start a regime knob?**
**A: Yes — it is THE knob** (exactly as kernel size was forward-path). Low data
wants a much stronger prior. A single λ0=1.0 is best only at 3–10%; it
undershoots at 1–2% and overshoots at 15–100%. — **SETTLED**

**Q3.4 — Is a weaker λ always safer at high data?**
**A: No — λ has an INTERIOR optimum per regime.** @25% constant λ: 0.02 −0.47 |
0.05 +0.11 | 0.10 +0.69 | 0.30 +0.46. **A too-weak aux is WORSE THAN NONE** — a
competing gradient with no payoff. And a gentle *schedule* beats the best
constant λ (+0.97 vs +0.69). — **SETTLED**

**Q3.5 — Which loss form?**
**A: MSE ≫ cosine** (@10% +2.81 vs +0.46). The magnitude *scale* of the moment
maps carries information; cosine discards it. — **SETTLED**

---

## 4. Defensibility — is it really the moments?

These are the controls that decide whether the paper has a contribution.

**Q4.1 — Is it just deep supervision / any auxiliary regression?**
**A: No.** Random-fixed target: +0.01@5%, +0.14@10% vs magnitude +3.31/+2.81.
**Moment gain +3.30/+2.67 over the structure control.** — **SETTLED**

**Q4.2 — Is it just distillation (a learned target would be better)?**
**A: No — and the negative is strong.** FitNets-style frozen same-data teacher's
layer3 features: −0.36@5%, +0.16@10%. **A learned target costing a whole extra
model does ~nothing, while the free hand-crafted moment gives +3.3/+2.8.** — **SETTLED**

**Q4.3 — Would any hand-crafted descriptor do (MaskFeat's HOG)?**
**A: No — HOG helps, but the moment is ~2× better.** HOG +1.22@5%, +1.37@10%.
So hand-crafted descriptors are a real family, and phase-invariant energy
specifically is the right member. — **SETTLED**

**Q4.4 — Which target family is best?**
**A: magnitude, decisively.** @10%, λ=0.3: magnitude +2.81 > structure +1.78 >
steerable +1.01 > rotinv +0.84 > gabor −0.24 > invariants −2.97.
Two lessons: the aux setting **rescues** features that were catastrophic
forward-path (rotinv −5.64→+0.84), and the best aux target is a **mild,
info-preserving nonlinearity** — raw edges and heavily-processed invariants are
both bad. — **SETTLED**

**Q4.6 — Are the low-data gains just memorised train/test duplicates?**
CIFAR-100's test set contains ~9% near-duplicates of training images
(927/10000). Every headline cell lives at 1-5% data, so this is a question any
reviewer will ask. ciFAIR (Barz & Denzler 2020) replaces exactly those images,
keeping labels and order -- a drop-in swap needing NO retraining.
**A: No. The gain survives deduplication essentially untouched.**

| cell | CIFAR | ciFAIR | drop |
|---|---|---|---|
| baseline @1% | 9.07 | 8.43 | -0.65 |
| aux @1% | 10.95 | 10.10 | -0.85 |
| baseline @5% | 25.31 | 23.26 | -2.05 |
| aux @5% | 30.72 | 28.63 | -2.09 |
| baseline @10% | 40.44 | 37.86 | -2.58 |
| aux @100% | 78.74 | 75.98 | -2.76 |

**Delta @5%: +5.41 -> +5.37. Delta @1%: +1.88 -> +1.67.** Both cells eat the
identical contamination, so it cancels in the delta.
Two predictions stated BEFORE running, both held: (i) the delta barely moves;
(ii) the ABSOLUTE drop GROWS with data (-0.65@1% -> -2.76@100%), because the
contamination is against the FULL train set -- at 1% only 500 training images
exist, so most duplicate sources are not in the subset at all. The penalty
therefore falls hardest on the high-data cells, which is where we claim
NEUTRALITY rather than gains. -- **SETTLED**
*(baseline @100%/@25% ciFAIR pending re-run: the first attempt read
`ablf_none/seed2` WHILE IT WAS STILL TRAINING and reported 70.39+/-14.03 for a
cell that is 78.43+/-0.07. Never evaluate a run dir a wave is writing to.)*

**Q4.5 — Is the FEATURE gain (not just accuracy) also the moments' doing?**
The probe (§7) shows aux improves features +4.71@1%, but ANY aux regression is a
regulariser, so a random target might do the same. Needed a random target at
*matched* λ0=2.0 cosine→0 (`auxrand_1pct_sched2`): the pre-existing random
control is λ=0.3 CONSTANT, so comparing it would confound target-type with
λ-schedule.
**A: Yes — magnitude gives 3.9x the feature gain of a random target.**

| @1%, matched λ0=2.0 | end-to-end | Δ | probe (features) | Δ |
|---|---|---|---|---|
| baseline | 8.90 ±0.09 | — | 26.80 ±0.22 | — |
| aux, **random** target | 9.48 ±0.07 | +0.58 | 28.00 ±0.41 | **+1.20** |
| aux, **magnitude** target | 10.81 ±0.09 | +1.90 | 31.50 ±0.34 | **+4.71** |

**But random is NOT zero, and that is a new fact.** A random fixed target does
improve features (+1.20, ~1/4 of the moment effect); it simply fails to reach
the logits (+0.58 e2e). The end-to-end controls (+0.14@10%, +0.01@5%) invited
the reading that a random target does NOTHING — it does not. The correct
statement is: any aux regression is a WEAK feature regulariser; the moment
structure is ~4x stronger and is what actually cashes in.
**It also independently reproduces the classifier bottleneck (Q7.2):**
realization at 1% is 1.90/4.71 = **40%** (magnitude) and 0.58/1.20 = **48%**
(random). Both cash <half of what they build — as predicted, since the
bottleneck is the 5-img/class classifier and is indifferent to WHY the features
improved. — **SETTLED**

---

## 5. Design — what is settled

**Q5.1 — Where to tap?**
**A: layer3 — and tap depth is NOT a regime knob** (the first knob here that
isn't). A broad plateau with one cliff, at BOTH ends of the data range:
@10% layer2 +2.83 ≈ layer3 +2.81, layer4 +0.92, multi-[2,3,4] +2.85 (no gain);
@1% layer1 +1.87, layer2 +1.65, layer3 +1.91.
*Reasoning note:* "SETTLED" was originally concluded from a **single** regime
(10%) in a study whose recurring finding is that nothing is settled from one
regime — bad justification that happened to reach the right answer. It has since
survived an out-of-regime test on evidence. — **SETTLED**

**Q5.2 — Do the model's params/FLOPs change?**
**A: No.** Deployed path is a bare ResNet: identical FLOPs, +0 inference params.
The aux head exists only during training. — **SETTLED**

---

## 6. Generalization

**Q6.1 — Does λ transfer across DEPTH?**
**A: Yes, untouched.** @10%, λ0=1.0: R18 +4.14, R34 +3.95. — **SETTLED**

**Q6.2 — Across BLOCK TYPE (ResNet-50 bottleneck)?**
**A: Not at first — R50 λ0=1.0 was −0.67 ±3.18 (unstable). Traced, then fixed.**
See Q8.1. With `head_norm`: **+3.93 ±0.32**. — **SETTLED**

**Q6.3 — Is there ONE λ for all backbones?**
**A: Yes — λ0=1.0 + `head_norm`.** R18 +4.31, R34 +3.95, R50 +3.93 — spread
inside seed noise, no per-family retuning.
`head_norm` **recovers gain rather than merely stabilising**: the tuned fallback
(R50 λ0=0.3) reached only +2.42 — weakening the prior suppressed the collapse
*and* the signal. Per-seed proof it is systematic, not a rare bad seed: R50
no-hn {41.22, **36.37**, 42.35} → hn {44.45, 44.95, 44.35}. **The best no-hn seed
is below the worst hn seed**; σ 3.18 → 0.32. — **SETTLED**

**Q6.4 — Is `head_norm` a safe always-on default?**
**A: Yes.** At 3 seeds it looked mean-neutral but with σ 0.33 → 1.04, which was
not resolvable (F(2,2) needs ~19×). At **6 seeds**: **+4.27 ±0.70** vs +4.14
±0.33 — mean unchanged (0.13, deep inside noise) and σ regressed 1.04 → 0.70.
F = 4.32 on df(5,2), nowhere near significance, so there is **no evidence that
`head_norm` destabilises R18**; the 3-seed σ was an artifact of seed0 (43.36)
landing low in a small sample (seeds 3-5: 44.29 / 44.83 / 44.14).
Seeds: 43.36, 44.70, 45.40, 44.29, 44.83, 44.14.
**The config is now ONE LINE for every backbone**: λ0=1.0, cosine→0, magnitude
target, tap layer3, MSE, head_norm on. Free on R18/R34; the difference between
+3.93 and −0.67±3.18 on R50. — **SETTLED**
*Lesson: a σ estimated from 3 seeds is nearly uninformative. Do not read one as
a finding in either direction.*

**Q6.5 — Across DATASET (CIFAR-10, champion transplanted verbatim)?**
**A: Yes.** 1%: 39.35 → 45.96 (**+6.61**, the largest gain on any dataset);
5%: 69.05 → 73.46 (+4.41); 10%: 80.71 → 81.80 (+1.09). — **SETTLED**

**Q6.6 — Across RESOLUTION (STL-10, 96×96)?**
**A: YES — resolution is irrelevant to the prior (2026-07-17).** STL-10 @10% is
500 imgs / 50-per-class / 600 steps — **identical to CIFAR-10 @1% on every axis
except resolution**. Predicted ~+6.6 in advance; reason to doubt it: k11 covers
34% of a 32×32 image but only 11.5% of a 96×96 one. LANDED: 41.58±1.31 →
47.51±0.60 = **+5.92 ±0.83** (3 seeds), within noise of CIFAR-10@1%'s +6.62
±0.18. Filter size is NOT a resolution knob; the prediction held on a dataset
with 3× the linear resolution and ImageNet-derived content. (stl@50% = 2500
imgs / 250-per-cls, the C10@5% mirror: +3.42 at 1 seed so far, vs C10@5%'s
+4.41 — same story at mid-data, pending seeds.) — **SETTLED**

**Q6.7 — Does the gain depend on CLASS GRANULARITY or on TOTAL DATA/COMPUTE?**
Undecidable from CIFAR-10-vs-100 (Q9.2), so `cifar100super` was built: CIFAR-100's
images with its 20 official coarse labels, reusing CIFAR-100's committed subset
indices, holding images+steps byte-identical while per-class count moves x5.
Fork stated in advance: ~+0.25 if gain follows per-class count; ~+5.30 if it
follows total data/compute.
**A: TOTAL DATA/COMPUTE. Decisive.** super5_none 42.66±0.68 -> super5_aux
48.51±0.40 = **+5.84** (3 seeds).

Same PER-CLASS count, 23x different gain:
| cell | images | steps | per-class | Δ |
|---|---|---|---|---|
| super @5% | 2500 | 3800 | 125 | **+5.84** |
| C100 @25% | 12500 | 19400 | 125 | **+0.25** |

Same DATA+COMPUTE, 5x different per-class -> same gain:
C10@5% (250/cls) +4.41 | super@5% (125/cls) +5.84 | C100@5% (25/cls) +5.30.
— **SETTLED**

**Q6.9 — THE SYNTHESIS: two mechanisms, one at each flank.**
One cell refuses to fit Q6.7: at 500 imgs / 600 steps, C10@1% gives +6.61
(50/cls) but C100@1% gives +1.49 (5/cls) — identical data AND compute, 4.4x
different gain. So granularity matters, but ONLY at the bottom. The account that
fits every cell:
  (a) **The prior's FEATURE benefit tracks TOTAL DATA/COMPUTE** — how much the
      data cannot teach by itself. Governs the RIGHT flank: as data suffices the
      prior goes redundant (+0.25@25%, +0.15@100% — neutral, see Q3.1).
  (b) **REALISING it needs >=~25 img/class** so the classifier can define a
      boundary. Governs the LEFT flank. Measured independently by the probe
      (Q7.2): realization 41% at 5/cls vs 84% at 25/cls.
This is why the curve is UNIMODAL with its peak near 25 img/class.
**LIVE PREDICTIONS (recorded before the runs landed):**
  - **tin @1%** = 1000 imgs / 1400 steps / **5 per class** -> below the
    threshold -> should be SUPPRESSED (~+1.5) like C100@1%, DESPITE having 2x
    the images of C100@1%. If it lands ~+5, (b) is WRONG.
  - **stl @10%** = 500 imgs / 600 steps / 50 per class -> matches C10@1% on
    every axis but resolution -> should land near **+6.6**.

**Q6.9a — did tin@1% hold? YES (2026-07-17).** 5.22±0.32 -> 6.95 = **+1.73**
(aux 1 seed so far; seeds 1-2 running). Squarely in the predicted suppressed
band, nowhere near the +5 that would have falsified (b). 2x the images and 2.3x
the steps of C100@1% bought ~nothing because per-class stayed at 5. This is (b)'s
first out-of-family test — Tiny-ImageNet (200 classes, 64x64) is not a dataset
the synthesis was built on. — **CONFIRMED (1 seed; 3-seed confirmation pending)**

**Q6.9b — THE MATCHED TRIPLE: the decisive evidence for (b).** All three cells
have IDENTICAL total images (1000) and IDENTICAL steps (1400), same champion
λ0=1.0 config; only per-class count differs:

| cell | per class | ways | baseline | aux | Δ |
|---|---|---|---|---|---|
| C10@2%  | 100 | 10  | 51.34±1.20 | 58.49±0.16 | **+7.15 ±0.70** |
| C100@2% | 10  | 100 | 14.17±0.15 | 16.67±0.23 | **+2.50 ±0.16** |
| tin@1%  | 5   | 200 | 5.22±0.32  | 6.95 (1 seed) | **+1.73 ±0.18** |

Monotone in per-class count at perfectly fixed data AND compute. The
C10-vs-C100 leg is **6.5σ** (+4.65 ±0.72). This is the MIRROR IMAGE of Q6.7:
that pair showed per-class ALONE doesn't determine gain (same per-class, 23x
different gain); this triple shows data+compute ALONE doesn't either (same
data+compute, 4.1x different gain). Together: **both factors are real, neither
suffices** — which is exactly the two-mechanism account, now with one decisive
experiment per flank.
**Reconciled by the ~25 img/cls threshold**, which both datasets now support:
at 2500 imgs every cell is at/above 25/cls (25/125/250) so all are realized and
the spread is flat (+5.30/+5.84/+4.41); at 1000 imgs the cells straddle it
(5/10/100) so the spread is steep (+1.73/+2.50/+7.15). Per-class count matters
enormously at 1000 imgs and barely at 2500 — what a threshold predicts, and why
neither factor alone ever fit the whole envelope.
**CAVEAT — do not over-read it.** At fixed total data, per-class count and CLASS
COUNT are perfectly anti-correlated (1000/10=100, 1000/100=10, 1000/200=5), so
the triple CANNOT separate "5 examples can't define a boundary" from "200-way is
a harder task"; tin also differs in resolution. What breaks the tie is Q6.7: at
2500 imgs, 20-way (+5.84) vs 100-way (+5.30) — 5x different class count, same
gain. So class count per se is not the driver at that scale.
— **CONFIRMED** (C10-vs-C100 leg 6.5σ; tin leg pending seeds)

**Q6.9c — is (b) a cliff or a ramp? UNRESOLVED, and the synthesis's own
prediction is nominally CONTRADICTED (2026-07-17).** Q6.9 predicted C10's summit
is at 0.5% (25/cls) and that "+6.61@1% is on the rising limb". Above the
threshold only (a) should govern (less data -> more gain), so C10@1% (50/cls)
ought to BEAT C10@2% (100/cls). Filling in 2% gives the reverse, nominally:
+7.15 vs +6.61. But the difference is **+0.53 ± 0.72 = 0.73σ — not
distinguishable**, so it adjudicates nothing in either direction. Either (b) is
GRADED and still biting at 50–100/cls, or the 1–2% band is a genuine plateau.
Resolving it needs ~10 seeds/cell (the σ is driven by the c10_none_2pct
baseline, ±1.20 from seeds 52.48/51.95/50.20). Until then the rising-limb claim
is **WITHDRAWN, not replaced**. See Q9.4. — **OPEN**

**Q6.9e — DOES THE PER-CLASS MECHANISM SURVIVE DATASET CONTROLS? The current
crisis (2026-07-17).** Filling in the tin envelope broke two things at once:
(1) **The ~25 img/cls threshold does NOT transfer.** Crossing 5→25 img/cls buys
    **+3.81** on C100 (+1.48→+5.30) but only **+0.53** on tin (+1.60→+2.13).
    tin@5% (25/cls) vs C100@5% (25/cls): +2.13 vs +5.30 — 8.1σ apart at the
    SAME per-class count.
(2) **The matched triple (Q6.9b) is CONFOUNDED by dataset identity.** At FIXED
    per-class count, changing only the dataset moves Δ by +2.48 (6.0σ, 50/cls:
    C10@1% vs C100@10%) and +3.17 (8.1σ, 25/cls: C100@5% vs tin@5%). The triple
    spans three datasets, so "monotone in per-class" cannot be attributed to
    per-class count from that design. Q6.9b's caveat was too mild: this is the
    same class of error as Q9.1 (a confounded comparison read as controlled).
WHAT SURVIVES UNTOUCHED — the left flank transplants PERFECTLY at matched
per-class AND matched images:
    5/cls:  C100@1% +1.48 ±0.12  vs  tin@1% +1.60 ±0.20   (0.5σ apart, despite
            2× images, 2× classes, 2× resolution — a universal ~+1.5 floor)
    50/cls: C10@1%  +6.62 ±0.18  vs  stl@10% +5.92 ±0.83  (0.8σ, despite 3×
            resolution and ImageNet content)
  So SOMETHING per-class-like governs the left flank; what fails is only the
  claim that a fixed count (~25) unlocks it dataset-independently.
THE FORK, with both tests RUNNING and predictions recorded in advance:
  (i) super@1% / super@2% (launched 2026-07-17) — C100's committed 1%/2% subsets
      with 20 coarse labels: BYTE-IDENTICAL pixels, identical steps; only
      granularity moves (5→25/cls at 1%, 10→50/cls at 2%). The only per-class
      manipulation possible BELOW the claimed threshold with dataset held fixed
      (super@5%/@10% sit above it at 125/250 per cls and tested nothing).
      PREDICTED: if granularity governs, super@1% ≈ +5 and super@2% ≈ +5.5..6.5;
      if the left flank is a dataset effect, both stay at C100's +1.5/+2.5.
      ~10σ between the outcomes; no result survives both hypotheses.
  (ii) probe decomposition of tin (probe2 wave, running) — splits tin@5%'s small
      +2.13 into feature gain × realization using Q7.3's machinery.
      PREDICTED: if tin's full-probe feature gap ≈ +6 (like C100@5%'s +6.35)
      with e2e only +2.13, REALIZATION is broken at 25/cls on tin → per-class
      thresholds scale with task difficulty (200 fine-grained classes need more
      examples per boundary). If the feature gap itself is ≈ +2.5, the PRIOR is
      weak on tin content → mechanism (a) is dataset-dependent, (b) survives.
  (iii) tin@10% (50/cls, running): graded-universal-realization predicts ≈ +4
      (matching C100@10% at 50/cls); a tin-specific cap predicts ≈ +2.
RIGHT FLANK — no single covariate collapses the datasets: at 5000 imgs Δ =
+1.09 (C10) / +2.13 (tin) / +3.19 (super) / +4.14 (C100); neither %, images,
steps, baseline acc, nor headroom orders all four. Redundancy onset is
task-specific. The left flank is where the universal structure lives. — **OPEN**

**Q6.9f — FORK (ii) RESOLVED BY THE PROBE (2026-07-17): REALIZATION IS
UNIVERSAL; THE FEATURE GAIN IS DATASET-DEPENDENT.** Decomposing Δ_e2e =
R(realization) × G(feature gain, full-label probe gap), 3 seeds throughout:

| cell | /cls | imgs | G (probe gap) | Δ e2e | R = Δ/G |
|---|---|---|---|---|---|
| C100@1% | 5 | 500 | +3.88 ±0.39 | +1.48 | **38%** |
| tin@1%  | 5 | 1000 | +4.33 ±0.30 | +1.60 | **37%** |
| C100@5% | 25 | 2500 | +6.35 ±0.34 | +5.30 | **83%** |
| tin@5%  | 25 | 5000 | +2.67 ±0.25 | +2.13 | **80%** |

**R matches across datasets to within noise at BOTH per-class counts** — the
realization curve R(per-class) is universal (mechanism (b) confirmed, now on
two datasets), while G varies ~2.4× with dataset at 5% (the prior improves tin
features far less than C100's). So the correct law is
    Δ(cell) = R(labels per class) × G(dataset, images),
with R universal and G task-specific. "The threshold does not transfer" (Q6.9e
item 1) is hereby REFINED: what failed to transfer was never realization — it
was G. tin@5% = 80% × 2.67 = +2.14 (measured +2.13). tin@1% has +2.7 pts of
feature gain sitting unclaimed (G +4.33, realized +1.60), like C100@1%.
G is NOT monotone in data and its shape is dataset-specific: C100 rises
3.88@500 → 6.35@2500; tin falls 4.33@1000 → 2.67@5000. The e2e peak location
is set by where G peaks (R saturates early).
**SHARPENED PREDICTIONS (v2, recorded 2026-07-17 from the tin probes, BEFORE
any super1/super2/tin@10% result; supersede Q6.9e's naive v1 numbers):**
  - R(50) ≈ 88–95% (from the R curve shape; three measurements incoming:
    C100@10%, C10@1%, stl@10% probes are queued). Implied G if R(50)=0.90:
    G(C100@10%) ≈ 4.6, G(C10@1%) ≈ 7.3, G(stl@10%) ≈ 6.6.
  - super@1% = R(25) × G(C100-pixels, 500 imgs) ≈ 0.82 × 3.9 ≈ **+2.5..+3.8**
    (v1 said ~+5 — that conflated G growth with R growth; G at 500 imgs is only
    ~3.9). Dataset-effect branch still predicts ~+1.5. Separation ~4σ.
  - super@2% = R(50) × G(C100, 1000) ≈ 0.9 × 4.5..5 ≈ **+4.0..+4.7** (v1 ~+6).
    Dataset-effect branch ~+2.5.
  - tin@10% = R(50) × G(tin, 10000). G(tin) is falling (4.33→2.67 over
    1000→5000), extrapolate 1.8..2.4 → Δ ≈ **+1.7..+2.2**. NOTE: the Q6.9e v1
    fork mislabeled ~+2 as the "tin-cap" branch — under the resolved model, +2
    CONFIRMS universal-R × falling-G(tin); only ~+4 would now surprise.
— **RESOLVED (R universal; G dataset-dependent); v2 predictions OPEN**

**AMENDED same day — the 50/cls probes BREAK the multiplicative form and fix
it into an ADDITIVE one.** All three R(50) measurements came in AT OR ABOVE
100%: C100@10% 113%, C10@1% 137%, stl@10% 126%. The v2 band "R(50) = 88–95%"
was WRONG, and with it the implied-G numbers (predicted 4.6/7.3/6.6, measured
3.66/4.81/4.70 — all derived through the wrong form). What replaces it, with
every probed pair (Δe2e = G + readout):

| cell | /cls | imgs | G | readout | Δe2e |
|---|---|---|---|---|---|
| C100@1% | 5 | 500 | +3.88 | −2.40 | +1.48 |
| tin@1%  | 5 | 1000 | +4.33 | −2.73 | +1.60 |
| C100@5% | 25 | 2500 | +6.35 | −1.06 | +5.30 |
| tin@5%  | 25 | 5000 | +2.67 | −0.54 | +2.13 |
| C100@10% | 50 | 5000 | +3.66 | **+0.48** | +4.14 |
| C10@1%  | 50 | 500 | +4.81 | **+1.80** | +6.62 |
| stl@10% | 50 | 500 | +4.70 | **+1.23** | +5.92 |

The readout term is MONOTONE in labels/class and CROSSES ZERO between 25 and
50: below, scarce labels leave feature gains unrealized (the left flank);
above, the e2e cell realizes MORE than its frozen-probe gap — aux features are
genuinely EASIER TO READ with few labels (self-realization at C10@1%: baseline
cashes 72.9% of its own probe ceiling, aux 78.2%). Same property Q7.3 saw from
the other side ("@5%-gains are 81% visible to a 5-shot head"). So the aux prior
confers TWO benefits: better features (G) and more label-efficient readout
(the positive term at ≥50/cls).
THREE STRUCTURAL CONFIRMATIONS: (i) the C10↔stl transplant holds
COMPONENT-WISE at matched 500 imgs / 50 cls — G 4.81 vs 4.70, readout +1.80 vs
+1.23 — not just in aggregate; (ii) G on C100 traces 3.88@500 → 6.35@2500 →
3.66@5000: G peaks exactly where the e2e envelope peaks (5%), so the envelope
shape IS the feature-gain curve; (iii) the universal-R statement at 5 and 25
/cls stands unchanged (38/37%, 83/80%).
CAVEAT: stl's "full-label" probe head is 500/cls (STL-10 has only 5k labeled
train images), vs 5000/cls (C10) and 500/cls (C100/tin per class varies) — G is
relative to each dataset's own probe ceiling.
PREDICTION BOOKKEEPING (against the super/tin cells still running): the sharp
fork vs the dataset-effect branch is UNCHANGED (super@1% ≈ +3 vs ≈ +1.5;
super@2% ≥ +4 vs ≈ +2.5). But the additive form shifts super@2% up to
~+5.0..+6.0 (G@1000 interpolates 3.9..6.4, readout ~+0.5..+1) — record BOTH
bands; and tin@10% is NO LONGER a sharp discriminator: additive predicts
+2.3..+3.4 (G(tin,10k)~1.8..2.4 + readout ~+0.5..+1) vs the old tin-cap ~+2 —
overlapping bands. Its value is now G/readout measurement (probe it when it
lands), not hypothesis selection. — **AMENDED; super cells remain the decisive
fork**

**Q6.9g — super@1% LANDED BETWEEN THE BRANCHES (2026-07-17): +2.09 ±0.40.**
Baseline 22.50±0.63 → aux 24.59±0.31 (3 seeds each). Scored against the
recorded branches: 2.5σ BELOW additive-granularity (+3.1), 1.5σ ABOVE
dataset-effect (+1.5) — the intermediate outcome the original ±10σ design
could not produce, made possible because the additive amendment narrowed the
granularity band. Against C100@1% (byte-identical pixels, fine labels, +1.48):
coarsening 5→25/cls moved Δ by **+0.61 ±0.42** — a granularity effect that
looks real but is MODEST, roughly a fifth of what the original threshold story
implied. NEITHER branch survives intact:
  - "realization unlocks at 25/cls" over-predicts (would need ≈+3.1 here);
  - "pure dataset effect, granularity irrelevant" under-predicts by 1.5σ.
THE SUSPECT ASSUMPTION, now measurable: the additive prediction transplanted
G = 3.88 measured under 100-way FINE CE onto 20-way COARSE training. If coarse
CE shapes features differently (less pressure toward fine distinctions → the
aux target overlaps more with what CE already teaches), G_coarse < G_fine and
the whole shortfall is in G, not readout. probe3 wave RUNNING on the super1
and super5 pairs; super2 pair will be probed when it lands. If G_coarse ≈ 2.9
with readout ≈ −0.8, the additive law survives with G label-space-DEPENDENT;
if G_coarse ≈ 3.9 with readout ≈ −1.8, the readout curve is not universal
across label spaces. super@2% branches restated BEFORE it lands:
multiplicative +4.0..4.7, additive +5.0..6.0, dataset-effect ~+2.5, and — if
the coarse-CE G-shrink seen here persists — an intermediate ~+3..3.5.
— **OPEN (probes + super@2% deciding)**

**Q6.9h — THE SUPER PROBES ANSWER THE FORK-WITHIN-THE-FORK (2026-07-17): G IS
LABEL-SPACE-INVARIANT; THE READOUT TERM FOLLOWS TASK PERFORMANCE, NOT LABEL
BUDGET.** Decomposing the coarse-label cells (same C100 pixels):

| | G_coarse | G_fine (same pixels) | readout | Δe2e |
|---|---|---|---|---|
| @500 imgs (25/cls) | +4.07 | +3.88 (C100@1%) | −1.98 | +2.09 |
| @2500 imgs (125/cls) | +5.44 | +6.35 (C100@5%) | +0.40 | +5.84 |

(1) **G is (approximately) label-space-invariant**: +0.19 ±0.5 apart at 500
imgs; −0.91 ±0.50 (1.8σ, mild shrink at most) at 2500. The aux prior shapes
features nearly identically whatever CE task runs alongside — G is a property
of (pixels, images, aux config). This kills the "coarse CE shrinks G"
explanation of Q6.9g's intermediate result.
(2) **Within a label space, readout rises monotonically with per-class count**
(20-way: −1.98@25 → +0.40@125), but the zero-crossing count is SPACE-DEPENDENT
(100-way: 25–50/cls; 20-way: ~25–125; 10-way: below 50).
(3) **The covariate that lines up all 9 probed cells: BASELINE ACCURACY.**
Readout is negative in every cell with baseline < ~25% and positive in every
cell with baseline > ~39% — the sign flips at base ≈ 30–35% in every label
space measured. Readout is a property of HOW WELL THE TASK IS LEARNED (how
good the e2e head itself is), not of the label budget per se. SIGN LAW ONLY:
magnitude below the crossing has residual structure (super1 −1.98 vs tin@5%
−0.54 at similar baselines).
REVISED LAW: Δe2e = G(pixels, images) + readout(task performance), with G
label-space-invariant and non-monotone in images, readout sign-governed by
baseline height. "Labels per class" was a PROXY: it correlated with baseline
height inside each dataset's envelope, which is why it looked universal on
C100/tin and then failed to transfer.
IN-FLIGHT PREDICTIONS UNDER THE REVISED LAW (recorded before results):
  super@2% (baseline ~35–38, G interp 4.3–4.7): **+4.3..+5.0** (supersedes the
    four earlier bands; dataset-effect ~+2.5 remains excluded-able);
  tin@10% (baseline ~28–30, G(tin,10k) ~1.8–2.4): **+1.5..+2.2**.
— **OPEN (super@2% and tin@10% are the next falsification tests)**

**Q6.9i — super@2% LANDED IN BAND: +5.08 ±0.20 (2026-07-17). THE DATASET-EFFECT
BRANCH IS DEAD (13σ).** 29.84±0.08 → 34.93±0.34, vs the revised-law band
+4.3..+5.0 recorded in Q6.9h (at its top edge). On byte-identical pixels and
identical steps, relabeling 100-way→20-way (10→50 per class) bought **+2.58**
over C100@2%'s +2.50 — granularity/readout is REAL and large at 1000 imgs,
completing the pattern from Q6.9g (+0.61 at 500 imgs, modest) and Q6.7
(nothing at 2500+, where readout has saturated on both sides).
ONE NUMBER LEFT UNEXPLAINED: baseline 29.84 sits below the ~30–35% readout
crossing, so the law wants readout ≤ 0 here, which forces G(C100-pixels,
1000 imgs, coarse) ≈ 5.1–5.5 — steeper than the 500→2500 interpolation
(4.07→5.44). Either G rises steeply below ~1000 imgs, or the 20-way crossing
sits lower (~28%). The super2 probes (running) decompose it directly.
Scorecard of the other predictions that landed simultaneously:
  - C10@25% = **−0.83 ±0.20** vs predicted −0.5..0 — direction and mechanism
    right (λ0=1.0 overshoot deep in sufficiency), magnitude slightly beyond
    the band. Second negative C10 cell; the λ0=0.3 rescue prediction stands.
  - stl@50% (3 seeds) = **+3.18 ±0.32** vs its C10@5% mirror +4.41 ±0.15 —
    3.5σ apart at matched 2500 imgs / 250-per-cls / 10-way. No law violation
    (G is dataset-dependent), but note the 500-img C10↔stl match was
    component-wise perfect while the 2500-img pair splits: G(stl) falls faster
    with data than G(C10). Probe pair queued if it matters later.
  - ConvNeXt-tiny + AdamW (diag-only): baseline 23.31±1.26 → aux 34.09±0.16 =
    **+10.77 ±0.74 (2 aux seeds, 3rd running)** — the largest gain measured in
    the study, first non-ResNet backbone, first non-SGD optimizer. The
    baseline is badly underfit (23% where R18-SGD gets 40.18), so read it as
    "the prior rescues an underfit modern backbone", not as a headline row;
    it does establish the mechanism is not ResNet- or SGD-specific.
— **fork CLOSED (granularity real, dataset-effect dead); G-vs-crossing tension
OPEN pending super2 probes**

---

## 7. Mechanism — why the curve has the shape it has

**Q7.1 — Why does the gain COLLAPSE at 1% (+1.91) vs 5% (+5.30)?**
First hypothesis: layer3 is *too late* — at 5 img/class the data cannot estimate
even early-layer features, so a mid-level prior arrives after the damage.
**A: FALSIFIED.** Tapping at layer1 — earliest possible, full 32×32, target
unpooled — recovers **nothing** (+1.87 vs +1.91). — **FALSIFIED**

**Q7.2 — Then why?**
**A: At 5 img/class the CLASSIFIER, not the features, is the bottleneck.**
Linear-probing frozen penultimate features on the FULL train set (removes the
classifier's data limit, so only feature quality is measured):

| | end-to-end | probe (frozen features) |
|---|---|---|
| @1% baseline | 8.90 | 26.81 ±0.24 |
| @1% aux λ0=2.0 | 10.81 (+1.91) | 31.51 ±0.37 (**+4.70**) |
| @5% baseline | 25.23 | 37.02 ±0.30 |
| @5% aux λ0=1.0 | 30.53 (+5.30) | 43.33 ±0.54 (**+6.31**) |

**The prior improves features at BOTH scales by comparable amounts. What differs
is REALIZATION: @5% 5.30/6.31 = 84%; @1% 1.91/4.70 = 41%.** The left flank is not
the prior failing — it is the classifier failing to cash it in.

Cross-checks: the probe tracks accuracy monotonically across λ (31.51 vs 30.68
mirrors +1.91 vs +1.49); it independently reproduces the tap null
(tap1 31.74 ≈ tap3 31.51); and it explains *why* Q7.1 failed — there was never a
feature deficit for an earlier tap to fix.

**Caveats:** the probe trains its head on 50k labels the cell never saw → a
DIAGNOSTIC, never a headline cell. The baseline's own 8.90→26.81 gap is not a
finding (any head with 100× labels does better); the finding is the aux-vs-
baseline delta under identical probing. Also note Q1.12: probe gains have failed
to survive end-to-end before in this very study. — **SETTLED** (control: Q4.5)

**Q7.3 — Does the aux-vs-baseline gap GROW with the head's label count?**
**A: YES at 1%, and the dose-response quantifies the whole left flank
(2026-07-17).** Refit the head on the *same frozen features* with k labels/class
(3 seeds each):

| head labels/cls | @1% gap (feat. trained 5/cls) | @5% gap (feat. trained 25/cls) |
|---|---|---|
| 5    | +2.08 ±0.17 | +5.13 ±0.44 |
| 25   | +3.02 ±0.10 | +5.31 ±0.43 |
| 100  | +3.75 ±0.31 | +5.88 ±0.47 |
| 500  | +4.71 ±0.25 | +6.32 ±0.35 |
| full | +4.71 ±0.23 | +6.35 ±0.34 |
| **e2e (own labels)** | **+1.91** | **+5.30** |

Two findings. (1) **The e2e classifier realizes exactly what an optimal linear
readout with the same label budget realizes**: e2e +1.91 ≈ the 5-shot probe gap
+2.08 (41% vs 44% of full); e2e +5.30 ≈ the 25-shot gap +5.31 (83% vs 84%).
So the left flank is NOT "SGD/CE fails to find the features" — a same-budget
LBFGS probe does no better. It is label scarcity at readout, full stop.
(2) **R(k) is not a universal curve**: the @5%-trained features' gain is already
81% visible to a 5-shot head, while the @1%-trained features' gain needs
hundreds of labels to cash. Feature gains earned at higher data are "linearly
shallow"; those earned at extreme scarcity are spread in directions a small
head cannot exploit. — **SETTLED**

**Q7.4 — Surviving account of the whole curve.**
Prior-shaped features commit during the high-LR phase. Beneficial when data
cannot estimate RGB statistics; costly forward-path at 10–25%; harmless at 100%.
conv1 usage ratio (logged per epoch) tracks pruning, but pruning conv1 does not
recover accuracy. — **SETTLED**

---

## 8. The R50 instability — a case study in tracing vs guessing

**Q8.1 — Why did ResNet-50 break at λ0=1.0?**
**A: A SCALE DEGENERACY in the aux objective.** ‖W·f − t‖² is invariant under
(f → f/c, W → c·W): SGD can minimise the aux by **collapsing the tapped features
and inflating the head**, learning nothing. Measured (300-step trace): R50
λ0=1.0 → layer3 std 0.596→**0.051** (12× collapse), ‖W_aux‖ 1.64→**11.2** (7×),
**CE stalls at chance (4.605) for 200+ steps**. Healthy comparators: R18 λ0=1.0
(0.59→0.40, W→3.05, CE→3.32); R50 λ0=0.3 (0.60→0.51, W→2.24, CE→3.52).
Bottleneck blocks are more collapsible (1×1 projections) — hence R50-only.
**Two fixes derived from the mechanism, both work:** (a) `head_norm` — project
‖W‖ back to init after every step, removing the degenerate direction; (b) cosine
loss — scale-free. — **SETTLED**

**Q8.2 — What was tried BEFORE tracing? (keep these dead)**
Three plausible guesses, all wrong — tracing is what actually solved it:
- **Magnitude/GradNorm balancing** — NO-OP. Aux magnitudes were already
  identical across backbones; there was nothing to balance.
- **fp32 / disabling AMP** — **WORSE**, not better (NaN@epoch 6 vs @27). Not a
  precision bug.
- **BatchNorm on the tap** — **MUCH worse** (NaN@21). BN *divides by* the
  collapsed 0.04 std: a **25× amplifier of the exact pathology** it was meant to
  cure. — **FALSIFIED**

---

## 9. Retractions — claims we made and then broke ourselves

**Q9.1 — "Gain tracks the DEFICIT the data leaves, not the data FRACTION."**
Committed to the ledger as a law on 2026-07-16. **RETRACTED the same day.**
- **Not monotone.** CIFAR-100 falsifies it alone: @1% the baseline is 8.90 — the
  largest deficit in the study — and the gain is the *smallest* non-zero one in
  the envelope. The curve is **unimodal**, peak at 5%. Below 25 img/class a
  bigger deficit gives a *smaller* gain. Holds with best-per-regime λ0 too
  (+1.91@1% < +5.30@5%), so not a λ artifact.
- **No cross-dataset x-axis** — see Q9.2.

**What survives, stated this narrowly:** on the **right flank**, gain decays to
~0 as data becomes sufficient — true on both datasets. The one prediction that
genuinely held (CIFAR-10 @10% → only +1.09 despite "10%" being CIFAR-100's peak
band, where percentage alone predicted ~+4) was a right-flank call. Real, but one
correct call was over-generalised into a two-sided law. — **RETRACTED**

**Q9.2 — Why can't CIFAR-10 vs CIFAR-100 settle the scaling variable?**
**A: It is STRUCTURALLY UNIDENTIFIABLE, not a missing variable.** Both datasets
are **exactly 50,000 images**, so a given % fixes total images *and* — with the
frozen recipe and `drop_last=True` — total optimizer steps, while per-class count
differs by exactly 10×. **Matching per-class count NECESSARILY unmatches total
data/steps by 10×.** You cannot match both from this pair.
The pair the retracted law rested on (C100@10% vs C10@1%: baselines 40.18 vs
39.35, both 50 img/class, +4.14 vs +6.61) differs **7800 vs 600 steps — 13×**. A
compute-mismatched pair was presented as a controlled comparison. Hence Q6.7. — **SETTLED**

**Q9.3 — Is step count a confound in cross-fraction claims?**
**A: Yes, first-order.** The frozen recipe ties steps to data (1% = 600 steps,
10% = 7800, 100% = 78000). The *same* 10% data at 800 epochs reaches 45.49 vs
40.18 at 200 — **training longer buys +5.31, more than the method's whole effect
at that cell (+4.14)**. Per-cell Δ stays valid (baseline gets identical steps),
but the *shape* of Δ vs "data" is really Δ vs "data AND compute jointly".
*Corollary:* sub-1% CIFAR-10 cells are **impossible** under the frozen recipe —
0.5% = 1 batch/epoch; 0.1% = **0 batches** (`drop_last=True`), an empty loader.
— **SETTLED**

**What survives (matched-%).** At matched percentage the comparison IS clean
(identical images, steps, recipe; only granularity differs): CIFAR-100 peaks at
5%; CIFAR-10 is already at its plateau by 1–2% and decaying by 5%. **At matched
data AND compute, the 100-class task needs more data before the prior pays off
than the 10-class task.** First confound-free cross-dataset statement we have,
and it survives Q9.4: it needs only CIFAR-10's plateau to sit LEFT of
CIFAR-100's, which holds however 1% vs 2% resolves.

**Q9.4 — "CIFAR-10 peaks at ≤1%" — WITHDRAWN (2026-07-17), and NOT replaced
with "peaks at 2%".** Filling in the 2% cell made it nominally the higher of the
two (+7.15 vs +6.61), which would have made "≤1%" a fourth retraction. It is not,
because the difference is **+0.53 ± 0.72 = 0.73σ** — the two are not
distinguishable. The 1–2% band is one flat plateau at +6.6..+7.1 and *which point
is the summit is unknown*. The σ comes entirely from the `c10_none_2pct`
baseline (±1.20; seeds 52.48/51.95/50.20 — one low seed), not from the aux cell
(±0.16).
**This is the same failure mode as Q9.1 (the retracted law), Q3.1 (the 100%
cell), and Q6.4 (the head_norm σ): a difference read off 3 seeds whose σ cannot
carry it.** The only thing different this time is that it was caught *before*
being written down. The standing lesson — *a σ from 2–3 seeds is nearly
uninformative in EITHER direction* — has now cost us four claims; treat any
3-seed difference under ~2σ as unmeasured, including ones that flatter the
method. — **WITHDRAWN** (resolvable: ~10 seeds on the 1% and 2% cells)

---

## 10. Where the moments still lose

**Q10.1 — Is MomentAux the best option everywhere?**
**A: No — 1–2% still belongs to the forward-path magnitude stem** (+2.55/+3.53
vs aux's +1.91/+3.14). At extreme scarcity a prior that never relaxes beats any
decaying one; fixed λ=2.0 @2% (+3.26) also beats the schedule (+2.50).
Crossover ≈3%. Consistent with Q7.2: the forward-path stem changes what the
classifier *sees*, rather than only shaping intermediate features. — **SETTLED**

**Q10.2 — Do the forward-path stem and the aux COMBINE at 1–2%?**
**A: NO — the combo is strictly worse than the forward-path stem alone
(2026-07-17).** combo = energy-magnitude stem in the forward path AND as the
aux target (λ0=2.0 schedule), the "they act through different mechanisms so
they may be additive" hypothesis from aux.py's docstring. CIFAR-100, 3 seeds:

| | @1% | @2% |
|---|---|---|
| forward-path stem alone | **+2.54 ±0.15** | **+3.52 ±0.28** |
| best aux alone (λ0=2.0) | +1.90 ±0.07 | +3.14 ±0.21 |
| combo | +1.73 ±0.23 | +2.98 ±0.15 |

Combo ≈ aux alone (the stem adds nothing once the aux is present) and is 2.9σ
WORSE than the stem alone @1%. Not additive — the aux constrains layer3 to
predict maps that are now directly present in the input, a redundant constraint
that only taxes CE. The hypothesis is falsified; the two mechanisms overlap
rather than compose. Vanilla-deploy remains intact as the method's story (the
combo forfeits it for a loss). — **SETTLED (negative)**

**Q10.3 — The champion config's zero-crossing is DATASET-DEPENDENT.**
CIFAR-10 @15% = **−0.66 ±0.22** (3σ negative) under champion λ0=1.0 — the first
negative cell in any champion envelope. "Positive up to 25%, neutral at 100%"
is a CIFAR-100 statement; on CIFAR-10 the crossing is between 10% and 15%.
Consistent with λ0 being a data-regime knob (C100@15% prefers λ0=0.3: +2.94 vs
+2.55; @25% +0.97 vs +0.25): C10's curve sits ~5× left of C100's, so C10@15%
(750/cls, baseline 85.7) is deep in overshoot. IMPORTANT NUANCE: λ→0 makes the
END of training pure CE but does NOT guarantee neutrality — the high-LR-phase
shaping can still cost when data is sufficient. Neutrality at C100@100% is an
empirical fact of that cell, not a structural theorem.
PREDICTION (recorded 2026-07-17, before any run): C10@15% at λ0=0.3 lands
+0.3..+1.0; C10@25% at λ0=1.0 (running) lands −0.5..0. — **OPEN**

---

## 11. Prior art (deep research, 2026-07-14)

Nearest neighbours, none of which is this: **Bhattarai 2023** (joint HOG aux —
but segmentation); **MaskFeat** (HOG target — but SSL pretraining);
**FitNets / Mostajabi** (intermediate-feature regression — but LEARNED targets).

**The novel intersection:** fixed hand-crafted **moment** target × joint aux+CE
**from scratch** × **vanilla deploy** × **provably scales with data**.

---

## 12. Open queue

| # | question | status |
|---|---|---|
| Q6.9e | **super@1%/@2%** — dataset-controlled granularity below the threshold (byte-identical pixels, coarse labels). THE fork; predictions on record | **running** |
| Q6.9e | **probe decomposition of tin** (+ R(k) anchors on C100@10%, C10@1%, stl@10%) — feature gain vs realization | **running** |
| Q6.9e | **tin@10%** — 50/cls on tin: graded-universal-R says ≈+4, tin-cap says ≈+2 | **running** |
| Q0.3 | **H3 re-opened**: CIFAR-C was only ever run on forward-path stems. `eval_robustness.py` never passed `moment_aux`, so aux checkpoints could not even load — the aux method has NEVER been tested under corruption. | running |
| Q6.8 | **ConvNeXt under AdamW** (diag-only) — first non-ResNet; SGD cells void (recipe is a ResNet recipe; SGD aux collapsed to chance) | running |
| — | CIFAR-10 envelope tail: 25% aux (prediction −0.5..0), 100% pair | running |
| Q9.4 | **~10 seeds on c10 1%/2%** — settles Q6.9c (cliff vs ramp) AND powers the variance-reduction claim (exact test currently p=0.073, suggestive). Cheap (600/1400 steps); wait for queue drain. | queued |
| Q6.9d | **tin20** — 1000 imgs / 1400 steps drawn from **20** of tin's 200 classes → 50/cls, 20-way, still 64×64. The within-tin granularity test (mirror of super@2%). If Δ jumps +1.60 → ~+6, granularity governs on tin too; if ~+2, tin is capped dataset-specifically. Needs a committed subset + 20-class wrapper. | queued |
| Q10.3 | **λ0=0.3 rescue at C10@15%/25%** — prediction on record: +0.3..+1.0 at 15%. Tests whether λ0 transplants by REGIME POSITION rather than percentage. | queued |
| — | **stl@20%** = 1000 imgs / 100-per-cls / 1400 steps — fourth dataset on the 1000-img rung (matched to C10@2%'s 100/cls: predict ≈+7 if per-class governs). The rung then spans 5/10/50(super2)/100 per cls at fixed data+compute. | queued |
| — | per-regime λ0 curves on CIFAR-10 / TIN (currently fixed-λ0 transplants) | not started |
| — | Tiny-ImageNet 25/100% (156k steps at 100%) | not started |

Settled this cycle: Q6.6 (stl, +5.92, prediction held), Q7.3 (dose-response:
e2e ≈ same-budget probe), Q10.2 (combo, negative), Q10.3 first half (C10@15%
−0.66: champion crossing is dataset-dependent).
