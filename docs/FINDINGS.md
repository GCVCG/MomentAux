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
| **Δ** | +1.91 | +3.14 | +3.68 | **+5.30** | +4.87 | +4.14 | +2.94 | +0.97 | +0.24 |
| λ0 | 2.0 | 2.0 | 1.0 | 1.0 | 1.0 | 1.0 | 0.3 | 0.3 | 0.1 |

Beats the best forward-path stem decisively at 3–15% (@10%: +4.14 vs −0.09) and
ties at 100%. — **SETTLED**

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

**Q4.5 — Is the FEATURE gain (not just accuracy) also the moments' doing?**
The probe (§7) shows aux improves features +4.70@1%. But any aux regression is a
regulariser — a random target might do the same. Requires a random target at
*matched* λ0=2.0 cosine→0; the existing random control is λ=0.3 constant, so
comparing them would confound target-type with λ-schedule. — **OPEN** (`auxrand_1pct_sched2`)

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
Mean-neutral on R18 (+4.31 vs +4.14) but σ 0.33 → 1.04 (~10× variance ratio;
F(2,2) needs ~19×, so not establishable from 3 seeds). Until resolved it is a
bottleneck-specific fix that happens to be free on R18. — **OPEN** (seeds 3–5 running)

**Q6.5 — Across DATASET (CIFAR-10, champion transplanted verbatim)?**
**A: Yes.** 1%: 39.35 → 45.96 (**+6.61**, the largest gain on any dataset);
5%: 69.05 → 73.46 (+4.41); 10%: 80.71 → 81.80 (+1.09). — **SETTLED**

**Q6.6 — Across RESOLUTION (STL-10, 96×96)?**
STL-10 @10% is 500 imgs / 50-per-class / 600 steps — **identical to CIFAR-10 @1%
on every axis except resolution**. CIFAR-10 @1% gave +6.61. Reason to doubt a
match: k11 covers 34% of a 32×32 image but only 11.5% of a 96×96 one, so the
prior describes much finer structure. A large shortfall would mean filter size
is a resolution knob and k11 was implicitly tuned to CIFAR's scale. — **OPEN**

**Q6.7 — Does the gain depend on CLASS GRANULARITY or on TOTAL DATA/COMPUTE?**
Undecidable from CIFAR-10-vs-100 (see Q9.2). `cifar100super` — CIFAR-100's
images with its 20 official coarse labels, reusing CIFAR-100's committed subset
indices — holds images and steps byte-identical and moves per-class count ×5.
Fork: ~0 if it follows per-class count (125/cls ≈ C100@25%, +0.25); ~+5.30 if it
follows total data/compute. — **OPEN**

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
to survive end-to-end before in this very study. — **SETTLED** (pending Q4.5)

**Q7.3 — Does the aux-vs-baseline gap GROW with the head's label count?**
The claim in Q7.2 rests on two points (41% vs 84%). The sharp version: refit the
head on the *same frozen features* with 5→500 labels/class. If the bottleneck is
the classifier's labels, the gap must grow. Protocol validated: probing the 1%
baseline at 5 labels/class gives **8.97**, reproducing its end-to-end **8.90**.
If instead the gap is flat, "available gain" is the wrong framing and the left
flank is a real ceiling. — **OPEN**

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
5%, CIFAR-10 at ≤1%. **At matched data AND compute, the 100-class task needs more
data before the prior pays off than the 10-class task.** First confound-free
cross-dataset statement we have.

---

## 10. Where the moments still lose

**Q10.1 — Is MomentAux the best option everywhere?**
**A: No — 1–2% still belongs to the forward-path magnitude stem** (+2.55/+3.53
vs aux's +1.91/+3.14). At extreme scarcity a prior that never relaxes beats any
decaying one; fixed λ=2.0 @2% (+3.26) also beats the schedule (+2.50).
Crossover ≈3%. Consistent with Q7.2: the forward-path stem changes what the
classifier *sees*, rather than only shaping intermediate features. — **SETTLED**

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
| Q4.5 | random target at matched λ0 — is the FEATURE gain the moments? | running |
| Q6.4 | R18 `head_norm` σ — safe always-on default? | running |
| Q6.6 | STL-10 96×96 — resolution transfer | running |
| Q6.7 | superclass fork — per-class count vs total data/compute | running |
| Q7.3 | shots dose-response — does the gap grow with head labels? | running |
| — | ciFAIR-100 re-eval of low-data points (train/test duplicate contamination) | not started |
