# Deep research prompt: grounding a knowledge-versus-data study in the information fusion literature

Copy everything below the line into Claude's deep research mode.

---

## Why I am asking

I have a manuscript under review at **Information Fusion** (Elsevier). A
referee has recommended the editor rule it **out of scope**, and their
single most damaging observation is factual and correct:

> "of 55 entries in `refs.bib`, none is from Information Fusion and none
> engages the fusion literature at all, classical or modern. The paper
> coins its own vocabulary of currencies, stacking and taxing without
> connecting it to any existing account of source combination."

They also note that nothing in the study is multi-sensor or multi-modal:
every experiment is single-image classification.

I need to know whether the paper's actual content has a genuine home in
the fusion literature, or whether the framing really is a wrapper. **I want
the honest answer, not a rescue.** If the literature does not support the
framing, say so plainly and I will change venue.

## What the paper actually does

Strip the framing and this is what remains.

- A **fixed, hand-crafted source of knowledge** (a pinned bank of Gabor
  and moment filters, derived from classical vision science and models of
  simple-cell receptive fields) is injected into a neural network's
  training objective as an auxiliary regression target. It is present only
  during training, its weight decays to exactly zero, and the deployed
  network is unchanged.
- This is compared, under **one frozen training recipe with committed data
  subsets** and at **declared training-compute multiples**, against the
  data-driven alternatives: self-supervised pre-training (SimCLR, SimSiam,
  DINO), ImageNet transfer, heavy augmentation, and a learned
  (FitNets-style) teacher. 2,800 cells, roughly 8,900 runs, 12 datasets,
  9 backbones, 500 to 1.28M images.
- **All pairwise combinations are also measured**, which is the part I
  believe is the actual contribution. The finding is a three-way outcome
  rule for combining two information sources:
  - **stack**: the sources supply different things and their gains
    compound (the fixed prior with heavy augmentation, 1.4 to 2.4 times);
  - **substitute**: they supply the same thing and the second adds nothing
    (the fixed prior with effective self-supervised pre-training, where a
    frozen-feature probe shows the two produce the *same* feature gain);
  - **tax**: injecting the prior into an already-informed initialization
    destroys what that initialization carried, in proportion to how much
    it carried (up to −17 points against ImageNet transfer, and a measured
    zero where ImageNet features do not transfer).
  We call the deciding property the "currency" each source supplies, and
  we measure it with a cheap linear evaluation of frozen features, which
  means the outcome of a combination can be **predicted before the
  combined system is trained**.

## What I need you to find out

### 1. Does *Information Fusion* itself publish this kind of work?

Search the journal's own record, not the general literature.

- Does it publish on **knowledge-driven and data-driven hybrid
  approaches**, **knowledge fusion**, **informed / knowledge-guided
  machine learning**, or **physics-informed learning**? Give me concrete,
  citable examples from the journal, with years, and say how central or
  peripheral they are to its output.
- Has it published work where the fused sources are **not** multiple
  sensors or modalities, for example fusing prior knowledge with learned
  representations, or fusing an inductive bias with data? If yes, these
  are precedents I must cite. If it consistently requires multiple
  physical sources, that is decisive and I need to know.
- Does it publish **benchmark or comparative-evaluation** papers as
  opposed to method papers?
- Find its current aims and scope, and any recent editorial that
  clarifies the boundary. Quote the scope text.

### 2. The classical fusion taxonomy: is my three-way rule a rediscovery?

This is the question I most want answered.

Classical multi-sensor fusion (I believe Durrant-Whyte, around 1988, and
subsequent taxonomies) classifies the relationship between data sources as
**complementary**, **redundant**, and **cooperative**. My stack / substitute
outcomes look suspiciously like complementary / redundant, arrived at
empirically and from a completely different direction.

- Confirm or correct that classical taxonomy: who proposed it, in what
  terms, and how is it stated in current fusion textbooks and surveys?
- **Is my mapping real or superficial?** Stack against complementary,
  substitute against redundant. Where does it break down? Is there an
  established term for my third outcome, where adding a second source
  actively destroys what the first supplied? (Candidates to check:
  destructive or negative fusion, catastrophic forgetting framing, the
  fusion literature on conflicting or unreliable sources, Dempster-Shafer
  conflict.)
- If the mapping holds, my contribution is not new vocabulary but an
  **empirical, controlled measurement of a classical distinction in a
  modern representation-learning setting**, plus a cheap predictor of
  which case you are in. Tell me honestly whether that is a fair claim or
  an overreach.

### 3. Information-theoretic accounts of redundancy and synergy

My "currency" idea is informal. Is there a formal account I should be
using, or at least citing?

- **Partial information decomposition** (Williams and Beer, and the
  subsequent literature) decomposes multi-source information into unique,
  redundant and synergistic components. Is that the formal version of what
  I am measuring empirically? Has it been applied in fusion or in deep
  learning, and is there an accepted estimator at the scale of neural
  network features?
- Any other formal treatment of when combining sources helps: rate-
  distortion, information bottleneck applied to multi-source settings,
  ensemble diversity theory, multi-view learning assumptions
  (view-sufficiency, view-redundancy).
- Would framing my result in these terms strengthen it, or would it
  overclaim, since I measure a proxy (a linear evaluation gap) rather than
  a mutual information?

### 4. Informed machine learning as the natural bridge

I believe von Rueden et al., "Informed Machine Learning: A Taxonomy and
Survey of Integrating Prior Knowledge into Learning Systems" (IEEE TKDE),
classifies exactly what I do: knowledge **source** (vision science),
knowledge **representation** (fixed filter bank producing target maps),
and **integration point** (the training objective).

- Verify that this taxonomy exists and describe its axes precisely.
- **Where does my method sit in it, and is that cell well populated or
  sparse?**
- Critically: does that literature, or any related one, already answer
  **when injecting knowledge pays versus when it is redundant with what
  the data supplies**? I believe it catalogues *where* knowledge can be
  injected but not *when it is worth injecting*. If that gap is real, it
  is my contribution's home. If someone has already answered it, I need
  the citation urgently.
- Related bodies to check: theory-guided data science (Karpatne et al.),
  knowledge-guided machine learning, physics-informed neural networks,
  neuro-symbolic integration, and any survey of inductive bias versus data
  scale.

### 5. Precedent for the specific empirical claims

- Has anyone run a **controlled, cost-normalized** comparison of
  hand-crafted priors against self-supervised pre-training and transfer,
  where compute is held constant or declared? I believe not, but I need to
  be sure.
- Is there prior work showing that **a hand-crafted prior and a learned
  initialization are redundant with each other**, or that combining them
  hurts? The negative-transfer and catastrophic-forgetting literatures may
  be the closest.
- Is there prior work predicting the **outcome of a combination before
  training it**, from a cheap measurement of each source?

### 6. The honest verdict

After all of the above, answer directly:

1. Is this paper genuinely within Information Fusion's scope, or is the
   framing a wrapper? Give a percentage-confidence and your reasoning.
2. If it is in scope, what is the **minimum** set of citations and
   conceptual connections that would make the framing substantive rather
   than decorative? Name the specific papers.
3. If it is not, which venue fits the content best, and is there a
   different fusion-adjacent venue (Information Sciences, Knowledge-Based
   Systems, Pattern Recognition, IEEE TKDE) where the knowledge-versus-data
   framing would land more naturally?
4. Is there a **reframing** that makes the fusion connection real rather
   than cosmetic, for example presenting the work as an empirical
   grounding of the complementary-versus-redundant distinction, or as the
   missing "when" for the informed machine learning taxonomy?

## Output format

For each numbered section: what you found, with full citations and links;
what it means for my manuscript; and a clear statement of what I should
cite and what I should claim. Distinguish throughout between what you
verified in a primary source and what you inferred.

End with the ten citations I most need to add, ranked by how much each
does to answer the scope objection, and for each one a single sentence I
could actually write in the related-work section that connects it to my
result. If fewer than ten are genuinely relevant, give fewer and say so.
Do not pad the list to make the framing look better supported than it is.
