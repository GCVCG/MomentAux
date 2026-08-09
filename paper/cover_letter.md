# Cover letter

**To:** The Editors, *Information Fusion*

**Manuscript:** When does fusing hand-crafted knowledge with learned
representations pay? A controlled, cost-normalized benchmark and a
measurable rule for stacking, substitution and interference

**Authors:** Ahmad AlMughrabi, Ricardo Marques, Petia Radeva

---

Dear Editors,

We submit the manuscript above for consideration as a research article in
*Information Fusion*.

**The question.** Fusing prior knowledge with data-driven learning is most
attractive where data is scarce, and yet there is no controlled account of
*when* such fusion pays, when it is merely redundant, and when it actively
harms. Dasarathy's framing of the field as what, where, why, when and how
to fuse notes that the *when* has had least attention, because unlike the
others it cannot be read off the architecture. That is the question this
paper takes up, in the setting where one source is hand-crafted and the
other is learned.

**What we contribute.** A controlled, cost-normalized benchmark of
data-efficiency interventions, and the rule that organizes its results. One
frozen training recipe with committed subset indices measures a fixed
hand-crafted source of knowledge against the data-driven alternatives
(SimCLR, SimSiam and DINO pre-training, ImageNet transfer, heavy
augmentation, learned teachers) across 13 datasets, 9 backbones, 500 to
1.28M images, 32 to 224 px and 5.7M to 86M parameters. Every pairwise
combination is measured too, and they follow one rule: sources carrying
different *currencies* stack, same-currency sources substitute, and fusing
into an already-informed initialization interferes in proportion to what
that initialization was contributing. Because each source's currency is
measurable on frozen features before the combination is built, the outcome
is predictable in advance, which we state as an explicit procedure.

**Why this journal.** The outcomes we measure are not new categories, and
we say so before claiming anything: the complementary, redundant and
cooperative triple has been in this field since Durrant-Whyte, and Smirnov
and Levashova's survey of knowledge-fusion patterns catalogs the same three
outcomes. What we add is operational. The classical taxonomy states what
sources *are*, given known sensing geometry; it offers no way to determine
which case holds when the sources are a hand-crafted prior and a learned
representation, whose relationship is not evident a priori and, as our
results show, not predictable from their family names either — the same
prior is redundant with effective self-supervision and complementary with
augmentation. A cheap frozen-feature measurement decides it in advance.

We also test the rule beyond the setting it was derived in. It holds where
the two sources are two Sentinel-2 band sets, where they are Sentinel-1
radar and Sentinel-2 optical, and where they are two trained classifiers
fused at the decision level.

**What we found, including against ourselves.** Two results of the paper
are negative and we consider them among its more useful contributions.
First, a second sensor does not always pay: splitting one instrument's
bands is complementary at every data scale, whereas pairing two satellites
never is, and the accuracy asymmetry between the sources predicts which
happens. Had we run only the first population we would have reported the
opposite. Second, when we gave the self-supervised comparators four times
their pre-training budget across the whole data envelope, contrastive
pre-training beat our prior at every convolutional cell. We withdrew the
corresponding accuracy claim and restated the convolutional case as one of
cost rather than accuracy.

The manuscript is explicit about these reversals rather than quiet about
them, and about three further limitations a reader would otherwise have to
discover: configuration selection used the test split, comparator tuning
parity is asymmetric by construction under cost normalization, and the
central decomposition's sign is informative only where the term it predicts
has not decayed to nothing. Predictions and falsifiers for each
experimental wave were recorded before results existed, and outcomes are
reported against them including the misses.

**Reproducibility.** The complete harness is released: configurations for
every cell, the committed subset indices, the fingerprinted filter banks, a
single training entry point, and the aggregation and audit scripts. Every
table and figure regenerates by command, and the central audit is itself a
released script rather than a manual query. The tagged release carries each
run's record, every linear-probe result, per-epoch curves and checksums.

**Declarations.** The work is original, is not under consideration
elsewhere, and all authors approve this submission. We declare no competing
interests. Use of generative AI in preparing the manuscript is disclosed in
the declaration required by Elsevier policy. The datasets are public
benchmarks; PathMNIST is a de-identified public benchmark requiring no
ethics approval.

We hope the manuscript is of interest to the journal's readership and look
forward to the reviewers' comments.

Yours sincerely,

Ahmad AlMughrabi, on behalf of the authors
Universitat de Barcelona — ahmad.almughrabi@ub.edu
