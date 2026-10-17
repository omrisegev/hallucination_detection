# Per-channel evidence with a position-conditional null — 2026-09-22 (Step 432)

Question: does a non-linear per-channel transform, learned without labels from the bank's
own fusion, change the decision among competing steps on the frozen eleven-channel bank; and
does conditioning the null distribution on the step's position in the answer (the label-free
position prior measured in Step 430) help or hurt?

Motivation. The argmax is invariant to any monotone transform of a single fused score, so a
per-step "how unusual" statistic of the fused score cannot move a decision. Two things can:
a non-linear transform applied per channel BEFORE summation, and a position-dependent null.
Step 430 measured the drift: on answers the frozen gate closes, every fused locator and most
channels start about +0.7 SD above the answer mean at step 0, fall through zero by step 2 and
sit near -0.3 SD afterwards; early misses of the fused locators land where that null is
higher (+0.48 SD) and late misses where it is lower (-0.46 SD).

Authorized bounded stage on branch `claude/readout-quickest-detection-v1`. No acceptance
threshold, no promotion, no new inference. The frozen readout-family profiles
(`results/readout_family_v1/profiles_full.npy`, sha asserted) are linked, not rebuilt.
Population, labels v3, source folds v2, frozen CT7 gate as in Steps 428-429.

## Method (declared before running)

Per outer fold, training answers only:

1. **Seed** (label-free, fit-free, answer-local): equal-weight mean over channels of the
   per-channel softmax over steps of the answer-standardised readout. Pseudo-positive step =
   earliest-tie argmax of the seed mass on training answers the frozen gate OPENS; every step
   of gate-closed training answers is negative (the closed population is the clean reference).
2. **Tables**: per channel, f1 (pseudo-positive steps) and f0 (all other steps) as smoothed
   histograms (alpha 0.5) on 32 pooled-quantile edges from the training steps; the
   position-conditional null estimates f0 per relative-position bin (8 bins) shrunk towards the
   pooled f0 with pseudo-count 32. f1 is pooled over positions.
3. **Evidence** of a test step = sum over channels of log f1_c / f0_c (plain) or
   log f1_c / f0_c(position) (conditional). Location = earliest-tie argmax; gate frozen.
   PRMBench per-step score = the same evidence; thresholds from training-fold scores as in
   cvf_v2 (inner-fold PRMScore selection unchanged).
4. **Iteration 2**: pseudo-labels from the plain evidence argmax on the same gate-open
   answers, tables refitted once. No further rounds.
5. **Fusion ablation**: cvf_v2 equal / spectral / continuous L-SML weights fitted on the
   evidence columns (rows = training steps, cell-balanced weights) versus the raw sum.

Readouts: top5 (roster `evidence`, primary) and top30 (`evidence30`, sensitivity).

Arms (roster__all__enc__kind): seed__equal; plain__{equal, equal_std, spectral,
continuous_lsml}; position__{same four}; plain2__equal; position2__equal;
randomseed__equal and randomseed_position__equal (one uniformly random pseudo-positive per
gate-open training answer: the null of the whole construction); prioronly__equal (the
position-conditional null alone with a flat f1: the prior without the per-channel
transform); ceiling__equal and ceiling_position__equal (tables from TRUE first-error labels
of the training answers: a label-selected ceiling, never unsupervised).

Guards and planned contrasts: Step 199 guard = agreement between the seed argmax and each
arm's argmax on test answers (recorded per job; an arm agreeing on more than 95 % of answers
has no algorithmic content); position minus plain; iteration 2 minus 1; evidence minus seed;
primary minus random-seed null; ceiling minus primary; prior-alone minus plain and position
minus prior-alone; learned minus equal on the evidence columns; new arms versus CT7,
Mind-the-Gap, token L-SML and the pmf equal reference; top30 minus top5. Endpoints and
uncertainty as in Step 429 (gate-free SLA macro8, common-gate F1, PRMBench within-answer
AUROC; 10,000 shared source-question draws, one Holm family); depth strata and late
fraction reported for every arm. Anchor parity against `results/readout_family_v1` on every
shared reference / control row.

Out of scope: retuning bins, alpha, pseudo-count or the number of iterations; CT7 as a seed;
any claim of confirmation (development data).

## Amendment A1 (2026-09-23), PRMBench pseudo-positives

The first full run showed that the frozen CT7 gate opens on **no** PRMBench answer (0 of
6,969; it was calibrated on ProcessBench). Under the declared rule the PRMBench tables were
fitted with zero pseudo-positives, so every pseudo-label arm collapsed to the same rarity
score (plain, iteration 2 and the random-seed null were identical on PRMBench: within-answer
AUROC 0.5125 at top5). The ProcessBench panel is unaffected (4,556 gate-open answers).
Amendment, declared before the rerun and applied to PRMBench only: the pseudo-positive step
is the seed argmax of **every** PRMBench training answer; there is no gate-closed reference
population there, so the "closed answers are all negative" element is absent on PRMBench and
the plain null is the marginal over all training steps. The PRMBench jobs (25 outer + 40
inner) were deleted and rerun under the amended driver; the ProcessBench jobs were kept and
one was replayed under the amended driver to confirm the ProcessBench path is byte-identical.
