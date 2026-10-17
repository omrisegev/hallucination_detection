# Cumulative-vote fusion: clarifications from the 2026-09-20/21 discussion with Omri

Companion to `docs/experiments/CUMULATIVE_VOTE_FUSION_V1.md` (Step 423) and
`docs/experiments/RAW_CHANNEL_READOUT_FUSION_V1.md` (Step 424). Nothing here changes a number;
it records what was clarified so the next session does not re-derive it.

## 1. In our L-SML the eigenvector is the answer, not an initialization

Checked in `spectral_utils/fusion_utils.py`:

- `sml_fuse_signed` returns the leading eigenvector v of the off-diagonal covariance and the
  score `X @ v`. The weights *are* v. No step follows.
- `lsml_fuse` (binary) and `lsml_continuous` run `detect_dependent_groups`, then
  `sml_fuse_signed` inside each group and across the group-level virtual classifiers. Two layers
  of eigenvectors, then a weighted vote. This is Jaffe-Fetaya-Nadler Algorithm 2 as deployed.
- The only iteration in the module is `_rank1_masked` (up to 100 alternating steps): a numerical
  rank-one solver with masked entries, not an EM over latent labels.
- No ψ / η (sensitivity / specificity) are estimated anywhere in `spectral_utils`, and no
  likelihood readout exists there. The only place in the repository with a ψ/η likelihood
  (Eq. 18 of the L-SML paper) is the October-2025 framework `hallucination_detection/core/lsml.py`
  on `main`, which nothing in the current pipeline uses.

Dawid & Skene (1979) is the classic EM for the same latent-class model: E-step posterior per
instance, M-step re-estimate ψ_j, η_j and the prior; converges to a local optimum and depends on
initialization. Parisi-Strino-Nadler-Kluger (2014) proposed SML as the spectral estimator for
that model and suggested initializing the EM from it. The "Dawid-Skene" rows in Steps 423/424 do
exactly that: SML initialization, then EM. The EM lives in `scripts/experiments/
cumulative_vote_fusion_v1.py::fit_dawid_skene` (not yet in `spectral_utils`).

What differs, in order of importance: DS estimates two numbers per classifier (ψ and η) where
SML estimates one (balanced accuracy); DS combines by likelihood ratios (asymmetric between a
"+1" and a "−1" vote of the same classifier) where SML combines linearly; SML is closed-form.
The case where they should differ is a classifier with an asymmetric error profile: a rarely
firing but precise onset/delta detector, or a systematically early/late localizer (in Step 423,
Unified-28 had η = 0.29 and Mind-the-Gap ψ = 0.26). On Step 423's five level localizers the two
gave the same answer (30.08 vs 29.76 SLA).

## 2. What the matrix is, and what the weights are between

Example: 30 answers, 10 features.

- Step 0 (token → step): each feature's token series is read out per official step (top5,
  onset80, …) and its argmax is ŝ_j. After this the token axis is gone; each answer is 10 step
  numbers.
- Step 1 (matrix): instances are (answer, n) pairs over the answer's disagreement region
  [min ŝ, max ŝ − 1]; columns are the 10 features; entry +1 if ŝ_j ≤ n else −1. About 3 rows per
  answer, so roughly 90 × 10 in {−1, +1}. The algorithm does not know which rows share an answer.
- Step 2 (fit): SML / L-SML / DS learn **one weight (or ψ, η) per feature**. The weighting is
  between features; nothing answer- or step-specific is learned.
- Step 3 (inference on a new answer): input is only the 10 ŝ_j and the 10 weights. For every n,
  F(n) = Σ_j w_j 1[ŝ_j ≤ n] (or the DS posterior). F rises from 0 to 1. The jump
  F(n) − F(n−1) is the fused mass on step n, equal to the total weight of the features whose
  ŝ_j = n. Prediction = argmax of the jumps (mode) or the first n with F(n) ≥ 0.5 (median). With
  equal weights the median readout is exactly the median of the ŝ_j.
- Soft variant: same rows and columns, but a feature "switches on" gradually according to its
  softmax-normalized step profile (entry 2F_j(n) − 1 ∈ [−1, 1]); the fit runs `sml_fuse_signed` /
  `lsml_continuous` on the continuous columns; steps 3's arithmetic is unchanged. As τ → 0 it
  reproduces the binary path (checked per fold). Soft DS was not implemented; standard DS needs
  ±1 votes and a soft-label extension is only worth building if binary DS beats binary SML on the
  full cells.

This is the reverse of Step 422, where the between-feature weights were learned on token
instances and a single Top-10 readout came after the fusion.

## 3. PRMBench: same fusion, different encoding

ProcessBench has one label per answer (the first erroneous step), so "is the error at a step
≤ n?" is the natural ordinal decomposition and the fused pmf sums to one. PRMBench labels every
step, several steps per answer may be erroneous (nine classes, ≈16% prevalence), and the metric
is a ranking of all steps (pooled step AUROC/AUPRC, plus within-answer AUROC, separate panel,
v3 labels via `spectral_utils.prm_label_contract.prm_error_flags`). The cumulative encoding is
therefore wrong for PRMBench: it assumes a single change point and would penalize a second error.

Planned `--task prmbench` mode of `raw_channel_readout_fusion_v1.py` (not implemented yet):

- Question per step: "is step s erroneous?". Instances = (answer, step) for **all** steps.
- Binary path: each feature's per-step readout profile is binarized within the answer by a
  label-free rule matched to prevalence — top-k steps per answer (k = 1..3) rather than a median
  threshold, which would set the positive rate to 50%. Same `sml_fuse_signed` / `lsml_fuse` /
  DS; the fused per-step score (weighted vote or DS posterior) is directly the ranking score. No
  PAVA, no mode/median.
- Soft path: the within-answer z-scored step profile itself is the column; `lsml_continuous`
  gives the weights; fused score = Σ_j w_j z_j(s). No softmax/cumsum needed.
- Carried over: the per-feature readout grid (selection criterion becomes per-step AUROC on
  training folds), answer-local standardization, source-group folds, DS for asymmetric features.
  The "late errors" question becomes AUROC by step position, since PRMBench errors sit at every
  position.
- Data: PRMBench × Qwen3-8B telemetry (6,969 answers) has the same rich-save schema (Step 421).
