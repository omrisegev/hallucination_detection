# Fusion window sampling pilot v1 — registered 2026-09-07

Question: can selecting fitting observations improve our existing answer-local
feature fusion? This is a supporting component, not a replacement detector.
Freeze this document, implementation and tests before scoring. This adaptive
development stage reuses all 58 IDs from representation pilot v1, release
`localization-cached-v1-20260907`. Labels were inspected in earlier stages;
score freezing here does not create an untouched test. No new inference.

## Fixed representation and budget

Use the parent's 27 moment coordinates, nonoverlapping width-8 fitting
windows, and scoring-only end window. For N original fitting rows select
M = min(N, max(32, ceil(N/2))). The floor is an engineering choice, not an
optimality claim or a guarantee of covariance identifiability. At N<=32 reuse
all parent results exactly. Selection is unsupervised within the current
answer; selected rows are sorted chronologically before fitting. No pooling.

Selectors (same selected rows for every fusion core):

1. `full`: all rows, exact parent score replay with the established peak locator.
2. `uniform`: M rounded equally spaced row indices, including endpoints.
3. `risk_top`: M highest raw entropy-level windows; chronological tie breaking.
4. `dufs_transposed`: the existing parameter-free DUFS adaptation, with windows
   as gated coordinates. First standardize each moment column across this
   answer's original fitting rows; remove constants and exact affine duplicates.
   Center each resulting window vector across coordinates and normalize its L2
   norm. Pass the resulting N-by-P array directly to `adapted_dufs_soft_gates`,
   whose interface is gated-features-by-observations. It returns N window gates
   and builds a graph on P feature coordinates. Seeds 0/1/2, 120 epochs, other
   existing defaults. Rank mean survival probabilities, retain top M.
5. `dufs_permuted`: permute those N probabilities over chronological positions
   with a SHA256(answer identity + '/sampling-permutation') seed; same budget.
6. `window_diffusion`: our geometric sampling adaptation. Use standardized,
   deduplicated moment rows as N graph nodes. Symmetric self-tuning k=7 nearest
   neighbor affinity, zero diagonal, no chronological edges. Form row-stochastic
   W and embedding W^2 / sqrt(stationary degree fraction). Select the medoid
   (minimum total squared diffusion distance), then greedily add the point
   farthest from its nearest selected point; chronological tie breaking.
   This is representative coverage of graph geometry, not learned correctness
   or a reproduction of DUFS or Shlezinger's graph compression algorithm.

Use all original rows to build selectors; fusion normalization, orientation,
groups, covariance and weights use only selected rows. Keep the explicit
negative-entropy sign anchor. Reuse unchanged `fit_local` kernels for equal,
IU, Joint lambda-zero, Joint graph lambda 0.1 and its node-permuted control.
No extra hyperparameter search. Four chronological validation blocks are now
quartiles of selected-row rank; report their actual time coverage separately.
The feature graph regularizer and the observation selector are distinct.

## Dense scoring and decisions

Score all original windows, including the end window, with the selected-fit
weights and normalization. Map windows to tokens by overlap mean, official
steps by maximum. Thus this stage reduces fitting rows; it does NOT reduce
feature extraction or constitute a sparse end-to-end detector.

Apply the same parent one-vs-two Gaussian mixture BIC rule to risk on ALL
original nonoverlapping windows (never just the selected high-risk rows).
If its original threshold readout says error, predict the maximum-risk step;
otherwise predict -1. Gate decisions may change after refitting, but the rule
and its data support do not. Log a secondary fixed-parent-gate prediction to
separate changes of location from changes of gate. Missing/invalid fits or
mixture convergence are failures, never silently replaced by the parent.
The full arm and all non-reducing selectors replay parent scores/decisions.

## Diagnostics and review

Save indices, selector probabilities, per-seed top-M Jaccard, selected time
quartile counts, largest token gap and all fusion diagnostics. For two fixed
SHA-seeded perturbations, resample four nonoverlapping two-token blocks
within each width-8 fitting window, compute its moment features again, and
rerun selection at the same budget. Report Jaccard stability; do not use it
to choose parameters. Copies do not add independent fitting observations
and do not establish LOCA's isotropic burst assumption. The permutation
control reuses perturbed DUFS probabilities under the same permutation.

After score freezing, measure first-error-step overlap with selected fitting
windows, including the <=32-token first-error subgroup. This diagnostic is
potential boundary support, not achieved recall for a sparse detector.
Report total and per-selector runtime including perturbation overhead.

Reuse parent's exact metrics: PRMB pooled step AUROC and available-answer
coverage; PB macro of four subset harmonic error/clean exact accuracies,
failures penalized on both classes. Paired PRMB uses common valid IDs; PB
retains every fixed ID. Bootstrap source groups 1,000 times with the frozen
parent evaluator. No significance claim based on point estimates alone.

Registered contrasts: each reduced selector vs full for every core; transposed
DUFS vs uniform/risk/permuted and window diffusion vs uniform for IU and Joint
graph; IU vs equal, Joint graph vs IU/lambda-zero/permuted for every selector.
Keep primary all-58 results plus descriptive sampling-eligible subset; the
subset must not replace the primary cohort. Show previous matched peak
scores and the earlier 30-long-answer pilot in a separately qualified panel.

Tests and independent review must check selector axes, budgets, determinism,
no duplicate indices, no fitting dependence on unselected values once indices
are fixed, exact replay, mapping, label joins, metrics, hashes and failure
accounting. Three CPU workers maximum, per-answer resumable checkpoints.
No edits to parent frozen code/protocol/results and no interaction with
Claude's worktree. No winner or transfer to 24 historical cells is licensed
by this development pilot alone.
