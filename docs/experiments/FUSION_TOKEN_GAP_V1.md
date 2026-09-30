# Provided-token preference gap inside our fusion — Step 315

Question: does expressing realized-token confidence relative to the model's
preferred token improve IU-PCR / Joint L-SML under the existing answer-local
window architecture? This is one representation change, not a new detector.

## Source/history audit and interpretation

The old nine-stream banks contain `spilled_series = -log p(provided token)`
and `top1_logprob_series = log p(most probable token)`. Their difference in
surprisal is `gap = spilled + top1_logprob = log(pmax / pprovided)`. It is
nonnegative, and zero when a provided token has maximum probability. A large
gap means model disagreement with that token, not a verified reasoning error.
The log normalizer cancels, so this equals the maximum-logit minus provided-
token-logit gap when both channels use the same distribution.

Audited sources: `cluster/backfill_views.py::candidate_quantities` computes
raw top-K log-softmax and provided-token negative log probability; the PB and
PRMB teacher-forced writers call it with `warpers=None` and save raw top-K.
`token_feature_views._logprob_token_series` takes the first retained logprob
as top1. Older generated/warped caches must not silently use this formula with
raw top1 and post-warper surprisal. Current entropy is top15 renormalized
entropy, not full-vocabulary entropy; top1/provided probabilities here use
the full unwarped normalization.

The local search inspected spectral_utils, docs/experiments and HISTORY for
regret/excess-surprisal/likelihood-gap/token-disagreement and source expressions
combining spilled/top1; the readonly Claude spectral_utils search was also
checked. It found the separate channels and ordinary prediction innovations,
but no direct maintained experiment for this reparameterization in that scope.
This is not a claim of literature novelty or a complete historical search.
`paper_exact/telemetry.py` also records separate provided-token and pmax data.

Target-free feasibility on the frozen110 inputs: all71,385 token gaps are
finite/nonnegative, min0, max42.34375;84.98% are within1e-5 of zero. No answer
has a constant gap stream. These are support properties, not accuracy results.
This view uses existing information; it is not an independent new observation.
Replacing the window SD also introduces cross-stream covariance into that
coordinate, so this is not merely a linear rotation of the old27 window values.

## Fixed quality experiment

- Same110 development answers, v3 PRMB labels and v2 source groups/folds.
  All98 corrected prior methods remain exact external anchors. No new inference.
- Original bank route stays81 moment /29 context. Replace only the three
  spilled coordinates with gap {mean,SD,slope} or {mean,EMA8,EMA32}. Keep
  P=27 and the original8-token fitting/scoring windows; no extra rows.
- Keep entropy orientation anchor, normalization, duplicate removal,
  grouping K={3,4,6,8}, minimum group size3, four chronological stability
  blocks, held admissibility.95, five starts/5000 sweeps and original seeds.
  Use the audited generic Step312 fit_bank; condition100 and graph lambda.1.
  Preserve the exact original permutation identity: scoring_namespace +
  '/' + cell + '/' + row_id + '/moments27_local8'.
- Seven fusion cores: equal, IU, Joint0, Joint graph, permuted Joint graph,
  equal graph, permuted equal graph. Every core uses the same new bank.
  Only an invalid Joint MODEL fit falls back to gap-bank IU. Numeric/graph
  or readout failure is retained, never a trigger for a different method.
- Two explicit scalar controls: gap mean and original surprisal mean. Fit
  each scale inside the answer; keep its declared larger-is-risk sign. They
  are diagnostics for attribution, not replacements presented as our method.
- Native GMM/step-max/peak readout is unchanged. Keep the original moment-IU
  gate as a separate diagnostic. Preserve pure fit failures and fallback counts.
- Freeze nine new outputs before this evaluator reads v3 targets. Existing
  data exposure is disclosed: this is adaptive development, not confirmation.
- Compare seven new cores with their exact original corresponding cores;
  compare learned fusion/equal and graph/zero/permuted/equal-graph; include
  scalar gap-versus-surprisal and fused-versus-gap controls. No label-selected
  winner or tuning grid. Paired1000-draw source-group intervals are exploratory.

## Review and gates

Meaningful tests verify the log-normalizer cancellation, greedy zero gap,
invalid inputs, unchanged feature count and untouched24 coordinates, and
that readout failures cannot invoke a different fusion fallback. Independently
rebuild all matrices, scores/step maps, native decisions and v3 metrics.
Verify top-K/provided-token consistency from original trusted cache arrays
on the selected110 rows, with support/censoring explicitly reported. Compare
corrected labels to raw annotations through the existing official port.
Reuse disclosed scientific kernels for representative fitting/graph replays;
use independent covariance/weight and metric algebra where applicable.

CPU only; three workers,1200s scoring cap, incremental checkpoints. No edits
to frozen sources/old releases or the readonly Claude worktree. A failed
variant does not close Joint/graph, IMM/LOCA/Flows/KalmanNet, sampling or the
full program. Corrected historical refits/comparators, untouched confirmation
and historical24 transfer remain open.
