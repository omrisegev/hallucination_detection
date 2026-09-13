# Direction 2: low-rank time/feature fusion, token scoring retained

Authorized continuation 2026-09-13 after hierarchical-family discussion. Base
2ec11dd0c; branch codex/rbm-two-axis-fusion-v1. Stop after this family.

## Question and scope

Can allowing feature reliability to vary with relative position inside a step
improve token fusion? The first family lost most performance when Top10 became
an all-token mean. Therefore every new arm here scores original tokens and uses
the same Top10 readout. No average of feature vectors replaces those tokens.

Use the existing answer-standardized 12-column bank of RBM12 (entropy15,
varentropy15, moment3, chosen-token surprisal/powers, moments/powers through 6).
Input columns, normalization, original RBM and reference scores remain frozen.
This is a new Gaussian factor-fusion fit on the RBM12 BANK, not a refitted RBM,
IU-PCR, Joint L-SML, or a paper-exact tensor regression network.

## Generative objective and rank

At relative-position region j: x_t = b_j h_t + epsilon_t,
h_t ~ N(0,1), epsilon_t ~ N(0,D), D diagonal and shared over positions.
The zero conditional mean is a modelling restriction: use second moments E[xx'],
not falsely centred covariances. Answer normalization is unchanged. The latent
factor is a shared feature signal; likelihood does not guarantee semantic risk.

The loading matrix B (16 positions x 12 features) factors as U V', rank R.
R=1: one feature direction with position-dependent amplitude. R=2: two separable
time/feature components, allowing the feature direction to vary with position.
R does NOT count hallucination classes or per-token latent units. This is a
low-rank loading-map adaptation of the proposed two-axis idea, not CP of step
means or a claim of equivalence to the earlier hierarchical model.

Optimize equal-position Gaussian marginal NLL:
mean_j 0.5 [logdet(b_j b_j'+D) + tr(inv(b_j b_j'+D) C_j)]
+ 0.5e-4 mean(B**2). Fixed numerical ridge, no lambda sweep. L-BFGS-B, float64,
maxiter1000, ftol1e-10, gtol1e-6. Two deterministic initializations: pooled leading
direction (rank2 adds a small second-direction slope), and regional leading
directions projected to rank R by SVD. SVD initializes; it is not IU-PCR.
Floor D=max(1e-8,1e-3*mean diagonal second moment). Save both starts, selected
NLL, convergence, projected gradients, residual misfit and coefficient spectrum.
Select by NLL only. Finite capped fits remain scored and flagged. Numerical
failure remains NaN and counted; no fallback to RBM or IU.

Orient each b_j towards its varentropy15 loading; if magnitude <=1e-12 use its
first nonzero loading, disclose ties. This fixed risk anchor is an assumption,
not learned semantic direction. Row signs preserve matrix rank. Score tokens by
the Gaussian posterior mean: c_j'x_t with c_j = D^-1 b_j/(1+b_j'D^-1 b_j).
No sigmoid, coefficient-sum rescaling or post-hoc sign from labels.

Sixteen equal relative-position regions use token/region interval overlap.
For fitting, average TOKEN OUTER PRODUCTS within each region, equal weight per
step, answer within canonical source group, then source group within cell.
For scoring, a token spanning regions uses the overlap-weighted coefficients.
Short steps need no zero padding. Tokens remain the units scored before Top10.

## Frozen roster and controls

- Original RBM12 Logit + Top10, answer-local weights, external calibration.
- Stationary shared factor fusion: B rows identical, same noise/objective/readout.
- Rank1 shared time-feature fusion.
- Rank2 shared time-feature fusion.
- Rank2 with deterministic joint permutation of token feature vectors within each
  step, in both training and scoring (not independent feature-column shuffling).
- All 13 frozen matched reference rows from the prior logit-readout experiment;
  additional historical leaders remain context through the completed ledger.

Primary contrasts: rank1 minus stationary; rank2 minus rank1. Rank2 minus
shuffled and all new arms minus RBM12 are descriptive. 10,000 paired canonical
source-group bootstrap draws; 97.5% primary CIs, 95% others. No success threshold.

## Access, metrics, validation

All 13,769 development answers, v3 labels, v2 source folds, PB8 cells/6800 answers,
PRMB1 cell/6969 answers. New fusion learns from other answers without labels.
No local parameter adaptation except the inherited within-answer normalization.
Each held f excluded from shared training; source groups never cross folds.
Same entropy gate q0.3 and argmax earliest tie; NO first_near_max.
PRMScore q0.8 uses inner h scores from models excluding f and h. Save excluded
groups, thresholds, coverage. PB macro/Q4/Q8/percell, PRMB within+pooled+fold AUC,
PRMScore, exact/early/late, failures, coefficients and uncertainty all reported.

Independent metric reconstruction, original RBM/Top10 replay <=1e-12; numerical
gradient and direct Gaussian inverse checks; rank1 stationary equivalence;
known rank2 synthetic data; short/constant/missing cases; group exclusion audit;
held-label mutation cannot change any fit inputs (fit API has no labels).
Smoke27 is numerical/runtime feasibility only, then full population. One process,
BLAS1, caches loaded one cell at a time after 4GiB RAM check; checkpoints and
8h invocation cap. No raw cache copies, no edits to Claude's runs.

## Interpretation boundary

This learns relative-position-dependent fusion, not token-context dynamics or
cross-step adjacency. Families 3 and 4 remain separate. A rank2 NLL improvement
without task gain is not a localization improvement. Lower scores than RBM do
not by themselves isolate the effect of pooling or changing the latent family;
the stationary/rank1/rank2 contrasts do isolate the registered loading structure.

Inspiration, not claimed reproduction:
- Gaussian factor observations: https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FactorAnalysis.html
- Low-rank parameter organization: Tensor Regression Networks,
  https://arxiv.org/abs/1707.08308 (supervised source; our objective is unlabeled).

Return results in chat and update HISTORY/PROGRESS/Research_Directions. Do not
start Conditional RBM, temporal convolution or another feature-bank experiment.

## Pre-run independent audit amendment — 2026-09-13

An independent plan-to-code audit stopped the first smoke while it was still
waiting for RAM (zero statistics, models and scores). The corrected runner now:

- records pure Gaussian NLL and the ridge penalty separately and selects the
  registered initialization by pure NLL;
- retains a health record for both deterministic starts, including a failed one;
- binds the exact ordered 12-feature roster, its producer and extraction code,
  and verifies inactive cache columns are zero before scoring;
- runs an actual held-fold label/target perturbation firewall check and saves
  explicit included and excluded source groups;
- adds grouped paired uncertainty for pooled PRMB AUC, fold-mean PRMB AUC and
  PRMScore, and independently reconstructs the registered per-cell and
  diagnostic endpoints; and
- tests fitted stationary/rank1 equivalence on position-identical moments.

The pre-audit smoke artifacts are retained under a separately named directory.
They contain only unit-test and manifest records and are not scientific results.
