# Continuous moment fusion: exact Gaussian RBM and B3

Authorized 2026-09-11 after explicit discussion of DEEM input semantics.
Parent d92f88f8, isolated codex/moment-rbm-fusion-v1. No native DEEM in this run.

## One fixed input bank

All T tokens in the same answer, six columns [H15,V15,m3_15,a,a^2,a^3].
H is mean surprisal over normalized Top15, V its central second moment,
m3 the raw third moment, a the original selected-token surprisal. Preserve
the existing q=p/(sum(p)+1e-12), y=-log(q+1e-12) numerical convention.
MGF identities describe the ideal distribution; these epsilons are the
frozen numerical approximation. V15 exactly replays the old contribution sum.
No tail, new ranks, windows, graphs, threshold sweeps or extra input banks.
Powers are dependent derived coordinates, not independent classifiers.
Z-score within the current answer; remove std<=1e-10 columns. Require >=3
rows and >=3 varying columns. No sampling or donor-answer fitting.

## Fixed methods and training

Four candidates: standardized mean, canonical IU-PCR (IU_FIT_DEFAULTS),
Gaussian-Bernoulli RBM with one hidden unit, continuous B3 with one group
containing all active moment coordinates. Two diagnostic controls: each
energy model before fitting, with its identical initialization.

RBM: E(x,h)=.5||x-a||^2-h(b+x.w), fixed identity visible variance.
Its partition function is analytic for one hidden unit. Optimize the exact
mean negative log likelihood using L-BFGS-B, maxiter100, ftol1e-10, gtol1e-6,
maxls40, float64. Initialize a=0,b=0,w=2/P. No CD approximation, restarts or
hyperparameter search. Store convergence, gradient, objective and parameters.
Finite nonconverged fits remain counted and flagged; nonfinite/increased
objective fails explicitly. No fallback or post-hoc selection.

B3: existing GenericEnergy/fit_generic, one group 'moments'. Default100 epochs,
lr1e-3, momentum0, width8, MALA5, delta.1, replay refresh.05, float64 CPU.
One fixed UID-derived seed, no seed search. Do not call B3 a canonical RBM:
it has a nonlinear logit network in its continuous energy function.

Common label-free orientation: correlate scores with mean standardized
input; complement posterior when negative, negate linear score when negative.
Constant/ambiguous scores retain orientation and are flagged. This is an
explicit high-risk prior. Latent classes are NOT identified as hallucination
by unsupervised training alone. Finite collapsed scores remain visible.
RBM and B3 use posterior scores; mean and IU use linear scores. Thus their
top10 readouts include a link-function difference. Initial-model controls
isolate learning from architecture/link effects. No added score tuning.

## Evaluation contract

Full cached localization development: 13769 model-answer rows,145597 steps,
PB8 and PRMBench. Frozen v3 labels, canonical source groups/folds, token spans,
top10 mean, earliest tie. Same saved dual__iu entropy q=.3 gate and held-group
q=.8 PRMScore calibration. No full-answer24 experiment in this run.
Nine existing Varentropy/IU/entropy references must reproduce old metrics.
Headline paired comparisons: RBM-IU and B3-RBM,10000 canonical-source bootstrap
draws,97.5% intervals. All other comparisons exploratory95%; no promotion
threshold or winner claim on a subset. Include all PB cells, Q4/Q8, clean/error
accuracy, early/late errors, within/pooled AUC, PRMScore, failures and coverage.
Preflight: shortest/median/95th-length row in each9cells; mechanics/runtime only.
Store states/normalization per answer. Full driver invokes the separate
arithmetic metric verifier automatically after evaluation. No HTML before chat.
