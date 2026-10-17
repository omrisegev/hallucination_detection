# Two requested RBM checks: distribution m3 ablation and raw degree-three powers

Authorized2026-09-11. Separate codex/rbm-m3-powers-v1 from47eb162f.
Scope is full matched LOCALIZATION13769 answers/145597 steps. No new global24
experiment, B3, higher degrees, fitting scopes, windows or hyperparameter grid.

## Test1: does the distribution third moment contribute?

Original6 columns=[H15,V15,m3_15,a,a^2,a^3]. Remove ONLY index2 to obtain
five columns=[H15,V15,a,a^2,a^3]. Selected-token third power remains present.
Use the same answer-local normalization and exact one-hidden-unit Gaussian RBM,
unit visible variance,L-BFGS-B maxiter100,ftol1e-10,gtol1e-6,maxls40,
initial a=0,b=0,w=2/P. Finite nonconverged fits retained/flagged, no fallback.

A feature removal must not silently change the latent label direction. Fit
both banks, orient BOTH against mean standardized surviving FIVE columns.
This common anchor has no m3 dependence. Also score the six-column model with
its ORIGINAL mean6 anchor as a bridge and require exact old step-score replay
for EVERY answer. Common-anchor six-column scores reuse the same fit/state;
only the declared direction can differ. This isolates the removal within the
fixed RBM recipe; it does not establish that m3 can never help another model.

## Test2: RBM on Y,Y^2,Y^3 at each rank

Reuse the exact old surprisal_power_fusion.representation(degree=3):
raw -log(p1..p15) plus selected surprisal, each at degrees1,2,3 -> T x48.
These are unweighted powers of original probabilities, not q-weighted moments
or Varentropy contributions. The selected-token columns are15,31,47 (zero-based).
Same exact RBM/defaults. Orient latent class against saved token entropy, the
same direction anchor as prior raw-power IU-PCR. Initial RBM receives the
same input/anchor with no optimizer steps. Replay all six old power mean/IU
metrics, particularly degree3 IU and mean, alongside the new RBM.

## Roster and comparisons

Five scored outputs: original6 replay,6 common-anchor,5 reduced,48 trained,
48 initial. Two primary paired contrasts:6 common-anchor minus5 (m3 value),
48 trained minus frozen48 IU (fusion comparison).10000 canonical source-group
bootstrap draws,97.5% intervals. Initial/bridge and Varentropy comparisons
exploratory95%. No threshold for promotion, no winner from feasibility runs.
All prior nine Varentropy/entropy/IU references, four current moment controls
and six raw-power references remain visible with frozen metrics replay.

Same label release,canonical source folds,spans,top10 mean/earliest tie,
saved dual__iu mean entropy q=.3 gate and held-group q=.8 PRMScore calibration.
Fitting uses current answer only; gate/calibration have declared external
access. Report PB8, Q4/Q8, clean/exact/early/late, PRMB pooled/within AUC,
PRMScore, coverage, convergence, parameters and direction changes. These are
development results and conditional fixed-fit intervals, not confirmation.

Preflight27 short/median/95th-length rows across9cells, then full execution.
Checkpoint WAL and bounded retries on Windows JSON replacement. Reuse frozen
model implementation; do not edit active-run files. Separate arithmetic metric
verifier automatically follows full evaluation. No new HTML before chat.
