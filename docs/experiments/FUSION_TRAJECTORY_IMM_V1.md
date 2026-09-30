# IU/Joint trajectories with a supporting IMM - Step318

Question: can combining complete existing fusion trajectories and interpreting
their chronological sequence improve both localization tasks, with learned
feature fusion contributing beyond matched simple aggregation?
Freeze this document, implementation, tests and roster before scoring.
Read the scoped history audit first:
`docs/reviews/trajectory_fusion_history_audit_2026-09-07.md`.

## Fixed inputs and fitting scope

Same current110 development answers (24 PRMB/86 PB), corrected v3 labels/v2
source groups. Keep all149 previous comparison entries. No new inference or
feature/Joint refit. Read original full-grid moment/context routed IU, Joint
condition100 at lambda0/.1/permuted graph, and equal/real/permuted graph scores.
The original81/29 bank route, width8 grid, seeds and declared entropy anchor
remain. Every new parameter is fixed engineering or fitted within the current
answer. Full-answer fusion and noise estimation make this offline, even though
the IMM recurrence itself advances chronologically. No causal-online claim.

Five paired input families:
1. IU + Joint graph (primary).
2. IU + Joint0 (graph attribution).
3. IU + Joint permuted graph (graph alignment control).
4. Equal + equal graph (feature-fusion attribution).
5. Equal + equal permuted graph (strong simple control).

Each has four outputs: normalized pointwise mean, correlated-noise GLS,
GLS with the temporal tail-hold policy, and IMM of that GLS sufficient
statistic. Three single-input families (IU, Joint graph, equal) each have
tail-hold and IMM outputs. Add one temporal-order permutation of the primary
pair IMM, separate from graph-node permutation.27 new entries,176 total.
This is one supporting IMM experiment with controls, not a learned search
over27 independent models. No alpha/lambda/Q/budget sweep.

## Measurement combination and chronological model

Source risk curves are already centered with unit population SD on original
fit windows; verify this. Exclude the extra overlapping end window from any
noise fit or chronological state update. Collapse exact/affine duplicate
tracks (positive correlation >=1-1e-10) before GLS/noise/IMM; retain one track
with its original risk sign, never count duplicates as independent evidence.
An exact negative-affine duplicate fails as conflicting risk orientation;
do not choose one of two opposite risk interpretations arbitrarily.
Failure of a required source is a failure, not an excuse to drop that source.
Original Joint->IU fit fallback is inherited and disclosed separately.

For each retained input j use the existing heuristic
R_jj=clip((median(abs(diff(y_j)))/(.67448975*sqrt(2)))^2,.05,1).
Set R_ij from first-difference Pearson correlation times sqrt(R_ii R_jj),
clipped only to[-1,1]. If a difference coordinate is constant, its off-diagonal
correlation is declared zero. Regularize the symmetric R to condition<=100
with analytic diagonal ridge and a relative1e-10 eigenvalue floor. This R is
a measurement-noise heuristic, not identified correctness noise.

GLS weights w=R^-1*1/(1'R^-1*1), r=1/(1'R^-1*1). The GLS trajectory is
normalized to mean0/SD1 on original fit rows; rescale r by that same SD^2.
Use the existing scalar two-mode IMM on this sufficient statistic:
Q=(.01r,r), transition self-probability .95, initial means0/variances1 and
mode probabilities(.5,.5). Modes interact by mixing means and covariances.
The common-covariance vector observation model is mathematically equivalent;
verify it by a separate vector update in tests/review.

Static mean/GLS score the original overlapping end window normally. Hold
controls and IMM expand the last full-window value into that end window;
there is no second state update for an overlapping observation. Normalize
the IMM level on all original fit positions before readout, to avoid promoting
between-answer location/scale changes like Step317's. Source single-hold
controls copy original fit scores without an unnecessary renormalization.
The time-permutation control keeps original R fixed, permutes fit observations
using SHA256(original identity+'/imm-time-permutation'), runs the same IMM,
then unpermutes the levels before tail projection/readout. No target enters.

## Decisions and evidence

Same original overlap-mean window-to-token projection, official step max,
and one-vs-two Gaussian mixture BIC no-error rule. GMM sees original fit
positions of the final curve; use the same frozen GMM settings. Also save
predictions under the original moment-IU binary gate to separate peak/gate
effects. Failures remain failures; do not change a candidate on its no-error
decision. All old149 metrics and row payloads replay exactly.

Register contrasts before scoring: primary mean/GLS/hold/IMM vs original IU
and Joint graph, IMM vs hold and GLS vs mean for each pair, graph/zero/
permutation and matched simple-family controls, paired IMM vs single IMM,
single IMM vs own hold, temporal permutation, and strong risk equal-permuted
anchor comparisons. Keep all110 primary, with explicit both-native-Joint
scope as a diagnostic. PRMB common-valid pooled and within-answer AUROC;
PB four-cell macro harmonic clean/exact-error accuracy, failures penalized.
1000 source-group bootstrap draws, exploratory unadjusted intervals.

Save R, ridge, weights, retained/duplicate sources, original correlation,
GLS scale, IMM level/variance/mode posterior and time permutation. Report
native parent coverage, failures, exact error/clean outcomes gained/lost,
first-error peaks, gate changes, time-order control and available short-error
subgroups. Old58 scalar IMM results stay separate historical context.

Three CPU workers,600-second submission cap, per-answer atomic checkpoints
with bounded PermissionError retries. Tests cover vector/scalar equivalence,
duplicate collapse, source failure, covariance positivity, all original fit
indices, tail handling and permutation alignment. Review original anchors,
raw targets, R/GLS, direct vector IMM, normalization/projection/GMM, all176
metric bundles and registered paired scopes/points; explicit source bootstrap
replays and a visual HTML report. No old frozen source or result edits.
No success claim without both task evidence and fusion attribution. Untouched
confirmation, complete comparators and historical24 transfer remain open.
