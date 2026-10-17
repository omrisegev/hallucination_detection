# Context-weighted original-feature fusion — frozen full quality experiment

Omri requests continuation of the pre-Claude contextual-IU line and retention
of the FM/DiFlo research line. This stage completes the Step384 quality question;
it does not repeat Claude's residual-moment experiments or restart the90-job queue.

## Fixed method and population

Innovation5 original oriented raw features, all13769 answers/145597 steps/
6968779 tokens, same tail15 gate percentile>=.33. No residual replacement,
feature-bank search, q tuning, eta tuning or correctness labels in fitting.
Use the frozen Step384 energy context: preceding16-token log energy, training
position/length profile removal, standardized position+energy coordinates,
K64 distinct source-group neighbors, Gaussian kernel, .5 covariance borrowing.
Reuse all45 checked single-fold-excluded fits/heads from energy_context_stability_v2.
New fits only for10 pairs of excluded folds in PRMB, to calibrate PRMScore without
training on outer held sources. Same preprocessing, settings and balancing.

Weights update at the existing up-to16 uniform landmarks t>=16. Between anchors
hold the MOST RECENT weight, never interpolate from a future anchor. Tokens0..15
use the static head. All tokens remain available for scoring, including tokens
between anchors. This is a16-anchor piecewise-constant policy, not exact every-token
context fitting. Long answers may contain short context shifts it misses. Relative
length and existing whole-answer orientation make the full algorithm offline.

## Heads, amplitude isolation and readout

Three existing heads, with no refit/selection on correctness:
native two-PC IU; full-C simplex tau1/eta.25; fixed one-parameter group simplex
{H0lim,VE0,innovation} vs {VE075,VE1}, matching Step384. Grouping is heuristic,
not a learned selector or assertion of conditional error independence.

Native a converts to raw weights a/sd, divided by the STATIC head's raw L1 norm.
This retains conditional amplitude changes. Simplex/group use their raw weights.
For each head define amplitude A_t=||sd*w_t||2/||sd*w_static||2, using training
scales. Six policies: static, energy-full, position-full, random-full,
energy-amplitude-only (A_t*w_static), energy-direction-only (w_t/A_t).
Before token16 all policies equal static. No zero-norm silent fallback; halt and
record any failure. Random uses exactly energy's kernel,64 distinct random groups.
For pair fits its deterministic seed is the fit name hash, as in the old driver.

Per step/per feature select ORIGINAL raw Top10 tokens (stable ascending index
tie break, taking the last10), then average their weighted contributions and sum
features: sum_k mean_{t in Top10(X_k)} w_tk*X_tk. Thus the selected evidence is
the same for every policy; constant equal weights recover innovation5. Negative
native weights do not select bottom tokens. No token compression or change to
step boundaries; weights/history span the entire answer. This is a declared new
dynamic readout, not Top10 on weighted contributions or fused tokens.

18 policies plus equal, with original4/innovation5, same-gate singles, entropy,
RBM12 and the historical secondary additive ridge. Retain Step385 outcomes as
context; no additional residual covariance axis in this stage.

## Evaluation and acceptance

Primary head fixed in advance: simplex. Primary energy-full minus static,
position-full, random-full, amplitude-only:4 contrasts x PB/within2 endpoints.
10000 source-group paired bootstrap draws,99.375% Bonferroni intervals. Native,
group, direction-only and baseline comparisons descriptive95%; no post-hoc
promotion of a secondary winner. PRMScore nested quantile.8 calibration, same
source folds and evaluation roster. Full data are development, not confirmation.

Tests: static/equal readout identity, deterministic ties, no future-anchor use,
direction/amplitude reconstruction, synthetic switching-expert versus static
weights and constant no-context identity, warm-up coverage and no dropped tokens.
Reuse current canonical/KKT/source-neighbor tests. Independently reconstruct
scores for audited answers via scalar loops; recompute PB and pairwise within
on ALL methods. Verify source fit/group hashes and whole-population coverage.
Small checks only mechanics/cost. No quality veto from synthetic observations.
Save scores, landmark weights/positions, pooled fit provenance and source hashes;
atomic completion per cell/exclusion pair, resumable, bounded memory/disk.

## FM/DiFlo integration remains open

FM is the base conditional flow-matching model; DiFlo adds auxiliary objectives;
DOT measures deviation of generated flow paths from their endpoint chord.
Flow-integration time is distinct from token time. They are not three standalone
temporal detectors. None has a full-population quality verdict from the paused
queue's four jobs. Current checkpoints and learned artifacts remain preserved.

Future bridge A (preferred): replace/augment the hand-defined energy context
with a source-excluded learned flow context, predictive moments or DOT descriptor,
then use the SAME neighbors->covariance->IU->original-feature readout with static,
position/random and amplitude controls. Scalar DOT alone cannot identify which
feature is reliable; its usefulness as a neighborhood coordinate is a hypothesis.
Current flow conditioning may include the current feature vector when predicting
the next token: align the integration contract explicitly, or lag the descriptor
when requiring strictly prior-token context. Never describe it as past-only
without checking the feature/target alignment.

Future bridge B: DOT as a bounded trust indicator controlling borrowing toward
static/equal weights, learned/calibrated without correctness labels. High DOT
does not establish a semantic error or trustworthy covariance; require a matched
constant-strength control and a full-population quality comparison. Auxiliary
loss activity/gradient scale and history sensitivity must be logged before a
costly restart. No combined model is fitted or validated in the current stage.
