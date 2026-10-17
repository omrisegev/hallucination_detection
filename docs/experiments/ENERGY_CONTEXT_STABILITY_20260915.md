# Real-bank unlabeled context stability — frozen diagnostic

Authorized continuation after Step383; retains the Step381 innovation/position
results as the real-data anchor. No correctness labels, step scoring, PB/AUC
evaluation, gate changes, CCA fitting or neural queue resumption in this stage.

## Population and access

Use only MANIFEST.json, METADATA.json and features.npy from the hashed,
label-free temporal_context_data_v1 bundle. Bank innovation5, original frozen
orientation/values. Verify the strict metadata whitelist and all source groups'
fold consistency. All13,769 answers enter the diagnostic, at up to16 distinct
uniform positions from token16 to the last token (zero-based). Every answer
currently has >16 tokens. These are landmark statistics, NOT full-token scoring
or subset quality evaluation. No step boundaries reset history. Whole-answer
feature orientation remains offline; new context uses strictly preceding16
tokens. Do not claim this is a complete causal-online system.

Fit independently in each of9 dataset/model cells and5 held-source folds.
All fit/preprocessing/reference rows exclude the held fold globally. Group,
answer and landmark weighting prevent repeated sources from counting as
independent. Each source contributes equal total weight to global moments.

## Fixed context and covariance

Global feature mean/SD come from training landmarks only. Context is the five
log1p mean squared standardized feature values in the preceding16 tokens.
This contains both background level and fluctuation around the training mean;
it is NOT variance alone. Keep current innovation as an original target feature.

Remove a training-only position/length mean profile from context:16 interpolated
position bins crossed with fixed length bins<=256/512/1024/>1024, weighted ridge
penalty .01 (global intercept unpenalized). Standardize residual context using
training SD. Position coordinates are relative position and log1p(answer length),
standardized on training. Weight each coordinate block by inverse sqrt(width).

Four arms: static; position; position+residual-energy; random reference groups.
K64 distinct source groups, nearest landmark from each group, exact neighbors.
Gaussian bandwidth = distance to64th group, so n_eff is recorded. Random arm
draws64 distinct groups uniformly and one landmark uniformly within each, and
uses EXACTLY the energy arm's sorted kernel weights. It preserves sample mass,
not position distribution; position is the separate main comparator.

Centered local covariance mixes .5 local + .5 global, FIXED, not selected by
quality or likelihood. Means also mix .5 for held-out Gaussian NLL. All arms
use diagonal numerical floor1e-6 in training standardized coordinates. Static
uses global moments. No local mean subtraction from a future detection score
is implied by this density diagnostic.

## Heads and stability

Canonical L2 IU moments, full pool, scale_ratio.25 with fixed global var_y
ceiling,300 grid points. Native PCR uses2 PCs. Full-C simplex uses tau1 and
eta.25. One-parameter simplex restricts within-group coefficients to be equal:
{H0lim,VE0,innovation} versus {VE075,VE1}. This is a stated heuristic grouping,
not learned GroupFS and not evidence of independent errors within/between groups.
Report standardized coefficients a as well as raw weights for simplex, and
unit direction/amplitude for native PCR. No coefficient clipping of native PCR.

For16 deterministic held query anchors per cell/fold, from distinct source
groups, draw64 bootstrap samples of the64 neighbor GROUPS. Recompute local
means/covariance and all heads, retaining the frozen global/profile estimates.
Report between-anchor variance divided by average within-anchor bootstrap
variance for covariance shape, rho direction, native direction, simplex direction
and group-head direction. Also report covariance magnitude and head amplitudes.
This is CONDITIONAL uncertainty given the fitted neighborhood/profile, not total
pipeline uncertainty. Overlapping neighborhoods make the ratio descriptive;
ratio>1 is not a hypothesis-test rejection or proof of correct reliability.

## Readouts and decision

Evaluate held-out FEATURE Gaussian NLL on every landmark, balanced by source
within cell; macro-average9 cells. Main paired contrast: energy minus position;
also energy minus static/random.10,000 source-group bootstrap draws, keeping
cross-cell occurrences of a source together;95% intervals descriptive. NLL is
not correctness, within-AUC, or a candidate-selection score for error detection.

Report all45 fits/cell heterogeneity, g2 ceilings/additive residuals, neighbor
support, weight direction/amplitude, group-head approximation and failure counts.
No automatic quality rollout based on one statistic. If context does not improve
held feature distribution or produces mostly unstable weight changes, stop the
complexity and report which link lacks support. Otherwise propose ONE frozen
matched full-quality comparison; do not retune banks, gate, eta, tau or K.

## Acceptance and artifacts

Scalar history replay, current/future mutation invariance, training/held-source
firewall, exact group kNN versus brute force, duplicated-source weighting,
canonical native/moment agreement, QP KKT and independent optimizer checks.
Independent held NLL reconstruction on saved query-targets/moments and scalar
group/cell aggregation. Save hashes, query/reference/group IDs, preprocessing,
moments/heads and bootstrap summaries, per-fit atomic completion, state/report.
Re-use only checkpoints with exact protocol/code/data hashes. No real labels
or original detector scores are opened. Single BLAS thread, bounded diagnostic.
