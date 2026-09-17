# Sparse loading membership inside Joint, 2026-09-17

Steps402-403 are verified progress, not completion: model-derived group weights
reduce noise damage, but initial fit failure and duplicate-dependent selection
remain. One next candidate learns noise-only rows through the Joint covariance
objective. No correctness labels or protected feature define membership.

Canonical model coordinates identify EXACT equal columns on training sources.
Aliases share one measurement/parameter; decoder distributes its weight equally.
At prediction, agreeing aliases use their common value; otherwise their mean.
This establishes exact-copy invariance, not near-collinear robustness. Only
training data determine aliases. Identical canonical training inputs/fold/seed
may reuse a fit; evaluation still uses that bank's own held-out coordinates.

Provisional sparse Joint minimizes
  .5 sum_(i<j) [C_ij-v_i v_j-1(group_i=group_j)u_i u_j]^2
  +lambda sum_i (|v_i|+|u_i|)
subject to v_i^2+u_i^2<=C_ii. Alternating scalar soft-threshold updates solve each
coordinate exactly; monotone descent required. Both loadings zero means a
diagonal-noise-only row. Groups start from unchanged training-fold-deletion
K={3,4} stability discovery. No initial unpenalized valid fit is required.

Penalty scale:31 independent feature-sign randomizations per source block,
preserving each channel's within-source temporal structure. For each randomized
off-diagonal covariance take the maximum absolute gradient against the feasible
initial global and masked local directions. Lambda is its95th percentile,
floored at1e-8. This is a Monte Carlo null-scale heuristic, not a formal test or
FWER guarantee. Seed404170+outer+100*round; no scale/exponent grid or outcome tune.

Five deterministic starts, <=3000 coordinate sweeps each; objective and parameter
stability for5 sweeps. Keep the UNION of nonzero rows across converged starts;
zero converged starts is a failure. At most3 monotone inclusion/discovery rounds;
stop early if support unchanged. Save all supports, convergence and objectives.

Finally rediscover groups on retained coordinates and debias using the ORIGINAL
checked unpenalized Joint estimator (five starts, profiled pair checks, all old
guards). Try admissible K candidates in the pre-existing stability order; first
valid is the native result. Report attempts explicitly. No valid structure -> H1
fallback, never counted native. Use Step403 group-reliability readout. H1 only
orients the score and may be removed from the active support. No95% per-factor
retention rule; weak local factors are not guaranteed preservation.

Full13769-answer matched baseBOCPD51/copies66/noise66 benchmark, frozen gate and
five source folds (hybrid, not answer-only), no digits. Same historical controls.
Alias-only control replays the full-support Step403 base head for base/copies
(canonical equality verified), noise full head for noise; preserves its failure.

Five primary contrasts x2 endpoints,10000 source-group bootstrap,99.5% intervals:
sparse versus frozen Step403 automatic group head on each bank (three), plus
sparse copies/noise versus sparse base (two). Same practical margins PB-1pp /
within-.002 and native13769. Also exact-copy score/peak equality, retained noise,
BOCPD retention, support size and runtime. Weak base quality cannot be called
success just because perturbation sensitivity is low. Full development evidence;
untouched confirmation and general redundancy robustness remain separate.
