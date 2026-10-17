# Procedure-matched gate calibration - Step320

Freeze before simulations. This is one candidate gate rule supporting our
existing fusion; it is not a new semantic detector or benchmark winner.
The current110/176-entry comparison stays unchanged during this stage.

## Why the gate must be replaceable

Step319 demonstrated mixture opening on stationary Gaussian sources after
filtering, including warm filters. Step318 also lost correctly located
errors behind its gate. Merely suppressing existing alarms cannot solve it.
Using current110 labels as an explicitly diagnostic oracle, accept every
clean answer but keep only the original exact-error successes. The resulting
PB upper bounds for closing-only gates are IU35.24123%, Joint graph37.43421%,
primary GLS37.86550%, primary IMM26.02124%, risk equal-permuted39.67517%.
Primary IMM cannot beat current IU30.15985% with a closing-only gate.
These are not candidates or attainable forecasts. Source: frozen Step318
EVALUATION.json. Recheck the per-subset harmonic formula in review.

## The fixed candidate and diagnostic reference

Observe one scalar fused trajectory. Standardize on all its N observations.
Fit AR(1) with an intercept by separately centering preceding/following
vectors; rho=sum((xprev-meanprev)*(xnext-meannext))/sum((xprev-meanprev)^2).
Clip to[-.95,.95] for a stationary generator and disclose every clipping.
This fit has no correctness labels or donor traces. It can mistake a mean
jump for dependence; the positive controls explicitly test that risk.

Generate B=39 stationary Gaussian sequences at the fitted rho and length N.
Each draw goes through the same normalization, noise estimation, cold IMM
(when applicable), final normalization and one/two-GMM fit as the observed
trajectory. Score T=BIC1-BIC2. Two readouts: raw and actual cold two-mode IMM,
with the exact Step319 parameters. Known-rho calibration repeats the same
procedure using the true synthetic source rho: diagnostic unavailable
information, not a candidate. Both get39 draws, with paired innovations.

p=(1+count(Tnull>=Tobserved-tolerance))/(B+1), tolerance=100*eps*abs(Tobserved).
Calibrated gate opens if p<=.05 and the observed curve exceeds the mean of
the fitted two-GMM means. It does NOT also require the old BIC gate to open.
Thus a calibrated gate may close or reopen; report both transitions. The
smallest p is.025 and only two ranks pass: a fixed coarse pilot budget, not
an optimal or publication calibration. Any missing/invalid required fit
makes that calibrated decision invalid; never drop a failed null draw.

Normalization means/SD, noise R and GMM parameters are refitted on every
draw. The statistic does not use rho explicitly; rho selects the generating
distribution, so a second unused rho fit is not substituted for uncertainty
assessment. Comparing fitted-rho to true-rho calibration on independent
trials measures sensitivity to estimation; no exact composite-null guarantee
is asserted. The empirical AR coefficient is not known true dependence.

## Independent evaluation design

Fresh seed namespace, independent of Step319. N16/64/256, true rho0/.6/.9,
plus rho0/+3SD halfway mean-jump control: all12 cells,32 evaluation trials
each,384 total. Within each cell evaluation replicates are independent.
All regimes share evaluation innovations at the same N/replicate for paired
source controls. No old Step319 evaluation trial is used for calibration.

Each evaluation trial receives its own39 new calibration innovation vectors,
independent of its observed data and of every other trial's calibration.
True-rho and fitted-rho use those same innovations as a matched control.
Use first128 SHA256 bits of versioned identity strings, with distinct
evaluation/calibration roles; persist/check seeds and their disjointness.
Initial Gaussian source state is stationary. No warmup data enters this gate.

Each readout has original BIC, fitted-rho calibrated, and true-rho reference
decisions. Six outputs/trial,2304 decisions. No alpha/rho-clip/N/B sweep.
Save observed/source and calibration curves, fitted parameters, BICs,pvalues,
open/close transitions and failures. Exact binomial intervals describe the
384-trial simulation evaluation, not real-answer false-positive guarantees.

## Predeclared advancement screen

Do not advance solely because fewer alarms occur. For a readout to earn
a real-benchmark gate-only pilot, require both: (a) in every stationary cell
where its original gate opens at least20% of trials, fitted-rho calibration
reduces openings by at least half; (b) in each of the three jump cells it
retains at least75% of original correctly detected jumps (count the SAME
trials, not only total detections). Report cells with low original sensitivity
and all intervals. This is an engineering screen on simulation, not a
confirmatory statistical test or permission to claim a5% semantic error rate.
Known-rho reference cannot itself earn advancement. If the rule fails,
retain that result rather than silently changing alpha or the null family.

## Review, budget and scope

Three CPU workers, per-trial atomic checkpoints,600s submission cap,
resumable completion. Estimated output well below0.5GB; local free space
21.74GiB at preflight. No model inference, external data transfer or changes
to Claude's worktree. Original IMM/R/GMM kernels stay frozen.
Tests cover seed separation, AR affine invariance, rank/tie accounting,
reopening a negative-BIC case, and failure preservation. Review all seeds,
source fits, null sources/normalizations, GMM likelihood/BIC, pvalue/rank
decisions, paired summaries and advancement criteria. Direct vector IMM
and actual GMM refits on representative cases; disclose shared kernels.
Recheck the closing-only ceiling using independent existing PB metrics.
Render a visual report with all conditions and unchanged benchmark anchors.

This adapts statistical calibration to the declared score-processing recipe.
It is not yet a calibration of the full fitted feature/Joint pipeline, the
paired-vector noise estimate of Step318, or a semantic model of correct
reasoning. Any subsequent benchmark adapter must verify its exact pipeline.

Primary statistical reference: SciPy's [goodness-of-fit documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.goodness_of_fit.html)
distinguishes known and estimated parameters, refits the statistic's fitted
quantities on resamples, and uses the plus-one Monte Carlo rank. It does not
validate our AR generator, BIC statistic, coarse budget or semantic target.
All wider fusion/Joint/IU, named supporting methods, comparator/refit,
untouched-confirmation and historical24 requirements remain active.
