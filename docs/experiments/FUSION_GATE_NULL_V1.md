# Fused-score mixture gate: controlled serial-dependence check - Step319

Freeze before generating simulations. This is a mechanism check supporting
IU-PCR / Joint L-SML, not a new benchmark method or a semantic-error model.
Keep the reviewed current110/176-entry Step318 evaluation as context. No
benchmark fit, score, peak, gate or label is changed or used in this simulation.

## History and decision question

Step302 already examined lag1, affine invariance and exact duplicated-data
BIC at fixed parameters. Step314 separated gate/location headroom. Step318
adds actual paired trajectories and fixed peak/gate exchanges: primary IMM
weakens both components, and clean lag1 rises .28649->.63670. Do not call
dependence a newly discovered concern or claim it is the sole cause.

Can one stationary Gaussian source, without any change in generating regime,
make the actual one/two-GMM gate open after dependence or filtering? Does
that persist after removing filter startup? A second, fixed mean-jump control
shows sensitivity to a known distribution change. Neither represents actual
correct reasoning or hallucinations. IMM is nonlinear; even a Gaussian input
need not yield a Gaussian output. A mixture preference is not automatically
an implementation bug or a semantic error. BIC is model selection, not a
test with a declared5% false-positive rate.

## Fixed design

N={16,64,256} observed windows; stationary Gaussian AR(1) rho={0,.6,.9};
64 independent seeds per cell. These nine null cells yield576 sequences.
Add a mean jump of3 marginal standard deviations halfway through the scored
segment, at rho0 only: three positive-control cells,192 sequences. Total768.
No sweep/tuning based on the outcomes. All12 cells and failures are reported.

Generate N+256 points with x0~N(0,1), x[t]=rho*x[t-1]+sqrt(1-rho^2)*z[t].
Use SHA256('fusion-gate-null-v1/'+N+'/'+replicate) first8 hex as RNG seed.
Use identical innovation draws across rho and jump conditions for pairing;
replicates within a cell remain independent. Jump affects only the second
half of the finalN points. Normalize using finalN mean/population SD, and
apply the same affine map to the preceding256 points.

Five fixed readouts: raw, ordinary single-mode Kalman cold/warm, two-mode
IMM cold/warm. Same existing noise_variance heuristic estimated on the
normalized finalN inputs. Kalman Q=.01R; IMM Q=(.01R,R), transition.95;
same initialmean0/variance1/modeprobability.5. Cold sees finalN only. Warm
sees allN+256 but scores finalN; its extra context is explicitly a synthetic
startup diagnostic, not an answer-only candidate with free external data.
Normalize each final curve on its N scores before the actual existing GMM.

Use the exact fusion_gate_interface_audit.inspect_mixture implementation:
one/two GaussianMixture, n_init3,max_iter300,reg_covar1e-4,seed2026090705.
Report component choice and actual gate opening separately. Treat windows
as the candidate positions for this gate-only check; no benchmark step
projection, localization score or first-error metric is defined here.

## Evidence and review

Save innovations, generated path, raw/five scored curves, observation noise,
mixture parameters/log likelihoods/BIC and validity for each trial. Freeze
code/protocol/test and parent reference hashes. Three CPU workers, per-trial
checkpoint,600s submission cap with resumable completion. No new inference.
Report gate-opening rates with exact binomial95% intervals per cell, paired
filter-vs-raw alarm gains/losses, cold/warm differences, and all failures.
Intervals describe simulation frequency, not project false-positive control.

Tests verify the exact stationary covariance implied by the generator,
normalization invariance, warm-state alignment against direct scalar Kalman
updates, and GMM BIC algebra. Review every saved path/normalization, scalar
Kalman trajectory and mixture likelihood/BIC; independently replay selected
vector IMM and GMM fits in every cell. Report shared kernels honestly.
No null-based threshold is selected here. If simulation shows a material
effect, next freeze one matched gate-only benchmark experiment with the
unchanged176 anchors and an explicit null-model limitation. Otherwise avoid
assuming dependence calibration will solve the problem.

## Primary sources checked

- scikit-learn's [GMM model selection example](https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_selection.html)
  selects the model with lower BIC. It does not establish a semantic-error
  threshold or a nominal false-alarm level for our dependent traces.
- SciPy's [Monte Carlo test documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.monte_carlo_test.html)
  makes the supplied null-generating distribution explicit. This motivates
  checking a declared synthetic null before proposing calibration; no
  Monte Carlo hypothesis test on benchmark answers is performed in Step319.

These sources support the statistical interfaces, not our chosen AR model,
IMM adaptation or a claim that a stationary Gaussian process describes a
correct answer. Those are declared diagnostic assumptions.
