# SDSF solver cycle v4 — fixed-stable features

## Provenance and scope

This cycle executes the solver-mechanism questions already preregistered in
`SPEC_SOLVER_MECHANISM_STUDY.md` on `confidence-orientation-v1` with the four
unstable raw views excluded. It was run only after the synthetic-selected
bootstrap stabilizer improved current SDSF but remained 2.91 AUROC points below
SU-PCR on the real artifact.

The data source is the tracked bundle
`results/dependency_fusion_raw/cells.npz`, introduced by commit `4316531` on
2026-08-05. It contains derived feature matrices and correctness labels; it
does not contain prompts, generations, raw traces, or model weights.

This is retrospective mechanism evidence. It cannot promote a tuned method.

## Hypotheses

1. **Tail mechanism:** admitting eigen-directions below the leading two causes
   the inverse solver's real-data loss.
2. **Head control:** rescaling the leading two directions with the registered
   ridge rule is approximately neutral.
3. **Amplification:** allowing a larger condition number makes the tail effect
   more negative.
4. **Structural versus finite-sample:** if the tail remains harmful on held-out
   rows as the training fraction grows, the full inverse is structurally
   mismatched rather than merely noisy.
5. **Different-family alternative:** directly fuse the first regularized CCA
   variates of the spectral-trace and energy/log-probability channels. This is
   exploratory; it tests multi-view fusion rather than repeating the earlier
   l0-CCA feature-selection experiment.

## Frozen design

- Full-data 2x2 at a fixed SU-PCR reliability estimate:
  PCR versus ridge scaling of the top-two head, crossed with tail absent versus
  present.
- Condition-number path: `{3, 10, 30, 100, 300}`.
- Held-out fractions: `{0.25, 0.50, 0.75}` with 50 deterministic repetitions
  per cell, train-fitted standardization, train-only global sign, and labels
  used only for test AUROC.
- Equal-family bootstrap confidence intervals over eight dataset families.
- Structural-mismatch decision: tail family CI is strictly negative, held-out
  tail effect at 75% training is below -0.5 points, and absolute head-rescaling
  effect is below 0.5 points.

Executable specification: `scripts/sdsf_solver_cycle_v4.py`.

## Results

The decision is **`ABANDON_FULL_INVERSE_SDSF`**.

- Head rescaling: -0.01 family-macro points, 95% CI [-0.02, +0.00].
- Tail addition: -2.01 family-macro points, 95% CI [-3.47, -0.77].
- Full inverse versus PCR: -2.04 family-macro points, 95% CI
  [-3.52, -0.78].
- The tail-effect slope is -0.617 points per log condition number: weaker
  regularization makes the tail more harmful.
- Held-out tail effects remain -4.44, -3.86, and -3.34 points at training
  fractions 0.25, 0.50, and 0.75. Head rescaling stays at zero.
- Direct two-channel CCA is also below PCR: -1.96 family-macro points, 95% CI
  [-3.67, -0.46].

## Scientific conclusion

The real-data failure is not principally inaccurate feature direction, ridge
rescaling of the leading signal, or insufficient sample size. It is the model
assumption that correctness information should be recovered by inverting the
low-eigenvalue covariance tail. Bootstrap shrinkage repairs part of the damage
but does not change that conclusion.

For the current feature family, full-inverse SDSF should be retired. SU-PCR's
low-dimensional final solver remains the leading method. A future dependency
contribution must modify reliability estimation while retaining low-dimensional
fusion, and must be tested on genuinely new dataset/model families. Re-running
the already failed clustered U-PCR or l0-CCA selection paths is not a new cycle.

Evidence: `results/sdsf_solver_cycle_v4/REPORT.md`, `summary.json`,
`per_cell.csv`, `kappa_path.csv`, and `heldout_repetitions.csv`.
