# CCA/IU isolation gate — frozen before execution, 2026-09-15

Authorized by Omri after Claude/Codex review. This bounded stage does not launch
PB/PRMB quality evaluation or resume the neural queue. Preserve historical S0.

## Design decision

The primary QP uses the FULL centered covariance, not a rank-two projection.
It minimizes .5*w' B C B*w - tau*w' B*rho on the nonnegative unit simplex,
B=diag(training feature SD)/mean(training feature SD), tau=1, eta=.25.
Final weights are .75*equal+.25*w_QP. No negative coefficient cancellation or
unbounded leverage is possible, but low-variance preference remains a risk.
Native rank-two IU is a separate diagnostic, not the QP's name or identity.
Use scale_ratio=.25 with the same frozen global var_y search ceiling for all
conditional fits. var_y bounds the g2 grid; g2 is a signal moment, NOT generally
Var(Y). Canonical additive identity: C_ij=rho_i+rho_j-g2. No rank-one product fit.
Tau=0 and linear-objective vertex limit are diagnostics, not tuned candidates.

## Data and stages

1. Independently replay frozen S0: seeds32000..32019, train320/test640, all
   informative/null/coherent_nuisance worlds. Call the original functions but
   write to a new directory; compare per-seed numbers and original gate verdict.
2. Use those EXACT current observations with analytic population covariance
   and supplied regime, then finite reference covariance. Compare equal,
   static/conditional full QP, native2PC, and known Cov(X,target) oracle QP.
   Oracle target moments/regimes are explicitly synthetic diagnostics only.
3. Given noisy DSP-only context (excludes current score/MAD), use K64 distinct
   independent reference groups, Gaussian bandwidth distance to the64th,
   and centered covariance with global shrinkage. Select alpha in{0,.5,1}
   using feature Gaussian NLL on training240/validation80, refit on320.
   Means are blended too for NLL. Test labels never enter these routines.
   Compare random neighborhoods with IDENTICAL sorted kernel weights/K and
   the same alpha selection budget. Each point is its own synthetic group.
   Compare a one-parameter fixed first3/last3 group mixture using the SAME
   objective restricted to equal coefficients within each block, eta unchanged.
4. Add independent16-observation histories under the SAME regime to the frozen
   current observations (new synthetic extension, not historical S0 replay).
   Compare oracle regime, raw linear CCA H->X, augmented [H,H^2]->X, and
   second-moment CCA H^2->X^2, rank2, joint covariance shrinkage .1.
   Center/scale all transformations using allowed training only. Squaring H
   alone cannot recover a zero conditional mean of X; the target moment must
   also change. A simple per-stream mean history energy is the direct control.
   Fit directions on inner train240 for NLL selection then refit320. Assess
   held-out canonical correlations, correlation with history mean/energy, and
   downstream fusion. This history has variance regimes, not semantic reasoning.

## Decision and outputs

20 seeds in every world and every method, no seed selection. Report mean AUC,
paired deltas, wins and10000 paired-seed bootstrap95% intervals (descriptive,
synthetic mechanism tests, not PB/PRMB confirmation). Save raw per-seed records,
fit telemetry, predictions/weights, input hashes and code/protocol hashes.
Original S0 gate criteria are replayed unchanged. Apply its informative gain
>=.005, wins>=18/20 and null/nuisance safety limits to each new method versus
its MATCHED static simplex, and report equal separately. No claim the original
S0 satisfies the additive IU assumptions. Known moments and known-rho oracle
separate identification/optimization from sample-size and representation issues.
If the proposed full method fails, stop full real-data evaluation and name the
failed link; do not tune to these seeds. If a simpler arm explains its benefit,
prefer that arm as the next proposal. Passing is permission to consider the
full unlabeled real-bank stability diagnostic, not automatic quality promotion.

## Acceptance checks

Independent QP solution/KKT, eta0 identity, alpha1 static identity, group-mixture
restriction, canonical rho/g2/native-weight equivalence for any accelerated
moment implementation, known population moment identities, train/test mutation
isolation, correct second-moment target, unchanged S0 sources/results. Numerical
failures abort visibly; no silent replacement with equal. Effective sample size
and g2 boundary hits are saved. Checkpoint per seed/world and resume only when
protocol/code/source hashes match. CPU threads1; no new hardware jobs.
