# Answer-local fusion with information about position and token context

Date: 2026-09-13. Base: answer-position-fusion-v1, commit 481381408.
This worktree prepares the follow-ups. It does not claim completed experiments.

## Where these directions fit

| Order | Experiment | What is learned | Current implementation state |
|---|---|---|---|
| Current | Whole-answer position fusion | Feature-weight maps from other answers, with Factor / exact H1 RBM / IU | Full driver exists; local smoke was waiting for RAM. Cluster portability prepared separately. |
| Next | Position-conditioned Shrinkage IU | Current-answer fusion plus a position prior from training answers | Covariance adapter and alpha-zero tests implemented; fold-aware experiment driver still needed. |
| Then | Graph-local IU | Feature weights from a token's graph neighborhood, regularized toward an answer/shared estimate | Design only; weighted sliding-window control required. |
| Then | Graph penalty on feature weights | A related family of token-specific weight vectors with limited jumps on graph edges | Design only; objective and solver must be registered before evaluation. |
| Separate later families | Conditional Gaussian RBM; Temporal Convolution | Short-range dynamics, including across official step boundaries | Remain in the four-direction roadmap; these IU studies do not replace them. |

The earlier hierarchical-time experiment fixed RBM feature weights and learned
positive averages of relative positions WITHIN each step. Its negative result
does not test the new hybrid or whole-answer position. The current experiment
instead preserves token scores and Top10, and learns feature weighting from
other answers by position in the COMPLETE answer. It is already a comparison
of several learning algorithms. The new hybrid adds current-answer fitting.

Code and data preparation may proceed in parallel. Scientific conclusions and
the next family specification follow the full current experiment and review.
There is no requirement to wait for it to write adapters or prepare compute.
There is also no automatic launch of unregistered graph families. A plan entry
is not a queued job. The cluster supervisor runs the CURRENT experiment only.

## First follow-up: preserve the answer-local baseline

Use the frozen 12-feature bank and answer-local normalization. Let C_a_base
be the covariance used by the chosen answer-local Shrinkage IU solver, after
its existing regularization. Define a NEW external-borrowing parameter alpha:

    C_a,t(alpha) = (1-alpha) C_a_base + alpha C_train(r_t)

r_t is relative token position in the WHOLE answer. Official step boundaries
are used only for the same Top10 readout. The training prior is estimated
without labels and excludes the entire test source-group fold. It must use
the same feature order, active-column mask, sign convention and answer-local
standardized coordinates. Answer and source-group balancing remain explicit.
Average within-answer regional centered covariances so variation between
regional means is not silently called residual noise. Mean/intercept changes
are excluded from the first contrast; a position mean is a separate control.

Alpha=0 must return the baseline weights and token/step scores exactly. Do
not recompute them and hope for approximate agreement. This includes the
baseline's own shrinkage setting, active columns, orientation and abstention.
The new alpha is NOT the baseline shrinkage parameter or LIU's lambda.

The prepared `borrowed_iu_weights` seam implements that exact zero branch.
Its positive-alpha default is canonical FULL covariance IU. If the selected
baseline uses solve-only or subspace shrinkage, supply the matching callback;
do not switch the solver while describing the result as borrowing alone.
The completed driver must verify the positive-alpha limit against its chosen
baseline, not use the exact zero branch to conceal an incompatible solver.
There is no label input to this estimator. Fold protection belongs to the
future driver; the small adapter by itself does not establish no leakage.

Required matched comparisons: baseline; pooled external covariance (no
position); position-specific external covariance; shuffled position prior.
Use the same borrowing strength for those three positive-alpha comparisons.
Register that strength and the baseline recipe before reading their results;
do not copy old stream-group labels onto the 12 moments. Do not search graphs,
feature banks and borrowing strengths together.

## Next two graph alternatives

**Graph-local IU** estimates covariance from weighted token neighbors, then
regularizes and solves for feature weights at the current token. A sliding
window is its simplest local-neighbor control. Start with adjacency/statistical
relationships already present in the traces. Do not claim syntactic dependency
edges without a parser. Do not construct edges from benchmark labels. Report
effective neighborhood weight concentration, since many near-duplicate tokens
are not many independent observations. Keep original token identity and step
membership. This changes coefficient estimation, not an average of output
scores. The full model and graph settings remain unimplemented.

**Graph regularization of weights** instead fits coefficient vectors together:

    sum_t fidelity(w_t; local moments and baseline)
      + lambda * sum_(t,s) edge_weight(t,s) ||w_t-w_s||_2

This is Network-Lasso / graph-total-variation inspiration, not yet an IU
derivation. The fidelity term must be specified consistently with the chosen
IU solve, and lambda=0 must recover that baseline. This penalty permits some
jumps, unlike requiring every adjacent score to be similar. It still may
attenuate useful events, so measure missed and displaced error steps. Compare
to local-window and shuffled-edge controls; do not call any smoother useful
merely because it yields stable weights or better covariance fit.

## Shared measurement contract

Keep all 13,769 answers, v3 labels, v2 source groups/folds, the saved entropy
q=0.3 gate, Top10 token mean and earliest argmax tie. No first_near_max. External
training and PRMScore q=0.8 calibration must be outer-fold-blind; use inner-fold
predictions whenever the risk model learns from other answers. Test-answer
features are allowed only for the declared local part. All failures remain
visible. Report PB all8/Q4/Q8/per-cell, exact/early/late/clean, PRMB within and
pooled/fold AUC, PRMScore, coverage, runtime and paired source-group uncertainty.
Keep current RBM and historical matched controls. No result is confirmation on
untouched data. No improvement is claimed from adapter tests or smoke runs.

## Cluster execution

AIRCC SSH and Slurm were verified on 2026-09-13. Non-interactive commands need
`SLURM_CONF_SERVER=controller-primary`. Live `sdata` reports account
`cycle3_tau_averbuch_prj`, QoS `owner_940`, partition `power-gpu`; the repository's
old owner_880 setting is stale for this user. The dedicated script records the
new allocation without editing Claude's shared scripts/configuration.

The local bottleneck is available RAM, not disk. Removing worktrees frees
storage; it does not free the memory occupied by active processes. The cluster
copy changes only the available-memory probe and its manifest hash list.
It keeps one process, BLAS=1, same fits and eight-hour invocation caps. Input
files are hash-verified and placed in an isolated directory under the existing
project workspace, never synced over the shared `code` directory. An initial
scheduled job requests 64 GiB and tests the platform plus original fixtures.
Resource requests are not reservations until Slurm starts the job. GPU is not
used by the algorithm. Queue policy may still require a GPU allocation.

Portable execution writes new results/manifests and preserves the waiting
Windows artifacts. Do not reuse a SQLite manifest with different absolute
paths. Full execution starts only after smoke PASS. Numerical reproduction of
the frozen Top10 and references remains required on Linux. No other method
family starts in that supervisor.

Inspiration, not a claim of reproducing these papers:
- Multi-Target Shrinkage: https://arxiv.org/abs/1412.2041
- Network Lasso: https://web.stanford.edu/~boyd/papers/network_lasso.html
- Existing project `upcr.py`, `shrinkage_iu.py`, `laplacian_upcr.py`.
