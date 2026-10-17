# Claude feature bank — covariance pooling scope v1

Status: development-only matched experiment.

## Question

Does the covariance and L-SML structure of the eleven token-risk views reflect
stable feature complementarity, or partly the mixture of model and dataset
populations in the global fit?

The frozen global experiment is reused exactly.  Three new source-fold OOF
arms change only which donor answers estimate standardization, covariance,
groups, and weights:

1. `global`: all ProcessBench and PRMBench cells together (frozen existing run).
2. `model`: one fit for Qwen3-4B and one for Qwen3-8B.  Under the matched
   existing population, the 8B scope also contains PRMBench.
3. `dataset`: one fit for each PB dataset, pooling Qwen3-4B and Qwen3-8B;
   PRMBench is its own dataset scope.
4. `cell`: one fit for each exact `(model, dataset)` cell.

For every scope key and each of the five source folds, the fit uses only the
other four folds from that same scope key.  Correctness labels never enter
standardization, covariance, group discovery, or weights.  The LOCO-5 gate,
eleven feature definitions, manual risk signs, 60,000-token fit cap, token
fusion, and step top-10 mean locator remain unchanged.

## Diagnostics

Two quantities must not be conflated:

- `covariance_participation_ratio` is the spectral quantity requested here,
  `(sum lambda)^2 / sum(lambda^2)`, calculated from the donor-standardized
  token covariance eigenvalues.
- `weight_participation_ratio` is `1 / sum((abs(w)/L1(w))^2)` and describes how
  many final coefficients carry mass.  The approximately 9.69 value reported
  by the first token experiment was this weight quantity, not covariance PR.

Each fit also records the complete eigenvalue spectrum, covariance condition
number, L-SML group assignment, selected group count, residual, coefficients,
and degeneracy flag.

## Interpretation

If covariance PR falls materially from global to cell-specific fits, the
global effective dimension was partly produced by model/domain mixture.  If PR
and group partitions remain stable, that supports genuine feature
complementarity.  Performance is reported for L-SML and the same-scope equal
mean so a local covariance improvement cannot be confused with a normalization
change.

The future hierarchical estimator
`C_cell = alpha*C_local + (1-alpha)*C_model_or_global` is explicitly out of
scope.  It will be considered only after this experiment establishes the
sample-size, PR, grouping-stability, and quality behavior of local fits.

## Result

The full matched roster contains 13,769 answers and 145,597 steps.  All 85
source-fold OOF fits completed without a degenerate L-SML solution.

| scope | covariance PR | weight participation | L-SML PB macro-F1 | equal PB macro-F1 | L-SML - equal |
|---|---:|---:|---:|---:|---:|
| global | 4.168 | 9.687 | 34.076% | 32.343% | +1.734pp |
| model | 4.526 | 9.036 | 33.731% | 32.402% | +1.329pp |
| dataset | 4.945 | 8.065 | 33.272% | 32.270% | +1.002pp |
| cell | 5.133 | 7.343 | 33.035% | 32.189% | +0.846pp |

Paired 10,000-draw source-group bootstrap contrasts against the frozen global
L-SML result are:

- model: -0.345pp, 95% CI [-0.708, +0.003]pp;
- dataset: -0.804pp, 95% CI [-1.354, -0.271]pp;
- cell: -1.041pp, 95% CI [-1.619, -0.478]pp.

Every scope still beats its own equal-mean baseline; all four paired bootstrap
intervals for L-SML minus equal are positive.  Therefore the local fits do not
make L-SML invalid, but they progressively reduce its advantage.

The requested mechanism test is decisive in direction: covariance PR rises,
rather than collapses, as pooling becomes more local.  The global effective
dimension is therefore not an artifact created by mixing model and dataset
populations.  At the same time, coefficient participation falls and the
selected group count becomes less stable (`K=3/4/6` locally versus fixed
`K=6` globally).  The local covariance has more spectral dimensions but gives
up useful cross-domain statistical strength when converted into L-SML groups
and weights.

Decision: retain the frozen global fit as the best current ProcessBench
contract.  Model-specific pooling is the closest local alternative but has no
matched gain.  The observed high local covariance PR plus degraded local
weights is a concrete reason to test the deferred hierarchical estimator next,
not to promote cell-specific fitting directly.

Canonical artifacts:

- `results/claude_feature_bank_pooling_scope_v1/RESULTS.json`
- `results/claude_feature_bank_pooling_scope_v1/OOF_SCORES.npz`
- `results/claude_feature_bank_pooling_scope_v1/AUDIT.json`
