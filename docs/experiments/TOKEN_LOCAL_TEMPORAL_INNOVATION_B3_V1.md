# Token-local temporal innovation B3 v1

**Status:** implemented as a target-free Phase-2 scorer and synthetic/mechanical
test lane.  No ProcessBench or PRMBench targets are opened by the fit runner.
Real-cell score freezes require a prepared localization release and an
independent pre-label audit.

**Protocol ID:** `TOKEN_LOCAL_TEMPORAL_INNOVATION_B3_V1`

## Scope

The experiment holds the equal-30 response head and all Phase-1 token
preparation/reducer contracts fixed.  It tests (a) token-native continuous B3
and (b) whether causal temporal innovation residuals add target-free nuisance
information beyond B3.  The innovation support is called a **predictive
nuisance-support graph**; it is not a causal or Granger graph.

## Causal innovation contract

The nine core streams are the Cartesian product of entropy, sampled-token
surprisal (`spilled`), and partition energy with level, sliding variance, and
absolute CUSUM.  For target `j` and token `t > 0`:

```text
xhat[t,j] = a[j] + c[j] log(1+t) + beta_self[j] x[t-1,j]
            + sum(beta_cross[j,i] x[t-1,i] for i in S[j])
u[t,j] = (x[t,j] - xhat[t,j]) / donor_RMS[j]
```

The self lag and log-time term are always present.  Optional `S[j]` contains
four lagged rook peers (same source or same operator) or a fixed four-peer
non-rook control.  No contemporaneous or future value is ever included.  The
first token has an exactly zero innovation and a hidden innovation branch.
Coefficients use `ridge=1.0`; scales are donor-question residual RMS values.

## Static stochastic support

There are 36 optional gates (four per target).  The Projected-STG implementation
uses clipped Gaussian gates with `sigma=0.5`, `mu=0.5`, 120 epochs, and the
registered penalty grid `(0.01, 0.03, 0.10, 0.30, 1.00)`.  Gates are static over
an entire trace.  Penalty selection is five-fold held-question MSE with a
one-standard-error rule, choosing the sparsest eligible support.

Support stability resamples whole donor questions, never tokens, for 20
replicates with seeds `2026082800..2026082819`.  An edge must have mean survival
at least `0.75` and cross the threshold in at least 15 replicates.  Because
each target has only four optional peers, all 16 subsets are enumerated as a
sanity audit.  Projected-STG fails closed unless it is within one held-MSE SE
of the optimal subset and uses at most one additional edge.

## Frozen arms

1. `LOCAL_TOKEN_B3` — continuous B3 on all 29 original streams.
2. `LOCAL_TOKEN_B3_SELF_INNOV` — self-lag innovation only.
3. `LOCAL_TOKEN_B3_ROOK_ALL_INNOV` — all four rook peers.
4. `LOCAL_TOKEN_B3_ROOK_PSTG_INNOV` — Projected-STG rook support (primary).
5. `LOCAL_TOKEN_B3_NONROOK_INNOV_CONTROL` — four non-rook peers.

All arms keep the 29 original streams intact.  Three operator subnetworks each
receive three atomic-source residuals; there is no scalar group compression.
The visible quadratic and orientation anchor use only the originals.  The
innovation logit gain is fixed at `1.0`; gain `0` delegates directly to B3 and
must reproduce it exactly.  B3 uses family width 8, 100 epochs, learning rate
`0.001`, MALA delta `0.1`, five MALA steps, replay refresh `0.05`, float64 CPU,
and seeds `(0,1,2,3,4)`.  Token risk is the five-seed posterior mean and the
step reducer is the span maximum.

## Cross-fitting and artifacts

Five question folds are shared by all arms.  Every fold fits on donor
questions and scores held questions using a deterministic cap of at most
60,000 donor tokens.  `scripts/reconstruction_benchmark/run_token_local_temporal_innovation_b3.py`
creates target-free A/B score trees.  Its token-only input loader validates the
complete NPZ roster but never indexes/materializes `response_scores`.

Each result record stores per-seed model states, innovation maps/supports,
fold/bootstrap diagnostics, and health/reconstruction checks.  Every freeze and
record binds the input manifest, environment snapshot/hash, source closure,
and score hashes.  Wall-clock diagnostics are excluded from the canonical
serialized payload so independent A/B runs can be byte-identical.

After the independent audit, `scripts/reconstruction_benchmark/evaluate_token_local_temporal_innovation_b3.py`
may join the Phase-2 step scores to the already frozen equal-global response
head and import targets.  Its preflight verifies both the Phase-2 and Phase-1
certificates before importing the target-bearing evaluator.  The companion
`scripts/reconstruction_benchmark/plot_token_local_temporal_innovation_b3.py`
creates target-free variant/correlation/support figures and, when supplied an
evaluation directory, post-audit F1/CI figures.  Plot files carry a manifest of
the score/evaluation hashes.

## Required gates before labels

- planted sparse-rook recovery: recall at least 80%, FDR at most 20%;
- self-only null has zero cross-channel edges;
- time-shift control loses the prediction advantage;
- future perturbation leaves all scores through token `t` unchanged;
- lag never crosses row boundaries;
- B3 zero-extension is exact/byte-identical to B3;
- additive-logit reconstruction is at most `1e-8`;
- median five-seed Spearman is at least `0.90`;
- all health, determinism, source/environment, and label-firewall checks pass.

Only after the independent pre-label audit may a separate evaluator join frozen
token scores to the equal-global response risk and import ProcessBench or
PRMBench targets.  Promotion thresholds remain those of the Phase-1 protocol;
PRMBench is secondary and cannot rescue a failed ProcessBench primary result.

## Design provenance

The static Gaussian gate follows the STG construction of Yamada et al. (ICML
2020): <https://proceedings.mlr.press/v119/yamada20a.html>.  Projected-STG is
used as a small linear-support optimizer and is checked against exact subset
enumeration: <https://arxiv.org/abs/2110.15960>.  Skeleton-then-pruning is an
architectural inspiration from GRACE, not a causal validity claim:
<https://arxiv.org/abs/2606.23880>.  Whole-question bootstrap follows the
temporal-resampling warning in Debeire et al. (CLeaR 2024):
<https://proceedings.mlr.press/v236/debeire24a.html>.

Dynamic NRI/dNRI, context-dependent gates, history-dependent variance, and
reducer changes are explicitly out of scope for this protocol.
