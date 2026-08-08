# Gray-box cross-view Phase 1 — conclusion

## Decision

**Stop the registered cross-view candidate. Do not run confirmation, implement
the Phase-2 trajectory bank, or open real hallucination data for this method.**

The experiment is internally valid but fails the scientific gates. Cross-view
transfer is a useful detector of pure-noise and measured-nuisance graphs; it is
not a target-relevance test and does not make Laplacian IU-PCR safe.

## Frozen primary result

All values are paired AUROC changes versus ordinary IU-PCR at the preregistered
`k=7`, `lambda=0.1`, over eight independent datasets per world.

| world | hard-veto consensus | direct G | direct A | mmDUFS-inspired | projected ridge |
|---|---:|---:|---:|---:|---:|
| aligned target | +0.377 +/- 0.151pp | +0.395 | +0.342 | +0.359 | +0.335 |
| discovery-specific nuisance | **-1.387 +/- 0.407pp** | -2.211 | **+1.850** | +0.196 | -0.185 |
| measured shared nuisance | 0.000pp (fallback 8/8) | -2.232 | -2.171 | -2.288 | -0.204 |
| pure noise | 0.000pp (fallback 8/8) | -0.001 | +0.001 | -0.001 | -0.008 |
| unmeasured shared nuisance | **-2.307 +/- 0.122pp** | -2.323 | -2.242 | -2.304 | -0.215 |

The aligned gain has a positive one-sided lower bound (+0.090pp), but misses the
registered +0.5pp meaningful-effect threshold and is not separated from
projected ridge: the paired candidate-minus-ridge lower bound is -0.008pp.

## The decisive failure

P1-B produces the wrong directional decision:

- the nuisance-dominated G graph is harmful (-2.211pp), but G->A passes the
  audit in 5/8 datasets and is therefore used;
- the target-dominated A graph is strongly beneficial (+1.850pp), but A->G is
  rejected in 8/8 datasets because the held-out G view is dominated by
  nuisance coordinates.

Thus the transfer statistic answers “does this graph contain some structure
that is visible in the other view?” It does not answer “is the shared structure
dominant, target-relevant, or safe to regularize toward?” A weak shared target
component can validate a graph whose dominant geometry is harmful.

P1-F establishes the stronger identifiability failure. When the same unmeasured
nuisance appears cleanly in both views, both directions pass in 8/8 datasets and
the loss is -2.307pp. The mmDUFS-inspired shared operator loses -2.304pp, showing
that extracting shared structure more directly does not fix the semantic
ambiguity.

## Why tuning lambda does not rescue it

| lambda | aligned target | discovery nuisance | unmeasured nuisance |
|---:|---:|---:|---:|
| 0.01 | +0.062pp | -0.150pp | -0.247pp |
| 0.03 | +0.164pp | -0.452pp | -0.758pp |
| 0.10 | +0.377pp | -1.387pp | -2.307pp |
| 0.30 | +0.637pp | -3.480pp | -5.731pp |
| 1.00 | +0.809pp | -6.785pp | -11.330pp |

The positive effect and nuisance harm are the same regularization-strength
tradeoff. No diagnostic lambda simultaneously reaches the +0.5pp positive gate
and the nuisance-safety gates. Selecting a smaller lambda would reduce both the
claimed mechanism and the harm; it would not identify the target manifold.

## What did work

- Algebra and leakage invariants pass exactly; fallback reproduces IU-PCR with
  zero score error.
- Aligned structure passes 8/8 and audit-row permutation destroys transfer in
  16/16 directional tests.
- Pure noise is rejected 8/8.
- The measured shared nuisance is recognized and vetoed 8/8 after cross-fitted
  nuisance residualization.
- An unconditional veto on significant nuisance CKA would also reject the
  P1-B G graph, but it cannot detect P1-F because the nuisance is unmeasured.

These are useful diagnostic results, but they do not satisfy the safety claim.

## Research recommendation

The evidence now points away from adding more zero-label probability views or
more elaborate shared-manifold operators. Entropy and realized-token
surprisal are linked by the sampling process, so a richer Phase-2 representation
would strengthen shared nuisance geometry as readily as correctness geometry.

The next primary direction should use a small amount of target information to
identify semantics directly. The repository already supplies two reasons:

1. target-anchored gates prove that a few labels can select the correct planted
   block, but U2 logistic used the same labels much more effectively
   (`+19.523pp` versus `+1.267pp` on target g);
2. the few-label stability-selected subset channel has measured room and
   recovered roughly 84% of it in the earlier split-half study.

The recommended next experiment is therefore a current-schema, recycling-safe
few-label comparison of:

- ordinary IU-PCR;
- U2 logistic;
- stability-selected feature-subset adaptation;
- their simplest predeclared combination.

Use nested budgets such as 4, 8, 16, and 32 labels and test whether the subset is
stable across calibration draws and cells. This addresses the demonstrated
bottleneck—target identification—rather than building a more accurate model of
an unlabeled but semantically ambiguous manifold.

A secondary zero-label branch is defensible only as a conservative trust-region
method that caps score/rank displacement from IU-PCR. It should be framed as
harm limitation, not target-manifold identification, and must first beat P1-F.

## Artifacts

- `DEVELOPMENT_REPORT.md`: frozen tables and gate results.
- `development_gate_decisions.json`: machine-readable gate record.
- `development_per_run.csv`: every arm and lambda.
- `development_audit_diagnostics.csv`: transfer, nuisance, residual, stability,
  and decision diagnostics.
- `development_figures/`: decision funnel, lambda paths, transfer/nuisance,
  stability, comparison, and evidence-convergence plots.

Confirmation seeds remain unopened.
