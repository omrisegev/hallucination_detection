# Laplacian IU-PCR synthetic falsification study

Version: `laplacian-upcr-synthetic-v3-2026-08-06`

## Scope and leakage boundary

This phase used generated data only; no hallucination artifact bundle or real
benchmark was read. Development and confirmation were executed as separate
commands. The first command persisted a frozen lambda and source/config hashes;
the second refused mismatches before opening disjoint confirmation seeds.
Labels are evaluation-only. `oracle_latent` is a synthetic mechanism ceiling,
never a deployable method.

## Frozen design

- Development replicates per world: 8
- Confirmation replicates per world: 8
- Samples per replicate: 360; features: 12; k-NN: 7
- Lambda grid: [0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
- Gate optimization: 3 seeds, 120 epochs each
- Choice rule: smallest positive lambda within one SE of the best positive
  smooth-signal development mean; otherwise lambda=0.
- The gate learner is the repository's parameter-free adapted DUFS: its kernel,
  optimizer, and CPU budget differ from paper-faithful DUFS. Its continuous gates
  feed a separate symmetric sparse k-NN graph; no feature is deleted.
- `projected_ridge` is trace-matched isotropic ridge in the identical U2 and
  isolates graph geometry from generic regularization.

## Development choice

| lambda | mean AUROC delta | SE |
|---:|---:|---:|
| 0 | +0.000 pp | 0.000 pp |
| 0.01 | +0.034 pp | 0.014 pp |
| 0.03 | +0.090 pp | 0.041 pp |
| 0.1 | +0.204 pp | 0.106 pp |
| 0.3 | +0.324 pp | 0.172 pp |
| 1 | +0.354 pp | 0.199 pp |
| 3 | +0.390 pp | 0.211 pp |
| 10 | +0.398 pp | 0.216 pp |

Frozen choice: **lambda=0.1**.

## Confirmation result at the frozen lambda

| world | DUFS graph | Ungated graph | Shuffled gates | Permuted graph | Projected ridge | Oracle latent graph |
|---|---:|---:|---:|---:|---:|---:|
| Smooth signal | +0.382 +/- 0.149 | +0.218 +/- 0.086 | +0.255 +/- 0.113 | -0.002 +/- 0.006 | +0.278 +/- 0.113 | +0.374 +/- 0.151 |
| Correlated errors | -0.009 +/- 0.008 | +0.009 +/- 0.010 | -0.000 +/- 0.006 | +0.001 +/- 0.005 | +0.000 +/- 0.002 | +0.035 +/- 0.008 |
| Nuisance manifold | -0.568 +/- 0.049 | -0.161 +/- 0.033 | -0.225 +/- 0.076 | +0.007 +/- 0.020 | -0.222 +/- 0.021 | +2.205 +/- 0.031 |
| Signal outside U2 | -0.050 +/- 0.077 | +0.006 +/- 0.063 | +0.007 +/- 0.068 | -0.006 +/- 0.022 | -0.006 +/- 0.002 | +0.055 +/- 0.113 |
| Disconnected graph | -0.035 +/- 0.008 | -0.034 +/- 0.008 | -0.035 +/- 0.007 | -0.006 +/- 0.003 | -0.008 +/- 0.005 | +0.100 +/- 0.026 |
| Pure noise | +0.010 +/- 0.014 | +0.003 +/- 0.008 | -0.000 +/- 0.020 | -0.003 +/- 0.005 | -0.005 +/- 0.004 | +0.003 +/- 0.008 |

Cells are paired AUROC changes in percentage points +/- one SE relative to
ordinary IU-PCR. Repeated lambda=0 arms are not independent observations.

### Absolute performance and secondary AUPRC

| world | IU AUROC | candidate AUROC | IU AUPRC | candidate AUPRC |
|---|---:|---:|---:|---:|
| Smooth signal | 0.8527 +/- 0.0058 | 0.8565 +/- 0.0049 | 0.8506 +/- 0.0071 | 0.8548 +/- 0.0065 |
| Correlated errors | 0.8840 +/- 0.0075 | 0.8839 +/- 0.0076 | 0.8901 +/- 0.0077 | 0.8900 +/- 0.0077 |
| Nuisance manifold | 0.7044 +/- 0.0100 | 0.6987 +/- 0.0102 | 0.6973 +/- 0.0108 | 0.6914 +/- 0.0109 |
| Signal outside U2 | 0.4964 +/- 0.0112 | 0.4959 +/- 0.0104 | 0.4931 +/- 0.0087 | 0.4927 +/- 0.0083 |
| Disconnected graph | 0.6768 +/- 0.0055 | 0.6764 +/- 0.0055 | 0.7091 +/- 0.0083 | 0.7088 +/- 0.0083 |
| Pure noise | 0.5080 +/- 0.0140 | 0.5081 +/- 0.0140 | 0.5197 +/- 0.0130 | 0.5198 +/- 0.0129 |

### DUFS specificity on the positive control

| comparator | paired advantage | SE | one-sided 95% lower |
|---|---:|---:|---:|
| Ungated graph | +0.164 pp | 0.067 pp | +0.036 pp |
| Shuffled gates | +0.127 pp | 0.044 pp | +0.043 pp |
| Permuted graph | +0.384 pp | 0.148 pp | +0.103 pp |
| Projected ridge | +0.103 pp | 0.036 pp | +0.035 pp |

The paired DUFS-minus-oracle gap was +0.007 +/- 0.008 pp; a negative value is the remaining graph-identification gap.

## Review-revised gates frozen before this rerun

- Algebraic/invariant gate: **PASS**
  (exact-copy score error 0.000e+00;
  unforced equation weight error 1.056e-13;
  minimum R eigenvalue 9.833e-03;
  disconnected connectivity 0.000e+00;
  all Laplacian energy paths monotone: True).
- Positive mechanism and DUFS-specificity gate: **FAIL**.
  It requires nonzero lambda, >0.5 pp mean smooth-signal gain with a positive
  one-sided 95% lower bound, and a positive paired lower bound versus every
  non-oracle control, including projected ridge.
- Graph-identification robustness gate: **FAIL**
  (baseline lower bound 0.6855
  must exceed 0.65; delta lower bound -0.661 pp
  must exceed -0.5 pp, the same magnitude as the minimum meaningful gain).
- Overall Phase-1 gate: **FAIL**.

The `Signal outside U2` world keeps the earlier chance-level construction as a
separate limitation test: a final penalty restricted to fixed U2 cannot recover
correctness signal that ordinary covariance excluded from that subspace.

## Diagnostic artifacts

- `01_confirmation_auroc_paths.png`: full confirmation paths.
- `02_mechanism_diagnostics.png`: target alignment and smoothness.
- `03_frozen_lambda_confirmation.png`: shared-scale frozen comparison.
- `04_gate_graph_diagnostics.png`: stability, connectivity, and graph separation.
- `05_gate_probabilities_by_planted_role.png`: per-role gate identification.
- Raw CSVs retain AUROC/AUPRC, weight diagnostics, conditioning, ordinary additive residual,
  per-feature gates, planted roles, graph diagnostics, and every path point.
- `run_metadata.json` records commands, dependency versions, config, Git HEAD,
  and hashes of the exact uncommitted source files.

## Interpretation discipline

Passing this phase establishes only a synthetic mechanism and failure boundary.
It does not establish improvement on hallucination detection. Phase 2 remains
closed until we discuss these results; later internal validation is not a pristine
publication test, and genuinely unseen external data remains necessary.
