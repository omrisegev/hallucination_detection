# Gray-box cross-view graph audit — development report

Version: `graybox-cross-view-phase1-v1-2026-08-06`

## Scope

Generated data only. The cross-view method was called once per dataset and
received only G, A, and N. Evaluator labels and oracle latents were joined
after method scores and hashes were frozen. Confirmation was not opened.

## Frozen design

- Dataset replicates per world: 8
- Samples per replicate: 360
- k path: [5, 7, 11]; primary k=7
- lambda path: [0, 0.01, 0.03, 0.1, 0.3, 1.0]; primary lambda=0.1
- 199 synchronized node permutations per graph
- transfer: p<=0.025 and robust Z>=2.0
- stability: identical decision across k and median affinity CKA>=0.75

## Audit decisions

| world | raw transfer (datasets) | final accepted |
|---|---:|---:|
| Aligned target | 8/8 | 8/8 |
| Discovery nuisance | 5/8 | 5/8 |
| Measured shared nuisance | 8/8 | 0/8 |
| Paired targets | 8/8 | 8/8 |
| Pure noise | 0/8 | 0/8 |
| Unmeasured shared nuisance | 8/8 | 8/8 |

## Primary performance at lambda=0.1

Values are mean paired AUROC changes in percentage points +/- one SE versus ordinary IU-PCR.

| world | consensus | direct G | direct A | mmDUFS-inspired | ridge |
|---|---:|---:|---:|---:|---:|
| Aligned target | +0.377 +/- 0.151 | +0.395 +/- 0.162 | +0.342 +/- 0.136 | +0.359 +/- 0.144 | +0.335 +/- 0.128 |
| Discovery nuisance | -1.387 +/- 0.407 | -2.211 +/- 0.046 | +1.850 +/- 0.040 | +0.196 +/- 0.021 | -0.185 +/- 0.018 |
| Measured shared nuisance | +0.000 +/- 0.000 | -2.232 +/- 0.088 | -2.171 +/- 0.079 | -2.288 +/- 0.083 | -0.204 +/- 0.015 |
| Paired targets | +0.211 +/- 0.081 | +0.228 +/- 0.089 | +0.188 +/- 0.076 | +0.196 +/- 0.079 | +0.177 +/- 0.076 |
| Pure noise | +0.000 +/- 0.000 | -0.001 +/- 0.002 | +0.001 +/- 0.003 | -0.001 +/- 0.003 | -0.008 +/- 0.007 |
| Unmeasured shared nuisance | -2.307 +/- 0.122 | -2.323 +/- 0.116 | -2.242 +/- 0.124 | -2.304 +/- 0.107 | -0.215 +/- 0.030 |

## Frozen gates

| gate | result |
|---|---:|
| `gate0_algebra_implementation` | **PASS** |
| `gate1_audit_premise` | **FAIL** |
| `gate2_positive_mechanism` | **FAIL** |
| `gate3_nuisance_safety` | **FAIL** |
| `gate4_attribution` | **FAIL** |
| `gate6_null_missingness_safety` | **PASS** |
| `overall_phase1_pass` | **FAIL** |

### Decisive diagnostics

- P1-A consensus acceptance: 8/8.
- P1-B nuisance G->A acceptance: 5/8.
- P1-C raw transfer/fallback: 8/8 / 8/8.
- P1-E consensus acceptance: 0/8.
- P1-F raw transfer: 8/8.
- P1-A mean/lower delta: +0.377 / +0.090 pp.
- P1-F mean/lower delta: -2.307 / -2.539 pp.
- Lambda=0 exact score error: 0.000e+00.
- Minimum roughness eigenvalue: 0.000e+00.

## Decision

At least one essential Phase-1 gate failed. Do not implement the trajectory
bank or group-gated DUFS as a rescue. Interpret the failed mechanism first.

## Figures

- `figures/01_decision_funnel.png`
- `figures/02_lambda_paths.png`
- `figures/03_transfer_vs_nuisance.png`
- `figures/04_graph_stability.png`
- `figures/05_primary_comparison.png`
- `figures/06_evidence_convergence.png`
