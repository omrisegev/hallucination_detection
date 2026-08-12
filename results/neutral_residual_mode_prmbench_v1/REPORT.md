# Frozen NRM-CS-IU confirmation on PRMBench/Qwen3-8B

**Decision: PASS.** NRM changed response-level correctness AUROC by **+0.460pp** versus IU; the paired source-group 95% interval is [+0.068, +0.841]pp.

| method | AUROC | AUPRC |
|---|---:|---:|
| `iu` | 0.720602 | 0.220771 |
| `nrm` | 0.725206 | 0.228811 |
| `cardinality` | 0.711966 | 0.222382 |

## Pre-registered gates

- **PASS — frozen score/source hashes verify**
- **PASS — telemetry-only fit payload**
- **PASS — numerical invariants**
- **PASS — positive overall PRMBench point delta**
- **PASS — positive source-group 95% lower bound**

## Error-class diagnostics

| error class vs correct | error N | IU AUROC | NRM AUROC | delta |
|---|---:|---:|---:|---:|
| circular | 758 | 0.835717 | 0.831032 | -0.469pp |
| confidence | 756 | 0.731089 | 0.730684 | -0.040pp |
| counterfactual | 757 | 0.701640 | 0.706974 | +0.533pp |
| deception | 749 | 0.629026 | 0.632769 | +0.374pp |
| domain_inconsistency | 757 | 0.801288 | 0.804650 | +0.336pp |
| missing_condition | 756 | 0.753541 | 0.750389 | -0.315pp |
| multi_solutions | 160 | 0.213696 | 0.264100 | +5.040pp |
| redundency | 758 | 0.707777 | 0.720559 | +1.278pp |
| step_contradiction | 757 | 0.710831 | 0.720981 | +1.015pp |

## Scope

This is a response-level correct-versus-error adaptation.  It is not PRMBench's official step-level metric.  Exactly the three readiness-identified alignment defects were excluded before scoring.
