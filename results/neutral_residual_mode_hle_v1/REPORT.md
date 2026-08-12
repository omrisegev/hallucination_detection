# Frozen NRM-CS-IU confirmation on HLE/Qwen2.5-72B

**Decision: FAIL.** NRM changed correctness AUROC by **+0.345pp** versus IU; the paired stratified 95% interval is [-0.898, +1.628]pp.

| method | AUROC | AUPRC |
|---|---:|---:|
| `iu` | 0.516775 | 0.034325 |
| `nrm` | 0.520229 | 0.034886 |
| `cardinality` | 0.505819 | 0.033403 |

## Pre-registered gates

- **PASS — frozen score/source hashes verify**
- **PASS — telemetry-only fit payload**
- **PASS — numerical invariants**
- **PASS — positive HLE point delta**
- **FAIL — positive paired 95% lower bound**

## Answer-type diagnostics

| answer type | N | correct | IU AUROC | NRM AUROC | delta |
|---|---:|---:|---:|---:|---:|
| exactMatch | 1645 | 31 | 0.564796 | 0.553983 | -1.081pp |
| multipleChoice | 513 | 37 | 0.546957 | 0.548433 | +0.148pp |

## Limitation

HLE has only 68 judged-correct answers here.  Labels come from one interim Codex judge rather than HLE's original GPT-4o protocol, so this is an independent-example/model confirmation under the stated judge, not a paper-faithful HLE result.
