# Direct DUFS explicit-length-drop ablation v1

**Decision:** `EXPLICIT_LENGTH_NOT_SOLE_NUISANCE_CHANNEL`

| Lane | Condition | Target effect | Held-out length effect | Target > length |
|---|---|---:|---:|---:|
| global24 | original | 18.2% | 81.0% | 0% |
| global24 | drop_length_fixed_gates | 18.2% | 63.6% | 5% |
| global24 | drop_length_refit_gates | 18.3% | 65.4% | 10% |
| processbench | original | 26.2% | 63.0% | 0% |
| processbench | drop_length_fixed_gates | 25.9% | 53.6% | 0% |
| processbench | drop_length_refit_gates | 25.7% | 53.6% | 0% |
| ragtruth | original | 31.3% | 94.0% | 0% |
| ragtruth | drop_length_fixed_gates | 30.4% | 87.7% | 0% |
| ragtruth | drop_length_refit_gates | 30.6% | 80.8% | 0% |

The held-out length variable is never used to construct a no-length graph. Residual length smoothness therefore measures indirect length information in the remaining features.

Global, ProcessBench, and RAGTruth remain separate estimands.
