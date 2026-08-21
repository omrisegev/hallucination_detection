# Direct DUFS train-fitted length residualization v1

**Decision:** `TRAIN_FITTED_RESIDUALIZATION_DOES_NOT_REMOVE_LENGTH_GEOMETRY`

| Lane | Condition | Target effect | Length effect | Target > length | Median |rho(feature,length)| | IU AUROC | DUFS-LIU AUROC |
|---|---|---:|---:|---:|---:|---:|---:|
| global24 | original | 18.2% | 81.0% | 0% | 0.389 | 0.7837 | 0.7835 |
| global24 | drop_length_refit_gates | 18.3% | 65.5% | 10% | 0.342 | 0.7787 | 0.7786 |
| global24 | train_residualized_refit_gates | 16.0% | 53.6% | 5% | 0.217 | 0.7188 | 0.7210 |
| processbench | original | 26.2% | 63.0% | 0% | 0.224 | 0.7880 | 0.7894 |
| processbench | drop_length_refit_gates | 25.7% | 53.5% | 0% | 0.218 | 0.7854 | 0.7862 |
| processbench | train_residualized_refit_gates | 22.3% | 37.2% | 0% | 0.099 | 0.7344 | 0.7370 |
| ragtruth | original | 31.4% | 93.9% | 0% | 0.298 | 0.7605 | 0.7629 |
| ragtruth | drop_length_refit_gates | 30.6% | 80.8% | 0% | 0.294 | 0.7605 | 0.7625 |
| ragtruth | train_residualized_refit_gates | 25.4% | 33.7% | 0% | 0.071 | 0.6950 | 0.6985 |

## Length reduction relative to dropping the explicit coordinate

- global24: 18.1%
- processbench: 30.5%
- ragtruth: 58.3%

Residualizer coefficients were fit without labels and only on the registered training cells/split. Held-out length was used after graph construction for the nuisance audit.

The residualizer reduces feature/length dependence in every lane, but target smoothness remains below length smoothness in every validation lane. Target smoothness and target-ranking performance also decline, so this transform does not reveal a hallucination-specific manifold.

Global, ProcessBench, and RAGTruth remain separate estimands.
