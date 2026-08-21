# Direct DUFS conditional graph-topology audit v1

**Decision:** `CONTROL_FAILURE_INVALIDATES_GEOMETRY_AUDIT`

This retrospective closure audit requires agreement between exact-length swaps and a cross-fitted flexible propensity CRT. Raw smoothness, coarse length bins, and a single tie resolution cannot establish a hallucination manifold.

## Candidate decisions

| Graph | Conditional geometry in all lanes | Detector utility | Joint pass |
|---|---:|---:|---:|
| union_knn_k7_self_safe | False | False | False |
| radius_edge_matched_k7 | False | False | False |
| adaptive_knn_mean7_k3_25 | False | False | False |
| diffusion_edge_matched_base25_t2 | False | False | False |

## Original-representation validation summary (worst tie seed)

| Lane | Graph | Raw effect | Exact effect | Exact sig. | CRT effect | CRT sig. | Healthy | LIU ΔAUROC |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| global24 | union_knn_k7_self_safe | 18.2% | 14.5% | 89% | 15.7% | 90% | 86% | +0.0002 |
| global24 | radius_edge_matched_k7 | 10.4% | 6.7% | 74% | 7.6% | 75% | 0% | -0.0001 |
| global24 | adaptive_knn_mean7_k3_25 | 19.3% | 14.8% | 89% | 16.5% | 85% | 81% | +0.0001 |
| global24 | diffusion_edge_matched_base25_t2 | 17.7% | 14.5% | 89% | 15.0% | 85% | 81% | +0.0001 |
| global24 | diffusion_edge_matched_base25_t4 | 16.5% | 13.9% | 89% | 14.8% | 85% | 81% | +0.0001 |
| global24 | deployed_union_knn_k7 | 18.3% | 14.5% | 0% | 15.7% | 0% | 86% | +0.0002 |
| global24 | mutual_knn_k7 | 19.9% | 13.6% | 0% | 16.2% | 0% | 62% | +0.0001 |
| global24 | length_only_knn_k7 | 3.3% | -0.2% | 0% | -1.4% | 0% | 29% | -0.0003 |
| global24 | permuted_self_safe_union_knn_k7 | -0.7% | -0.4% | 0% | -0.5% | 0% | 86% | -0.0000 |
| processbench | union_knn_k7_self_safe | 26.4% | 15.0% | 100% | 23.0% | 100% | 100% | +0.0021 |
| processbench | radius_edge_matched_k7 | 20.9% | 12.0% | 100% | 20.7% | 100% | 0% | +0.0004 |
| processbench | adaptive_knn_mean7_k3_25 | 25.6% | 15.3% | 100% | 22.6% | 100% | 100% | +0.0021 |
| processbench | diffusion_edge_matched_base25_t2 | 25.0% | 14.4% | 100% | 22.3% | 100% | 100% | +0.0019 |
| processbench | diffusion_edge_matched_base25_t4 | 25.2% | 14.6% | 100% | 22.6% | 100% | 100% | +0.0017 |
| processbench | deployed_union_knn_k7 | 26.3% | 15.0% | 0% | 22.9% | 0% | 100% | +0.0021 |
| processbench | mutual_knn_k7 | 26.3% | 15.4% | 0% | 23.4% | 0% | 100% | +0.0016 |
| processbench | length_only_knn_k7 | 6.4% | 0.0% | 0% | -3.1% | 0% | 0% | -0.0024 |
| processbench | permuted_self_safe_union_knn_k7 | -0.9% | -0.8% | 0% | -1.0% | 0% | 100% | -0.0001 |
| ragtruth | union_knn_k7_self_safe | 31.4% | 22.5% | 100% | 23.0% | 100% | 100% | +0.0024 |
| ragtruth | radius_edge_matched_k7 | 25.0% | 18.8% | 100% | 19.0% | 100% | 0% | +0.0012 |
| ragtruth | adaptive_knn_mean7_k3_25 | 31.0% | 21.8% | 100% | 22.5% | 100% | 100% | +0.0024 |
| ragtruth | diffusion_edge_matched_base25_t2 | 29.2% | 21.2% | 100% | 21.6% | 100% | 100% | +0.0025 |
| ragtruth | diffusion_edge_matched_base25_t4 | 27.7% | 19.8% | 100% | 20.0% | 100% | 100% | +0.0024 |
| ragtruth | deployed_union_knn_k7 | 31.4% | 22.5% | 0% | 23.0% | 0% | 100% | +0.0024 |
| ragtruth | mutual_knn_k7 | 28.6% | 19.5% | 0% | 19.9% | 0% | 100% | +0.0014 |
| ragtruth | length_only_knn_k7 | 12.6% | -0.4% | 0% | -0.5% | 0% | 0% | -0.0024 |
| ragtruth | permuted_self_safe_union_knn_k7 | -1.2% | -0.7% | 0% | -1.2% | 0% | 100% | -0.0000 |

## Controls

- Overall fail-closed control gate: False.
- Length-only graph: median length effect 71.7%; exact/CRT false-positive fractions 8.3%/5.6%.
- Permuted-union raw false-positive fraction: 3.8%.
- Exact edge budgets: True; radius boundary proof: True; adaptive mean-k: True; deployed-score reproduction: True.

## Interpretation

At least one predeclared positive, negative, eligibility, construction, or replay control failed. Geometry outcomes are therefore invalidated rather than interpreted.

Global answer hallucination, ProcessBench process error, and RAGTruth response hallucination remain separate estimands. See the machine-readable tables for the drop-length and train-residualized robustness arms.
