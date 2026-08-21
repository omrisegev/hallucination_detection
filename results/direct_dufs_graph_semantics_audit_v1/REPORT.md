# Direct DUFS graph-semantics audit v1

This retrospective diagnostic reconstructs the actual DUFS kNN graph and asks what is smooth on it. Lanes are not pooled.

## Validation decisions

| Lane | Decision | Target smoothness effect | Length smoothness effect | Target > length | LIU ΔAUROC |
|---|---|---:|---:|---:|---:|
| global24 | TARGET_ALIGNED_BUT_NUISANCE_DOMINATED | 18.2% | 81.0% | 0% | +0.0001 |
| processbench | TARGET_ALIGNED_BUT_NUISANCE_DOMINATED | 26.2% | 63.0% | 0% | +0.0022 |
| ragtruth | TARGET_ALIGNED_BUT_NUISANCE_DOMINATED | 31.4% | 93.9% | 0% | +0.0024 |

## Interpretation

Positive smoothness z means that neighbours are more similar on that variable than under row permutation. It does not identify the cause of that similarity.

A target-aligned graph is useful to LIU only when the target is smoother than competing nuisances and the resulting Laplacian correction improves ranking. The per-cell tables preserve the cases where those conditions disagree.

Across every validation lane, the target is smoother than chance but never smoother than length in the primary cells. The LIU increment is correspondingly small and target smoothness does not predict a larger increment.

## kNN topology sensitivity

| Lane | Graph | k | Target aligned | Target smoother than length | Median components |
|---|---|---:|---:|---:|---:|
| global24 | union_knn | 7 | 95% | 0% | 1.0 |
| global24 | mutual_knn | 7 | 95% | 0% | 60.0 |
| processbench | union_knn | 7 | 100% | 0% | 1.0 |
| processbench | mutual_knn | 7 | 100% | 0% | 34.5 |
| ragtruth | union_knn | 7 | 100% | 0% | 1.0 |
| ragtruth | mutual_knn | 7 | 100% | 0% | 67.0 |

The ordinary union-kNN conclusion is stable for k in {3,5,7,10,15,25}. Mutual-kNN does not rescue target specificity and fragments the graph into many components at small k.

Global answer hallucination, ProcessBench process-error localization, and RAGTruth response hallucination are different estimands and must remain separate.
