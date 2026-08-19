# Exact-row white-box versus gray-box comparison

**Status: POSTHOC / PRELIMINARY / WHITE VALIDATION BLOCKED.**

This comparison uses the exact intersection of **31,440 candidates** in the same 13 dataset/model cells. Hallucination is the positive class (`1 = incorrect`) for both methods. The historical gray-box AUPRC is not reused because it treated correctness as positive.

| Method | Macro AUROC | Hallucination AUPRC |
|---|---:|---:|
| White pure distributed-depth U-PCR | 0.781690 | 0.677048 |
| Gray mixed-v2 DUFS-LIU | 0.782994 | 0.687731 |
| Gray mixed-v2 deployed-U-PCR control | 0.780940 | 0.686236 |
| Exploratory equal-z white + gray | 0.790203 | 0.690580 |

## Decision

White-box alone has no reliable aggregate performance advantage over the final gray-box system: AUROC delta -0.001304, 95% equal-cell bootstrap interval [-0.016931, +0.012300]; AUPRC delta -0.010683 [-0.035363, +0.011078]. The same-U-PCR control is also a practical tie.

White-box does have a coverage advantage: 42,238 scorable candidates versus 31,467 under the gray 30-feature complete-case contract. The exploratory two-score average gains +0.007209 AUROC, with interval [+0.000101, +0.014105]. Its lower bound is close to zero and the comparison is post-hoc, so it is not a promoted result.

Mean per-cell Spearman correlation between final risk scores is 0.8677; most information is shared, with bounded complementary signal.

## Leakage and claim boundary

- Fit reconstructed row availability and produced scores without reading correctness labels.
- Score bundles were hashed before evaluation opened labels.
- Candidate IDs and gray raw/bundle label order match exactly.
- The comparison and hybrid were proposed after observing both component studies; all inference is descriptive.
- The white capture still lacks corrected live Gate B and the architecture-fidelity pilot.
