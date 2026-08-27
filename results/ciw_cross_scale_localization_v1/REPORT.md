# CIW cross-scale localization v1

The target-free input layer predicts each of 29 token coordinates from its
whole-answer mean and frozen CIW-DEEM answer risk.  A bounded OOF-R2 gate mixes
the original coordinate with its standardized local innovation before the
token fusion head.

| Arm | ProcessBench macro F1 | PRMBench step AUROC | PRMBench step AUPRC |
|---|---:|---:|---:|
| CIW cross-scale + token IU-PCR | 0.308301 | 0.582489 | 0.196327 |
| CIW cross-scale + token SU-PCR | 0.306202 | 0.582516 | 0.196347 |
| B3 incumbent | 0.310228 | 0.584218 | 0.197104 |
| IU-PCR incumbent | 0.308194 | 0.598834 | 0.208690 |
| DUFS-LIU incumbent | 0.309731 | 0.600431 | 0.209774 |

Decision: no promotion.  The IU input layer improves PRMBench by about
`+0.0013` AUROC when applied retrospectively with each of B3, IU-PCR, and
DUFS-LIU response heads, but it does not transfer to ProcessBench first-error
localization.  See
`docs/experiments/CIW_CROSS_SCALE_LOCALIZATION_V1.md` for the complete method,
boundary, and post-opening diagnostic.
