# Automatic group-free IU — Phase A4

## Outcome

- Shared-repeatability premise: **CLOSE_SHARED_REPEATABLE_COMPONENT_PREMISE**
- Detector/target verdict: **CLOSE_NO_TARGET_CONTRAST**
- Correctness or step labels accessed: **no**

The primary residualized CorrCA component has Qwen repeatability
**0.9979** and one-Llama external structural correlation
**0.9555**. The preselected strongest paired baseline has
Llama correlation **0.9669**. That baseline was `single:1`, which is
`trace_length`, in all five outer folds;
the observed delta is **-0.0114**
with 95% interval **[-0.0160,
-0.0090]**.

The training-pair null 95th percentile is **0.4830**
and the held-Llama conditional-pair null 95th percentile is
**0.4237**. Minimum outer-fold loading squared
cosine is **0.9992**.

## Gates

| gate | pass |
|---|---:|
| material_delta_vs_preselected_paired_baseline | FAIL |
| positive_every_subset_and_macro_intervals | PASS |
| feature_level_text_length_confound_control | PASS (formal; invalidated for a non-length claim) |
| conditional_pair_nulls | PASS |
| leave_one_subset_transfer | PASS |
| outer_loading_stability | PASS |

## Interpretation boundary

Under the frozen residualization, CorrCA recovered a stable coordinate that
was significantly worse than `trace_length` alone. A post-held adversarial
diagnostic found that `trace_length` is exact generated-token count, the two
Qwen views have identical counts for all 3,400 responses, and the CorrCA
loading on this coordinate is 0.9979--0.9993 across folds. Trace-only gives
Qwen/Llama correlations 0.999999/0.9669. Removing only the trace-length term
from each frozen loading, without refitting or reselection, lowers them to
0.9900/0.8667.

The nuisance design modeled log-count terms and their squares with ridge
shrinkage; it did not eliminate the standardized linear token-count feature.
Consequently, the formally passing confound gate re-tests the same restricted
nuisance basis, and the length-decile pair null destroys fine-grained exact
length within its coarse strata. Neither supports a non-length interpretation.
The high repeatability, null margins, stability, and leave-one-subset results
therefore describe a coordinate dominated by incompletely removed trace
length, not a content-independent shared telemetry mechanism. The ablation
shows residual shared structure, but it is a post-held diagnostic rather than
a registered candidate and cannot rescue A4.

The exact frozen verdicts remain
`CLOSE_SHARED_REPEATABLE_COMPONENT_PREMISE` and
`CLOSE_NO_TARGET_CONTRAST`. This experiment does not identify a complementary
scorer-sensitive component, and neither shared nor residual variation is
identified as hallucination. Because the fixed responses contain no legal
target-changing contrast, the detector closure holds regardless of the
structural result. The diagnostic is preserved in
`POST_HELD_TRACE_LENGTH_DIAGNOSTIC.json`; A5 begins next.
