# Data Readiness Report

**Date:** 2026-08-11

**Schema:** `data-readiness-v1-2026-08-11`

**Registry fingerprint:** `a219757cacf901e88ed3a0f3e271414335fc0e9d00c6a2c6e679a87ba87a1c6e`

## Purpose

This report validates the collected data before any hallucination method is run. It contains no U-PCR, DUFS, detection, or localization result. Raw artifacts were read but not changed.

A package is **READY** when its expected files, rows, labels and structural alignment passed. **READY_WITH_LIMITATIONS** means evaluation is allowed only within the stated scope and with the caveat attached; it does not mean the package is unusable. **INCOMPLETE** means files are missing. **BLOCKED** means a required label or integrity condition is invalid.

## Headline

- 15 packages were inspected.
- 15 are structurally loadable now (9 READY and 6 READY_WITH_LIMITATIONS).
- 0 package(s) are incomplete and 0 package(s) are blocked.
- Class balance was measured, not modified. No example was resampled or deleted.
- Review queues were prepared under ignored `local_cache`; protected benchmark text is not copied into this report.

## Readiness matrix

| Dataset package | Kind | Count | Balance | Status |
|---|---|---:|---|---|
| `frozen_24cell` | answer-level feature matrices | 48,607 | positive: 21,940, negative: 26,667, positive_rate: 45.1% | **READY_WITH_LIMITATIONS** |
| `semgrad_sciq` | answer-level generation telemetry | 1,000 | positive: 612, negative: 388, positive_rate: 61.2% | **READY** |
| `semgrad_truthfulqa` | answer-level generation telemetry | 817 | positive: 315, negative: 502, positive_rate: 38.6% | **READY** |
| `hle_qwen72b` | answer-level generation telemetry | 2,158 | correct: 68, incorrect: 2,090, accuracy: 3.2% | **READY_WITH_LIMITATIONS** |
| `ragtruth_full_evidence` | RAG evidence-condition telemetry | 2,700 | hallucinated: 943, clean: 1,757, hallucinated_rate: 34.9% | **READY_WITH_LIMITATIONS** |
| `gasp_ragtruth_400` | RAG evidence-condition telemetry | 400 | hallucinated: 200, clean: 200, hallucinated_rate: 50.0% | **READY_WITH_LIMITATIONS** |
| `refchecker_claims` | claim-level evidence telemetry | 10,733 | Contradiction: 935, Entailment: 7,176, Neutral: 2,622 | **READY_WITH_LIMITATIONS** |
| `processbench_qwen3_4b` | first-error benchmark telemetry | 3,400 | first_error_present: 2,221, fully_correct: 1,179, error_rate: 65.3% | **READY** |
| `processbench_qwen3_8b` | first-error benchmark telemetry | 3,400 | first_error_present: 2,221, fully_correct: 1,179, error_rate: 65.3% | **READY** |
| `processbench_qwen25math7b_predictions` | competitor prediction cache | 3,400 | — | **READY** |
| `processbench_qwen3_8b_judge_control` | competitor prediction cache | 3,400 | — | **READY** |
| `processbench_qwen72b_critic` | competitor prediction cache | 3,400 | — | **READY** |
| `prmbench_qwen3_8b_telemetry` | every-step benchmark telemetry | 6,969 | — | **READY_WITH_LIMITATIONS** |
| `prmbench_qwen25math7b_predictions` | competitor prediction cache | 6,969 | — | **READY** |
| `ragtruth_lettucedetect_predictions` | competitor span predictions | 2,700 | hallucinated: 943, clean: 1,757, hallucinated_rate: 34.9% | **READY** |

## Can these packages be used for evaluation?

Yes. Both READY and READY_WITH_LIMITATIONS packages can support evaluation. The latter must be used only for the claim stated below; INCOMPLETE and BLOCKED packages cannot be used yet.

| Package | Evaluation use |
|---|---|
| `frozen_24cell` | Yes — development/exploratory evaluation only; not an independent confirmation set. |
| `semgrad_sciq` | Yes — use under its frozen benchmark protocol. |
| `semgrad_truthfulqa` | Yes — use under its frozen benchmark protocol. |
| `hle_qwen72b` | Yes — interim evaluation only; not a paper-faithful HLE score until the original GPT-4o judge is run. |
| `ragtruth_full_evidence` | Yes — exploratory evaluation only because these labels were already opened. |
| `gasp_ragtruth_400` | Yes — protocol-level comparison only; the paper's exact 400 IDs and splitter are unavailable. |
| `refchecker_claims` | Yes — fixed-claim verification only; do not claim claim-extraction performance. |
| `processbench_qwen3_4b` | Yes — use under its frozen benchmark protocol. |
| `processbench_qwen3_8b` | Yes — use under its frozen benchmark protocol. |
| `processbench_qwen25math7b_predictions` | Yes — use under its frozen benchmark protocol. |
| `processbench_qwen3_8b_judge_control` | Yes — use under its frozen benchmark protocol. |
| `processbench_qwen72b_critic` | Yes — use under its frozen benchmark protocol. |
| `prmbench_qwen3_8b_telemetry` | Conditionally — first resolve or explicitly exclude the three identified alignment defects. |
| `prmbench_qwen25math7b_predictions` | Yes — use under its frozen benchmark protocol. |
| `ragtruth_lettucedetect_predictions` | Yes — use under its frozen benchmark protocol. |

## Required data work

1. **PRMBench Qwen3-8B telemetry:**
   - Failed checks: step_alignment.

## Dataset details

### Frozen 24-cell development collection

- **Package ID:** `frozen_24cell`
- **Status:** READY_WITH_LIMITATIONS
- **Observed:** {"cells": 24, "features_max": 30, "features_min": 19, "rows": 48607}
- **Balance:** positive: 21,940, negative: 26,667, positive_rate: 45.1%
- **Checks passed:** 5/5
- **Limitations:** The cells use heterogeneous datasets and graders and were repeatedly used during method development.

### SemGrad SciQ with BEM labels

- **Package ID:** `semgrad_sciq`
- **Status:** READY
- **Observed:** {"duplicate_question_texts": 0, "rows": 1000, "trace_failures": 0}
- **Balance:** positive: 612, negative: 388, positive_rate: 61.2%
- **Checks passed:** 3/3

### SemGrad TruthfulQA with BEM labels

- **Package ID:** `semgrad_truthfulqa`
- **Status:** READY
- **Observed:** {"duplicate_question_texts": 0, "rows": 817, "trace_failures": 0}
- **Balance:** positive: 315, negative: 502, positive_rate: 38.6%
- **Checks passed:** 3/3

### Humanity's Last Exam Qwen2.5-72B

- **Package ID:** `hle_qwen72b`
- **Status:** READY_WITH_LIMITATIONS
- **Observed:** {"duplicate_question_texts": 0, "interim_judge": {"accuracy": 0.03151065801668211, "agreement_with_provisional_rouge": 0.9304911955514366, "by_answer_type": {"exactMatch": {"accuracy": 0.01884498480243161, "correct": 31, "total": 1645}, "multipleChoice": {"accuracy": 0.07212475633528265, "correct": 37, "total": 513}}, "correct": 68, "incorrect": 2090, "judge_correct_rouge_incorrect": 27, "judge_incorrect_rouge_correct": 123, "rows": 2158}, "interim_judge_manifest": "results/data_readiness_2026_08_11/hle_codex_5p6_sol_xhigh_manifest.json", "rows": 2158, "trace_failures": 0}
- **Balance:** correct: 68, incorrect: 2,090, accuracy: 3.2%
- **Checks passed:** 16/16
- **Limitations:** The complete labels come from an interim gpt-5.6-sol xhigh Codex judge, not the original paper's GPT-4o judge. Preserve both label sets when paper-faithful grading becomes available.

### RAGTruth full test evidence-condition cache

- **Package ID:** `ragtruth_full_evidence`
- **Status:** READY_WITH_LIMITATIONS
- **Observed:** {"conditions": 16200, "duplicate_conditions": 0, "key_mismatches": 0, "missing_full_noctx_pairs": 0, "noncontiguous_loo": 0, "responses": 2700, "sources": 450, "task_counts": {"Data2txt": 900, "QA": 900, "Summary": 900}, "token_mismatches_across_conditions": 0, "trace_failures": 0}
- **Balance:** hallucinated: 943, clean: 1,757, hallucinated_rate: 34.9%
- **Checks passed:** 8/8
- **Limitations:** RAGTruth labels have already been opened in earlier exploratory work.

### GASP-style balanced RAGTruth cohort

- **Package ID:** `gasp_ragtruth_400`
- **Status:** READY_WITH_LIMITATIONS
- **Observed:** {"conditions": 2508, "duplicate_conditions": 0, "key_mismatches": 0, "missing_full_noctx_pairs": 0, "noncontiguous_loo": 0, "responses": 400, "sources": 228, "task_counts": {"Data2txt": 214, "Summary": 186}, "token_mismatches_across_conditions": 0, "trace_failures": 0}
- **Balance:** hallucinated: 200, clean: 200, hallucinated_rate: 50.0%
- **Checks passed:** 8/8
- **Limitations:** The paper did not publish its 400 response IDs or sentence splitter; this is a protocol-level reproduction.

### RefChecker fixed human-labelled claims

- **Package ID:** `refchecker_claims`
- **Status:** READY_WITH_LIMITATIONS
- **Observed:** {"claims": 10733, "conditions": 21466, "incomplete_pairs": 0, "settings": {"accurate_context": 3994, "noisy_context": 3420, "zero_context": 3319}, "token_mismatches_across_conditions": 0, "trace_failures": 0}
- **Balance:** Contradiction: 935, Entailment: 7,176, Neutral: 2,622
- **Checks passed:** 6/6
- **Limitations:** The gold labels cover the fixed shipped claims. This is claim checking, not claim extraction.

### ProcessBench telemetry (qwen3 4b)

- **Package ID:** `processbench_qwen3_4b`
- **Status:** READY
- **Observed:** {"alignment_failures": 0, "missing_lfs_objects": [], "rows": 3400, "subset_counts": {"gsm8k": 400, "math": 1000, "olympiadbench": 1000, "omnimath": 1000}, "subsets": 4, "trace_failures": 0}
- **Balance:** first_error_present: 2,221, fully_correct: 1,179, error_rate: 65.3%
- **Checks passed:** 5/5

### ProcessBench telemetry (qwen3 8b)

- **Package ID:** `processbench_qwen3_8b`
- **Status:** READY
- **Observed:** {"alignment_failures": 0, "missing_lfs_objects": [], "rows": 3400, "subset_counts": {"gsm8k": 400, "math": 1000, "olympiadbench": 1000, "omnimath": 1000}, "subsets": 4, "trace_failures": 0}
- **Balance:** first_error_present: 2,221, fully_correct: 1,179, error_rate: 65.3%
- **Checks passed:** 5/5

### ProcessBench Qwen2.5-Math-PRM predictions

- **Package ID:** `processbench_qwen25math7b_predictions`
- **Status:** READY
- **Observed:** {"manifest": true, "rows": 3400, "subset_counts": {"gsm8k": 400, "math": 1000, "olympiadbench": 1000, "omnimath": 1000}, "subsets": 4}
- **Balance:** —
- **Checks passed:** 3/3

### ProcessBench Qwen3-8B judge-control predictions

- **Package ID:** `processbench_qwen3_8b_judge_control`
- **Status:** READY
- **Observed:** {"manifest": true, "rows": 3400, "subset_counts": {"gsm8k": 400, "math": 1000, "olympiadbench": 1000, "omnimath": 1000}, "subsets": 4}
- **Balance:** —
- **Checks passed:** 3/3

### ProcessBench Qwen2.5-72B critic predictions

- **Package ID:** `processbench_qwen72b_critic`
- **Status:** READY
- **Observed:** {"manifest": true, "rows": 3400, "subset_counts": {"gsm8k": 400, "math": 1000, "olympiadbench": 1000, "omnimath": 1000}, "subsets": 4}
- **Balance:** —
- **Checks passed:** 3/3

### PRMBench Qwen3-8B telemetry

- **Package ID:** `prmbench_qwen3_8b_telemetry`
- **Status:** READY_WITH_LIMITATIONS
- **Observed:** {"classes": {"circular": 758, "confidence": 757, "correct": 758, "counterfactual": 757, "deception": 750, "domain_inconsistency": 757, "missing_condition": 756, "multi_solutions": 160, "redundency": 758, "step_contradiction": 758}, "misaligned_ids": ["confidence_confidence_prm_train_p1_303", "deception_deception_prm_test_p1_87", "step_contradiction_step_contradiction_prm_test_p2_991"], "misaligned_rows_or_span_sets": 3, "official_error_class_steps": 83371, "rows": 6969, "step_spans": 94203, "trace_failures": 0}
- **Balance:** —
- **Checks passed:** 4/5
- **Limitations:** Three rows were reported as misaligned; they must be identified and explicitly resolved or excluded before evaluation.

### PRMBench Qwen2.5-Math-PRM predictions

- **Package ID:** `prmbench_qwen25math7b_predictions`
- **Status:** READY
- **Observed:** {"labels": 94203, "rewards": 94203, "rows": 6969}
- **Balance:** —
- **Checks passed:** 2/2

### RAGTruth LettuceDetect prediction package

- **Package ID:** `ragtruth_lettucedetect_predictions`
- **Status:** READY
- **Observed:** {"malformed_spans": 0, "rows": 2700, "truncated": 0, "unique_response_ids": 2700}
- **Balance:** hallucinated: 943, clean: 1,757, hallucinated_rate: 34.9%
- **Checks passed:** 4/4

## Canonical data contract

Future consumers must address records by stable `dataset_id`, `record_id`, `source_id`, split, task, model, condition and parent ID. Large token arrays remain inside the immutable source artifact and are addressed by artifact path and row key.

Labels use a separate sidecar contract containing `record_id`, label space, value and provenance. This prevents a future label-free fitting program from receiving labels by accident.

## Decision

HLE now has a complete interim Codex-judge label sidecar, but it is not the paper-faithful GPT-4o label set. Resolve any incomplete package required by the intended experiment. Packages marked READY may be used unchanged once the benchmark protocol is frozen. Packages marked READY_WITH_LIMITATIONS may also be evaluated now, but only for the documented scope and with that limitation included in every result.
