# RAGTruth Evidence-Contrast Experiment

> **Audit notice:** this original blind run used stored full-vocabulary entropy
> for two EC columns instead of the registered top-50-plus-tail formula. The
> formula-faithful correction is in
> `../ragtruth_evidence_contrast_v1_top50_correction/`. It gives the same
> scientific conclusion, but it is not a new blinded confirmation because the
> original test labels had already been opened.

**Experiment:** `ragtruth-evidence-contrast-v1-2026-08-10`  
**Final visible split:** `test`

## Result

The development gate passed. The unchanged test scores show that Evidence-Contrast fusion beats the approximate GASP baseline, but DUFS-LIU does not beat IU-PCR. The feature construction worked; the registered graph mechanism did not.

On the LOO sentence cohort, EC-DUFS-LIU reached AUROC **0.703**. The change was **+0.031** versus GASP-top50 and **-0.0005** versus EC-IU-PCR.

The registered method-level success rule failed. The Evidence-Contrast contract plus IU-PCR is useful, but the DUFS-gated Laplacian did not add value. The result should be described as a **feature-contract success and mechanism failure**.

The separately hashed post-hoc intrinsic mixed-v2 audit reached pooled response AUROC 0.763, but only 0.434 on Data-to-Text. EC-DUFS-LIU reached 0.748 pooled and 0.706 on Data-to-Text. The pooled old-method value is therefore not a stable cross-task baseline.

The self-contained visual report is [`REPORT.html`](REPORT.html). Definitions and mathematical provenance are in [`METHODS.md`](METHODS.md). Exact execution commands are in [`RUNBOOK.md`](RUNBOOK.md).

## Boundaries

- GASP-top50 approximates, but does not reproduce, full-vocabulary GASP JSD.
- Scores come from a Qwen2.5-1.5B teacher-forced scorer over fixed RAGTruth answers.
- Results measure agreement with the available RAGTruth annotations.
- Test labels are opened only if every registered development check passes.
- Response-level performance is more sensitive to chunk count and context length than sentence-level performance.
- The original intrinsic mixed-v2 response baseline is reported only as a separately hashed post-hoc audit and does not enter the registered decision.
- The cache identifies omitted chunks by index but does not contain their exact text or metadata; example cards therefore show only the largest-drop chunk index.
