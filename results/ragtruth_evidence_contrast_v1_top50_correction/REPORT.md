# RAGTruth Evidence-Contrast Experiment

**Experiment:** `ragtruth-evidence-contrast-v1-top50-tail-protocol-correction-2026-08-10`  
**Final visible split:** `test`

**Protocol status:** fixed-formula correction performed after the original test labels had been opened. The approved top-50-plus-tail entropy formula was not selected from labels, but this is not a new blinded confirmation.

## Result

The development gate passed. The frozen corrected test scores show that Evidence-Contrast fusion beats the approximate GASP baseline, but DUFS-LIU does not beat IU-PCR. The feature construction worked; the registered graph mechanism did not.

On the LOO sentence cohort, EC-DUFS-LIU reached AUROC **0.703**. The change was **+0.031** versus GASP-top50 and **-0.0006** versus EC-IU-PCR.

The registered method-level success rule failed. The Evidence-Contrast contract plus IU-PCR is useful, but the DUFS-gated Laplacian did not add value. The result should be described as a **feature-contract success and mechanism failure**.

The separately hashed post-hoc intrinsic mixed-v2 audit reached pooled response AUROC 0.763, but only 0.434 on Data-to-Text. EC-DUFS-LIU reached 0.748 pooled and 0.703 on Data-to-Text. The pooled old-method value is therefore not a stable cross-task baseline.

The self-contained visual report is [`REPORT.html`](REPORT.html). Definitions and mathematical provenance are in [`METHODS.md`](METHODS.md). Exact execution commands are in [`RUNBOOK.md`](RUNBOOK.md).

## Boundaries

- GASP-top50 approximates, but does not reproduce, full-vocabulary GASP JSD.
- Scores come from a Qwen2.5-1.5B teacher-forced scorer over fixed RAGTruth answers.
- Results measure agreement with the available RAGTruth annotations.
- Test labels are opened only if every registered development check passes.
- Response-level performance is more sensitive to chunk count and context length than sentence-level performance.
- The original intrinsic mixed-v2 response baseline is reported only as a separately hashed post-hoc audit and does not enter the registered decision.
