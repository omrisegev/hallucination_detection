# Family15 tail20 external transfer: complete exploratory results

All ten frozen alternatives were scored on the complete available Hard2Verify/Qwen3-8B and Socratic-PRMBench/Qwen3-8B, QwQ-32B populations. Existing teacher-forced telemetry was reused. There was no new GPU inference, model training, target-label fitting or threshold recalibration. Predictions were sealed before the external evaluator read labels. The external datasets had already been discussed before this follow-up, so these are exploratory transfer results, not fresh untouched confirmation.

| Frozen method | Hard2Verify Balanced F1 | Socratic Qwen3 PRMScore | Socratic QwQ PRMScore |
|---|---:|---:|---:|
| Family15 tail20 L-SML | 41.020 | 59.921 | 62.690 |
| Bank11 L-SML | **43.670** | **63.221** | **64.238** |
| Family15 equal control | 42.380 | 61.124 | 62.937 |
| Family15 covariance L-SML | 42.021 | 60.023 | 61.154 |
| Key28 covariance L-SML | 41.868 | 60.565 | 61.486 |
| Key28 equal control | 42.081 | 61.250 | 63.003 |
| Bank11 equal control | 40.882 | 60.791 | 61.503 |
| Bank11 partition-equal control | 39.758 | 61.270 | 62.110 |
| All48 equal control | 39.697 | 59.504 | 61.601 |
| CT7 reference | 37.751 | 58.753 | 60.166 |

Scores are percentages. Hard2Verify Balanced F1 is the harmonic mean of correct/error recall; Socratic PRMScore is the mean of correct/error F1. The two benchmark metrics must not be averaged. The official scoring implementation was replayed on all 30 method/cell rows. The paired primary comparison family has 18 contrasts, 100,000 source-question bootstrap draws and Bonferroni adjustment. Family15 tail20 minus Bank11 L-SML is −2.650 points on Hard2Verify (adjusted interval [−6.806,+1.511]), −3.301 on Socratic/Qwen3 ([−4.085,−2.532]), and −1.549 on Socratic/QwQ ([−2.309,−0.792]). The Hard2Verify interval includes zero; the Socratic intervals do not.

The method's Socratic correct/error F1 components are 78.461/41.380 (Qwen3) and 79.795/45.585 (QwQ), versus Bank11's 79.717/46.726 and 80.060/48.417. The error-class decline drives much of the aggregate PRMScore gap. Family15 tail20 does beat its matched continuous Family15 covariance model on QwQ by 1.536 points (adjusted interval [+0.933,+2.146]), but loses to Bank11 and has lower point estimates than the Family15 equal control in every cell. Equal fusion remains a control rather than a proposed leading method. Bank11 remains the leading frozen L-SML transfer candidate.

There are 200 Hard2Verify answers and 1,860 included steps; each Socratic backbone has 2,995 answers and 26,055 included steps. An exact observed-overlap exclusion leaves 2,553 Socratic answers per backbone. This sensitivity analysis does not establish the absence of semantic or pretraining contamination. Question-group bootstrap resamples all variants of a source question together. Full details, category scores, within-answer ranking, coverage, failure diagnostics and all adjusted contrasts are in the [generated report](../../results/family_tail_external_v1/REPORT.md) and [interactive HTML gallery](../../results/family_tail_external_v1/REPORT.html). Machine-readable components are in [METRICS.json](../../results/family_tail_external_v1/METRICS.json), [CONTRASTS.json](../../results/family_tail_external_v1/CONTRASTS.json) and [DISJOINT_CONTRASTS.json](../../results/family_tail_external_v1/DISJOINT_CONTRASTS.json).

The gallery includes all-method scores, class-specific PRMScore F1 and precision/recall, Hard2Verify components, paired contrasts, category/length/position panels, feature diagnostics and published-comparator context. Published comparator values come from the [Hard2Verify paper](https://arxiv.org/html/2510.13744v1) and [Socratic-PRMBench paper](https://arxiv.org/html/2505.23474v1). They are contextual paper values, not reproduced runs; target-threshold access, models and prompts differ. Missing published component values are left missing. No state-of-the-art claim follows from these plots.

The source feature reconstruction gate covered 13,769/13,769 answers and 145,597 steps, with maximum absolute deviation 6.66e−14 against 1e−6 tolerance. The external CPU scoring pass covered 6,190/6,190 records and took 3,217.83 wall seconds with four workers; the source reconstruction pass took 3,557.81 wall seconds. Both used zero new GPU-hours. Reproduction manifests, code identities, method locks and score files are under [results/family_tail_external_v1](../../results/family_tail_external_v1/). The large arrays are excluded from Git but included in a 190,131,044-byte, 6,312-member archive. Its Drive copy has matching size and SHA256; exact location and restore steps are in [ARCHIVE.json](../../results/family_tail_external_v1/ARCHIVE.json) and [RESTORE.md](../../results/family_tail_external_v1/RESTORE.md).

The user explicitly deferred further self-checks to another agent. The main official evaluator completed, and three separate full-population component reviews report PASS for metric recomputation, coverage/contracts and null/math replay. One independent metric-audit script received post-seal integrity guards, documented with original/revised hashes in `independent_metrics/AUDIT_AMENDMENT.json`; the frozen predictions and primary evaluator did not change. See [review status](../../results/family_tail_external_v1/AUDIT_DEFERRED.md). A final combined independent sign-off is still deferred. A presentation-only code amendment after the analysis freeze is recorded in `PRESENTATION_AMENDMENT.json`.

Hebrew interpretation: [INTERPRETATION_HE.md](../../results/family_tail_external_v1/INTERPRETATION_HE.md). Future work should be developed and calibrated on source data, then assessed on fresh unexposed data. External labels in this study have now been examined.
