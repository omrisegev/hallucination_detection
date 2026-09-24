# Literature audit for family-tail external transfer

Verified 2026-09-24. This inventory covers the two specified version-1 benchmark papers, not an exhaustive current leaderboard. No new external method scores were inspected. Machine-readable values are in [LITERATURE_CONTEXT.json](LITERATURE_CONTEXT.json); all are percentages.

## Metric and plotting contract

With the correct step as the positive class:

- Socratic PRMScore is the arithmetic mean of correct-class F1 and error-class F1. Plot these two components separately; show both precisions and recalls as additional diagnostics.
- Hard2Verify step Balanced F1 is the harmonic mean of correct-step recall and error-step recall. These are different components and a different aggregate from PRMScore.
- Error categories describe subsets of Socratic examples. They are not the algebraic components of PRMScore. Compute pooled confusion counts within each subset; do not average category scores to reconstruct the overall score.

The repository metric implementation matches these formulas for nondegenerate confusion matrices. The pinned Socratic toolkit uses negative-one sentinels for some degenerate precision/recall/F1 cases; the repository helper uses zero for zero denominators. Consequently, official replay must cover category rows too, or explicitly flag degenerate rows. This is an edge-case difference, not evidence that earlier nondegenerate full-set scores are wrong. [Official evaluator](https://github.com/ssmisya/PRMBench/blob/a1ae4eab803e14c4c6316bf89d1679b0486faa8c/mr_eval/tasks/prmtest_classified/task.py).

## Hard2Verify literature

Tables 2 and 5 report 29 models; Table 5 supplies the two class recalls. The JSON captures every step-level row. Examples:

| Method | Balanced F1 | Correct recall | Error recall |
|---|---:|---:|---:|
| Qwen3-8B critic | 53.51 | 92.96 | 37.56 |
| Qwen2.5-Math-PRM-7B | 42.37 | 87.13 | 27.99 |
| UniversalPRM-7B | 60.27 | 80.00 | 48.35 |
| gpt-oss-20B critic | 70.93 | 93.06 | 57.31 |
| GPT-5 critic | 85.83 | 94.35 | 78.72 |

Section E.1 selects PRM thresholds using 100 target responses, unlike our source-only calibration. Critics use a verification prompt; Qwen3 thinking is enabled. Exact PRM evaluation masks/counts were not established here. Rounded response-level recalls suggest differing subsets, but this is an inference, not a confirmed split. Keep these as paper-reported context, with no paired significance claim. Precision and class F1 are not supplied as numeric table columns. [Hard2Verify, Tables 2/5 and Sections 4.1/4.2/E.1](https://arxiv.org/html/2510.13744v1).

## Socratic-PRMBench literature

Table 3 supplies all 11 overall scores and 20 error-category scores per model. Table 4 additionally gives class-specific accuracy, corresponding to recall, for eight models; it does not report class F1 or precision. Missing entries remain null.

| Method | PRMScore | Correct recall | Error recall |
|---|---:|---:|---:|
| Skywork-PRM-7B | 43.6 | 22.7 | 93.0 |
| ReasonEval-7B | 61.9 | 87.3 | 35.7 |
| MathShepherd-Mistral-7B | 64.4 | 73.3 | 56.0 |
| Qwen2.5-Math-PRM-7B | 68.0 | 90.8 | 42.9 |
| GPT-4o critic | 70.8 | 83.0 | 57.5 |
| QwQ-32B critic | 73.8 | 83.9 | 63.1 |
| Gemini-2.5-Pro critic | 73.5 | 83.6 | 62.8 |
| o3-mini critic | 75.7 | 82.6 | 69.0 |

The other overall scores are RLHFlow-Mistral 48.8, RLHFlow-Deepseek 51.5 and Deepseek-R1 73.0. Appendix A identifies the PRMBench toolkit and critic temperature 1.0; model-specific thresholds and exact checkpoint revisions are not fully documented. Original PRMBench category scores must not populate Socratic panels. [Socratic paper, Tables 3/4/5 and Section 4.2/Appendix A](https://arxiv.org/html/2505.23474v1).

## Presentation requirements

Show every registered internal alternative. Distinguish paper-reported rows visually and label their access conditions. Literature precision/F1 panels must show unavailable values rather than estimates; published recalls can be compared descriptively. Do not reconstruct unpublished confusion counts from rounded numbers. A primary-score advantage over a published row is a descriptive comparison, not matched reproduction or a paired statistical result.

Validation: 29 Hard2Verify rows, 11 Socratic rows, 20 categories per Socratic row, 220 category values; class precision/F1 absent in all literature rows. The two requested paper versions were checked against primary-source HTML tables.
