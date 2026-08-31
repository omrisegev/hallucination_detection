# Reasoning Localization 0.3662 — H2/H3 PRMBench Diagnostic Results V2

Status: `COMPLETE / PRMBENCH_SPECIALIST / NO_PHASE4_PROMOTION`.

This is a frozen cross-task mechanism diagnostic on historically opened
PRMBench labels. It is not fresh confirmation and does not open Phase 4.
ProcessBench first-error F1 and PRMBench every-step AUROC/AUPRC remain separate
estimands and are never averaged.

## Pre-label contract correction

The first executable attempt hard-failed before labels because it incorrectly
required the new top-ten H0 score to alias the Phase-1 top-five R2 score. The
maximum discrepancy was `0.23125777605843761`; no PRMBench label artifact was
loaded and no scientific score was opened. Amendment V2 added a non-rankable
top-five control, which reproduced Phase-1 R2 at maximum absolute error `0`.
The top-ten H0/H2/H3 candidates were otherwise unchanged. All three imported
Qwen ProcessBench score artifacts also reproduced at maximum absolute error
`0` before PRMBench labels opened in this run.

## Frozen method ladder

- H0: current five-family token curve and top-ten step mean.
- H2: remove sampled-token energy, remove the partition `energy_series` view,
  and insert frozen C7 EDIS onset inside entropy dynamics.
- H3: equal within-response rank fusion of H2 and frozen C8 self-innovation.

The response detector is identical across arms. No PRMBench label selected a
feature, transform, reducer, orientation, weight, or threshold.

## Population and inference

The evaluator contains 83,280 annotated steps in 6,208 paired `source_idx`
groups. There are 13,144 positive error steps and 70,136 negative steps.
Inference uses 20,000 paired whole-source draws with seed `2026083102` and
Bonferroni-simultaneous intervals across the three frozen contrasts separately
for each metric. Eight error families are evaluable; `multi_solutions` is
single-class and remains visible as undefined rather than being zero-filled.

| Arm | AUROC | AUPRC |
|---|---:|---:|
| H0 family6/top-ten | 0.592057 | 0.209760 |
| H2 cleanup+C7 | 0.597871 | 0.210778 |
| H3 equal+C8 | **0.619469** | **0.225194** |

Paired simultaneous contrasts:

| Contrast | AUROC delta [CI] | AUPRC delta [CI] | AUROC W/T/L | Worst AUROC family |
|---|---:|---:|---:|---:|
| H2−H0 | +0.005814 [+0.004710,+0.006973] | +0.001018 [−0.000034,+0.002062] | 8/0/0 | +0.000921 |
| H3−H0 | +0.027412 [+0.023675,+0.031091] | +0.015434 [+0.011378,+0.019404] | 8/0/0 | +0.021655 |
| H3−H2 | +0.021598 [+0.017653,+0.025457] | +0.014417 [+0.010252,+0.018475] | 8/0/0 | +0.011403 |

H2 clears the frozen `+0.003` practical-benefit bound for AUROC, although its
AUPRC interval crosses zero and is not called zero or rejected. H3 clears the
same AUROC bound versus both H0 and H2, improves AUPRC in both comparisons,
and improves AUROC in all eight evaluable families.

## Verdict and boundary

H3 is a supported `PRMBENCH_SPECIALIST`. This result strengthens the premise
that C8 contributes dense error-step ranking information that was not visible
as confirmed incremental ProcessBench first-error F1 on Llama. It does not
make H3 a universal winner: ProcessBench promotion is still absent, the
PRMBench population was historically opened, `prm_train`/`prm_test` source
membership is unavailable in the sealed evaluator, and the outcome-selected
ancestry requires independent confirmation.

No Phase-3 or Phase-4 branch is promoted by this diagnostic. A future
fresh-question contract must retain H0, H2 and H3 as separate arms; a future
task-general architecture may use task-specific output roles, but it may not
select that design from these opened labels and may not average the two tasks.

Canonical artifacts are under
`results/reasoning_localization_03662_v1/phase_2/transfer/h3_prmbench_v2/`.
The standalone plot is `evaluation/H3_PRMBENCH_RESULTS.svg`.
