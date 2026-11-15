# Simes answer gate inside the standalone pipeline (2026-11-15)

**Result: the fixed gate (alpha = .05) closes 94-95% of all answers. PRMBench PRMScore drops by about 19 points
in all 4 cells. ProcessBench average F1 goes from 0 to 5.4-6.8.** The gate does what Simes does under its null:
it opens about alpha of the answers. Data: development data; labels used only for scoring.

## Definitions

| Quantity | Definition |
|---|---|
| Base models | Qwen2.5-Math-7B-Instruct, Qwen2.5-Math-72B-Instruct, Qwen2.5-Math-7B (base), Llama-3.1-8B-Instruct (standalone telemetry; file keys qwen_prm_7b, qwen_prm_72b, pure_prm, rlhflow_prm). No PRM score is used. |
| Ranking step score | TELE16 L-SML on R0 columns (each raw channel z-scored inside its answer) = arm "F2 L-SML, R0" of gsp_status_review_v1. |
| GAP flags | Sort the step scores of one answer; flag the steps above the largest gap. Always flags at least 1 step per answer. |
| Gate step score s_g | L-SML on the same 16 channels after global rank -> N(0,1) over all steps of the benchmark, no per-answer z (cross_model_gate_check_v1/steps). |
| Step p-value | p_i = share of all steps of the benchmark (same base model) with s_g >= s_g,i. |
| Simes | Simes = min over i of n p_(i) / i, on the n steps of one answer (p sorted ascending). Low = suspicious. |
| Gate | Fixed before the run: Simes > .05 -> answer called clean -> no flag in it. Same alpha for all 8 cells. |
| PRMScore | Official PRMBench metric: 50 x (F1 of valid steps + F1 of error steps), all steps pooled. |
| PB average F1 | Official ProcessBench: per subset, harmonic mean of error accuracy (exact first-error step on wrong answers) and correct accuracy (predict -1 on clean answers); mean of 4 subsets. Prediction = first flagged step, else -1. |
| CI | 95% paired bootstrap, 2,000 draws. PRMBench: source-question groups. ProcessBench: answers within each subset. |

## Parity

- R0 columns equal the stored N16 channels in 8 of 8 cells (max diff 3.8e-7).
- Step labels of the gate file equal the cell labels in 8 of 8 cells.
- No-gate baseline equals gsp_status_review_v1 "F2 L-SML, R0": PRMBench 7B 68.26 (stored 68.26), ProcessBench SLA 7B 38.47 (stored 38.47).

## Main result (alpha = .05)

| Benchmark | Metric | 7B-Inst | 72B-Inst | 7B-base | Llama-8B |
|---|---|---|---|---|---|
| PRMBench | PRMScore no gate | 68.3 | 67.8 | 67.6 | 68.2 |
| PRMBench | PRMScore with gate | 49.1 | 49.1 | 48.9 | 48.9 |
| PRMBench | gate - no gate [95% CI] | -19.2 [-19.9, -18.4] | -18.7 [-19.4, -18.0] | -18.8 [-19.4, -18.1] | -19.3 [-20.0, -18.6] |
| ProcessBench | avg F1 no gate | 0.0 | 0.0 | 0.0 | 0.0 |
| ProcessBench | avg F1 with gate [95% CI] | 6.8 [5.2, 8.6] | 6.4 [4.9, 7.9] | 5.7 [4.3, 7.2] | 5.4 [4.1, 6.8] |
| ProcessBench | error acc no gate -> gate | 38.5 -> 3.6 | 38.1 -> 3.3 | 31.3 -> 2.9 | 34.3 -> 2.8 |
| ProcessBench | correct acc no gate -> gate | 0 -> 99.3 | 0 -> 98.8 | 0 -> 98.5 | 0 -> 98.6 |

## What the gate closes

| Benchmark | true clean share | closed, all answers | closed among clean | closed among wrong |
|---|---|---|---|---|
| PRMBench | 13.4% | 94.1-94.3% | 98.6-99.7% | 93.2-93.4% |
| ProcessBench | 34.7% | 94.3-94.9% | 98.5-99.3% | 92.1-92.9% |

Answer AUROC of Simes (not length-stratified): PRMBench .73-.77, ProcessBench .70-.75.

## Why alpha = .05 closes almost everything

- The reference distribution for the p-values is the benchmark itself (all steps, mostly from wrong answers:
  87% of PRMBench answers and 65% of ProcessBench answers are wrong).
- Under "the steps of this answer look like random steps of the pool", Simes is about uniform. A typical answer,
  wrong or clean, looks like the pool. So about alpha of the answers open, whatever their label (here 5-6%).
- Thus alpha is not a calibrated error rate here. It is an opening budget: "open about alpha of the answers".
- The ranking signal exists (AUROC .70-.77), but it is spread over the whole range, not in the extreme tail.

## Diagnostic only: the same gate at other alpha values (SWEEP.csv, fig3)

Not the main result; alpha was not tuned.

| alpha | PRMBench PRMScore (4 models) | PB avg F1 (4 models) | closed (PB) |
|---|---|---|---|
| .05 | 48.9-49.1 | 5.4-6.8 | 94-95% |
| .3 | 60.1-60.9 | 21.4-28.7 | 71-75% |
| .5 | 64.7-65.3 | 29.0-36.7 | 55-60% |
| .7 | 66.7-67.3 | 33.6-39.6 | 38-45% |
| 1.0 (no gate) | 67.6-68.3 | 0 | 0% |

- PRMBench: every alpha < 1 is worse than no gate.
- ProcessBench: the best tested value (.7) gives 33.6-39.6. Context: uPRM reports 39.8-58.3, trained PRMs 73-78.

## Files

| File | Content |
|---|---|
| run.py | Pipeline + gate, RESULTS.csv, BOOT.csv, SWEEP.csv, GATES.json, cells/*.npz (Simes per answer) |
| figures.py | fig1_main.png, fig2_gate_closed.png, fig3_sweep_diagnostic.png |
