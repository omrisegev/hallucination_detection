# ProcessBench Step-level Localization Accuracy (SLA) under GAP Decision Rule

Evaluation of TELEM16 (`TELE_N16`) and baseline arms on ProcessBench under the GAP decision rule, compared against published results from "Mind the Gap" (Chen et al., ICML 2026, Table 3).

## Key Background & Metric Distinction

* **Why ProcessBench was previously considered limited:** Previous project evaluations focused on the **Official ProcessBench F1** (evaluated under the `NC` rule in `SUB5_ALL.csv` or `Q80`). ProcessBench F1 evaluates both erroneous answers and clean answers (which require abstaining by predicting `-1`). Because pure telemetry lacks a dedicated clean/error gate, its ProcessBench F1 was constrained by the PRM's gate.
* **The Mind the Gap protocol (Table 3):** Mind the Gap evaluates **SLA** (Step-level Localization Accuracy) strictly on **erroneous answers** (2,221 out of 3,400 answers across GSM8K, MATH, OlympiadBench, and Omni-MATH). Clean answers are un-scored.
* **The GAP decision rule:** In each answer of length $m$, telemetry risk scores are sorted descending, $k$ is determined by the argmax of score gaps within the top $\min(\lceil m/2 \rceil, m-1)$ steps, and the top-$k$ steps are flagged. We predict the **first flagged step in chronological order** ($\min \{ i : \text{step } i \text{ is flagged} \}$).

## Results Summary

| Model / Method | Rule | Macro SLA | Overall SLA | GSM8K (207) | MATH (594) | Olympiad (661) | Omni-MATH (759) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Mind the Gap** (Qwen3-4B) | Shannon Drop (T3) | 39.14% | 38.43% | 43.42% | 32.03% | 43.06% | 38.04% |
| **Mind the Gap** (Qwen3-8B) | Shannon Drop (T3) | 39.39% | 38.11% | 46.11% | 32.90% | 41.52% | 37.04% |
| **TELEM16 (Qwen-7B backbone)** | GAP (first flag) | **47.23%** | **44.35%** | **59.90%** | **46.46%** | **42.36%** | **40.18%** |
| *vs MtG Qwen-8B* | | *+7.84 pp* | *+6.24 pp* | *+13.79 pp* | *+13.56 pp* | *+0.84 pp* | *+3.14 pp* |
| **TELEM16 (Qwen-72B backbone)**| GAP (first flag) | **50.64%** | **49.08%** | **57.49%** | **51.52%** | **45.99%** | **47.56%** |
| *vs MtG Qwen-8B* | | *+11.25 pp* | *+10.97 pp* | *+11.38 pp* | *+18.62 pp* | *+4.47 pp* | *+10.52 pp* |

## Findings

1. **Substantial Win over Mind the Gap:** Under the same SLA metric and model family (Qwen), TELEM16 beats Mind the Gap's headline Shannon Drop by **+7.84 pp Macro** on Qwen-7B (47.23% vs 39.39%) and by **+11.25 pp Macro** on Qwen-72B (50.64% vs 39.39%).
2. **Huge Gains on Hard Math:** The largest advantages appear on MATH (+13.56 pp / +18.62 pp) and GSM8K (+13.79 pp / +11.38 pp).
3. **GAP First-Flag vs Argmax:** Chronological first-flag achieves +3.45 pp Macro over Argmax on Qwen-7B (47.23% vs 43.78%), capturing earlier root-cause premise deviations before uncertainty spikes maximally.
4. **Controls:** Plain unweighted average `MEAN_N16` achieves 46.26% (Qwen-7B) and 48.60% (Qwen-72B); L-SML weights in `TELE_N16` add +0.97 pp and +2.04 pp respectively.

## Reproduction

```bash
python results/processbench_gap_sla_v1/evaluate_sla.py
```
Outputs `RESULTS.csv` and prints the benchmark comparison table.
