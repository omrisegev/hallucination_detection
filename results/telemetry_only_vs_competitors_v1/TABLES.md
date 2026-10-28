# Telemetry only (standalone) vs the paired PRM, under PRISM, SLIDER and CROP protocols (2026-10-28)

**What this is.** Each base model's own telemetry (no PRM in the score) against the PRM built on that base model, on
every benchmark and metric that PRISM, SLIDER and CROP report and for which we have data. CPU only, stored scores,
no refit and no inference. Published numbers sit beside ours, with the paper and table.

| standalone model (telemetry source) | paired PRM |
|---|---|
| Qwen2.5-Math-7B-Instruct | Qwen2.5-Math-PRM-7B |
| Qwen2.5-Math-72B-Instruct | Qwen2.5-Math-PRM-72B |
| Qwen2.5-Math-7B | PURE-PRM-7B |
| Llama-3.1-8B-Instruct | RLHFlow-PRM-Mistral-8B |

Arms: **TELE16 (L-SML)** = `TELE_N16`, label-free fusion of the 16 standalone channels; **TELE16 plain average** =
`MEAN_N16`, the same channels with equal weights; **PRM, native threshold** (reward ≤ 0.5 = error); **PRM, same rule**
= the PRM under the same label-free per-answer GAP rule as the telemetry arms (ProcessBench: Q80).
TELE16 was chosen on PRMBench, so PRMBench is in-sample for it; everything else is out of sample.

**Parity (PARITY.json): 128 checks, 0 failed, every difference 0.0.** Before anything new, the run reproduces:
- the PRM rows on PRMBench: PRMScore, FPR, TPR, TNR, Precision (`COMP_PRISM_T1_T2`), the nine Table 7 categories
  (`COMP_PRISM_T7_CATEGORIES`), the matched-TPR FPR (`COMP_PRISM_MATCHED_TPR`), CROP Direct-PRM kept (`COMP_CROP_T6`) and
  the SLIDER LR accuracy (`COMP_SLIDER`), for all four PRMs;
- the stored standalone TELE16 and plain-average PRMScore (`prm_standalone_telemetry_v1/tables/STANDALONE_MAIN.csv`),
  their matched-TPR PRMScore and CROP kept (`prm_sub5_all_benchmarks_v1/SUB5_ALL.csv`);
- ProcessBench standalone TELE16 avg F1 (NC) and the PRM native avg F1 (`SUB5_ALL.csv`);
- the CROP code path: standalone TELE16 and plain-average kept % on CRV-Arith, CRV-GSM8K and PRM800K (`CROP_TABLE6_CAB.csv`).
Our PRM-7B row also reproduces PRISM's published baseline (PRMScore 65.49 vs 65.50; ProcessBench GSM8K/MATH exactly).

## Summary

1. **PRMBench PRMScore (PRISM Table 2).** TELE16 standalone scores 67.6–68.3. That is above its paired PRM at the native
   threshold in 3 of 4 pairs (all but Qwen-PRM-72B, 68.18), above the PRM under the same rule in 3 of 4 (Qwen-PRM-72B
   69.49 wins), and at PRISM's 68.0 for two pairs (68.26 Qwen-7B, 68.20 Llama). The plain average is 0.4–1.1 lower in
   all 4 pairs. In-sample for TELE16.
2. **At PRISM's operating point (TPR 87.3, threshold set with labels)** TELE16 is 65.8–66.3, below PRISM (68.0) and below
   its paired PRM in 3 of 4 pairs (it beats only RLHFlow). Its gain in (1) is partly the decision rule.
3. **Categories (PRISM Table 7).** Telemetry is strong on redundancy and circular logic (NR 73–74, NCL 69–72, vs NR 46–60 and NCL 47–65
   for every PRM and 52–54 / 64–66 for PRISM) and weak on empirical soundness, deception and multi-solution.
4. **SLIDER (redundancy).** TELE16 reaches 76–79% accuracy and 76–77% balanced accuracy with the LR threshold (74–75% balanced with the label-free rule), vs
   Step-RRI 61.91. The paired PRMs reach 53–66% balanced accuracy. Plain accuracy is weak evidence here: always saying
   "not redundant" gives 86.8%.
5. **ProcessBench (PRISM Table 6, CROP Table 6).** Telemetry alone does not work: pure-telemetry avg F1 7–11 vs
   29–78 for the PRMs (it almost never says "no error"); with the PRM's gate (NC) 34–54. CROP kept 10.5–11.8% vs 21–50%
   for the PRMs; pooled AUROC .735–.757 vs .70–.91.
6. **CROP on PRMBench.** TELE16 keeps 16.4–18.2% at ≈5% risk: above CROP's Token/format (16.2) and Trace-feature (15.9)
   LRs, below its supervised Trace+PRM LR (21.8). Against the paired PRM on the same grid it keeps more for Qwen-PRM-7B (18.2 vs 17.2) and RLHFlow (18.2 vs 8.7), less for Qwen-PRM-72B and PURE.
7. **CROP on CRV-Arith, CRV-GSM8K, PRM800K.** Telemetry is far below every PRM and CROP proxy (pooled AUROC .41–.66).
   Cross-domain (Table 17) it fails like CROP does (Arith → GSM8K risk 10–15%).

Bottom line: telemetry alone matches or beats the paired PRM on PRMBench's own metric (in-sample) and clearly on
redundancy; on every other benchmark here the PRM is much stronger. L-SML adds 0.4–1.1 PRMScore over the plain average
on PRMBench; elsewhere the two are close and the average is sometimes higher.

## Access, per row

- Ours (TELE16, plain average): label-free; one teacher-forced pass of the base model over the given answer. The CROP
  and SLIDER rows add those protocols' label-calibrated thresholds; the matched-TPR row sets one global threshold with labels.
- The PRMs: trained by their authors with step labels.
- PRISM: re-trains Qwen-PRM-7B on PRM800K step pairs. CROP: every threshold calibrated with labels; its LR proxies are also
  trained on 60% of each dataset. SLIDER: training-free score plus a threshold fitted on labelled data, with extra
  GPT-4o-mini calls.

## Not covered

- PRISM's downstream tests (Best-of-N, guided decoding, GRPO) need generation: not computed.
- CROP's Math-Shepherd set (its PRM is excluded from this study), boundary deviation and repair accuracy: not computed.
- CROP Table 17 does not name its proxy; FA and accept follow our reading of the column names.
- SLIDER's protocol is under-specified, so its comparison is approximate.

Files: `compute.py` (all numbers), `RESULTS.csv` (1,360 rows, long format), `PARITY.json`, `tables.py` (writes the tables below the summary, as TABLES_BODY.md).

## PRISM, Table 2: PRMBench error-detection rates

Positive class = a correct step. FPR = error steps passed as correct. Pool: 6,211 non-control answers, 83,371 steps.

| pair (standalone → PRM) | arm | access | FPR | FNR | Precision | TPR | TNR | PRMScore |
|---|---|---|---|---|---|---|---|---|
| published | Qwen-PRM-7B baseline (PRISM T2) | trained PRM | 69.34 | 4.27 | 89.40 | 95.36 | 30.66 | 65.50 |
| published | PRISM, re-trained Qwen-PRM-7B (PRISM T2) | trained on PRM800K pairs | 47.13 | 12.69 | 91.93 | 87.30 | 52.86 | 68.00 |
| Qwen2.5-Math-7B-Instruct → Qwen-PRM-7B | PRM, native threshold | trained PRM | 69.42 | 4.60 | 88.01 | 95.40 | 30.58 | 65.49 |
| Qwen2.5-Math-7B-Instruct → Qwen-PRM-7B | PRM, same rule as telemetry | trained PRM, label-free rule | 58.45 | 7.86 | 89.38 | 92.14 | 41.55 | 68.01 |
| Qwen2.5-Math-7B-Instruct → Qwen-PRM-7B | TELE16 (L-SML) | label-free | 50.29 | 11.50 | 90.38 | 88.50 | 49.71 | 68.26 |
| Qwen2.5-Math-7B-Instruct → Qwen-PRM-7B | TELE16 plain average | label-free | 52.66 | 11.68 | 89.96 | 88.32 | 47.34 | 67.14 |
| Qwen2.5-Math-72B-Instruct → Qwen-PRM-72B | PRM, native threshold | trained PRM | 61.34 | 6.22 | 89.09 | 93.78 | 38.66 | 68.18 |
| Qwen2.5-Math-72B-Instruct → Qwen-PRM-72B | PRM, same rule as telemetry | trained PRM, label-free rule | 54.73 | 8.08 | 89.97 | 91.92 | 45.27 | 69.49 |
| Qwen2.5-Math-72B-Instruct → Qwen-PRM-72B | TELE16 (L-SML) | label-free | 51.62 | 11.40 | 90.16 | 88.60 | 48.38 | 67.80 |
| Qwen2.5-Math-72B-Instruct → Qwen-PRM-72B | TELE16 plain average | label-free | 52.42 | 11.50 | 90.02 | 88.50 | 47.58 | 67.39 |
| Qwen2.5-Math-7B → PURE-PRM-7B | PRM, native threshold | trained PRM | 63.43 | 8.24 | 88.54 | 91.76 | 36.57 | 65.31 |
| Qwen2.5-Math-7B → PURE-PRM-7B | PRM, same rule as telemetry | trained PRM, label-free rule | 55.34 | 11.43 | 89.53 | 88.57 | 44.66 | 66.24 |
| Qwen2.5-Math-7B → PURE-PRM-7B | TELE16 (L-SML) | label-free | 54.07 | 10.43 | 89.84 | 89.57 | 45.93 | 67.63 |
| Qwen2.5-Math-7B → PURE-PRM-7B | TELE16 plain average | label-free | 54.99 | 10.64 | 89.67 | 89.36 | 45.01 | 67.06 |
| Llama-3.1-8B-Instruct → RLHFlow-PRM-8B | PRM, native threshold | trained PRM | 82.12 | 9.72 | 85.45 | 90.28 | 17.88 | 54.43 |
| Llama-3.1-8B-Instruct → RLHFlow-PRM-8B | PRM, same rule as telemetry | trained PRM, label-free rule | 68.07 | 13.03 | 87.22 | 86.97 | 31.93 | 59.39 |
| Llama-3.1-8B-Instruct → RLHFlow-PRM-8B | TELE16 (L-SML) | label-free | 53.40 | 10.11 | 89.99 | 89.89 | 46.60 | 68.20 |
| Llama-3.1-8B-Instruct → RLHFlow-PRM-8B | TELE16 plain average | label-free | 55.12 | 10.13 | 89.70 | 89.87 | 44.88 | 67.45 |

## PRISM, same operating point: global threshold set (with labels) so TPR ≈ 87.30

Same threshold access for every row; the PRM row moves the untouched PRM's threshold.

| pair | arm | FPR | FNR | Precision | TPR | TNR | PRMScore |
|---|---|---|---|---|---|---|---|
| published | PRISM (its own operating point) | 47.13 | 12.69 | 91.93 | 87.30 | 52.86 | 68.00 |
| Qwen2.5-Math-7B-Instruct → Qwen-PRM-7B | PRM, native threshold | 47.70 | 12.69 | 90.72 | 87.31 | 52.30 | 68.26 |
| Qwen2.5-Math-7B-Instruct → Qwen-PRM-7B | TELE16 (L-SML) | 52.82 | 12.71 | 89.82 | 87.29 | 47.18 | 66.21 |
| Qwen2.5-Math-7B-Instruct → Qwen-PRM-7B | TELE16 plain average | 54.55 | 12.68 | 89.53 | 87.32 | 45.45 | 65.52 |
| Qwen2.5-Math-72B-Instruct → Qwen-PRM-72B | PRM, native threshold | 45.25 | 12.67 | 91.16 | 87.33 | 54.75 | 69.22 |
| Qwen2.5-Math-72B-Instruct → Qwen-PRM-72B | TELE16 (L-SML) | 53.94 | 12.69 | 89.63 | 87.31 | 46.06 | 65.77 |
| Qwen2.5-Math-72B-Instruct → Qwen-PRM-72B | TELE16 plain average | 54.85 | 12.68 | 89.48 | 87.32 | 45.15 | 65.40 |
| Qwen2.5-Math-7B → PURE-PRM-7B | PRM, native threshold | 52.24 | 12.75 | 89.92 | 87.25 | 47.76 | 66.41 |
| Qwen2.5-Math-7B → PURE-PRM-7B | TELE16 (L-SML) | 54.43 | 12.72 | 89.54 | 87.28 | 45.57 | 65.54 |
| Qwen2.5-Math-7B → PURE-PRM-7B | TELE16 plain average | 55.29 | 12.70 | 89.40 | 87.30 | 44.71 | 65.20 |
| Llama-3.1-8B-Instruct → RLHFlow-PRM-8B | PRM, native threshold | 77.14 | 12.70 | 85.80 | 87.30 | 22.86 | 55.26 |
| Llama-3.1-8B-Instruct → RLHFlow-PRM-8B | TELE16 (L-SML) | 52.72 | 12.70 | 89.84 | 87.30 | 47.28 | 66.26 |
| Llama-3.1-8B-Instruct → RLHFlow-PRM-8B | TELE16 plain average | 54.24 | 12.68 | 89.58 | 87.32 | 45.76 | 65.65 |

## PRISM, Table 7: PRMBench PRMScore by error category

NR redundancy, NCL circular logic, ES empirical soundness, SC step consistency, DC domain consistency, CI confidence invariance, PS prerequisite sensitivity, DR deception resistance, MS multi-solution (no error steps: valid-step F1). NR and NCL go through the validity path (no redundancy head), as in every earlier table.

| pair | arm | Overall | NR | NCL | Simplicity | ES | SC | DC | CI | Soundness | PS | DR | MS | Sensitivity |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| published | Qwen-PRM-7B baseline (PRISM T7) | 65.5 | 49.1 | 55.0 | 52.1 | 71.7 | 67.4 | 66.3 | 78.5 | 71.0 | 57.7 | 69.1 | 99.7 | 75.5 |
| published | PRISM 132k curriculum-4 (PRISM T7) | 68.0 | 53.5 | 66.0 | 59.7 | 73.8 | 67.5 | 67.1 | 75.6 | 71.0 | 62.4 | 70.1 | 98.7 | 77.1 |
| published | PRISM 132k curriculum-3 (PRISM T7) | 68.0 | 52.3 | 63.7 | 58.0 | 73.9 | 68.5 | 67.4 | 77.4 | 71.8 | 61.1 | 70.6 | 99.3 | 77.0 |
| Qwen2.5-Math-7B-Instruct → Qwen-PRM-7B | PRM, native threshold | 65.5 | 49.0 | 54.9 | 52.0 | 71.7 | 67.3 | 66.4 | 78.5 | 71.0 | 57.8 | 69.1 | 99.9 | 75.6 |
| Qwen2.5-Math-7B-Instruct → Qwen-PRM-7B | PRM, same rule as telemetry | 68.0 | 55.2 | 63.1 | 59.1 | 73.1 | 70.1 | 68.2 | 79.3 | 72.7 | 63.4 | 70.6 | 94.6 | 76.2 |
| Qwen2.5-Math-7B-Instruct → Qwen-PRM-7B | TELE16 (L-SML) | 68.3 | 73.0 | 71.5 | 72.2 | 66.0 | 71.2 | 71.2 | 71.2 | 69.9 | 63.2 | 60.8 | 89.0 | 71.0 |
| Qwen2.5-Math-7B-Instruct → Qwen-PRM-7B | TELE16 plain average | 67.1 | 71.3 | 70.3 | 70.8 | 64.5 | 69.8 | 70.1 | 69.4 | 68.5 | 62.4 | 61.0 | 89.6 | 71.0 |
| Qwen2.5-Math-72B-Instruct → Qwen-PRM-72B | PRM, native threshold | 68.2 | 50.4 | 58.7 | 54.6 | 73.7 | 71.1 | 72.1 | 78.5 | 73.8 | 60.3 | 71.2 | 99.4 | 77.0 |
| Qwen2.5-Math-72B-Instruct → Qwen-PRM-72B | PRM, same rule as telemetry | 69.5 | 59.8 | 64.9 | 62.3 | 74.0 | 70.9 | 72.0 | 79.1 | 74.0 | 63.0 | 71.1 | 94.6 | 76.2 |
| Qwen2.5-Math-72B-Instruct → Qwen-PRM-72B | TELE16 (L-SML) | 67.8 | 73.8 | 70.3 | 72.0 | 65.5 | 70.0 | 70.6 | 71.4 | 69.4 | 62.8 | 60.1 | 88.8 | 70.6 |
| Qwen2.5-Math-72B-Instruct → Qwen-PRM-72B | TELE16 plain average | 67.4 | 71.2 | 70.2 | 70.7 | 65.3 | 69.3 | 70.5 | 70.0 | 68.8 | 63.0 | 61.2 | 90.1 | 71.4 |
| Qwen2.5-Math-7B → PURE-PRM-7B | PRM, native threshold | 65.3 | 49.1 | 55.1 | 52.1 | 71.1 | 68.9 | 64.0 | 76.9 | 70.2 | 60.3 | 69.3 | 97.9 | 75.8 |
| Qwen2.5-Math-7B → PURE-PRM-7B | PRM, same rule as telemetry | 66.2 | 53.5 | 60.4 | 56.9 | 71.4 | 68.9 | 64.3 | 77.5 | 70.5 | 62.3 | 69.5 | 94.5 | 75.4 |
| Qwen2.5-Math-7B → PURE-PRM-7B | TELE16 (L-SML) | 67.6 | 73.1 | 69.9 | 71.5 | 63.8 | 70.9 | 71.4 | 72.1 | 69.5 | 62.3 | 59.9 | 90.9 | 71.0 |
| Qwen2.5-Math-7B → PURE-PRM-7B | TELE16 plain average | 67.1 | 71.3 | 69.2 | 70.3 | 64.1 | 70.6 | 70.1 | 70.7 | 68.9 | 62.4 | 60.4 | 90.4 | 71.1 |
| Llama-3.1-8B-Instruct → RLHFlow-PRM-8B | PRM, native threshold | 54.4 | 46.2 | 47.4 | 46.8 | 56.6 | 55.2 | 54.4 | 63.8 | 57.5 | 51.5 | 56.2 | 97.9 | 68.5 |
| Llama-3.1-8B-Instruct → RLHFlow-PRM-8B | PRM, same rule as telemetry | 59.4 | 49.6 | 58.3 | 53.9 | 61.9 | 60.6 | 59.1 | 67.7 | 62.3 | 57.1 | 60.5 | 91.2 | 69.6 |
| Llama-3.1-8B-Instruct → RLHFlow-PRM-8B | TELE16 (L-SML) | 68.2 | 73.8 | 70.5 | 72.1 | 66.1 | 71.8 | 72.3 | 72.1 | 70.6 | 61.7 | 60.0 | 90.1 | 70.6 |
| Llama-3.1-8B-Instruct → RLHFlow-PRM-8B | TELE16 plain average | 67.5 | 72.5 | 69.5 | 71.0 | 65.7 | 70.4 | 70.5 | 70.1 | 69.2 | 61.7 | 61.2 | 90.6 | 71.2 |

## PRISM, Table 6 (appendix 10.2): ProcessBench

TPR = accuracy on samples with an error (exact first-error step), TNR = accuracy on error-free samples, F1 = their harmonic mean. PRISM prints Omni-MATH and OlympiadBench cells that do not fit F1 = harmonic mean (Omni-MATH 54.8/66.1 → 83.4); they are copied as printed. Q80 is pure telemetry (first flagged step, or "no error" when nothing is flagged). NC uses the PRM's native gate and error count with the arm's order, so it is NOT telemetry only.

| pair | arm | rule | GSM8K TPR | GSM8K TNR | GSM8K F1 | MATH TPR | MATH TNR | MATH F1 | Omni-MATH TPR | Omni-MATH TNR | Omni-MATH F1 | OlympiadBench TPR | OlympiadBench TNR | OlympiadBench F1 | avg F1 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| published | Qwen-PRM-7B baseline (PRISM T6) | native | 71.0 | 96.4 | 81.8 | 67.0 | 91.1 | 77.2 | 54.8 | 66.1 | 83.4 | 54.6 | 85.5 | 66.7 | 77.3 |
| published | PRISM CL1 (PRISM T6) | native | 72.0 | 94.8 | 81.8 | 69.4 | 86.9 | 77.2 | 55.6 | 66.2 | 81.7 | 58.2 | 79.6 | 67.3 | 77.0 |
| published | PRISM CL2 (PRISM T6) | native | 73.4 | 94.3 | 82.6 | 70.5 | 85.0 | 77.1 | 58.2 | 65.8 | 75.5 | 58.9 | 75.5 | 66.1 | 75.3 |
| published | PRISM CL3 (PRISM T6) | native | 73.4 | 93.8 | 82.4 | 72.6 | 82.0 | 77.0 | 57.8 | 64.8 | 72.6 | 58.5 | 72.6 | 64.8 | 74.2 |
| Qwen2.5-Math-7B-Instruct → Qwen-PRM-7B | PRM, native threshold | native | 71.0 | 96.4 | 81.8 | 67.0 | 91.1 | 77.2 | 54.7 | 83.4 | 66.1 | 54.8 | 85.3 | 66.7 | 72.9 |
| Qwen2.5-Math-7B-Instruct → Qwen-PRM-7B | PRM, same rule as telemetry | Q80 | 70.5 | 40.4 | 51.4 | 63.5 | 22.2 | 32.9 | 51.6 | 19.1 | 27.9 | 56.1 | 17.7 | 26.9 | 34.8 |
| Qwen2.5-Math-7B-Instruct → Qwen-PRM-7B | TELE16 (L-SML) | Q80 (telemetry only) | 48.3 | 7.8 | 13.4 | 33.2 | 5.9 | 10.0 | 34.0 | 6.2 | 10.5 | 32.5 | 3.5 | 6.4 | 10.1 |
| Qwen2.5-Math-7B-Instruct → Qwen-PRM-7B | TELE16 plain average | Q80 (telemetry only) | 51.7 | 8.3 | 14.3 | 35.5 | 6.7 | 11.2 | 35.3 | 7.1 | 11.8 | 31.8 | 3.5 | 6.4 | 10.9 |
| Qwen2.5-Math-7B-Instruct → Qwen-PRM-7B | TELE16 (L-SML) | NC (PRM gate + count) | 46.9 | 96.4 | 63.1 | 39.9 | 91.1 | 55.5 | 35.6 | 83.4 | 49.9 | 33.4 | 85.3 | 48.0 | 54.1 |
| Qwen2.5-Math-7B-Instruct → Qwen-PRM-7B | TELE16 plain average | NC (PRM gate + count) | 48.8 | 96.4 | 64.8 | 40.7 | 91.1 | 56.3 | 34.3 | 83.4 | 48.6 | 32.7 | 85.3 | 47.2 | 54.2 |
| Qwen2.5-Math-72B-Instruct → Qwen-PRM-72B | PRM, native threshold | native | 78.3 | 97.9 | 87.0 | 73.9 | 87.9 | 80.3 | 65.0 | 79.3 | 71.4 | 67.8 | 81.7 | 74.1 | 78.2 |
| Qwen2.5-Math-72B-Instruct → Qwen-PRM-72B | PRM, same rule as telemetry | Q80 | 70.5 | 35.8 | 47.5 | 66.7 | 20.4 | 31.3 | 59.3 | 14.9 | 23.9 | 63.7 | 6.8 | 12.3 | 28.7 |
| Qwen2.5-Math-72B-Instruct → Qwen-PRM-72B | TELE16 (L-SML) | Q80 (telemetry only) | 45.4 | 5.7 | 10.1 | 37.4 | 6.4 | 10.9 | 33.5 | 5.8 | 9.9 | 30.7 | 4.7 | 8.2 | 9.8 |
| Qwen2.5-Math-72B-Instruct → Qwen-PRM-72B | TELE16 plain average | Q80 (telemetry only) | 45.9 | 6.2 | 11.0 | 38.0 | 5.9 | 10.2 | 34.8 | 6.2 | 10.6 | 31.8 | 4.7 | 8.2 | 10.0 |
| Qwen2.5-Math-72B-Instruct → Qwen-PRM-72B | TELE16 (L-SML) | NC (PRM gate + count) | 34.3 | 97.9 | 50.8 | 38.2 | 87.9 | 53.3 | 32.7 | 79.3 | 46.3 | 28.9 | 81.7 | 42.7 | 48.3 |
| Qwen2.5-Math-72B-Instruct → Qwen-PRM-72B | TELE16 plain average | NC (PRM gate + count) | 34.8 | 97.9 | 51.3 | 39.1 | 87.9 | 54.1 | 33.7 | 79.3 | 47.3 | 29.0 | 81.7 | 42.9 | 48.9 |
| Qwen2.5-Math-7B → PURE-PRM-7B | PRM, native threshold | native | 47.8 | 93.8 | 63.3 | 51.3 | 81.3 | 62.9 | 25.4 | 80.9 | 38.7 | 31.3 | 82.0 | 45.3 | 52.6 |
| Qwen2.5-Math-7B → PURE-PRM-7B | PRM, same rule as telemetry | Q80 | 48.3 | 8.8 | 14.9 | 44.9 | 11.8 | 18.7 | 30.3 | 3.7 | 6.6 | 39.8 | 5.0 | 8.9 | 12.3 |
| Qwen2.5-Math-7B → PURE-PRM-7B | TELE16 (L-SML) | Q80 (telemetry only) | 40.6 | 4.7 | 8.4 | 34.5 | 5.9 | 10.1 | 30.2 | 7.5 | 12.0 | 25.6 | 6.5 | 10.4 | 10.2 |
| Qwen2.5-Math-7B → PURE-PRM-7B | TELE16 plain average | Q80 (telemetry only) | 45.4 | 4.7 | 8.5 | 35.5 | 4.7 | 8.3 | 32.8 | 4.6 | 8.0 | 28.0 | 4.7 | 8.1 | 8.2 |
| Qwen2.5-Math-7B → PURE-PRM-7B | TELE16 (L-SML) | NC (PRM gate + count) | 22.2 | 93.8 | 35.9 | 32.2 | 81.3 | 46.1 | 21.1 | 80.9 | 33.4 | 19.8 | 82.0 | 31.9 | 36.8 |
| Qwen2.5-Math-7B → PURE-PRM-7B | TELE16 plain average | NC (PRM gate + count) | 26.1 | 93.8 | 40.8 | 32.2 | 81.3 | 46.1 | 22.0 | 80.9 | 34.6 | 21.0 | 82.0 | 33.5 | 38.7 |
| Llama-3.1-8B-Instruct → RLHFlow-PRM-8B | PRM, native threshold | native | 35.3 | 97.9 | 51.9 | 22.7 | 70.9 | 34.4 | 9.9 | 43.6 | 16.1 | 8.5 | 44.8 | 14.3 | 29.2 |
| Llama-3.1-8B-Instruct → RLHFlow-PRM-8B | PRM, same rule as telemetry | Q80 | 48.8 | 1.0 | 2.0 | 29.1 | 1.2 | 2.4 | 11.7 | 0.8 | 1.6 | 10.7 | 0.3 | 0.6 | 1.6 |
| Llama-3.1-8B-Instruct → RLHFlow-PRM-8B | TELE16 (L-SML) | Q80 (telemetry only) | 44.9 | 3.6 | 6.7 | 36.7 | 5.7 | 9.8 | 29.2 | 1.7 | 3.1 | 28.1 | 5.0 | 8.5 | 7.0 |
| Llama-3.1-8B-Instruct → RLHFlow-PRM-8B | TELE16 plain average | Q80 (telemetry only) | 46.9 | 3.6 | 6.7 | 36.2 | 6.7 | 11.2 | 29.8 | 2.1 | 3.9 | 28.3 | 4.1 | 7.2 | 7.3 |
| Llama-3.1-8B-Instruct → RLHFlow-PRM-8B | TELE16 (L-SML) | NC (PRM gate + count) | 22.2 | 97.9 | 36.2 | 25.9 | 70.9 | 38.0 | 27.3 | 43.6 | 33.5 | 22.7 | 44.8 | 30.1 | 34.5 |
| Llama-3.1-8B-Instruct → RLHFlow-PRM-8B | TELE16 plain average | NC (PRM gate + count) | 20.8 | 97.9 | 34.3 | 25.6 | 70.9 | 37.6 | 28.1 | 43.6 | 34.1 | 21.6 | 44.8 | 29.2 | 33.8 |

## SLIDER / Step-RRI: PRMBench redundancy class (758 answers, positive = redundant step)

SLIDER calibrates a threshold with a linear classifier on a held-out labelled set and reports "accuracy"; split, class weighting and step-1 handling are not stated. Our reading: 10 trace-level 50/50 splits, 1-D logistic regression on the score. About 87% of steps are not redundant, so plain accuracy rewards "never flag" (majority row). steps ≥ 2 variants are in RESULTS.csv.

| pair | arm | acc, LR threshold (balanced) | balanced acc, LR (balanced) | acc, LR threshold (unweighted) | acc, label-free rule | balanced acc, label-free rule |
|---|---|---|---|---|---|---|
| published | Step-RRI (SLIDER Fig. 7) | 61.91 | — | — | — | — |
| published | InfoGain, QwQ-32B (implied: 61.91 − 13.53) | 48.38 | — | — | — | — |
| published | Cosine, MiniLM (implied: 61.91 − 10.31) | 51.60 | — | — | — | — |
| ours | always "not redundant" (majority) | 86.76 | — | — | — | 50.00 |
| Qwen2.5-Math-7B-Instruct → Qwen-PRM-7B | PRM, native threshold | 50.44 | 60.36 | 86.80 | 85.49 | 50.73 |
| Qwen2.5-Math-7B-Instruct → Qwen-PRM-7B | PRM, same rule as telemetry | 78.81 | 54.54 | 86.80 | 81.83 | 54.74 |
| Qwen2.5-Math-7B-Instruct → Qwen-PRM-7B | TELE16 (L-SML) | 76.15 | 75.95 | 89.10 | 86.71 | 74.70 |
| Qwen2.5-Math-7B-Instruct → Qwen-PRM-7B | TELE16 plain average | 74.62 | 74.06 | 88.39 | 86.00 | 72.69 |
| Qwen2.5-Math-72B-Instruct → Qwen-PRM-72B | PRM, native threshold | 69.83 | 65.77 | 86.79 | 85.24 | 51.44 |
| Qwen2.5-Math-72B-Instruct → Qwen-PRM-72B | PRM, same rule as telemetry | 82.93 | 56.09 | 86.80 | 83.91 | 58.72 |
| Qwen2.5-Math-72B-Instruct → Qwen-PRM-72B | TELE16 (L-SML) | 76.41 | 75.98 | 88.45 | 87.16 | 75.40 |
| Qwen2.5-Math-72B-Instruct → Qwen-PRM-72B | TELE16 plain average | 74.21 | 73.92 | 88.20 | 85.79 | 72.87 |
| Qwen2.5-Math-7B → PURE-PRM-7B | PRM, native threshold | 59.28 | 62.06 | 86.80 | 83.30 | 50.16 |
| Qwen2.5-Math-7B → PURE-PRM-7B | PRM, same rule as telemetry | 75.14 | 57.18 | 86.80 | 79.80 | 53.29 |
| Qwen2.5-Math-7B → PURE-PRM-7B | TELE16 (L-SML) | 76.46 | 75.68 | 88.39 | 87.05 | 74.20 |
| Qwen2.5-Math-7B → PURE-PRM-7B | TELE16 plain average | 74.23 | 73.32 | 88.10 | 86.43 | 72.00 |
| Llama-3.1-8B-Instruct → RLHFlow-PRM-8B | PRM, native threshold | 54.56 | 52.72 | 86.80 | 80.48 | 47.45 |
| Llama-3.1-8B-Instruct → RLHFlow-PRM-8B | PRM, same rule as telemetry | 54.56 | 52.72 | 86.80 | 76.22 | 49.55 |
| Llama-3.1-8B-Instruct → RLHFlow-PRM-8B | TELE16 (L-SML) | 78.58 | 77.47 | 88.46 | 87.79 | 74.08 |
| Llama-3.1-8B-Instruct → RLHFlow-PRM-8B | TELE16 plain average | 76.72 | 75.50 | 88.09 | 86.96 | 73.16 |

## CROP, Table 6: conformal clean prefix at α = 0.05

Step AUROC pooled over test steps; kept = mean retained-prefix fraction; risk = share of test traces whose kept prefix contains an error (target ≤ 5%). 10 splits 60/20/20, seeds 2806–2815. CROP calibrates every row's threshold with labels on the 20% calibration split; their LR rows are also TRAINED on the 60% split. Our rows use no training split.

### PRMBench

Our PRMBench uses all 6,211 non-control answers (CROP sampled 5,000), split assignment re-implemented.

| pair | arm | step AUROC | kept % | risk % |
|---|---|---|---|---|
| published | Random (CROP T6) | 0.4994 | 8.7 | 4.7 |
| published | Token/format LR (CROP T6) | 0.7751 | 16.2 | 5.0 |
| published | Trace features LR (Llama-3.1-8B) (CROP T6) | 0.7812 | 15.9 | 4.6 |
| published | Direct PRM (Qwen-PRM-7B) (CROP T6) | 0.8010 | 4.8 | 0.6 |
| published | Trace+PRM LR (CROP T6) | 0.8336 | 21.8 | 5.0 |
| Qwen2.5-Math-7B-Instruct → Qwen-PRM-7B | PRM (1 − reward), CROP fixed 101 grid | 0.7999 | 4.8 | 0.6 |
| Qwen2.5-Math-7B-Instruct → Qwen-PRM-7B | PRM (1 − reward), 201-quantile grid | 0.7999 | 17.2 | 3.9 |
| Qwen2.5-Math-7B-Instruct → Qwen-PRM-7B | TELE16 (L-SML), 201-quantile grid | 0.7440 | 18.2 | 4.9 |
| Qwen2.5-Math-7B-Instruct → Qwen-PRM-7B | TELE16 plain average, 201-quantile grid | 0.7302 | 17.9 | 4.8 |
| Qwen2.5-Math-72B-Instruct → Qwen-PRM-72B | PRM (1 − reward), CROP fixed 101 grid | 0.7844 | 20.4 | 7.3 |
| Qwen2.5-Math-72B-Instruct → Qwen-PRM-72B | PRM (1 − reward), 201-quantile grid | 0.7844 | 20.4 | 7.3 |
| Qwen2.5-Math-72B-Instruct → Qwen-PRM-72B | TELE16 (L-SML), 201-quantile grid | 0.7316 | 16.7 | 4.7 |
| Qwen2.5-Math-72B-Instruct → Qwen-PRM-72B | TELE16 plain average, 201-quantile grid | 0.7255 | 17.2 | 5.0 |
| Qwen2.5-Math-7B → PURE-PRM-7B | PRM (1 − reward), CROP fixed 101 grid | 0.7811 | 13.7 | 3.4 |
| Qwen2.5-Math-7B → PURE-PRM-7B | PRM (1 − reward), 201-quantile grid | 0.7811 | 16.9 | 4.6 |
| Qwen2.5-Math-7B → PURE-PRM-7B | TELE16 (L-SML), 201-quantile grid | 0.7167 | 16.4 | 4.9 |
| Qwen2.5-Math-7B → PURE-PRM-7B | TELE16 plain average, 201-quantile grid | 0.7127 | 16.1 | 4.8 |
| Llama-3.1-8B-Instruct → RLHFlow-PRM-8B | PRM (1 − reward), CROP fixed 101 grid | 0.6307 | 8.3 | 4.5 |
| Llama-3.1-8B-Instruct → RLHFlow-PRM-8B | PRM (1 − reward), 201-quantile grid | 0.6307 | 8.7 | 4.7 |
| Llama-3.1-8B-Instruct → RLHFlow-PRM-8B | TELE16 (L-SML), 201-quantile grid | 0.7343 | 18.2 | 4.8 |
| Llama-3.1-8B-Instruct → RLHFlow-PRM-8B | TELE16 plain average, 201-quantile grid | 0.7271 | 18.1 | 4.7 |

### ProcessBench

Same 3,400 traces / 25,697 steps as CROP Table 4; labels mark the first-error step only (8.6% of steps here; CROP lists 6.6%, not explained).

| pair | arm | step AUROC | kept % | risk % |
|---|---|---|---|---|
| published | Random (CROP T6) | 0.4887 | 9.8 | 4.9 |
| published | Token/format LR (CROP T6) | 0.7057 | 10.6 | 4.8 |
| published | Trace features LR (Llama-3.1-8B) (CROP T6) | 0.7058 | 11.3 | 4.5 |
| published | Direct PRM (Qwen-PRM-7B) (CROP T6) | 0.8801 | 46.7 | 4.5 |
| published | Trace+PRM LR (CROP T6) | 0.8935 | 42.4 | 4.5 |
| Qwen2.5-Math-7B-Instruct → Qwen-PRM-7B | PRM (1 − reward), CROP fixed 101 grid | 0.8834 | 44.0 | 5.1 |
| Qwen2.5-Math-7B-Instruct → Qwen-PRM-7B | PRM (1 − reward), 201-quantile grid | 0.8834 | 44.2 | 5.2 |
| Qwen2.5-Math-7B-Instruct → Qwen-PRM-7B | TELE16 (L-SML), 201-quantile grid | 0.7352 | 10.8 | 4.9 |
| Qwen2.5-Math-7B-Instruct → Qwen-PRM-7B | TELE16 plain average, 201-quantile grid | 0.7390 | 11.1 | 5.0 |
| Qwen2.5-Math-72B-Instruct → Qwen-PRM-72B | PRM (1 − reward), CROP fixed 101 grid | 0.9071 | 49.7 | 4.9 |
| Qwen2.5-Math-72B-Instruct → Qwen-PRM-72B | PRM (1 − reward), 201-quantile grid | 0.9071 | 49.6 | 4.8 |
| Qwen2.5-Math-72B-Instruct → Qwen-PRM-72B | TELE16 (L-SML), 201-quantile grid | 0.7525 | 11.1 | 4.9 |
| Qwen2.5-Math-72B-Instruct → Qwen-PRM-72B | TELE16 plain average, 201-quantile grid | 0.7567 | 11.8 | 5.0 |
| Qwen2.5-Math-7B → PURE-PRM-7B | PRM (1 − reward), CROP fixed 101 grid | 0.7646 | 27.6 | 4.7 |
| Qwen2.5-Math-7B → PURE-PRM-7B | PRM (1 − reward), 201-quantile grid | 0.7646 | 28.1 | 4.9 |
| Qwen2.5-Math-7B → PURE-PRM-7B | TELE16 (L-SML), 201-quantile grid | 0.7351 | 10.5 | 5.3 |
| Qwen2.5-Math-7B → PURE-PRM-7B | TELE16 plain average, 201-quantile grid | 0.7416 | 11.1 | 5.3 |
| Llama-3.1-8B-Instruct → RLHFlow-PRM-8B | PRM (1 − reward), CROP fixed 101 grid | 0.6989 | 20.6 | 4.6 |
| Llama-3.1-8B-Instruct → RLHFlow-PRM-8B | PRM (1 − reward), 201-quantile grid | 0.6989 | 20.8 | 4.7 |
| Llama-3.1-8B-Instruct → RLHFlow-PRM-8B | TELE16 (L-SML), 201-quantile grid | 0.7306 | 10.8 | 5.3 |
| Llama-3.1-8B-Instruct → RLHFlow-PRM-8B | TELE16 plain average, 201-quantile grid | 0.7294 | 10.8 | 4.9 |

### CRV-Arithmetic

Inputs rebuilt with CROP's selection rules (counts match Table 4). CROP used 20 seeds on CRV; we use 10.

| pair | arm | step AUROC | kept % | risk % |
|---|---|---|---|---|
| published | Random (CROP T6) | 0.5049 | 62.0 | 4.6 |
| published | Token/format LR (CROP T6) | 0.8168 | 77.5 | 4.7 |
| published | Trace features LR (Llama-3.1-8B) (CROP T6) | 0.8564 | 90.3 | 4.5 |
| published | Direct PRM (Qwen-PRM-7B) (CROP T6) | 0.9421 | 97.5 | 4.2 |
| published | Trace+PRM LR (CROP T6) | 0.9701 | 97.1 | 4.1 |
| Qwen2.5-Math-7B-Instruct → Qwen-PRM-7B | PRM (1 − reward), CROP fixed 101 grid | 0.9433 | 97.4 | 4.0 |
| Qwen2.5-Math-7B-Instruct → Qwen-PRM-7B | PRM (1 − reward), 201-quantile grid | 0.9433 | 96.9 | 3.4 |
| Qwen2.5-Math-7B-Instruct → Qwen-PRM-7B | TELE16 (L-SML), 201-quantile grid | 0.6137 | 43.4 | 4.9 |
| Qwen2.5-Math-7B-Instruct → Qwen-PRM-7B | TELE16 plain average, 201-quantile grid | 0.5483 | 44.8 | 4.7 |
| Qwen2.5-Math-72B-Instruct → Qwen-PRM-72B | PRM (1 − reward), CROP fixed 101 grid | 0.9783 | 96.7 | 2.4 |
| Qwen2.5-Math-72B-Instruct → Qwen-PRM-72B | PRM (1 − reward), 201-quantile grid | 0.9783 | 97.0 | 3.3 |
| Qwen2.5-Math-72B-Instruct → Qwen-PRM-72B | TELE16 (L-SML), 201-quantile grid | 0.5304 | 36.4 | 4.2 |
| Qwen2.5-Math-72B-Instruct → Qwen-PRM-72B | TELE16 plain average, 201-quantile grid | 0.5033 | 41.7 | 4.5 |
| Qwen2.5-Math-7B → PURE-PRM-7B | PRM (1 − reward), CROP fixed 101 grid | 0.8772 | 90.4 | 4.2 |
| Qwen2.5-Math-7B → PURE-PRM-7B | PRM (1 − reward), 201-quantile grid | 0.8772 | 90.3 | 4.1 |
| Qwen2.5-Math-7B → PURE-PRM-7B | TELE16 (L-SML), 201-quantile grid | 0.4101 | 49.9 | 4.9 |
| Qwen2.5-Math-7B → PURE-PRM-7B | TELE16 plain average, 201-quantile grid | 0.4128 | 50.4 | 5.0 |
| Llama-3.1-8B-Instruct → RLHFlow-PRM-8B | PRM (1 − reward), CROP fixed 101 grid | 0.8604 | 95.6 | 4.7 |
| Llama-3.1-8B-Instruct → RLHFlow-PRM-8B | PRM (1 − reward), 201-quantile grid | 0.8604 | 95.1 | 4.4 |
| Llama-3.1-8B-Instruct → RLHFlow-PRM-8B | TELE16 (L-SML), 201-quantile grid | 0.4854 | 52.5 | 4.4 |
| Llama-3.1-8B-Instruct → RLHFlow-PRM-8B | TELE16 plain average, 201-quantile grid | 0.4724 | 52.2 | 4.2 |

### CRV-GSM8K

As CRV-Arithmetic.

| pair | arm | step AUROC | kept % | risk % |
|---|---|---|---|---|
| published | Random (CROP T6) | 0.5046 | 27.2 | 4.3 |
| published | Token/format LR (CROP T6) | 0.7081 | 39.8 | 4.9 |
| published | Trace features LR (Llama-3.1-8B) (CROP T6) | 0.7495 | 43.2 | 4.3 |
| published | Direct PRM (Qwen-PRM-7B) (CROP T6) | 0.8659 | 81.7 | 5.1 |
| published | Trace+PRM LR (CROP T6) | 0.8812 | 63.1 | 4.5 |
| Qwen2.5-Math-7B-Instruct → Qwen-PRM-7B | PRM (1 − reward), CROP fixed 101 grid | 0.8486 | 79.8 | 5.0 |
| Qwen2.5-Math-7B-Instruct → Qwen-PRM-7B | PRM (1 − reward), 201-quantile grid | 0.8486 | 80.9 | 5.2 |
| Qwen2.5-Math-7B-Instruct → Qwen-PRM-7B | TELE16 (L-SML), 201-quantile grid | 0.5541 | 27.6 | 5.2 |
| Qwen2.5-Math-7B-Instruct → Qwen-PRM-7B | TELE16 plain average, 201-quantile grid | 0.5539 | 28.8 | 5.1 |
| Qwen2.5-Math-72B-Instruct → Qwen-PRM-72B | PRM (1 − reward), CROP fixed 101 grid | 0.8792 | 80.8 | 5.2 |
| Qwen2.5-Math-72B-Instruct → Qwen-PRM-72B | PRM (1 − reward), 201-quantile grid | 0.8792 | 81.4 | 5.2 |
| Qwen2.5-Math-72B-Instruct → Qwen-PRM-72B | TELE16 (L-SML), 201-quantile grid | 0.5336 | 33.2 | 5.1 |
| Qwen2.5-Math-72B-Instruct → Qwen-PRM-72B | TELE16 plain average, 201-quantile grid | 0.5333 | 31.8 | 4.9 |
| Qwen2.5-Math-7B → PURE-PRM-7B | PRM (1 − reward), CROP fixed 101 grid | 0.8471 | 62.5 | 5.5 |
| Qwen2.5-Math-7B → PURE-PRM-7B | PRM (1 − reward), 201-quantile grid | 0.8471 | 63.0 | 5.5 |
| Qwen2.5-Math-7B → PURE-PRM-7B | TELE16 (L-SML), 201-quantile grid | 0.5050 | 32.2 | 4.8 |
| Qwen2.5-Math-7B → PURE-PRM-7B | TELE16 plain average, 201-quantile grid | 0.5071 | 32.4 | 4.4 |
| Llama-3.1-8B-Instruct → RLHFlow-PRM-8B | PRM (1 − reward), CROP fixed 101 grid | 0.7967 | 59.8 | 4.5 |
| Llama-3.1-8B-Instruct → RLHFlow-PRM-8B | PRM (1 − reward), 201-quantile grid | 0.7967 | 62.3 | 4.8 |
| Llama-3.1-8B-Instruct → RLHFlow-PRM-8B | TELE16 (L-SML), 201-quantile grid | 0.5196 | 33.3 | 5.1 |
| Llama-3.1-8B-Instruct → RLHFlow-PRM-8B | TELE16 plain average, 201-quantile grid | 0.5208 | 32.1 | 5.1 |

### PRM800K

Inputs rebuilt with CROP's selection rules (counts match Table 4).

| pair | arm | step AUROC | kept % | risk % |
|---|---|---|---|---|
| published | Random (CROP T6) | 0.5002 | 20.5 | 5.4 |
| published | Token/format LR (CROP T6) | 0.9919 | 89.0 | 4.8 |
| published | Trace features LR (Llama-3.1-8B) (CROP T6) | 0.9925 | 89.4 | 4.8 |
| published | Direct PRM (Qwen-PRM-7B) (CROP T6) | 0.9327 | 67.2 | 4.5 |
| published | Trace+PRM LR (CROP T6) | 0.9934 | 89.5 | 4.6 |
| Qwen2.5-Math-7B-Instruct → Qwen-PRM-7B | PRM (1 − reward), CROP fixed 101 grid | 0.9299 | 68.3 | 4.7 |
| Qwen2.5-Math-7B-Instruct → Qwen-PRM-7B | PRM (1 − reward), 201-quantile grid | 0.9299 | 69.1 | 5.0 |
| Qwen2.5-Math-7B-Instruct → Qwen-PRM-7B | TELE16 (L-SML), 201-quantile grid | 0.5003 | 14.7 | 5.4 |
| Qwen2.5-Math-7B-Instruct → Qwen-PRM-7B | TELE16 plain average, 201-quantile grid | 0.6535 | 20.1 | 5.1 |
| Qwen2.5-Math-72B-Instruct → Qwen-PRM-72B | PRM (1 − reward), CROP fixed 101 grid | 0.9500 | 75.8 | 5.0 |
| Qwen2.5-Math-72B-Instruct → Qwen-PRM-72B | PRM (1 − reward), 201-quantile grid | 0.9500 | 75.4 | 5.0 |
| Qwen2.5-Math-72B-Instruct → Qwen-PRM-72B | TELE16 (L-SML), 201-quantile grid | 0.4975 | 17.9 | 5.3 |
| Qwen2.5-Math-72B-Instruct → Qwen-PRM-72B | TELE16 plain average, 201-quantile grid | 0.6336 | 19.9 | 5.2 |
| Qwen2.5-Math-7B → PURE-PRM-7B | PRM (1 − reward), CROP fixed 101 grid | 0.8781 | 61.6 | 5.4 |
| Qwen2.5-Math-7B → PURE-PRM-7B | PRM (1 − reward), 201-quantile grid | 0.8781 | 61.8 | 5.4 |
| Qwen2.5-Math-7B → PURE-PRM-7B | TELE16 (L-SML), 201-quantile grid | 0.5251 | 17.3 | 5.5 |
| Qwen2.5-Math-7B → PURE-PRM-7B | TELE16 plain average, 201-quantile grid | 0.6137 | 17.1 | 5.2 |
| Llama-3.1-8B-Instruct → RLHFlow-PRM-8B | PRM (1 − reward), CROP fixed 101 grid | 0.7477 | 36.9 | 4.6 |
| Llama-3.1-8B-Instruct → RLHFlow-PRM-8B | PRM (1 − reward), 201-quantile grid | 0.7477 | 37.3 | 4.7 |
| Llama-3.1-8B-Instruct → RLHFlow-PRM-8B | TELE16 (L-SML), 201-quantile grid | 0.5532 | 22.8 | 5.1 |
| Llama-3.1-8B-Instruct → RLHFlow-PRM-8B | TELE16 plain average, 201-quantile grid | 0.6612 | 24.2 | 5.1 |

## CROP, Table 17: cross-domain transfer (calibrate on one CRV set, test on the other)

Exchangeability is broken on purpose; CROP presents this as a failure diagnostic. The paper does not name the proxy in this table. FA = whole traces accepted that contain an error; accept = whole traces accepted (our reading of the columns; % of target test traces).

| pair | arm | calibrate → test | risk % | kept % | FA % | accept % |
|---|---|---|---|---|---|---|
| published | CROP (CROP T17) | CRV-Arith → CRV-GSM8K | 14.8 | 69.9 | 13.3 | 62.5 |
| published | CROP (CROP T17) | CRV-GSM8K → CRV-Arith | 5.1 | 69.0 | 4.4 | 54.7 |
| Qwen2.5-Math-7B-Instruct → Qwen-PRM-7B | PRM (1 − reward), fixed grid | CRV-Arith → CRV-GSM8K | 15.6 | 95.8 | 14.0 | 92.3 |
| Qwen2.5-Math-7B-Instruct → Qwen-PRM-7B | PRM (1 − reward), q201 | CRV-Arith → CRV-GSM8K | 14.5 | 94.8 | 12.6 | 90.7 |
| Qwen2.5-Math-7B-Instruct → Qwen-PRM-7B | TELE16 (L-SML), q201 | CRV-Arith → CRV-GSM8K | 11.2 | 54.3 | 9.2 | 35.5 |
| Qwen2.5-Math-7B-Instruct → Qwen-PRM-7B | TELE16 plain average, q201 | CRV-Arith → CRV-GSM8K | 10.7 | 53.6 | 8.6 | 33.2 |
| Qwen2.5-Math-72B-Instruct → Qwen-PRM-72B | PRM (1 − reward), fixed grid | CRV-Arith → CRV-GSM8K | 13.5 | 94.2 | 11.6 | 89.5 |
| Qwen2.5-Math-72B-Instruct → Qwen-PRM-72B | PRM (1 − reward), q201 | CRV-Arith → CRV-GSM8K | 14.2 | 94.7 | 12.5 | 90.4 |
| Qwen2.5-Math-72B-Instruct → Qwen-PRM-72B | TELE16 (L-SML), q201 | CRV-Arith → CRV-GSM8K | 11.1 | 58.4 | 9.1 | 40.1 |
| Qwen2.5-Math-72B-Instruct → Qwen-PRM-72B | TELE16 plain average, q201 | CRV-Arith → CRV-GSM8K | 12.6 | 65.7 | 11.2 | 51.0 |
| Qwen2.5-Math-7B → PURE-PRM-7B | PRM (1 − reward), fixed grid | CRV-Arith → CRV-GSM8K | 11.1 | 89.0 | 8.8 | 78.8 |
| Qwen2.5-Math-7B → PURE-PRM-7B | PRM (1 − reward), q201 | CRV-Arith → CRV-GSM8K | 10.9 | 88.9 | 8.7 | 78.6 |
| Qwen2.5-Math-7B → PURE-PRM-7B | TELE16 (L-SML), q201 | CRV-Arith → CRV-GSM8K | 10.8 | 61.1 | 9.2 | 48.0 |
| Qwen2.5-Math-7B → PURE-PRM-7B | TELE16 plain average, q201 | CRV-Arith → CRV-GSM8K | 10.0 | 57.6 | 8.2 | 42.3 |
| Llama-3.1-8B-Instruct → RLHFlow-PRM-8B | PRM (1 − reward), fixed grid | CRV-Arith → CRV-GSM8K | 14.1 | 92.1 | 11.5 | 86.9 |
| Llama-3.1-8B-Instruct → RLHFlow-PRM-8B | PRM (1 − reward), q201 | CRV-Arith → CRV-GSM8K | 13.7 | 91.5 | 11.1 | 86.1 |
| Llama-3.1-8B-Instruct → RLHFlow-PRM-8B | TELE16 (L-SML), q201 | CRV-Arith → CRV-GSM8K | 14.8 | 77.1 | 13.1 | 64.6 |
| Llama-3.1-8B-Instruct → RLHFlow-PRM-8B | TELE16 plain average, q201 | CRV-Arith → CRV-GSM8K | 15.4 | 77.8 | 13.8 | 65.3 |
| Qwen2.5-Math-7B-Instruct → Qwen-PRM-7B | PRM (1 − reward), fixed grid | CRV-GSM8K → CRV-Arith | 0.7 | 74.3 | 0.4 | 64.6 |
| Qwen2.5-Math-7B-Instruct → Qwen-PRM-7B | PRM (1 − reward), q201 | CRV-GSM8K → CRV-Arith | 0.7 | 77.3 | 0.4 | 67.3 |
| Qwen2.5-Math-7B-Instruct → Qwen-PRM-7B | TELE16 (L-SML), q201 | CRV-GSM8K → CRV-Arith | 1.7 | 28.5 | 0.7 | 12.9 |
| Qwen2.5-Math-7B-Instruct → Qwen-PRM-7B | TELE16 plain average, q201 | CRV-GSM8K → CRV-Arith | 1.3 | 28.2 | 0.7 | 13.6 |
| Qwen2.5-Math-72B-Instruct → Qwen-PRM-72B | PRM (1 − reward), fixed grid | CRV-GSM8K → CRV-Arith | 0.3 | 73.4 | 0.2 | 65.8 |
| Qwen2.5-Math-72B-Instruct → Qwen-PRM-72B | PRM (1 − reward), q201 | CRV-GSM8K → CRV-Arith | 0.3 | 73.5 | 0.2 | 65.7 |
| Qwen2.5-Math-72B-Instruct → Qwen-PRM-72B | TELE16 (L-SML), q201 | CRV-GSM8K → CRV-Arith | 1.3 | 19.1 | 0.8 | 2.2 |
| Qwen2.5-Math-72B-Instruct → Qwen-PRM-72B | TELE16 plain average, q201 | CRV-GSM8K → CRV-Arith | 1.4 | 20.2 | 0.9 | 4.3 |
| Qwen2.5-Math-7B → PURE-PRM-7B | PRM (1 − reward), fixed grid | CRV-GSM8K → CRV-Arith | 0.7 | 32.6 | 0.0 | 7.4 |
| Qwen2.5-Math-7B → PURE-PRM-7B | PRM (1 − reward), q201 | CRV-GSM8K → CRV-Arith | 0.7 | 33.2 | 0.0 | 7.5 |
| Qwen2.5-Math-7B → PURE-PRM-7B | TELE16 (L-SML), q201 | CRV-GSM8K → CRV-Arith | 1.3 | 23.2 | 1.0 | 13.7 |
| Qwen2.5-Math-7B → PURE-PRM-7B | TELE16 plain average, q201 | CRV-GSM8K → CRV-Arith | 2.5 | 34.8 | 2.1 | 23.6 |
| Llama-3.1-8B-Instruct → RLHFlow-PRM-8B | PRM (1 − reward), fixed grid | CRV-GSM8K → CRV-Arith | 0.3 | 30.8 | 0.3 | 29.9 |
| Llama-3.1-8B-Instruct → RLHFlow-PRM-8B | PRM (1 − reward), q201 | CRV-GSM8K → CRV-Arith | 0.3 | 34.1 | 0.3 | 33.0 |
| Llama-3.1-8B-Instruct → RLHFlow-PRM-8B | TELE16 (L-SML), q201 | CRV-GSM8K → CRV-Arith | 0.5 | 28.4 | 0.2 | 9.9 |
| Llama-3.1-8B-Instruct → RLHFlow-PRM-8B | TELE16 plain average, q201 | CRV-GSM8K → CRV-Arith | 0.5 | 27.0 | 0.1 | 8.6 |
