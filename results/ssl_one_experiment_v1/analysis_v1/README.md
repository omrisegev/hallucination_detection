# One SSL experiment: findings (2026-11-18)

Full summary of all runs and dev diagnostics (2026-11-18/19): `SUMMARY_2026-11-19.md`.

Status: PRMBench is complete under the pre-registered rules. ProcessBench has only the parity check (P0), because its V2 GPU run waits for Omri's decision.
Method (what each step calculates): `METHOD.md`. Code and outputs: branch `claude/ssl-one-experiment-2026-11-18`, folder `results/ssl_one_experiment_v1/` (code commit 641b67d, outputs commit 2f97b5e).

## 1. Summary

- **The synthetic errors do not look like real errors.** A classifier separates them from real errors with AUROC .82 (step level) and .72 / .81 (answer level, V2 / V1). The pre-registered limit is .65. So, by the stop rule, no self-supervised stage ran on PRMBench (stages S, S2, A and G).
- **The method of record stays.** Test PRMScore 68.05. It is better than all simple baselines (the equal mean of the 16 channels is −1.00, CI [−1.52, −0.47]).
- **On PRMBench a gate can add almost nothing.** A label-perfect gate adds only +0.62 (CI [+0.47, +0.79]), because 86.7% of the test answers contain an error. The gate question is a ProcessBench question.
- **The answer score separates wrong from clean answers with AUROC .70 (test), but this drops with answer length**: .86 for answers with ≤ 7 steps, .64 for answers with > 18 steps.
- **Parity passes.** Our GPU re-run gives the stored numbers: PRMBench PRMScore 68.26 (stored 68.26), PB SLA 38.62 (stored 38.47).

## 2. What ran, and where

| Part | Where | Result |
|---|---|---|
| GPU telemetry, 6 runs (PB, PRMBench × C, V1, V2) | AIRCC cycle4, B200 | 5 of 6 pass all gates. PB V2 stopped at the smoke check (1 of 12 prefix pairs differs at bf16 level) |
| Step tables, upload | AIRCC → Drive (`aircc/`, 143 files) and → Omri's PC | rclone check OK |
| CPU analysis | Omri's PC, Python 3.13.3 | PB: P0 only (14 s). PRMBench: full run (36 s, because the stop rule fired early) |

## 3. Parity check (P0)

The method of record on the full C pool, with our new GPU telemetry, against the stored numbers:

| Benchmark | Metric | Ours | Stored | Readouts vs stored (Spearman per channel) |
|---|---|---|---|---|
| PRMBench | PRMScore | 68.26 | 68.26 | .986-1.000 (11 token channels) |
| ProcessBench | SLA | 38.62 | 38.47 | .98-1.00 |

PB F1 is 0.0 for the method of record. GAP flags at least one step in every answer, so no clean answer is ever correct. This is the known reason for the gate.

## 4. Realism check (C2ST, dev pool only)

| Comparison | Classifier AUROC | Correlation-pattern agreement | Pass (AUROC ≤ .65 and agreement ≥ .5) |
|---|---|---|---|
| V2 step k vs real wrong steps | .821 | .835 | No |
| V2 answers vs real wrong answers | .723 | .925 | No |
| V1 answers vs real wrong answers | .806 | .894 | No |

- The channels correlate in the same pattern (agreement .84-.92), but the values differ: the synthetic errors are easier to see than the real errors.
- Consequence (pre-registered stop rule): stage S needs V2 to pass at step level; stages A and G need V2 or V1 to pass at answer level. None passed. The "final" readout therefore equals the method of record.

## 5. Test pool results (PRMBench, 3,470 answers, scored once)

TPR = error steps flagged (%). FPR = valid steps flagged (%). Open = answers where GAP may flag.

| Arm | PRMScore | Δ vs method of record [95% CI] | TPR | FPR | Open % |
|---|---|---|---|---|---|
| Method of record (top10, R0, L-SML, GAP) | 68.05 | - | 49.9 | 11.9 | 100 |
| + label-perfect gate (oracle) | 68.67 | +0.62 [+0.47, +0.79] | 49.9 | 11.2 | 86.7 |
| + Simes gate α .05 | 49.11 | −18.95 [−19.97, −17.96] | 3.6 | 0.4 | 6.1 |
| + Simes gate, best α (oracle) | 68.05 | 0 (best α = 1.0, that is no gate) | 49.9 | 11.9 | 100 |
| Equal mean of 16 R0 channels + GAP | 67.05 | −1.00 [−1.52, −0.47] | 47.0 | 11.7 | 100 |
| Mean entropy (q15_H1) + GAP | 59.58 | −8.47 [−9.36, −7.56] | 33.3 | 13.7 | 100 |
| Mean surprisal (perplexity) + GAP | 58.36 | −9.69 [−10.62, −8.78] | 28.5 | 12.2 | 100 |
| Step length only + GAP | 56.77 | −11.28 [−12.35, −10.17] | 32.1 | 17.1 | 100 |
| Step position only + GAP | 51.76 | −16.29 [−17.36, −15.25] | 14.4 | 10.9 | 100 |

The dev pool gives the same picture: method of record 68.33, oracle gate +0.60 [+0.46, +0.75], Simes α .05 −19.31.

## 6. Other measurements (PRMBench)

| Quantity | Dev | Test |
|---|---|---|
| Answer score AUROC (wrong vs clean answers; answer L-SML with M = 1) | .676 | .700 |
| Answer AUROC by step-count quintile (edges 7, 9, 13, 18 steps) | .87 / .85 / .80 / .69 / .65 | .86 / .85 / .80 / .71 / .64 |
| Conditional mean \|Spearman\| between the 16 step channels (wrong .306, clean .274 on dev) | .290 [.286, .295] | .287 [.283, .292] |
| Conditional mean \|Spearman\| between the 17 answer features | .448 | .417 |
| True share of wrong answers π | .865 | .867 |

## 7. Figures

| File | What it shows |
|---|---|
| `F1_prmbench.png` | Ofir's histograms. Left: fused step score, real clean vs real wrong steps (test). Right: answer score, real clean vs real wrong answers (test). |
| `F2_prmbench.png` | Bracha's plane: error steps flagged vs valid steps flagged, with iso-PRMScore lines. The method of record and the oracle gate are almost the same point. |
| `F4_cond_corr.png` | Conditional dependence before / after calibration. Before = after here, because no stage ran. |

F3 (π̂ vs true π) is not made, because stage G did not run.

## 8. Open decisions (for Omri)

1. PB V2: re-run with a bf16 tolerance in the smoke check, or drop V2 for PB.
2. The realism stop on PRMBench: accept it as the finding, or change the synthetic errors or the limit. A change of the limit after this result is a deviation from the pre-registered rule and must be reported as such. The PRMBench test pool was scored only with arms that do not depend on any SSL choice.
