# algorithm_external_v1: the frozen candidates on the external benchmarks (all digit banks)

Omri, 2026-09-29: send the frozen candidates to the external benchmarks on every feature combination that includes the digit
features. Exposure: Hard2Verify and Socratic-PRMBench were evaluated earlier in the project, so this is an exploratory transfer test of
methods selected on PRMBench/ProcessBench only, not an untouched confirmation.

- Protocol `PROTOCOL.json` (b3ef5c2e1; amendment A1 before scoring; NOTES.md). Features `results/external_banks_v4` (source parity
  PASS). Frozen source bundle `BUNDLE.json` (folds 0-3 fit, fold-4 q80 thresholds). Scoring without labels; pre-seal review (no
  blocker); predictions sealed at 9beae6efe before any label was read; evaluation 775ec15a7 with the official evaluator replayed
  exactly. Red team `run_20260929/RED_TEAM.md`.
- Cells: Hard2Verify (Qwen3-8B, 200 answers, 79 question groups), Socratic (the same 2,995 answers scored with Qwen3-8B and with
  QwQ-32B - one dataset seen twice).

## Main numbers (official metric: Hard2Verify balanced F1 / Socratic PRMScore; within-answer AUC)

| Arm | H2V off. | Soc Q8 off. | Soc QwQ off. | H2V AUC | Soc Q8 AUC | Soc QwQ AUC |
|---|---:|---:|---:|---:|---:|---:|
| **step index alone** (reference added by the red team) | - | - | - | **0.8649** | **0.7307** | **0.7307** |
| frozen bank11 L-SML (earlier external run) | **0.4367** | 0.6322 | 0.6424 | 0.6343 | **0.6836** | **0.7011** |
| 13+d, DS filter + plain average (primary candidate) | 0.3973 | 0.6321 | 0.6440 | 0.6139 | 0.6820 | 0.6996 |
| 13+d, grouped DS-estimate weights | 0.4185 | **0.6401** | **0.6485** | 0.6048 | 0.6820 | 0.6986 |
| 20+d plain average | 0.4095 | 0.6196 | 0.6316 | 0.6184 | 0.6604 | 0.6869 |
| 32+d plain average | 0.3953 | 0.5792 | 0.5947 | 0.5941 | 0.6163 | 0.6380 |
| 32+d grouped | 0.4085 | 0.6115 | 0.6246 | 0.6270 | 0.6588 | 0.6784 |
| 51+d plain average | 0.3918 | 0.5992 | 0.6173 | 0.6074 | 0.6343 | 0.6647 |
| bank11 plain average (earlier) | 0.4088 | 0.6079 | 0.6150 | 0.6051 | 0.6512 | 0.6650 |
| ct7 (earlier incumbent) | 0.3775 | 0.5875 | 0.6017 | 0.6079 | 0.6452 | 0.6727 |

## Reading
1. **Against ct7:** the candidate is higher on Socratic (+0.0446 / +0.0424 PRMScore, Bonferroni) and on 20+d / 51+d, not on
   32+d; nothing is decisive on Hard2Verify. But most of the advantage is step position (content share 8% / 24% of the within-AUC
   gain) and 15-25% of the PRMScore gap is ct7's larger predicted-correct share. Net of position the candidate is about equal to ct7.
2. **Against the earlier frozen bank11 L-SML:** equal on Socratic in raw numbers (cancellation across categories), lower on
   Hard2Verify at unadjusted 95%. L-SML's external edge over plain averaging is a late-step tilt (it silences the channels that prefer
   early steps); net of position the candidate leads L-SML by about +0.014 / +0.008 within-AUC.
3. **Grouping:** the estimate-weighted grouping helps on 32+d (+0.032 / +0.030 PRMScore; within-AUC positive on all 3 cells),
   broadly and about half content - the same bank family where it helped in development. It does not lift that bank above 13+d.
4. **Banks:** on Socratic 13+d > 20+d > 51+d > 32+d, as in development; nothing is ordered on Hard2Verify.
5. **Access:** the label-free refit on the target is not better than the frozen development fit.
6. **Stopping rule:** never switched on externally (between-group dependence above the development threshold everywhere).
7. **Step position alone beats every method** on these benchmarks (errors run to the end of the answer in about a third of Socratic
   answers). Any future external protocol must include the step-index row and position nulls.

Predictions: P1 held (13+d above the bank11 average in all 3 cells); P2 partly (grouping positive on 32+d, but also on 13+d);
P3 failed (the switch never turned on); P4 failed (refit vs frozen differ by more than 0.005 in several cells).
