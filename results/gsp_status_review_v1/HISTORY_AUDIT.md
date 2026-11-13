# History audit: GSP adoption 2026-10-30 .. 2026-11-13 (agent 1)

Verdict: yes, we kept working on a GSP version that did not deliver. The freeze (11-07) was justified by PRMBench numbers WITH the PRM, in-sample, against AUTO which differed in 4 ways (R1 vs R0, extra raw-p PRM column, own count vs GAP). First matched L-SML/mean arms appeared 11-12 (after the advisor meeting). Later work (readout A0-A4, joint B1/B2) was measured against GSP-T A0, never against TELE16 or mean. How C4U was selected (Mac-only branch claude/rigorous-recipe-2026-11-04, Step 514): UNKNOWN.

## Matched head-to-head, standalone, no PRM, GAP (7B/72B/PURE/RLHF)
R1 (gsp_ablation_baseline_v1/RESULTS_drop3.csv):
| Bench | GSP-T | L-SML R1 | mean R1 | GSP>L-SML | GSP>mean |
|---|---|---|---|---|---|
| PRMBench | 67.12 | 67.25 | 66.35 | 1/4 | 3/4 |
| Socratic | 64.25 | 62.96 | 61.77 | 4/4 | 4/4 |
| Hard2Verify | 43.28 | 44.36 | 42.93 | 1/4 | 3/4 |
| PB SLA (+Qwen3) | 31.54 | 35.48 | 36.41 | 0/5 | 0/5 |

R0 (TELE16 stored files; GSP-T from RESULTS_r0.csv):
| Bench | TELE16 R0 | mean R0 | GSP-T R0 | GSP>TELE16 | GSP>mean |
|---|---|---|---|---|---|
| PRMBench | 67.97 | 67.26 | 67.53 | 0/4 | 3/4 |
| Socratic | 63.71 | 62.96 | 66.08 | 4/4 | 4/4 |
| Hard2Verify | 42.57 | 41.52 | 43.56 | 3/4 (181 vs 177 answers) | 3/4 |
| PB SLA | 35.54 | 36.54 | 30.45 | 0/4 | 0/4 |

## Components
- (a) Parisi within-group vs L-SML signed: no gain (Parisi-within + continuous SML vs L-SML: PRMB 1/4, Soc 2/4, H2V 0/4, PB 0/5; RESULTS_signed.csv).
- (b) Jaffe binary cross-group vs continuous SML: Socratic 4/4 gain; PRMB 2/4, H2V 2/4; PB 1/5 (Qwen3 22.1 vs 35.2). Only component with a gain, only on Socratic.
- (c) Own count vs GAP: GAP better (Socratic 7B +38, H2V +0.7..+11, PB 4/4).
- (d) R1 vs R0: R0 mostly better (GSP PRMB 3/4, Soc 4/4; L-SML PRMB 4/4, Soc 4/4); PB mixed.
- (e) PRM column: source of the C4U headline (PRMB +2..+3).

## Best PRM-free label-free result per benchmark (mean of 4 cells)
| Bench | Best | Method |
|---|---|---|
| PRMBench | 67.97 | TELE16 L-SML R0 + GAP |
| Socratic | 66.08 | GSP-T R0 + GAP |
| Hard2Verify | not separable (42.6-44.9, CI ~ +-4) | - |
| PB SLA | 38.02 | change-point free EM on TELE16; GAP-only best: mean R0 36.54 |
| PB F1 | 25.7/24.0/22.8/20.4 | CP GSP-T pi=1/2 |
| DeltaBench | ~54.8 (not official protocol) | TELE16 sa |

## Claims that overstate GSP
MEMORY "FROZEN LEADING RECIPE C4U"; plan step 3 "GSP ranking"; c4u_summary "wins 4/4 ... above AUTO" (confounded, with PRM; SLIDER vs AUTO 1/4); "wins PRMBench vs PRISM/SLIDER/CROP" (with PRM); ABLATION "GSP-T better than PRM 3/4" (TELE16 also); "hybrid cutoff fixes Socratic 7B" (GAP already does); readout README "worth adopting" (baseline GSP-T A0 is below TELE16).
