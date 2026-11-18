# Realism of 3 new synthetic errors V3, V4, V5 (dev only, 2026-11-18)

Status: the GPU runs and the realism check are complete. No test data was used.
Decisions (Omri, 2026-11-18): the decision card "Numeric edit + on-topic donor" chose V3 and V4. V5 was Omri's own idea in the thread. For V4, Omri approved in writing that the numeric prefix limit is report-only.
Code: branch `claude/ssl-one-experiment-2026-11-18`: `make_variants_v34.py`, `make_variants_v5.py` and `realism_v34.py`, at commits aad94fc, 59ae1ce and c289fce. Outputs: `analysis_v1/REALISM_V34/` (commit 4fed69f).
Background: `DIAG_REALISM.md` (V2 is about 1.7 times louder than a real error).

## 1. What the variants are

| Variant | What changes in the answer | Dev answers built (PB / PRMBench) |
|---|---|---|
| V1 (from before) | Steps 2..n are put in a random order | 1,689 / 3,499 |
| V2 (from before) | Step k is replaced by a step of a different question | - / 3,499 (the PB V2 run stopped at its smoke check) |
| V3 | In step k, the last number changes by ±1 | 1,690 / 3,489 |
| V4 | Step k is replaced by a step at a different position of another answer to the same question | 421 / 3,418 |
| V5 | Every number from step k to the end changes by ±1. The same old number always becomes the same new number. Same answers and same k as V3 | 1,690 / 3,489 |

- No labels were used to build any variant. On PB, V4 covers only 25% of the dev answers, because most PB questions have only one answer.
- V5 changes a median of 19 numbers in 3 steps (PB) and 15 numbers in 4 steps (PRMBench).

## 2. Runs

| Job | Smoke check of the equal prefix | Step table |
|---|---|---|
| PB V3 365414, PRMBench V3 365416 | Pass, strict limit 1e-4 (gap 0.0) | GATE PASS 1,690 / 3,489 |
| PB V5 365443, PRMBench V5 365444 | Pass, strict limit 1e-4 (gap 0.0) | GATE PASS 1,690 / 3,489 |
| PB V4 365415, PRMBench V4 365417 | Stopped. Token ids and step bounds were equal in 24 of 24 pairs. The numeric gap was above 1e-4 in 3 of 24 pairs (max: logsumexp .125, entropy .055) | not built |
| PB V4 365436, PRMBench V4 365437 | Rerun with the numeric limit as report-only (Omri's written approval). Ids and bounds equal; same gaps | GATE PASS 421 / 3,418 |

- Deviation from the pre-registered smoke rule (V4 only): `$W/DEVIATION_V4_prefix_tol.txt`. The strict result is kept in `SMOKE_strict_1e-4.json`.
- Code check: the same functions give the earlier V1/V2 numbers exactly (PRMBench step V2 .8209, answer V2 .7232, answer V1 .806; PB answer V1 .898).

## 3. Results (classifier AUROC of variant vs real wrong; pass = AUROC ≤ .65 and pattern agreement ≥ .5)

| Benchmark | Variant | Step level | Answer level | Answer vs its own original (check) | Loudness of step k (median, real error = 1) |
|---|---|---|---|---|---|
| PRMBench | V1 | - | .806 fail | .822 | - |
| PRMBench | V2 | .821 fail | .723 fail | .749 | 1.67 |
| PRMBench | V3 | .699 fail | **.558 pass** | .572 | 0.60 |
| PRMBench | V4 | .654 fail | **.617 pass** | .641 | 1.04 |
| PRMBench | V5 | .730 fail | .785 fail | .800 | 0.82 |
| PB | V1 | - | .898 fail | .921 | - |
| PB | V3 | .676 fail | **.637 pass** | .610 | 0.49 |
| PB | V4 | .787 fail | .675 fail | .756 | 1.38 |
| PB | V5 | .721 fail | .768 fail | .826 | 1.22 |

Pattern agreement is .84 to .99 for all rows: every variant moves the channels in the same direction as real errors. The difference is in the size of the move.
Dev pool: PRMBench 3,499 answers (6,548 real wrong steps, 3,026 real wrong answers); PB 1,695 answers (1,095 wrong).

## 4. Findings

1. **No variant passes at step level.** The closest result is PRMBench V4: .654 against the limit .65, with a loudness of 1.04, which is the size of a real error. The same V4 on PB is too loud (1.38, AUROC .787).
2. **One changed number is almost invisible to the telemetry.** The 11 token channels of V3 step k compared with the same step in the original answer give AUROC .48 to .61. Only realized_z and realized_drv see the change (.62 to .67). V3 is therefore too quiet (0.5 to 0.6 times a real error).
3. **Changing all numbers from step k (V5) is detected by the digit channels.** digit_alternative, digit_spread and digit_alternative_innovation move 1.3 to 1.5 (PRMBench) and 2.4 to 3.2 (PB) clean SDs, which is about twice a real error. The model notices that a changed number is not the digit it expected. At answer level V5 is the easiest of the new variants to detect (.785 / .768).
4. **The answer-level pass of V3 is weak evidence.** V3 answers are hard to tell from their own originals (.572 PRMBench, .610 PB), and most originals are already wrong (86% PRMBench, 65% PB). So V3 answers look like the pool, not specifically like real wrong answers. If V3 is used as the "known wrong" set for π̂ (BBE), π̂ is expected to be biased toward 1. This is an expectation, not a measured result.
5. **The V4 answer-level pass on PRMBench is more informative:** .617 against real wrong answers and .641 against its own originals.

## 5. Label checks (gold labels used as a check only)

| Check | PRMBench | PB |
|---|---|---|
| Step k is already an error step of the original answer (V3, V5) | 22% | 49% at or after the first error |
| Same for V4 | 20% | 46% |
| V4 donor step is itself a labeled error | 29% | 45% at or after the donor's first error |
| Mean out-of-fold P(synthetic) of V4 steps: donor step wrong vs not wrong | .538 vs .534 | .658 vs .613 |

The V4 result does not come from the V4 donor steps that are real error steps: those look as synthetic as the other donor steps.

## 6. What this means for the pre-registered plan

- Stage S (step readout) needs a variant that passes at step level. None passes, so stage S does not run.
- Stages A and G (answer score, π̂, gate) name V2 or V1 as the known-wrong set. V3 (both benchmarks) and V4 (PRMBench) pass at answer level. To use them would be a design change, so Omri must decide. The PRMBench gate can add at most +0.62 PRMScore (oracle gate, README section 5), because 86.7% of the PRMBench answers are wrong. The gate question is a PB question. On PB, only V3 passes, and finding 4 applies to it.
- A cheap way to find out whether V3 can calibrate the PB gate: compute π̂ with V3 as the known-wrong set on PB dev and compare it with the true π (.65). Omri chose this check. It ran and failed (section 8).

## 7. Figure

`R1_realism_v34_final.png`. Left: classifier AUROC per variant (dark = step level, light = answer level, dashed line = pass limit). Right: mean shift of each channel at step k, compared with real wrong steps.

## 8. π̂ check: V3 as the known-wrong set on PB dev (2026-11-18, Omri's choice on the decision card)

Code: `pihat_dev.py` (commit 86d99a7). Output: `analysis_v1/PIHAT_V3/` (commit aeff3ac). CPU, 5.5 s. Dev only, no test data.
Method: the same computation as stage G of `run_analysis.py`, with M = 1 (stage A did not run). Answer score = L-SML of the 17 answer features of the dev C pool. 2-fold cross-fit by question group. In each fold, BBE(scores of the V3 answers of the other fold, scores of the C answers of the other fold). π̂ = mean of the 2 folds. Gold labels give only the true π and the AUROC values.

| Measure | Value |
|---|---|
| π̂ (fold 1 / fold 2) | .884 / .894, mean .889 |
| True π (share of wrong answers, PB dev, 1,695 answers) | .646 |
| Error | .243 (limit .10): **FAIL** |
| Answer score AUROC: V3 answers vs the C pool | .539 |
| Answer score AUROC: real wrong vs clean answers | .821 |
| PU posterior P(wrong): AUROC | .794 |
| Gate opens on | 88.9% of the answers (70.5% of them are really wrong) |
| Closed answers that are really clean | 82.5% |

- The V3 answers get almost the same scores as the whole pool (.539). BBE therefore sees the "known wrong" set as a sample of the pool, and π̂ goes toward 1. This is the bias that finding 4 expected.
- The 2 folds agree within .01, so the error is systematic, not noise.
- The ranking of the answer score is good (.821). The problem is only the threshold: V3 cannot set it on PB.
- Result: no synthetic variant (V1 to V5) passes the realism check and also calibrates the PB gate. The stop rule of the pre-registered plan stays in effect.

## 9. V3 + V4 mixture as the known-wrong set (2026-11-19, Omri in the project: "check the V3 + V4 mixture, and whether it also passes the clean proxy")

Code: `mix_v34.py` (commit 4d57e57). Output: `analysis_v1/MIX_V34/` (commit 051b8da). CPU, 19.5 min. Dev only, no test data.
Mixture = all dev V3 answers + all dev V4 answers. PRMBench: 6,907 answers, 49% V4. PB: 2,111 answers, 20% V4 (V4 exists for 25% of the PB dev answers only).
The checks are the unchanged stop rule (section 3), the π̂ computation of section 8, and, only where the step level passes, stages S and S2 of `run_analysis.py` with the mixture in place of V2 (same lambdas, guard, keep rules; 500 steps, 2,000 bootstrap draws). Code check: V3 alone and V4 alone give the section 3 and section 8 numbers exactly (all pass).

| Benchmark | Set | Step level | Answer level | Answer vs own original | π̂ (true π) |
|---|---|---|---|---|---|
| PRMBench | V3 | .699 fail | .558 pass | .572 | .933 (.865) |
| PRMBench | V4 | .654 fail | .617 pass | .641 | .793 (.865) |
| PRMBench | V3 + V4 | **.649 pass** | **.573 pass** | .601 | **.863 (.865)**, folds .853 / .874 |
| PB | V3 | .676 fail | .637 pass | .610 | .889 (.646) |
| PB | V4 | .787 fail | .675 fail | .756 | .526 (.646) |
| PB | V3 + V4 | .669 fail | .629 pass | .628 | .813 (.646) |

Stage S on PRMBench (mixture in place of V2):

| Readout | Dev PRMScore | Conditional abs. correlation of the 16 channels (lower = closer to independent) |
|---|---|---|
| top10 (method of record) | 68.33 | .290 |
| Stage S, chosen λ = 1: N ≈ 1 in all 11 token channels (≈ the largest token of the step) | 67.95 | .239 (drop .051, CI .049 to .053): **kept** |
| Stage S2 = stage S + clean proxy (bottom 30% of s_g; 95% of these steps are really clean) | 67.90 | .246 (worse than S by .007, CI .006 to .008): **not kept** |

Findings:
1. On PRMBench the mixture passes all three checks on dev. The step-level pass is at the limit (.6494 against .65).
2. The PRMBench π̂ lies between V3 (too high) and V4 (too low). The mixture share (49% V4) comes from the coverage of V4, not from a design choice. With another share, π̂ moves between .79 and .93. So the match with the true π can depend on this share. This is not measured yet.
3. On PB the mixture fails the step level and π̂ (.813 against .646). The gate question is a PB question, so the mixture does not solve it.
4. The clean proxy (S2) does not help: it makes the conditional correlation worse, so the pre-registered rule does not keep it.
5. Stage S is kept by the pre-registered rule, but it is not stable: the other half of the questions (half B) chooses N ≈ 39 in channels 3 and 5 and N ≈ 18 in channel 9 (max difference 38). The λ = .1 and .3 fits of half A also choose such large N. The PRMBench gate can add at most +0.62 PRMScore (README section 5).

