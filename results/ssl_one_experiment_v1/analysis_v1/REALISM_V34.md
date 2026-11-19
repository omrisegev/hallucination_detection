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


## 10. The answer-level gate on dev, and the synthetic answers split by the label of their parent (2026-11-19, Omri in the thread: "יש אישור")

Code: `GATE_DEV/gate_dev.py` (commit a04fc0c) and `parent_split.py` (commit fc17711, figure fix 31d5072). Output: `analysis_v1/GATE_DEV/` and `analysis_v1/PARENT_SPLIT/` (commit 35d8264). CPU, less than 1 min. Dev only, no test data.

The gate is stage G of `run_analysis.py`, unchanged, with M = 1:
1. Answer features: for each of the 16 channels, the pool ECDF value of the most extreme step of the answer; plus the Simes value of the fused step score. Total 17.
2. Answer score sA: z-score of each feature, then answer-level L-SML. No labels.
3. Threshold (the SSL part): π̂ by BBE, with the synthetic answers as the known-wrong set and all real dev answers as the unlabeled set. Two gate folds by question group; π̂ = mean of the two folds.
4. Decision: open the π̂ share of answers with the highest sA. Flag steps (GAP) only in open answers.

The synthetic answers were made from ALL dev answers as parents, without labels. On PB 65% of the parents are wrong, on PRMBench 86%. Here the synthetic answers are split by the gold label of the parent. This split uses labels, so it is a diagnostic and not a label-free method. A reference row uses the real wrong answers of the other gate fold as the known-wrong set ("perfect synthetic set").

Table A. PB dev (1,695 answers, true π .646, sA AUROC wrong vs clean .82):

| Known-wrong set | n | π̂ (per fold) | Wrong answers opened | Clean answers closed | F1 | SLA |
|---|---|---|---|---|---|---|
| No gate | – | – | 100% | 0% | 0.00 | 38.23 |
| V3, all parents | 1,690 | .889 (.884 / .894) | 97% | 26% | 30.90 | 36.83 |
| V3, clean parents | 599 | .984 (.987 / .981) | 100% | 4% | 6.63 | 37.91 |
| V3, wrong parents | 1,091 | .651 | 82% | 66% | not run | not run |
| V4, all parents | 421 | .526 (.530 / .521) | 71% | 80% | 40.21 | 26.80 |
| V4, clean parents | 149 | .661 (.662 / .660) | 83% | 65% | 41.91 | 30.90 |
| V4, wrong parents | 272 | .417 | 58% | 88% | not run | not run |
| V3 + V4, all parents | 2,111 | .813 (.778 / .848) | 93% | 39% | 37.69 | 35.31 |
| V3 + V4, clean parents | 748 | .982 | 100% | 5% | 6.97 | 37.91 |
| Perfect synthetic set (real wrong answers, other fold) | 550 / 545 | .686 (.641 / .730) | 84% | 61% | not run | not run |
| True π, same sA ranking | – | .646 | 82% | 66% | 41.65 | 30.30 |
| Oracle gate (labels) | – | – | 100% | 100% | 54.82 | 38.23 |

Table B. sA of the synthetic answers against the real answers (AUROC of synthetic vs real wrong answers: .5 = the same distribution, lower = quieter, higher = louder):

| Benchmark | Set | All parents | Clean parents | Wrong parents |
|---|---|---|---|---|
| PB | V3 | .420 | .221 | .529 |
| PB | V4 | .595 | .469 | .663 |
| PRMBench | V3 | .505 | .354 | .529 |
| PRMBench | V4 | .561 | .465 | .576 |

Table C. PRMBench dev (3,499 answers, true π .865, sA AUROC .676). PRMScore against no gate (68.33):

| Known-wrong set | π̂ | PRMScore | Difference [95% CI] |
|---|---|---|---|
| V3, all parents | .933 | 68.18 | −0.14 [−0.33, +0.01] |
| V4, all parents | .793 | 67.06 | −1.26 [−1.68, −0.90] |
| V3 + V4, all parents | .863 | 67.78 | −0.55 [−0.84, −0.28] |
| V3 / V4 / V3 + V4, clean parents | .999 / .989 / .999 | 68.32 / 68.28 / 68.32 | about 0 (the gate stays almost fully open) |
| Perfect synthetic set | .904 | not run | – |
| True π, same sA ranking | .865 | 67.76 | −0.56 [−0.85, −0.29] |
| Oracle gate | – | 68.93 | +0.60 |

Findings:
1. BBE works on this answer score when the known-wrong set is like the real wrong answers: the perfect synthetic set gives π̂ .686 on PB and .904 on PRMBench (both +.04 over the true π). The single-fold values differ by up to .09 on PB. Thus the large π̂ errors of sections 8 and 9 come from the synthetic sets, not from BBE.
2. V3 is too quiet. One changed number in a clean answer moves sA only a little (PB AUROC .221 against real wrong answers). V3 on wrong parents is near the real wrong answers (.529) and gives π̂ .651, but only because the parent is already a real wrong answer. That part does not give information from the injected error.
3. V4 on clean parents is near the real wrong answers (PB .469) and gives π̂ .661 (true .646). V4 on wrong parents is too loud (.663): the foreign step adds a second extreme step to an answer that already has an error. This is why V4 with all parents gives a π̂ that is too low (.526).
4. On PB, the gate with V4 on clean parents gives F1 41.91. This is the same as the gate with the true π (41.65). The gap to the oracle gate (54.82) comes from the ranking quality of sA (AUROC .82), not from the threshold. F1 is flat near the true π: V4 with all parents (π̂ .526) gives 40.21.
5. On PB, F1 and SLA go in opposite directions. SLA counts only wrong answers, so each wrong answer that the gate closes lowers SLA (no gate 38.23, true-π gate 30.30).
6. On PRMBench the gate does not help at any tested threshold. With the true π, PRMScore goes down by 0.56. sA (AUROC .676) is too weak to close the 13.5% clean answers without closing wrong answers. The oracle limit is +0.60.
7. Limits: "clean parents" uses the gold label of the parent. V4 on clean parents has only 149 PB answers. The change point with prior 1 − P was weaker than GAP (V4 clean parents: F1 37.08, SLA 23.99). This dev data was inspected before.

Next step (not run, needs Omri's OK): V4 on parents that are chosen without labels as "probably clean" (for example, the answers with the lowest sA), on PB dev. Check: |π̂ − π| ≤ .10 and F1 against the true-π gate.

## 11. The error direction and the reaction to several error types (2026-11-19, Omri's ideas in the thread at 10:45 and 10:54 Israel)

Idea: measure the telemetry of an answer, inject an error, measure again. The difference may contain the "direction" of an error, and it may help to learn the answer-level threshold. Omri's second idea: inject several different error types into the same answer, and fuse the differences with SML, because different types may give independent views.
Code: `analysis_v1/DIRECTION_QUICK/pair_direction.py` and `pair_sml.py` (quick checks, saved unchanged, code base 4d57e57), `reaction_ds.py` (commit 804cd4d). Output: `analysis_v1/DIRECTION_QUICK/` and `analysis_v1/REACTION_DS/` (commit 1e966d11). CPU, less than 1 min each. Dev only, no test data. Code check: the true-π gate gives the GATE_DEV numbers exactly (PB F1 41.653, PRMBench PRMScore 67.7636).

### 11.1 One error direction (Omri's first reading)

Direction = mean of the difference of the 17 z-scored answer features, variant minus its own parent. Learned on one gate fold, applied to the other. Score of an answer = its projection on the direction. No labels.

| AUROC, wrong vs clean answers | PB | PRMBench |
|---|---|---|
| Current answer score (answer L-SML) | .820 | .676 |
| Direction from V3 | .817 | .701 |
| Direction from V4 | .804 | .732 |
| Direction from V3 + V4 | .814 | .725 |
| Direction from V3 + V4, whitened (Σ⁻¹ × difference) | .686 | .740 |
| Supervised ceiling (logistic regression with labels, cross-fit) | .865 | .822 |

- The direction is stable: cosine between the two folds .997 to .999 (whitened PRMBench V4: .548).
- π̂ by BBE on the direction score is still wrong: PB .875 / .528 / .788 (V3 / V4 / V3 + V4, true .646); PRMBench .891 / .703 / .796 (true .865).

### 11.2 One direction per error type, fused by SML (Omri's second reading, first form)

| | PB (V1, V3, V4, V5) | PRMBench (V1 to V5) |
|---|---|---|
| AUROC of each type's direction | .800 to .822 | .701 to .738 |
| Cosine between the directions of two types | .82 to .97 | .76 to .97 |
| Conditional Spearman between the type scores (inside wrong / inside clean) | .976 / .972 | .962 / .975 |
| SML = L-SML = equal mean | .813 | .727 |

SML gives the same weight to all types (.50 PB, .44 to .45 PRMBench).

### 11.3 Reaction of each answer to each error type, fused by Dawid-Skene (second form)

Reaction r = sA(variant) − sA(parent). Small reaction = "the answer is already wrong". Each type is one view. Binary Dawid-Skene EM with votes cut at the 35%, 50% and 65% quantile of each view, and a Gaussian latent-class EM. No labels in any fit.

| | PB (π .646) | PRMBench (π .865) |
|---|---|---|
| Error types (answers) | V1 1,689, V3 1,690, V4 421, V5 1,690 | V1 3,499, V2 3,499, V3 3,489, V4 3,418, V5 3,489 |
| AUROC of −r, wrong vs clean parent | .73 to .79 | .63 to .74 |
| The same inside 5 strata of the parent's sA | .54 to .58 | .56 to .71 (V1 .71) |
| Spearman(r, parent sA) inside wrong / inside clean | −.62 to −.81 / −.40 to −.72 | −.49 to −.70 / −.55 to −.80 |
| Conditional Spearman between types, mean absolute value (range) | .50 (.21 to .72) | .56 (.35 to .69) |
| DS π̂, votes cut at 35% / 50% / 65% (all answers) | .35 / .47 / .61 | .36 / .45 / .57 |
| Gaussian latent-class π̂ (all answers) | .52 | .56 |
| Gate with these π̂: PB F1 (true-π gate 41.65) / PRMBench PRMScore (no gate 68.33) | 29.3 / 37.1 / 41.4; latent class 39.8 | 57.8 / 60.5 / 63.2; latent class 63.1 |

Findings:
1. The error direction improves the answer ranking on PRMBench (.676 to about .73; the supervised ceiling is .82). On PB it gives the same ranking as the current score (.80 to .82).
2. All error types move the answer in almost the same direction: a general rise of the model's uncertainty. The types differ in the size of the move (V3 small, V4 large), not in its direction. So the types are not independent views, and SML on them equals the mean.
3. The reaction of an answer mostly measures how much room the answer has to rise: it correlates −.4 to −.8 with the parent's own score. Inside strata of the parent score it adds little on PB (.54 to .58). On PRMBench it adds some (up to .71 for V1). This is a possible lead for the PRMBench ranking, not for the threshold.
4. The reactions to different types are conditionally dependent (.50 PB, .56 PRMBench), because all of them come from the same parent, the same model and the same channels.
5. The DS π̂ follows the vote cut (PB .35 / .47 / .61 for cuts at 35 / 50 / 65%). So π is not identified. This is the same failure as the cross-model Dawid-Skene check of 2026-11-14 (π̂ ≈ 1 − q).
6. No label-free route in sections 8 to 11 identifies π on PB. The only set that gave π̂ near the true π was V4 on clean parents chosen by the gold label (section 10).
