# Advisor document flow v2, meeting 2026-10-15 (PROPOSAL, not applied)

Compare with v1: `FLOW_OUTLINE_2026-10-15.md`. List of v1 gaps: `FLOW_OUTLINE_DROPPED_ITEMS_2026-10-12.md`.
Rules for this version (Omri, 2026-10-12):
- PRMBench only. No ProcessBench. No Mind the Gap.
- Digits are a normal part of the feature set. We discuss them only when we analyze one channel's behavior (for example, the LR weights).
- Use a graph when we can. Each graph has a short "Experiment" text before it (what we tested and why) and exactly 3 conclusion bullets after it.
- Old 8 October values go to the appendix. The message is "the picture did not change" and the next steps.
- Freeze: all numbers come from saved results. "New figure" means a new plot of existing numbers.

## Story map

| # | Question | Graph(s) | Answer |
|---|---|---|---|
| 0 | Where did we stop on 8 October? | Methods × banks AUC heatmap | The fusion methods are close to each other; LR is above them. |
| 1 | Did the corrections change the picture? | Corrected vs old AUC per method | All methods move down by the same amount; the order stays. |
| 2 | What is a PRM, and what does PRMBench test? | Benchmark schematic + two examples | Background to read sections 3-8. |
| 3 | Where are we on the official metric? | Leaderboard bar | A tie with Qwen2.5-Math-PRM-7B. |
| 4 | Which errors do we find, and which signals find them? | Category radar; matched controls; channel × type heatmap | Redundancy is our robust strength; different channels catch different error types. |
| 5 | Does it transfer outside PRMBench? | External benchmarks bar | Yes on Socratic; Hard2Verify is too small and position-driven. |
| 6 | What does supervision add? | LR gap across banks; four-way top1 by type; 4B change per bank | LR gains are small at N16 and do not give a weighting we can copy. |
| 7 | Why is the score only a tie? | Budget curves | The global threshold is not the problem. |
| 8 | What is left to gain, and where did we stop? | Ceiling ladder; what we tried; errors per answer vs length | The per-answer count is worth +6.3; our count rules failed. Open direction. |

---

## 0 · Where we stopped on 8 October

**Experiment.** We fuse the step signals of one teacher-forced pass with label-free methods. On 8 October we
compared all methods on all feature banks with within-answer AUC. We repeat the same comparison on the corrected input.
The purpose is to give the advisors the same starting view as last time.

**Graph.** Heatmap: rows = 10 banks, columns = label-free methods, then the bank mean, then LR. Row "method mean" at
the bottom. N16 marked. Source: `controls/ALL_BANK_TOP1_AUC.csv` (v12 appendix "Full matrices"). New figure.

**Conclusions.**
1. All label-free methods are within about 0.01 AUC of each other in each bank. The choice of fusion method has a small effect.
2. The best fusion method is above the plain average in every bank, by about 0.01 at N16 (0.780 vs 0.771).
3. Supervised LR on the same features is above all label-free methods in every bank (N16: 0.801). This is the headroom.

**Not in the main text:** top1 and PRMScore matrices (tabs or appendix D). The 8 Oct table (appendix A).
Note on the graph: the 8 Oct control was "DS-signed"; the current one is "DS-filter". They are different controls.

---

## 1 · Did the corrections change the picture?

**Experiment.** After 8 October we found that the teacher-forced prompt did not have the empty think block that
Qwen3 expects. We fixed the prompt, ran the forward pass again and scored all methods again. We also masked the
first answer token. The purpose: to make sure the numbers are valid, and to see if the conclusions change.

**Graph.** N16 dumbbell plot: old input vs corrected input, per method (plain average, DS-filter, L-SML, IU-PCR, LR),
with PRM-7B as a fixed reference line. Source: v12 section 02 table. New figure.

**Conclusions.**
1. The fix lowers all methods by almost the same amount (about 1.5-2 AUC points). The order of the methods does not change.
2. The decline is not a first-token effect. Without step 0 the decline is larger (−2.2 vs −1.6 points). The first-token mask changes AUC by less than 0.1 point.
3. The same conclusions hold on Qwen3-4B (N16 L-SML 0.775 vs 0.779) and on a held-out half of the source groups (IU-PCR 0.786).

**Box, no graph: "Without teacher forcing".** On answers that the model wrote itself (GSM8K/MATH, not PRMBench),
the signal gives AUC 0.757, against 0.742 with teacher forcing on ProcessBench answers. Difference +0.015,
95% CI [−0.021, +0.052]. We found no difference, but we do not claim equivalence.

**Box, 3 lines: "Claims we withdraw".** realized_drv leads 6 of 8 types (now 0 of 8). "All ~40 literature rows are
weaker on Simplicity" (10 of 15 rows in Table 3). Equivalence of self-generated and teacher-forced answers.

**Not in the main text:** step-0 table, prefix/mask attribution, Q80 provenance, LOTO and half-split tables (appendix A).

---

## 2 · Background: what is a PRM, and what does PRMBench test?

Omri asked for this deep dive (plan item 0b). v12 has only parts of it (examples and a construction note in the appendix).

**Text (short).**
- A process reward model (PRM) gives a score to each step of a solution. It is trained on step labels.
  Qwen2.5-Math-PRM-7B gives each step a probability of being correct; a step below 0.5 is an error.
- Our method does not train. It reads token probabilities from one teacher-forced pass of Qwen3 and fuses them without labels.
- PRMBench: GPT-4o inserted errors into correct PRM800K solutions, and the results were filtered. QwQ supplied
  correct alternative solutions for MS. Each step has a label. Our population: 6,211 answers with errors + 758 clean controls (the paper has 6,216).
- Nine error types in three groups: Simplicity (NR, NCL), Soundness (ES, SC, DC, CI), Sensitivity (PS, DR, MS).
- The official metric, PRMScore, is the mean of the F1 on valid steps and the F1 on error steps. It needs a flag rule.
  Within-answer AUC and top1-on-error need no rule.

**Graph.** Schematic of the benchmark: a correct solution → inserted error → nine types in three groups
(the v12 SVG schematic, with the legend open). Below it, the two worked examples (NR and NCL).

**Conclusions.**
1. NR and NCL steps can be wrong without a wrong number: a removable step, or a step that assumes the answer. The final answer can stay correct.
2. CI, counterfactual and DR errors are written to look plausible. A model can read them with low surprise.
3. PRMScore rewards both finding errors and not flagging valid steps. So it tests the flag rule, not only the ranking.

**To check before writing (no new runs):** what the PRMBench paper says about why models are weak on Simplicity.
v12 says only: "correctness-focused training is a possible mechanism; not established".

---

## 3 · Where are we on the official metric?

**Experiment.** We computed the official PRMScore with the PRMBench evaluator on all answers. Our flag rule is Q80
(fixed before the corrected results). We ran Qwen2.5-Math-PRM-7B on the same answers with its native 0.5 threshold.
The purpose: to place a label-free method on the published leaderboard.

**Graph.** Horizontal bars, Overall PRMScore: our two methods, PRM-7B (our run), and the paper Table 3 rows
(open PRMs and LLM judges). Source: v12 leaderboard table. New figure.

**Conclusions.**
1. Our Overall score is 65.4; Qwen2.5-Math-PRM-7B gets 65.5. The paired difference is −0.08 [−0.77, +0.64]: a tie.
2. We are above all open PRMs in the paper (best 60.5) and below GPT-4o (66.8) and o1-mini (68.8). We use one pass, no labels and no training.
3. On ranking only, PRM has the higher AUC (0.801 vs 0.779), but our top-ranked step is an error more often (59.9% vs 57.8%).

**Under the graph:** one small table (method × PRMScore / AUC / top1). Full Table 3 to appendix D.

---

## 4 · Which errors do we find, and which signals find them?

### 4a · Category profile

**Experiment.** We split the official PRMScore by the nine error types, for us and for PRM-7B.
The purpose: to see what is behind the tie.

**Graph.** Category radar (in v12).

**Conclusions.**
1. We lead in Simplicity: NR 70.0 vs 49.0 and NCL 68.0 vs 55.0. No row in paper Table 3 is higher on NR.
2. PRM-7B leads on the plausible-looking errors: Soundness 71.0 vs 66.9 and Sensitivity 75.5 vs 69.3.
3. Our MS score is low (88 vs 99.9), because Q80 flags a step in almost every clean answer. Section 8 comes back to this.

### 4b · Is the Simplicity lead real?

**Experiment.** Step position, step length and text repetition can predict some error types. We matched error and
valid steps on relative position and length inside each answer, and compared AUC on the matched pairs.
The purpose: to test if the lead comes from the signal or from these simple cues.

**Graph.** Nuisance controls (in v12, `nuisance_controls.svg`).

**Conclusions.**
1. The NR lead stays under every control: +19.7 AUC points over PRM [15.9, 23.6]. This is our strongest result.
2. The NCL lead shrinks to +4.3 points, and the interval crosses zero. We do not claim it.
3. On all answers, position matching removes PRM's AUC lead and our lead over the plain average (subset of 4,691 answers; a caution, not a final result).

### 4c · Which signals find which errors?

**Experiment.** We computed the within-answer AUC of each of the 56 channels alone, per error type, with a
label-free sign rule. Step index is a baseline row. The purpose: to see which signal families catch which error types.

**Graph.** Channel × error-type heatmap (in v12). Omri: main graph of this section.

**Conclusions.**
1. No channel wins on all types. Uncertainty channels lead on redundancy (0.84), domain (0.81) and circular (0.77).
2. Step index alone reaches 0.87 on confidence, above every channel, but it is near chance on domain (0.54) and circular (0.57). So our strong types are not position effects.
3. One channel (rolling spectral entropy) is below 0.5 on all types: the label-free sign rule turns it the wrong way.

---

## 5 · Does it transfer outside PRMBench?

**Experiment.** We froze the L-SML weights from PRMBench and applied them, without changes, to two other benchmarks:
Socratic-PRMBench (same error taxonomy, 2,995 answers) and Hard2Verify (200 answers). Each uses its own official
metric. Baselines: equal weights, the CT7 baseline and step index. The purpose: to test the method on data that we did not use for development.

**Graph.** Grouped bars per benchmark: frozen L-SML, equal weights, CT7, step index; 8B and 4B side by side.
Published Qwen2.5-Math-PRM-7B values as a dashed line (context, other protocol). Source: v12 4B section and appendix. New figure.

**Conclusions.**
1. On Socratic, frozen L-SML (0.628) is above equal weights (0.600), CT7 (0.566) and step index (0.580). It stays the same on 4B (0.631).
2. On Hard2Verify, step index alone (0.483) is above all our methods (0.421). With 200 answers, this benchmark is not a reliable test for us.
3. The frozen weights were fitted on the old (pre-fix) input, and we cannot fit them again. Show this caveat under the graph.

---

## 6 · What does supervision add?

### 6a · The size of the gap

**Experiment.** We trained logistic regression (LR) on the same features with step labels, with source-group
cross-fitting, in all ten banks. The purpose: to measure how much a supervised fusion of the same signals can add.

**Graph.** LR gap across the ten banks, AUC and top1 side by side (in v12).

**Conclusions.**
1. The AUC gap is smallest at N16 (0.021) and largest at N40 (0.063).
2. On top1 we lead LR up to N25. At N16, L-SML is 2.0 points above LR [1.0, 3.0].
3. LR does not show a better weighting that we can copy: its weights agree with ours at cosine 0.03, and 88% of its weight is on a +/− pair of two almost identical channels.

### 6b · Where each one is right

**Experiment.** For each answer with an error, we checked if the top-ranked step is an error, for L-SML and for LR.
We split the answers into four groups (both right, only ours, only LR, both wrong) and by error type.
The purpose: to see if the two methods find different errors.

**Graph.** Four-way top1 outcomes by error type, N16 (in v12).

**Conclusions.**
1. L-SML alone finds the error in 485 answers that LR misses; LR alone finds 366.
2. We win on redundancy (90 vs 27), domain (79 vs 32) and circular (77 vs 42). This matches the channel heatmap in 4c.
3. LR wins on confidence (60 vs 28) and deception (75 vs 58), the plausible-looking errors.

### 6c · Model size

**Experiment.** We repeated the full pipeline with Qwen3-4B as the teacher-forcing model. Answers, labels, banks and
methods are the same. The purpose: to see if the results depend on the model that produces the features.

**Graph.** AUC change from 8B to 4B per bank, one dot per method (in v12).

**Conclusions.**
1. The label-free methods change by less than 0.01 AUC in every bank, in both directions.
2. LR is lower on 4B in all ten banks (−0.003 to −0.010). It depends on the model more than our methods do.
3. The category profile and the PRMScore do not change (65.4 → 65.0; NR stays at about 70).

**Not in the main text:** LR weight plots, family mass and omission, selection × weighting crossover,
conditional correlations (appendix B and C).

---

## 7 · Why is the score only a tie?

**Experiment.** We moved one global threshold over all steps and computed PRMScore at each flag rate, for us and
for PRM-7B. The purpose: to see if a better global threshold closes the gap.

**Graph.** Budget curves (in v12), with our Q80 point and the PRM 0.5 point marked.

**Conclusions.**
1. At the same flag rate (about 18% of steps), PRM gets 68.3 and we get 65.7.
2. PRM's 0.5 threshold flags only 8.7% of steps, far from its best point. That is why the default scores are equal.
3. Our Q80 point is almost at the top of our curve: the best global threshold adds only 0.3 points. The problem is inside each answer.

---

## 8 · What is left to gain, and where did we stop?

### 8a · The value of the per-answer count

**Experiment.** We kept our ranking and changed only the number of flags in each answer, using label-based
oracles as ceilings. The purpose: to find which part of the decision has the largest headroom.

**Graph.** Ceiling ladder (bars): Q80 65.4 → best global cut 65.7 → constant 2 flags 67.9 → count from error type
68.9 → true count 71.7 → perfect ranking at our counts 81.8. Bars that use labels or post hoc choices have a different
color. Source: v12 "Open direction" table. New figure.

**Conclusions.**
1. With the same ranking, the true number of errors per answer adds +6.3 points. This is above PRM-7B at its best threshold (68.3).
2. A constant two flags per answer gives 67.9. This is post hoc, not a method, but it shows that the right count is small.
3. Perfect ranking adds more (+16.4), but the LR gap shows that our features do not have this information now. The count is the part we can work on without new signals.

### 8b · What we tried

**Experiment.** We tested label-free rules that choose the count per answer: two registered rules (R2_COUNT,
entropy gate), MoM on the L-SML group votes, and four estimator families from Codex (A-D).
The purpose: to find a label-free count that beats Q80.

**Graph.** Bars: PRMScore change against Q80 for each rule (R2_COUNT −0.05, entropy gate −0.5, best Codex arm −0.48,
MoM count −1.2), with TOP_2 (+2.5, post hoc) as a marked reference. Source: v12 "Open direction" part 2. New figure.

**Conclusions.**
1. No label-free rule beats Q80 on PRMScore.
2. R2_COUNT and the entropy gate make many clean answers clean again (false flags 99.5% → 58.6%; MS 88 → 98), at a small cost (−0.05 and −0.5). They help a different goal: the clean-answer decision.
3. The MoM error rate follows the vote quota, not the data (quotas 10/20/30% give 13.6/22.9/32.6%).

### 8c · Why they failed

**Experiment.** We compared the true number of errors per answer with answer length, and with the counts that our rules give.
The purpose: to explain the failure in 8b.

**Graph.** Errors per answer vs number of steps: the truth and the Q80 count (data: `results/count_structure_check_v1`,
commit 17aa1e4; check if a plot exists, else plot from that file).

**Conclusions.**
1. In PRMBench, an answer has about 2 errors (mean 2.12). This number almost does not depend on length (Spearman 0.09).
2. Q80 and all the estimators that we tried give more flags in longer answers.
3. A rule with counts that grow with length cannot match this benchmark. A good count rule needs a different signal.

**Closing box: "Open direction, for discussion".**
1. First check if the fused scores in an answer (for example the gap between the top steps) predict the count beyond length. Use a supervised count model as the ceiling. If this fails, move the effort to ranking.
2. If the information exists: estimate the count from the continuous scores (for example a mixture model per answer), not from thresholded votes.
3. Keep two problems apart: how many errors a wrong answer has (PRMScore), and when an answer is clean (MS and clean controls).

---

## Appendix (collapsed)

- A. Corrections and robustness: the 8 Oct table with old values, prefix/mask attribution, step-0 table, half-split, LOTO, Q80 provenance.
- B. LR analysis: weights N16/N56, family mass and omission, selection × weighting crossover.
- C. Decision rules and estimators: per-rule nine-category tables, estimator accuracy, MoM quota, conditional correlations.
- D. Full matrices: AUC / top1 / PRMScore for all banks × 14 methods; full paper Table 3; 4B tables.
- E. Provenance: dataset and literature audit, task status (16 rows), source files.

## Changes from v1

- Removed: ProcessBench, Mind the Gap, separate digits discussion, old values in section 0.
- Added: section 2 (PRM and PRMBench background), section 5 (external benchmarks), 6b (four-way by type), 6c (4B), 8b (what we tried, as a graph), box "without teacher forcing".
- Channel × type heatmap moved from optional to main (4c).
- Each graph now has "Experiment" text and exactly 3 conclusions.
- New figures from existing numbers: 0 heatmap, 1 dumbbell, 3 leaderboard bars, 5 external bars, 8a ladder, 8b bars (and 8c if no plot exists).
