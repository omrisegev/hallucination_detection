# Advisor document flow, meeting 2026-10-15 (PROPOSAL, not applied)

Status: outline for Omri's approval. No page edits, no new experiments (freeze until the meeting).
Sources: published v12 (`published_v12_2026-10-12.html`), `V11_REVIEW_2026-10-12.md`, `SUMMARY_8B_vs_4B.md`,
`analysis/prmbench_think_block/FINDINGS.md`, 8 Oct page `docs/meetings/advisor_meeting_2026-10-08/findings_page/findings.html`
and its `README_HE.md` (what the advisors asked for).

## The story in one line per section

| # | Question the section answers | One graph | Answer in one sentence |
|---|---|---|---|
| 0 | Where did we stop on 8 October? | Methods × banks AUC table (corrected) | The fusion methods are all within about 0.01; LR is above them all. |
| 1 | Can we trust the numbers now? | Old → corrected AUC per method (N16) | The think-block fix lowers all methods by about the same amount; the order does not change. |
| 2 | How do we compare with a supervised PRM on the official metric? | PRMScore leaderboard bar | A tie with Qwen2.5-Math-PRM-7B, above all open PRMs in the paper. |
| 3 | Where do we win and where do we lose? | Category radar + matched controls | We win on redundancy, also after controls; PRM wins on the "plausible" error types. |
| 4 | Is the gap to supervision in the signals or in the fusion? | LR gap across 10 banks | The gap is small at N16 (0.021) and LR does not find a better weighting we can copy. |
| 5 | If the ranking is close, why is PRMScore only a tie? | Budget curves | At an equal flag budget PRM leads (68.3 vs 65.7); our global threshold is already at its best. |
| 6 | What can still be gained, and where did we stop? | Ceiling ladder | The number of flags per answer is worth up to +6.3 points; all label-free count rules we tried failed. Open direction. |

Appendix: all long tables, per-rule category tables, LR weights, estimators, correlations, provenance, task status.

---

## Section 0: Where we stopped on 8 October

**Question:** What did the advisors see last time, and what does the same view look like now?

**Graph:** The same PRMBench "methods × banks, within-answer AUC" table from 8 October, rebuilt on the corrected
input (source: `controls/ALL_BANK_TOP1_AUC.csv`, already in the v12 appendix "Full matrices"). Recommend a heatmap
with the bank mean column, the method mean row, LR as the last column and N16 marked. Put the 8 Oct values
in a tooltip or a small "old" sub-row, not in a second table.

**Bullets:**
- All label-free methods are within about 0.01 AUC of each other in every bank. Fusion saturates. (N16: plain average 0.771, best fusion 0.781.)
- Fusion beats the plain average by about 0.01 at N16 (0.780 vs 0.771). The gain is small; the best fusion method is above the plain average in every bank.
- Supervised LR on the same channels is above every label-free method in every bank (N16 0.801; mean over banks 0.804 vs 0.769 best label-free).

**Notes:**
- The 8 Oct table used the DS-signed average; v11/v12 use the DS-filter average. They are different controls. Name it on the graph.
- Same view for top1-on-error as a second tab, not a second table on the page.
- ProcessBench: one sentence only ("no gain over the plain average, so we focus on PRMBench"), as the advisors decided.

**Drop / move:** the PRMScore × banks matrix and the top1 matrix go to the appendix.

---

## Section 1: Can we trust the numbers? (corrections since 8 October)

**Question:** The 8 Oct numbers used a teacher-forcing prompt without the empty think block. What changed?

**Graph:** N16, old vs corrected AUC per method (dumbbell plot: plain, DS-filter, L-SML, IU-PCR, LR).
Values are in the v12 section 02 table. New figure from existing numbers.

**Bullets:**
- The fix lowers all methods by about the same amount (L-SML 0.796 → 0.779, plain 0.787 → 0.771, LR 0.822 → 0.801). The order of the methods does not change.
- It is not a first-token effect: without step 0 the decline is larger (−2.2 vs −1.6 points). Masking the first token changes AUC by less than 0.1 point.
- The results repeat on Qwen3-4B (N16 L-SML 0.775 vs 0.779) and on a held-out half of the source groups (IU-PCR 0.786).

**Box "Claims we withdraw"** (short, 3 lines): realized_drv leads 6/8 types (now 0/8); "all ~40 literature rows are weaker on Simplicity" (10 of 15 in Table 3); self-generated vs teacher-forced "equivalence".

**Drop / move:** step-0 selection table, prefix/mask attribution table, 4B tables (keep one line + link), LOTO and half-split tables, Q80 provenance. All to the appendix "Robustness". The advisors asked for these checks, so they must be easy to reach, but they do not move the story.

---

## Section 2: Against a supervised PRM on the official metric

**Question:** Where are we on the PRMBench leaderboard metric?

**Graph:** Horizontal bar chart of Overall PRMScore: our L-SML / IU-PCR, Qwen2.5-Math-PRM-7B (our run) and the
paper Table 3 rows (open PRMs, GPT-4o, o1-mini, Gemini). Numbers are in the v12 leaderboard table. New figure from existing numbers.

**Bullets:**
- Our Overall 65.4 equals Qwen2.5-Math-PRM-7B (65.5); the paired difference is −0.08 [−0.77, +0.64].
- We are above all open PRMs in the paper (best 60.5) and below GPT-4o (66.8) and o1-mini (68.8). We use no labels and no training.
- On ranking only: PRM has the higher AUC (0.801 vs 0.779), but our top step is an error more often (59.9% vs 57.8%).

**Drop / move:** the full 15-row Table 3 with nine columns goes to the appendix. Keep the three-method summary table (PRMScore / AUC / top1) under the graph.

---

## Section 3: Where we win and where we lose

**Question:** Which error types drive the tie? Is our advantage real or a position artifact?

**Graph A (main):** Category radar (already in v12): ours vs PRM-7B, nine types.

**Bullets A:**
- We lead in Simplicity: redundancy 70.0 vs 49.0 and circular 68.0 vs 55.0. No row in paper Table 3 is higher on redundancy.
- PRM leads on the "plausible" errors: confidence, counterfactual, deception (Soundness 71.0 vs 66.9, Sensitivity 75.5 vs 69.3). A surprise signal has no reason to catch an error that the model finds fluent.
- Our MS is low (88 vs 99.9): Q80 flags a step in almost every clean answer. This links to section 6.

**Graph B:** Nuisance controls (already in v12, `nuisance_controls.svg`), position + length matched.

**Bullets B:**
- Redundancy survives every control: +19.7 AUC points over PRM [15.9, 23.6]. This is our strongest result.
- Circular shrinks to +4.3 points and the interval crosses zero. We do not claim it.
- On all answers, position matching removes both PRM's AUC lead and our lead over the plain average (subset of 4,691 answers; a caution, not a final result).

**Optional Graph C:** channel × error-type heatmap (in v12). Use only if time allows; one bullet: uncertainty channels lead redundancy (0.84), step index alone leads confidence (0.87).

**Drop / move:** the two worked NR/NCL examples can stay as a collapsed box (they help the advisors read the category). Category legend collapsed.

---

## Section 4: Is the gap to supervision in the signals or in the fusion?

**Question:** LR on the same channels is better. Can we learn from it?

**Graph:** LR gap across the ten banks (v12 "New OOF gap across ten banks"), AUC and top1 side by side.

**Bullets:**
- The AUC gap is smallest at N16 (0.021) and grows with bank size to a peak at N40 (0.063).
- On top1 we lead LR up to N25: at N16 L-SML is +2.0 points [1.0, 3.0] above LR.
- LR's weights do not show a better weighting we can copy: 88% of its weight is on two nearly identical digit channels with opposite signs, and its weights agree with ours at cosine 0.03. On 4B, LR drops in all ten banks while our methods do not.

**Drop / move:** N16 and N56 weight plots, family mass, family omission, crossover selection × weighting table, four-way top1 by type, historical bank panels. All to the appendix "LR analysis" (the advisors asked for it, keep it reachable). Keep one sentence: "selection and weighting interact; giving LR's feature half to our fuser does not close the gap (0.774 vs 0.797)".

---

## Section 5: If the ranking is close, why is PRMScore only a tie?

**Question:** Is it the threshold?

**Graph:** Budget curves (in v12): PRMScore vs share of steps flagged, ours and PRM, with the Q80 and PRM 0.5 points marked.

**Bullets:**
- At the same flag budget (about 18%), PRM scores 68.3 and we score 65.7. PRM's default 0.5 rule flags only 8.7% of steps, which is why the default scores tie.
- Our Q80 point is almost at the top of our curve. The best global threshold adds only 0.3 points for us (2.9 for PRM).
- So a better global threshold is not the fix. The question moves inside each answer: how many steps to flag.

**Drop / move:** the registered-rule table (R2_COUNT, entropy gate) moves to section 6 as "what we tried". The 3 × 9 per-rule category table goes to the appendix.

---

## Section 6: What can still be gained, and where we stopped (end of the story)

**Question:** With the same ranking, how much is the per-answer count worth, and why did our count rules fail?

**Graph A:** Ceiling ladder (bar chart of the v12 "Open direction" table): Q80 65.4 → best global cut 65.7 →
constant 2 flags 67.9 → count from error type 68.9 → true count per answer 71.7 → perfect ranking at our counts 81.8.
Color the bars that use labels differently. New figure from existing numbers.

**Bullets A:**
- The true number of errors per answer adds +6.3 points with the same ranking. That alone would put us above PRM-7B at its best threshold (68.3).
- Two flags per answer give 67.9, already above PRM-7B's default. This is post hoc, not a method, but it shows the right count is small.
- Perfect ranking is worth more (+16.4), but the LR gap shows our channels do not hold that information today. The count is the part we can still attack without new signals.

**Graph B:** Errors per answer vs answer length: truth vs our count rules (data in `results/count_structure_check_v1`, commit 17aa1e4; check if a plot exists, else plot from that file).

**Bullets B:**
- In PRMBench the number of errors per answer is almost constant (mean 2.1) and does not depend on length (Spearman 0.09).
- Q80 and every estimator we tried give counts that grow with length. This is a simple reason why they failed.
- What we tried (one short table): R2_COUNT (clean-answer false flags 99.5% → 58.6%, PRMScore −0.05), entropy gate (MS 88 → 98, −0.5), MoM count (64.2), Codex estimators A–D (best −0.48). None beat Q80.

**Closing box, "Open direction for discussion"** (from v12 part 3, shortened):
1. Check if the fused score inside an answer holds count information beyond length; a supervised count model is the ceiling.
2. If yes: a count estimator on the continuous score, not on thresholded votes.
3. Separate "how many errors" (what PRMScore rewards) from "is the answer clean" (what ProcessBench and MS reward).

**Drop / move:** the MoM quota section, the estimator accuracy table, the "Groups reduce the prevalence bias" section. Appendix. Keep one line: "the MoM error rate follows the vote quota (10/20/30% → 13.6/22.9/32.6%), not the data".

---

## Appendix (collapsed, in this order)

- A. Robustness: prefix/mask attribution, step-0 table, 4B vs 8B tables, half-split, LOTO, externals (Hard2Verify, Socratic; with the step-index caveat).
- B. LR analysis: weights N16/N56, family mass and omission, crossover, four-way top1 by type.
- C. Decision rules: per-rule nine-category tables, estimator accuracy, MoM quota, conditional correlations.
- D. Full matrices: AUC / top1 / PRMScore × all banks × 14 methods; full paper Table 3.
- E. Provenance: Q80 chronology, dataset construction and literature audit, task status (16 rows), source files.

## What changes against v12

- New order: v12 starts with the PRM comparison and the prefix repair. The proposal starts with the 8 Oct table, so the advisors see continuity.
- Four new figures, all from existing numbers: section 0 heatmap, section 1 dumbbell, section 2 leaderboard bar, section 6 ceiling ladder (plus section 6 graph B if no plot exists).
- About 15 tables move from the main text to the appendix. The main text keeps about 8 graphs and 3 small tables.
- The "Accountability" task table moves to the appendix.
- Page owner: the "Advisor page, PRMBench focus" thread. Merge rule: check the live artifact equals `published_v12_2026-10-12.html` before any update.

## Open points for Omri

1. Section 0: heatmap of the corrected numbers with the old values as a sub-row (recommended), or the exact 8 Oct table next to the corrected one?
2. Section 3 graph C (channel × type heatmap): keep or appendix? Recommend appendix.
3. ProcessBench: one sentence in section 0 (recommended) or a small appendix table?
