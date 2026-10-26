# Pre-declaration: descriptive 8-cell evaluation of reasoned channel sets, with and without position neutralization (2026-10-26)

Written before any of the arms below was computed. Not committed (Omri asked to review the note first). The
evaluation script records this file's sha256 in its output.

## Why

Omri asked on 2026-10-26:
- how the proposed channels perform alone and fused with the PRM;
- to take into account the fork's finding that on Socratic the channels track step position, and its
  position-neutralization probe;
- to restrict attention to the four PRMs other than Math-Shepherd.

This is a descriptive evaluation of a few declared sets, not a search. Everything is in-sample development data.

## What had been seen before writing this

- All stage 0–2 outputs and the fork's position probe. The probe covered three subsets, the stage-1 9/10 subset,
  "alt8" and all 16 channels, with channels residualized and the PRM untouched.
- Label-free:
  - the within-answer correlation of every view with relative position, per cell;
  - the consensus signs.
- With labels, as diagnostics only:
  - the within-answer AUC of relative position itself: PRMBench .662, Socratic .731;
  - the within-class position correlations;
  - the composition of the 9 stage-1 subsets that win all 8 cells. 8 of the 9 contain top15_turnover,
    dominant_freq16 and bocpd_p0.

Not seen before writing this: any number for the rule set S_R or for C3; SUB5 in the position-neutral
representation; the Jaffe-assumption diagnostics below; error-category or competitor-protocol numbers for any new arm.

## Cells

PRMBench and Socratic × {qwen_prm_7b, qwen_prm_72b, pure_prm, rlhflow_prm}: 8 cells.

Math-Shepherd is dropped by Omri's decision of 2026-10-26. The stated reasons, from the stage-1b proposal:
- its PRM is position-driven (correlation with relative position +.47 to +.50);
- old general base model;
- Monte-Carlo automatic labels.

It is not computed here.

## Representations

**RAW.** Stage 1's cache: answer-z values; PRM-anchored orientation; per-answer GAP votes.
- Telemetry-only arms use stage 2's cache, with the stored orientation.

**POS.** The fork's probe procedure (`position_probe/probe.py: residual_cell`), copied unchanged:
- every channel's answer-z value is residualized on a cubic in relative step position, by pooled least squares over
  the pool, label-free;
- the PRM is left untouched;
- channels are re-oriented against the PRM; GAP votes are recomputed.
- Telemetry-only arms: the same residualization of stage 2's values, keeping the stored orientation (no PRM).

**Why POS is the principled representation for estimation.** Write each channel as
channel = u·y + β·position + noise. After residualizing the channel, cov(PRM, channel) = u_PRM · u_channel · Var(ỹ) +
shared noise, where ỹ is the part of the error label not explained by position. So every cross-covariance that SML
and DS-EM use is free of position, even though the PRM itself is not residualized.

## Rule R (label-free; the PRM is used, labels are not)

1. **Families,** fixed from the code definitions:
   - uncertainty: q15_H1, q15_VE1, logprob_margin, true_tail50, energy_level;
   - realized: chosen_surprisal, realized_z, realized_drv;
   - dynamics: energy_innovation, top15_turnover, top50_js, dominant_freq16, bocpd_p0;
   - digit: digit_alternative, digit_spread, digit_alternative_innovation.
2. **Orientation audit.** Keep a channel only if its PRM-anchored sign in POS equals its stored sign in all 8 cells.
   Reason: the set must also work without the PRM, and no telemetry-internal rule can repair a stored sign. The
   consensus sign agrees with the stored sign even for the two channels that are below chance.
3. **Accuracy proxy** (Jaffe 2016, Lemma 1): r_0k, the covariance between the PRM's vote and channel k's vote in POS,
   per cell.
4. **Pick.** In each family, keep the audited channel with the largest median r_0k over the 8 cells. A family with
   no audited channel is skipped.
5. **S_R** = the picks, shared by all cells.
6. **Identifiability requirement.** |S_R| ≥ 3, because the two-coin Dawid–Skene model needs at least 3 conditionally
   independent voters.

## Sets

- **S_R:** the rule's output.
- **C3** = {q15_H1, chosen_surprisal, digit_spread}: the earlier proposal.
- **SUB5** = {q15_H1, chosen_surprisal, top15_turnover, dominant_freq16, bocpd_p0, digit_spread}: the only set that
  meets both goals in RAW in the 8 cells.

## Fusions

**F_CI, for S_R and C3.** Each channel is its own group, i.e. Jaffe's model with K = m, which is the Dawid–Skene
model. No clustering.
- **With the PRM:** two-coin DS-EM on [PRM vote, channel votes], anchored on the PRM.
- **Alone:** DS-EM on the channel votes, with the majority label switch.
- **Weights:** clipped ML log-odds, applied to the continuous values (primary). The binary weighted vote is
  secondary.
- **DS-EM probability floor** 1e-6 as in stages 1–2, with 1e-4 as a sensitivity.

**L-SML, for SUB5.** The stage-1 and stage-2 methods unchanged (K ≥ 3 telemetry groups), in RAW and in POS.

## Arms per cell

- **References and context:**
  - the PRM alone (RAW);
  - relative position as a score, later = riskier. This is context, not a method: it imposes a direction prior;
  - CHAIN_N16 and TELE_N16 from STATUS_MAIN.
- **For each set and representation:**
  - set + PRM, fused;
  - mean(PRM, set);
  - set alone, fused;
  - mean(set).

## Metrics

- **All 8 cells:**
  - official PRMScore under GAP;
  - within-answer AUC;
  - TPR and FPR in the official orientation (positive = valid step);
  - the PRM's share of the weight.
- **PRMBench, 4 PRMs:**
  - PRMScore per error category (`category_scores`);
  - PRISM's FPR at matched TPR 87.3;
  - ARES-protocol macro-F1;
  - SLIDER redundancy accuracy;
  - CROP kept at α = 5%, plus the cross-domain Socratic risk.
  These use the stage-2 `competitors.py` functions, for the S_R and C3 arms in POS and RAW and SUB5 in POS.
  SUB5 in RAW already exists in `stage2/COMPETITORS.csv`.

## Jaffe-assumption diagnostics, RAW against POS

Labels are used as diagnostics only, never for choosing.
- **D1. Better than random:** the balanced accuracy of every vote.
- **D2. Conditional independence across groups:** the mean |within-class correlation| of votes for:
  - PRM–channel pairs;
  - channel pairs from different families;
  - channel pairs from the same family.
- **D3. Lemma 1, label-free:** the rank-one misfit of the off-diagonal vote covariance of [PRM, S_R].

## Goals (as in stages 1–2, restricted to 8 cells)

- **G1:** set + PRM beats both the PRM alone and mean(PRM, set) in all 8 cells.
- **G2:** set alone beats mean(set) in all 8 cells.

## Predictions

- **P1.** SUB5 + PRM in POS loses to the PRM in at least 2 of the 3 strong-PRM Socratic cells (Qwen-7B, Qwen-72B,
  PURE), with a PRM share below .5 there.
- **P2.** POS lowers the mean |within-class correlation| of cross-family and PRM–channel pairs in at least 6/8 cells.
- **P3.** The rank-one misfit of [PRM, S_R] votes is lower in POS than in RAW in at least 6/8 cells.
- **P4.** S_R contains digit_spread and chosen_surprisal.
- **P5.** S_R + PRM beats the PRM in 4/4 PRMBench cells in both representations. In POS it does not beat the PRM in
  at least 2 of the 3 strong-PRM Socratic cells.
- **P6.** S_R alone is within 1 PRMScore point of mean(S_R) in at least 6/8 cells. With few, roughly conditionally
  independent voters of similar accuracy, DS weights are close to equal (FRAMING_THEORY P3).

All arms are reported, whatever they show. No arm is selected after the fact.
