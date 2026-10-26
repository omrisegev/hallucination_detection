# View selection stage 2: telemetry-only L-SML, exhaustive subset search, plus comparisons (CPU, 2026-10-26)

**Pre-registration.** `docs/experiments/PRM_VIEW_SELECTION_STAGE2_PREREG_2026-10-26.md` (92cc8faa; pre-run checks
6cd68544). The stage-1b amendment proposed after stage 1 is recorded, not run, in
`PRM_VIEW_SELECTION_STAGE1B_PROPOSAL_2026-10-26.md`.

**Method.**
- **Search.** The stage-1 algorithm with no PRM anywhere, on all 64,839 subsets (4–16 channels) × 10 cells.
- **Channels.** Stored (frozen-bank) orientation, per-answer GAP votes.
- **Fusion.** L-SML groups (K ≥ 3), SML within groups, DS-EM between groups (label switch by majority), weights
  applied to the continuous answer-z values.
- **Primary success.** Beat the plain mean of the same channels in all 10 cells.

**Run and checks.**
- Runtime: 69 minutes on 14 workers.
- Parity of the vectorized GAP: 610/610.
- Exact re-check of every named subset through the evaluator loops: 193/193 comparable values. Two stage-1 values
  have no search value, because the pre-registered early exit skips the remaining cells once one cell is invalid.

Figures are in `figures/`.

## Stage 2 results

| | Subsets |
|---|---|
| Valid in all 10 cells | 19,254 |
| **Beat the same-set mean in all 10 cells (primary)** | **380** |
| Same, with binary scoring | 4 |
| Beat max(PRM, same-set mean) in all 10 (secondary) | **0** |
| Primary on all 5 PRMBench cells / all 5 Socratic cells / 8 cells without Math-Shepherd | 2,282 / 1,177 / 1,078* |

**Best primary subset:** q15_VE1 + chosen_surprisal + logprob_margin + energy_innovation + top50_js (5 channels).
- It wins all 10 cells, with a min-margin of +1.62 against its own mean.
- The same subset is chosen when selecting on Socratic only, and in the leave-one-PRM-out runs for Qwen-7B and PURE.
- Selected on Socratic, it also wins PRMBench 5/5 (+2.5 to +5.4).
- Selected on PRMBench, the choice wins Socratic 3/5.

**Honest reading.** This subset beats its *own* plain mean everywhere: label-free weighting adds value over equal
weights. But the subset itself is weak in absolute terms.

| Qwen-7B | PRMBench | Socratic |
|---|---|---|
| this subset, fused | 67.30 | 65.07 |
| its own mean | 62.72 | 60.22 |
| TELE_N16 (all 16) | 69.26 | 64.47 |
| MEAN_N16 | 68.60 | 63.37 |
| PRM alone | 68.01 | 70.10 |

- **Telemetry alone does not beat the PRM on Socratic strong PRMs.** The share of valid subsets beating the PRM is 0%
  on Socratic Qwen-72B, PURE and Math-Shepherd, and 2% on Socratic Qwen-7B.
- **Prediction T2 holds.** T1 is wrong in its second half: on Socratic the weighting beats the mean more often (24–76%
  of subsets) than T1 expected.

\* 1,073 in `ANALYSIS.json`, which follows the search's pre-registered early exit: once one cell is invalid, the
  remaining cells are not evaluated. A cell-4 (PRMBench Math-Shepherd) failure therefore hides Socratic cells. The
  verification agent completed those hidden cells and found 5 more 8-cell successes. The 8-cell both-goals count (1)
  and the JOINT 8 subset do not change.

## Comparison with stage 1 (with the PRM)

Over the 8,194 subsets valid in both runs:
- **The two runs' worst-cell margins correlate weakly:** Spearman +.26 (prediction T3 holds).
- **The top 1% of the two runs do not overlap at all:** 0 of 123 and 193; Jaccard 0.
  - Stage 1's top subsets carry realized_z (85%) and true_tail50 (67%).
  - Stage 2's 380 successes carry energy_innovation (88%), realized_drv (68%) and chosen_surprisal (67%); realized_z
    appears in only 10%. See figure 4.
- **No subset meets both goals in all 10 cells.**
- **Exactly one meets both in the 8 cells without Math-Shepherd:**
  **q15_H1 + chosen_surprisal + top15_turnover + dominant_freq16 + bocpd_p0 + digit_spread**.
  - With the PRM, it beats both the PRM and its mean in 8/8 (+0.15 to +1.29).
  - Without the PRM, it beats its mean in 8/8 (+0.09 to +2.86).
  - It fails Socratic Math-Shepherd with the PRM (−16.2) and PRMBench Math-Shepherd without it (−0.53).
  - Its Socratic margins with the PRM are tie-level (+0.15 to +0.62). There the PRM holds most of the weight, and
    stage 1's DS-EM probability-floor caveat applies.
- **Joint best over all 20 margins:** q15_VE1 + chosen_surprisal + energy_innovation + top15_turnover + realized_z +
  digit_alternative + digit_alternative_innovation. Its min over 20 is −1.03.

## Against CROP, ARES, PRISM and SLIDER (`COMPETITORS.csv`, figure 6)

**Setup.**
- Five outstanding subsets:
  - SUB1: stage-1 best;
  - SUB2: stage-1 best on 8 cells;
  - SUB3: stage-2 best;
  - SUB4: joint best;
  - SUB5: joint 8-cell.
- Each enters "+PRM" (the stage-1 fusion) and "alone" (the stage-2 fusion, telemetry only).
- They are scored under the competitors' own protocols, re-used from `collection_competitors.py`. Parity: 40/40
  existing CHAIN_N16 / TELE_N16 table values reproduced exactly.
- **All PRMBench numbers are in-sample:** the subsets were selected with PRMBench labels.

The table is for Qwen-7B, the PRM the competitors used.

| Protocol (metric) | Paper | PRM | CHAIN_N16 | Best +PRM | Best alone |
|---|---|---|---|---|---|
| PRISM (PRMScore, GAP) | 68.0 | 68.01 | 70.40 | 70.55 (SUB4) | 68.78 (SUB5) |
| PRISM (FPR at TPR 87.3, lower better) | 47.13 | 47.70 | 47.78 | **46.31 (SUB5)** | 52.18 (SUB5) |
| CROP (% kept at risk 5%) | 21.8 | 4.83 | 23.33 | 23.67 (SUB5) | 18.93 (SUB5) |
| CROP cross-domain: Socratic risk at α 5% | 14.8 (theirs) | 2.07 | 6.71 | 6.48 (SUB3) | 6.71 (SUB3) |
| ARES protocol (Macro-F1) | n/a (different subset) | 68.17 | 68.10 | 68.81 (SUB5) | 66.58 (SUB5) |
| SLIDER (redundancy accuracy) | Step-RRI 61.9 | 78.81 | 74.05 | 75.43 (SUB5) | 75.48 (SUB5) |

**Reading.**
- **With the PRM.** The best subsets slightly improve on CHAIN_N16 under PRISM's matched-TPR reading, giving an FPR
  0.8 below PRISM's. CROP and ARES are ties within about 0.6.
- **Alone, without the PRM:**
  - SUB5 reaches a PRMScore above the PRM's (68.78 vs 68.01), and a SLIDER accuracy far above Step-RRI.
  - It is behind the PRM under PRISM's matched TPR, under CROP and under ARES.
- **CROP cross-domain.** Calibrated on PRMBench and tested on Socratic, most +PRM subsets break the 5% risk target
  badly (up to 74% on PURE). The "alone" versions break it less.
- **Other PRMs.** On the weak PRMs (RLHFlow, Math-Shepherd), both versions beat the PRM by 6.6 to 11.6 Macro-F1 under the
  ARES protocol.

## Verification and caveats

**Independent re-implementation** (own code, not importing the stage scripts):
- every headline count matches (19,254 / 380 / 0 / 0 / 1);
- the best subset and its +1.62 min-margin match;
- Spearman +.26 matches;
- the best subset on two cells matches exactly;
- on 60 random subsets: K and validity 507/507, mean 487/487, fused 486/487.

The one fused difference is 0.003, from where DS-EM stops iterating. The competitor claims (SUB5 +PRM matched-TPR FPR
46.31; SUB5 alone 68.78) also match.

**Fixed after the check:**
- **Channel frequency** is now counted over the 380 successes, as pre-registered (it had used the top 1%; same
  picture).
- **The PRM's PRISM matched-TPR threshold** now uses the PRM's raw score, as in the existing table (RLHFlow and
  Math-Shepherd had used answer-z). Competitor parity is now 75/75, including the PRM rows.

**Carried over from stage 1, as disclosed there:**
- DS-EM soft-majority start;
- tolerance on the mean log-likelihood;
- probability floor 1e-6;
- the PRMBench pool excludes the label-known control answers.

## Files

- `SEARCH.npz` (all subsets × cells), `ANALYSIS.json`, `COMPARISON.json`, `SUBSETS_NAMED.csv`, `NAMED_EXACT.csv`,
  `CHANNEL_FREQ.csv`, `COMPETITORS.csv`, `COMPETITORS_PARITY.json`, `PARITY.json`, `CACHE_META.json`.
- `figures/1-6_*.png`.
- Code: `stage2.py`, `analyze.py`, `competitors.py`.
- Logs: `run.log`, `exact.log`, `competitors.log`.
