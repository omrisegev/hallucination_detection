# PRM-anchored view selection, stage 1: exhaustive subset search (CPU, 2026-10-25/26)

Pre-registration: `docs/experiments/PRM_VIEW_SELECTION_STAGE1_PREREG_2026-10-25.md` (0124cfe6; pre-run checks
0a6be2e6).

**Method.** All 64,839 subsets of the 16 N16 channels with |S| ≥ 4, each fused with the PRM in each of 10 cells (5 PRMs
× PRMBench and Socratic), refitted from scratch per cell:
- PRM-anchored orientation; per-answer GAP votes;
- L-SML groups (Jaffe 2016, Algorithm 1), with K ≥ 3 required in every cell;
- SML within groups and two-coin DS-EM between groups, both non-negative;
- binary-estimated weights applied to the continuous answer-z values;
- GAP flags; official PRMScore.

**Success.** Beat both the PRM alone (same rule) and the plain mean of the same views, in all 10 cells.

**Run.** 72 minutes on 14 workers, from light per-cell caches; the cache reproduces chunk 0 bit for bit.
- Vectorized GAP: identical to the evaluator on 345/345 checks.
- Every subset reported by name was recomputed through the evaluator's own loops: exact (`NAMED_DETAIL.csv`,
  `exact_ok`).

## Result: no subset meets the goal

| Count | Subsets |
|---|---|
| Valid (K ≥ 3 in all 10 cells) | 12,247 |
| **Beat both PRM and the same-set mean in all 10 cells** | **0** |
| Same, with binary scoring / HEM (top 300 + successes) | 0 / 0 |
| Beat the PRM alone in all 10 cells | 2 (near-identical; they differ only by one digit channel) |
| Beat the same-set mean in all 10 cells | 18 |
| Beat both in all 5 PRMBench cells | 815 |
| Beat both in all 5 Socratic cells | 4 |
| Most cells won | 9/10 (5 subsets) |

- **Best min-margin over the 10 cells:** −0.61, for an 8-channel subset that wins 6/10.
- **Every 9/10 subset fails one cell.** Three fail Socratic Math-Shepherd (−15.7 to −16.2); two fail Socratic
  Qwen-7B (−3.4).

### Socratic is the bottleneck

The table is over the 12,247 valid subsets.

| Cell | PRM alone | Share of subsets beating it | Best fused − PRM |
|---|---|---|---|
| Socratic Qwen-7B | 70.10 | 43% | +2.34 |
| Socratic Qwen-72B | 71.72 | 28% | +0.83 |
| Socratic PURE | 70.59 | 11% | +0.62 |
| Socratic RLHFlow | 50.38 | 35% | +7.50 |
| Socratic Math-Shepherd | 65.52 | 1.7% | +0.43 |

On PRMBench, 81–96% of valid subsets beat the PRM.

### Selection does not transfer

Margins below are fused − max(PRM, same-set mean) in the held-out cells.
- **Selected on PRMBench, tested on Socratic:** −1.3, −6.3, −5.0, −4.4, −15.1. Against the PRM alone: −1.3 to −15.1.
- **Selected on Socratic, tested on PRMBench:**
  - against the PRM alone it wins 5/5 (+0.8 to +6.1);
  - against the same-set mean it loses 5/5 (−0.35 to −6.0).
- **Leave-one-PRM-out:** the held-out PRMBench cell is won 5/5 (+0.2 to +3.1); the held-out Socratic cell is lost 5/5
  (−1.3 to −20.5).

What can be won on Socratic depends on the specific PRM. It is found by the search, not by a rule that carries over.

### Mechanism of failure

- **Socratic.** Where a fused score beats the PRM, the DS-EM between-group step gave the PRM 70–86% of the weight. The
  fusion survives by staying close to the PRM.
- **When the PRM's share collapses,** PRMScore collapses with it. For example, the leave-one-PRM-out subset for PURE
  on Socratic PURE: PRM share .08, PRMScore 50.1 against the PRM's 70.6.
- **How often:** between-group weights were clipped (negative raw weight) in 37% of valid subset × cell fits.
- This is the stage 0 mechanism again: the label-free latent class follows the telemetry consensus.

### Context, not criteria (`NAMED_DETAIL.csv`)

- **Supervised LR on the same views** (5-fold GroupKFold) beats the PRM in most Socratic cells, for example 71.66 vs
  70.10 on Qwen-7B.
- **CHAIN_N16** is ahead of every subset on Socratic RLHFlow (57.06).

## Predictions

| | Prediction | Verdict |
|---|---|---|
| M1 | Diversity: at most one of the digit channels over-represented | Holds weakly. In the top 1% (123 subsets; there are no successes), digit_spread .58 vs .51 baseline; the other two at chance |
| M2 | Channels above chance on Socratic over-represented, sub-chance ones under-represented | **Mixed.** Over-represented: realized_z .85, chosen_surprisal .77, true_tail50 .67. Under-represented: q15_H1 .15, energy_level .09, logprob_margin .15, bocpd_p0 .33. But realized_z was below chance for Math-Shepherd on Socratic in stage 0 |
| M3 | The PRM keeps the largest between-group weight in successful subsets | **Fails on PRMBench** (top-1% median PRM share .08–.21). Holds on Socratic where fusion helps (.53–.86) |
| M4 | All-10 success rare or absent | **Holds: absent** |

## Omri's structural conditions

These are label-free, on binary votes:
- each channel correlates with the PRM more than its mean correlation with the rest of S;
- low pairwise correlation.

Neither predicts success:
- Spearman with the min-margin over valid subsets: −.06 for the first condition, +.27 for the second (more
  correlation goes with a slightly better min-margin).
- The top 1% satisfies the first condition only slightly more often: .73 of channels, against .65 overall.

## Independent verification and caveats

The verification agent used its own re-implementation and did not import stage1.py, analyze.py or
prm_anchored_lsml.py. Its code is in the session scratchpad.
- **Every count above recomputes exactly.** That includes 0 successes, 2 beating the PRM in all 10 cells, 815, 4,
  and a maximum of 9.
- **Re-implementation.** It matches `SEARCH.npz` in 4/4 cells once the same probability floor is used. The repo code
  replays `SEARCH.npz` with zero difference.
- **No bug found** in orientation, ties, clipping, label switching, the Socratic fast metric or the vectorized GAP.
  Labels enter only the metric.

**Caveats it raised:**
1. **An unregistered probability floor decides the small Socratic wins.**
   - DS-EM clips ψ, η and the prevalence to [1e-6, 1−1e-6]. In Socratic cells the PRM's ML estimate often sits on
     that boundary (in 15–61 of 200 random valid subsets per Socratic cell, against 0–5 on PRMBench). Its
     log-odds weight is then set by the floor.
   - The "2 subsets beat the PRM in all 10" and "4 win all 5 Socratic cells" claims hold at floors of 1e-12, 1e-9
     and 1e-6, but **break at 1e-4**.
   - The winning Socratic margins are mostly 0.002–0.06 PRMScore points, except RLHFlow (1.4–1.7). They are
     tie-level and depend on the estimator settings.
   - The headline (no subset meets the goal) does not depend on this.
2. **The PRMBench fitting pool is label-informed.** `keep` excludes the 758 control answers, which are known to be
   error-free. This was pre-registered, as in stage 0 amendment 2, but it contradicts "no label enters".
   - With all answers in the pool, PB Qwen-7B on the best_all10 subset moves from +0.025 to −0.118.
   - The PRM scale (probability vs logit) was also picked with labels earlier (disclosed).
3. **Minor DS-EM deviations from the pre-registration text.**
   - Initialization is a soft majority (vote fraction clipped to [.05, .95]), not the hard majority.
   - The tolerance is on the mean, not the total, log-likelihood.
   - The 500-iteration cap is hit in up to 48 of 200 fits (Socratic RLHFlow).
   - Effect: < .005 PRMScore on PRMBench, up to about .04 in the Socratic boundary cells.
4. **An unregistered rule that never fired.** A cell is invalid if every between-group weight clips to 0. Never
   triggered.

## Side analyses from a fork of this session (exploratory, not pre-registered)

These were forwarded 2026-10-26. The 8-cell counts were re-derived from `SEARCH.npz` here and match exactly.

1. **Why Math-Shepherd fails on Socratic: step position.**
   - Its PRM correlates +.47 to +.50 with relative step position, against +.24 to +.34 for the Qwen PRMs.
   - On Socratic, the PRM-anchored orientation flips 10/16 channels (2/16 on PRMBench).
   - Residualizing the channels only on a cubic in relative position (label-free) brings the flips to 2/16. Channels
     above chance go from 9/16 to 15/16.
   - Code and data: `../position_probe/probe.py`, `PROBE.csv` (not committed by this session).
2. **Rerunning the method on position-residualized channels** fixes the orientation, but not the weighting.
   - Socratic Math-Shepherd rises (the 9/10 subset: 49.9 to 56.2), still below its PRM (65.5).
   - The same subset collapses on Socratic Qwen-7B, Qwen-72B and PURE. The channels now agree more with the PRM, and
     DS-EM moves the weight to them: the PRM's share falls from .85 to about .2.
3. **Without Math-Shepherd (8 cells).**
   - 16,730 subsets are valid; 9 beat both the PRM and the mean in all 8; 85 beat the PRM in all 8.
   - Best: q15_VE1 + chosen_surprisal + top15_turnover + dominant_freq16 + bocpd_p0. Margins: PRMBench +0.63/+0.82/
     +0.18/+1.63; Socratic +0.29/+0.21/+0.23/+2.97 (Qwen-7B, 72B, PURE, RLHFlow).
   - On Socratic Qwen/PURE, the PRM holds 70–86% of the weight in these fits, so those margins are essentially the
     PRM. The only substantial Socratic gain is on RLHFlow.
   - Dropping a PRM was not pre-registered. Adopting it needs an amendment with a stated reason, plus the
     cross-checks rerun on the 8 cells.

## Files

- `SEARCH.npz`: all subsets × cells.
- `ANALYSIS.json`, `SUBSETS_NAMED.csv`, `NAMED_DETAIL.csv`, `CHANNEL_FREQ.csv`, `STRUCTURE.csv`, `HEM.csv`, `PARITY.json`.
- Code: `stage1.py`, `analyze.py`; `cache/` (light per-cell inputs).
- Logs: `run.log`, `hem.log`, `detail.log`.
- Helpers: `spectral_utils/prm_anchored_lsml.py`.
