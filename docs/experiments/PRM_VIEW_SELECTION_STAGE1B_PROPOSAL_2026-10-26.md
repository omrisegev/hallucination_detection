# View selection stage 1b: proposed amendment after stage 1 (2026-10-26). PROPOSED, NOT RUN

Status: recorded at Omri's request ("תתחיל מלתעד את הריצה הקודמת ואת התוכנית שהagent הציע", i.e. "start by
documenting the previous run and the plan the agent proposed"). Nothing here has been run.

## Stage 1 in one paragraph

Details are in Step 479 and `results/prm_view_selection_v1/stage1/README.md`.

Method and search:
- An exhaustive search over all 64,839 subsets of the 16 N16 channels (|S| ≥ 4).
- PRM-anchored binary L-SML: groups as in Jaffe 2016, Algorithm 1, with K ≥ 3; SML within groups; DS-EM between
  groups; non-negative weights applied to the continuous values.
- Refitted in each of 10 cells (5 PRMs × PRMBench and Socratic).

Results:
- **No subset beats both the PRM and the same-set mean in all 10 cells:** 0 of 12,247 valid. The best wins 9/10.
- **Socratic is the bottleneck:** 1.7–43% of subsets beat the PRM there, against 81–96% on PRMBench.
- **Selection transfers neither across benchmarks nor across PRMs.**

Verification:
- An independent re-implementation matched.
- The small Socratic wins depend on DS-EM's unregistered probability floor (1e-6).

## What a fork of this session found (exploratory)

1. **Math-Shepherd fails on Socratic because of step position.**
   - Its PRM correlates +.47 to +.50 with relative position, against +.24 to +.34 for the Qwen PRMs.
   - The PRM-anchored orientation flips 10/16 channels on Socratic (2/16 on PRMBench).
   - Residualizing the channels only, on a cubic in relative position (label-free, pooled least squares), brings
     the flips to 2/16. Channels above chance go from 9/16 to 15/16.
2. **Position cleaning fixes orientation, not weighting.**
   - With residualized channels, the 9/10 subset rises on Socratic Math-Shepherd (49.9 → 56.2), but stays below its
     PRM (65.5).
   - It collapses on Socratic Qwen-7B, Qwen-72B and PURE: DS-EM moves the weight from the PRM (share .85) to the
     channels (about .2).
   - Code and data: `results/prm_view_selection_v1/position_probe/`.
3. **Without Math-Shepherd (8 cells):** 9 subsets beat both comparators.
   - Best: q15_VE1 + chosen_surprisal + top15_turnover + dominant_freq16 + bocpd_p0.
   - But on Socratic Qwen/PURE the PRM holds 70–86% of the weight. Those margins (+0.2 to +0.3) are essentially the
     PRM; the only substantial Socratic gain is on RLHFlow.

## The proposed amendment (to pre-register before any rerun)

- **A PRM weight anchor.** Either a floor on the PRM's between-group weight share, or the PRM placed inside every
  group. Rationale: stage 0 and stage 1 both show that label-free CI estimators over-trust the telemetry consensus.
- **Position residualization of the channels:** a cubic in relative step position, label-free. The PRM is left
  untouched; residualizing only the channels leaves cov(PRM, channel) unchanged.
- **Sensitivity runs:** DS-EM probability floor of 1e-12 / 1e-6 / 1e-4; a fitting pool that includes the PRMBench
  control answers.
- **Cross-checks** (PRMBench ↔ Socratic, leave-one-PRM-out) on 10 cells, and on 8 cells if Math-Shepherd is dropped.
  Dropping Math-Shepherd needs a stated reason: position-biased PRM, old general Mistral base, Monte-Carlo
  auto-labels.

## Decision

Omri chose first to run the same search without the PRM: telemetry-only L-SML, stage 2,
`PRM_VIEW_SELECTION_STAGE2_PREREG_2026-10-26.md`. Stage 1b stays proposed.
