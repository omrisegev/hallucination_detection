# Cumulative-vote fusion of first-error localizers — v1 (Step 423)

Date: 2026-09-20. Population: ProcessBench × Llama-3.1-8B, the committed fair-comparison
localization lane (`results/fair_paper_exact_comparisons_v1/lanes/localization/PER_QUESTION_LONG.csv`),
3,400 questions, 2,221 erroneous. Protocol: Mind-the-Gap Step-level Localization Accuracy
(erroneous answers only, no gate), per subset. Every fit is label-free and out-of-fold on the
file's five source folds. Reproduce with

```
python scripts/experiments/cumulative_vote_fusion_v1.py
python scripts/experiments/analyze_cumulative_vote_fusion_v1.py
```

Deterministic: a second run reproduces every number in `REPORT_pooled.json`.

## The method

Each localizer j emits one step ŝ_j. It answers every binary question "is the first error at a
step ≤ n?" at once: vote(n) = +1 if ŝ_j ≤ n else −1 (an ordinal-to-binary decomposition). Instances
are (question, n) pairs restricted to the disagreement region [min ŝ, max ŝ − 1]; outside it all
votes coincide. Fits: SML signed weights (`sml_fuse_signed`), L-SML with group discovery
(`lsml_fuse`), and a Dawid-Skene EM (ψ_j = P(vote ≤ n | truly ≤ n) "not late", η_j = P(vote > n |
truly > n) "not early") initialised from SML. Readouts: mode of the fused pmf (weighted plurality)
and median of the fused CDF (weighted median), after pool-adjacent-violators. Majority vote on
these votes is exactly the median of the five positions. A bias-shift variant subtracts each
localizer's consensus-relative median offset before voting.

Localizers (all label-free, single-trace telemetry): `family6_level_step_top5mean` (dedicated
Local incumbent), `unified28`, `gl_liu_v1_replay`, `max_entropy_step_top5mean`,
`mind_the_gap_common_replay` (derivative readout: EMA + worst drop per step).

## Headline (SLA %, pooled fit, out-of-fold)

| rule | gsm8k | math | olympiadbench | omnimath | all |
|---|---:|---:|---:|---:|---:|
| family6 (incumbent) | 44.44 | 31.14 | 29.20 | 26.22 | **30.12** |
| max_ent | 42.51 | 30.30 | 28.29 | 26.22 | 29.45 |
| gl_liu | 37.20 | 29.29 | 25.11 | 23.58 | 26.83 |
| mind_gap | 28.50 | 25.59 | 21.79 | 21.48 | 23.32 |
| unified28 | 39.61 | 26.09 | 18.91 | 17.39 | 22.24 |
| median of positions (majority vote) | 42.51 | 31.99 | 28.59 | 24.51 | 29.40 |
| SML weighted median | 43.48 | 31.82 | 29.20 | 24.90 | 29.76 |
| L-SML mode | 42.51 | 32.32 | 29.05 | 25.30 | 29.90 |
| Dawid-Skene mode | 44.93 | 31.14 | 29.20 | 25.96 | 30.08 |
| bias-shift variants | = unshifted (every fold's shift rounds to 0) | | | | |

Paired question-level bootstrap, 4,000 draws:

| comparison | all | gsm8k | math | olympiadbench | omnimath |
|---|---|---|---|---|---|
| DS mode − family6 | −0.05 [−0.27, +0.18] | +0.48 [0.00, +1.45] | 0.00 | 0.00 [−0.61, +0.61] | −0.26 [−0.66, 0.00] |
| DS mode − mind_gap | +6.75 [+4.50, +9.00] | +16.43 | +5.56 | +7.41 | +4.48 |
| DS mode − median | +0.68 [−0.63, +1.98] | | | | |

**Decision: `FUSION_REPRODUCES_INCUMBENT`.** No fused rule beats the incumbent; the best of them
is statistically identical to it. Per-subset fitting does not help (gsm8k has ~300 instances per
fold and the DS fit degrades to 38.65).

## Why: the label-free consensus *is* the incumbent

Fitted parameters, pooled scope, mean over folds:

| localizer | SML w | ψ (not late) | η (not early) |
|---|---:|---:|---:|
| family6 | +0.60 | 0.968 | 0.945 |
| max_ent | +0.54 | 0.718 | 0.845 |
| gl_liu | +0.50 | 0.730 | 0.791 |
| unified28 | +0.15 | 0.755 | 0.289 |
| mind_gap | **−0.25** | 0.261 | 0.601 |

L-SML finds K = 3 on every fold: {family6, gl_liu}, {unified28, mind_gap}, {max_ent}. Dawid-Skene
assigns family6 near-perfect sensitivity and specificity, i.e. the consensus of the five is
family6 itself, so the likelihood readout returns family6. Unified-28 is an *early* localizer
(η = 0.29; it predicts step 0 for 34% of erroneous answers against a 12% label rate, a
first-step prior). Mind-the-Gap is a *late* localizer (ψ = 0.26) and receives a negative SML
weight everywhere. A weighted median cannot move a position; it can only choose among them.

## What is different on the long-chain subsets

1. **Accuracy falls with chain depth, and the misses are late.** Incumbent SLA by depth (max
   locator across the five): 45.5% at depth 0–2, 33.7% at 3–4, 21.9% at 5–7, 16.4% at 8+; the
   late fraction rises 0.22 → 0.40 → 0.52 → 0.60 while the early fraction stays at 0.24–0.32.
   Within subset, the longest token-length tercile is where the incumbent collapses (math 12.7%
   vs 41.0% mid; olympiadbench 18.0%; omnimath 14.3% with 61% late).
2. **The lateness exceeds argmax noise.** A uniform guess over [0, max locator] is itself late
   when the true error is early, so the offset is compared to that null by true position. On the
   long subsets the level localizers sit +0.5 to +0.8 steps above the null at positions 1–4
   (family6: +1.67 vs +1.19 at position 1; +1.14 vs +0.40 at 2; +0.42 vs −0.38 at 3), and
   Mind-the-Gap +1.1 to +1.4 above it. The residual late bias is largest when the first error is
   early in a deep chain: 120 + 319 + 321 of the 1,420 long-subset errors sit at steps 0–2.
3. **The derivative readout is later, not earlier.** In this common-protocol replay on Llama the
   Mind-the-Gap locator is the latest of the five on every subset (late fraction 0.46–0.55, mean
   offset +0.5 to +1.3) and the second-weakest overall. The paper's long-chain advantage on Qwen
   (Step 422 table) does not appear here; with EMA span 5 and flux attributed to the step holding
   token j+1, a lag of about one step is the expected signature of that pipeline, testable once
   per-step scores are available.
4. **Diversity rises but in one direction.** Pairwise exact agreement family6/max_ent falls from
   0.75 (gsm8k) to 0.57 (omnimath), family6/mind_gap from 0.41 to 0.26, yet all five share the
   late drift, so their disagreements are not the independent errors SML can exploit.
5. **Reading the fused CDF at a lower quantile rebalances early/late but does not buy exact
   accuracy.** The label-selected sweep (descriptive only): second-earliest position +1.13 pp
   [−0.85, +3.10] on the long subsets, SML quantile 0.2 +0.85 [−0.99, +2.75]; both move the
   long-subset late/early split from 0.45/0.28 to about 0.35/0.37. The earliest position is
   harmful (−5.14 [−7.75, −2.54]).

## Boundaries

- Llama-3.1-8B only; the Qwen cells whose OOF step scores drove Step 422 are not in git.
- Binary votes only: per-step score profiles are not committed, so the soft cumulative-curve
  version (`lsml_continuous` on 2F−1) is not run here. It needs the cluster OOF step scores.
- Mind-the-Gap numbers are the project's common-protocol replay, not the paper's own scores.
- The lower-quantile sweep uses labels to choose q and is a ceiling, not a candidate.
- Labels enter only evaluation and the bootstrap. The clean/error split is the gate's job and
  is outside this study (SLA protocol).

## What this says for the next experiment

The long-chain deficit is a positional bias shared by every level localizer plus argmax
dilution, not a reliability-weighting problem; reweighting cannot fix it because the consensus
is already the best member. The candidates that target the bias directly are (i) an onset
readout at token level (earliest step whose fused risk crosses a label-free threshold) instead
of the peak, (ii) a delta-shaped operator (Laplacian-of-Gaussian) as a fourth column of the
source × operator grid, and (iii) the soft cumulative-curve fusion on the Qwen OOF step scores,
which is where the derivative-versus-level asymmetry was actually observed.

Files: `REPORT_pooled.json`, `REPORT_per_subset.json`, `PREDICTIONS_*.csv` (per-question OOF
predictions for every rule), `ANALYSIS.md` / `ANALYSIS.json` (sections 1–7 of the diagnostics),
`RUN_STDOUT.txt`.
