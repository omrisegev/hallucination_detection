# er_generality_v1: is the stage-B method tailored to the 13-channel bank?

Omri, 2026-09-27: "What worries me most is that the new method was tailored to exactly the 13 features, and that on
20 or 30 features it will not work well." This experiment answers it on larger banks that already existed, with the
frozen chain of expectation_realization_v1 stages B/B2 and **no parameter changed**.

- Protocol: `PROTOCOL.json`, frozen at 41408b2d7; one hygiene rule (drop zero-variance channels) added before freezing,
  after the fold-0 smoke exposed a dead pool channel. Pre-run review fixes: 17e3f4cae.
- Run: `run_20260927/`, commit 2b3a8e862. COMPLETE, 0 failed fits; B13 reproduces stage B exactly (8.9e-16, including the
  binary-partition arm); B20 and B32 reproduce the Step 438/439 calfix runs exactly (within-AUC diff 0.0).
- Verification: three independent agents (`RED_TEAM.md`); every number below reproduces.
- Worktree `.worktrees/er-generality-v1` was deleted after this document was committed and merged into
  `claude/ssl-pseudolabel-residual-v1`. The npz outputs were copied to that worktree's `results/er_generality_v1/run_20260927/`
  (gitignored).

## Banks (all pre-existing, none built for this question)

| Bank | Channels | Origin | Orientation |
|---|---:|---|---|
| B13 | 13 | stage A/B bank (bank11 + realized_z + realized_drv) | as stage B |
| B20 | 20 | Step 438 bank20 (bank11 + 4 CT7 streams, 4 q15_H1 shape features, evidence_drop_risk) | as Step 438 |
| B32 | 32 | Step 439 B11_IND_lf (bank11 + 21 entropy-independent channels) | label-free sign of Step 439 |
| B51 | 51 | the 52-channel step pool minus one constant channel | native (arbitrary) |

## Results (PRMBench within-answer AUC, 6,030 eligible answers; PRMScore; ProcessBench SLA macro 8 cells)

| Arm | B13 | B20 | B32 | B51 |
|---|---:|---:|---:|---:|
| Average of all channels | 0.7749 | 0.7532 | 0.7440 | 0.7425 |
| **DS filter, then average** | 0.7802 | 0.7540 | 0.7475 | 0.7553 |
| Simple filter (corr. with mean of others < 0), then average | 0.7803 | 0.7557 | 0.7323 | 0.7450 |
| DS filter, binary-mark partition, equal group weights | 0.7797 | 0.7472 | 0.7445 | 0.7444 |
| DS filter, continuous L-SML partition, equal group weights | 0.7748 | 0.7419 | 0.7340 | 0.7297 |
| L-SML on all channels | 0.7765 | 0.7599 | 0.6722 | 0.7216 |
| DS filter, then L-SML | 0.7751 | 0.7556 | 0.7185 | 0.7499 |
| PRMScore: all / DS filter | 0.6527 / 0.6565 | 0.6353 / 0.6358 | 0.6230 / 0.6286 | 0.6269 / 0.6378 |
| ProcessBench SLA: all / DS filter | 0.3698 / 0.3750 | 0.3612 / 0.3629 | 0.3515 / 0.3475 | 0.3442 / 0.3505 |

References: fam421 0.7801 / 0.6573 / 0.3980; CT7 0.7724 / 0.6458 / 0.3989; step index alone 0.6617.

## Answer, part by part

### 1. The DS filter's SELECTION is not tailored to 13 channels (holds on every bank)
- In every bank and fold it never dropped a channel with true balanced accuracy >= 0.52, and it dropped every channel
  below 0.475. B20: exactly the 2 reversed channels. B32: the 4 strongly reversed ones always, a borderline one
  (0.48) in 4/5 folds. B51: 8 channels, 7 reversed plus one borderline at 0.513-0.520 (the < 0.52 margin is 0.00013 in
  one fold).
- Channels near the 0.5 cut are unstable (the random tie key changes one B32 decision); the cut has no margin.
- Estimated prevalence is 0.28-0.30 against a true 0.14 in every bank: the estimates are biased, but the keep/drop
  decisions for clearly reversed channels are right.
- The simple correlation filter misfires on larger banks. On B32 it catches 1 of 5 reversed channels (the weakest),
  misses the 4 strongest, and drops a good channel (single-channel AUC 0.640) and a near-chance one; on B51 it misses
  all reversed channels. (The red team refuted the mechanism I first proposed - that the level family dominates the
  mean; the simple filter's decisions follow the non-level majority. Its statistic is informative in general,
  Spearman 0.70-0.79 with single-channel AUC, but misfires near its zero threshold.)

### 2. The filter's GAIN in score depends on the bank
| Bank | DS filter - all | Interval | Position (whole-answer same-length label swap) | Robustness |
|---|---|---|---|---|
| B20 | +0.0008 | adj. [-0.0018, 0.0033] | swap mean +0.0032 | no effect (position-adjusted +0.0001) |
| B32 | +0.0035 | adj. [0.0006, 0.0063]; PRMScore +0.0056 | swap mean -0.0116: not positional; position-adjusted +0.0095 | concentrated: 86% from 61 answers, +0.0005 without them, 5%-trimmed +0.0026 |
| B51 | +0.0128 | 95% [0.0103, 0.0152]; PRMScore +0.0109 | swap mean +0.0089: about 70% positional, content about +0.0039; position-adjusted +0.0058 | broad: 29% from top 1%, 5%-trimmed +0.0112 |
- DS vs the simple filter: +0.0152 on B32 and +0.0103 on B51 (about 46% of the latter positional), but -0.0018 on B20
  (adj. [-0.0033, -0.0003]; only 1,299 answers differ).
- ProcessBench: mixed (B32 macro -0.40 points, B51 +0.63 points, 2 of 4 datasets positive).

### 3. Grouping does not transfer
- The partitions are no longer identical across folds (B20 binary: 7 groups in four folds, 4 in one; B51 continuous: 4
  distinct partitions); on B13 they were.
- Equal weights per discovered group are BELOW plain averaging after the filter: B20 -0.0068 (robust), B32 -0.0030
  (fragile), B51 -0.0109 (robust); B13 a tie. Grouping acts purely as a reweighting that moves weight away from the
  strongest channels.
- Omri's intuition that binary-mark clustering is better than continuous holds at equal group weights on every bank
  (+0.0050, +0.0052, +0.0105, +0.0147; adjusted intervals above 0 on B20 and B32) - but both lose to not grouping.

### 4. L-SML does not transfer
- It collapses on B32 (0.6722): 4 channels get the wrong sign, including the reversed energy_innovation and top50_js,
  and most weight goes to near-chance groups. The DS filter removes the two wrong-signed reversed channels and lifts it
  to 0.7185, still 0.0290 below plain averaging (mostly positional). B51: 0.7216 -> 0.7499, still below averaging.

## Frozen interpretation rules (PROTOCOL.json)
| Rule | Verdict |
|---|---|
| filter_generalizes (both B20 and B32: no good channel dropped, and adjusted gain > 0) | **not established**: selection holds on both; the gain is established on B32 only (fragile), absent on B20 |
| DS_beats_simple_filter | **not established**: much better on B32 (and B51, secondary), slightly worse on B20 |
| binary_grouping_helps | **no** |
| binary_beats_continuous_partition | **yes** (B20 and B32 adjusted intervals above 0) |
| lsml_beats_averaging | **no** |

Predictions: P1 held; P2 not held strictly (one reversed channel kept in 2 of 5 folds); P3 held; P4 not held (B32);
P5 not held (B20 point above); P6 held; P7 held.

## Bottom line for Omri's concern
- **Not tailored:** the label-free DS filter's choice of channels - it finds the reversed channels on 20, 32 and 51
  channels without any change and never removed a good one. Adding channels, however, did not help: every larger bank
  scores below B13.
- **Fragile / does not transfer:** the gain in score (none on B20, small and concentrated on B32, mostly positional on
  B51), grouping (unstable partitions, below plain averaging) and L-SML (collapses without the filter, below averaging
  with it).
- **Idea, not tested:** on B51 the filter drops strongly reversed channels that would score up to 0.61 when flipped;
  flipping instead of dropping is a separate variant for a separate discussion.
