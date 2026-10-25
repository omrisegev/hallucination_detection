# Telemetry class-conditional correlation on PRMBench (diagnostic, 2026-10-25)

**This is a label-using diagnostic, not a method. Nothing here feeds any fusion weights.**

Question (from the Ising-model label-aggregation paper, Balasubramanian et al., arXiv 2601.22336): is the
correlation among our 16 telemetry channels (+ PRM reward) different on erroneous vs correct steps, and does a
quadratic rule find detection signal that a linear rule misses?

Short answer: **yes, the correlation structure differs clearly and robustly (all 5 PRMs; errors have stronger
positive inter-channel correlation), but the extra detection signal it carries is small**: logistic regression
with cross products beats linear+squares by +0.0014 to +0.0104 within-answer AUC, while a supervised *linear* LR
already beats the label-free L-SML score by +0.0075 to +0.0338. QDA is worse than LDA on every PRM.

Run: `.venv/bin/python results/telemetry_class_correlation_v1/class_correlation.py` (CPU, 288 s total, ~58 s per
PRM; seed 20261025). No model inference. All alignment asserts passed (`row_ids`/`offsets` of every channels file
equal `.ids`/`.off`; finite channels).

## Population (identical for all 5 PRMs)

| | answers | steps | error steps | correct steps |
|---|---|---|---|---|
| used (non-control, >= 3 steps) | 6,209 | 83,368 | 13,147 | 70,221 |
| excluded: control answers | 758 | | | |
| excluded: < 3 steps | 2 | 3 | 2 | |

6,033 of the 6,209 answers have at least one error step (176 non-control answers have no in-range error step).
Part B within-answer AUC uses the 6,029 test answers that have both classes.
Zero within-answer std (z set to 0), answer x channel cells: `dominant_freq16` 94-215 per PRM, `top15_turnover`
0-24, `prm_reward` 25 (7B) / 47 (72B), three digit channels 1 each (RLHFlow). See `RUN_META.json`.

## Part A: correlation difference (answer-z values)

`frob` = Frobenius norm of off-diagonal R_err - R_ok (both triangles). Permutation null = labels shuffled within
answer (error count preserved), 1000 draws; min p = 0.001. CI = 1000-draw cluster bootstrap over source groups.
`mean r` = mean off-diagonal correlation, raw channel orientation.

| PRM | view | frob [95% CI] | perm null mean / q95 | p | mean r err | mean r ok | diff [95% CI] | p |
|---|---|---|---|---|---|---|---|---|
| Qwen-7B | N16 | 1.02 [0.96, 1.10] | 0.39 / 0.44 | .001 | .245 | .221 | +.024 [.017, .031] | .001 |
| Qwen-7B | N17 | 1.11 [1.05, 1.20] | 0.43 / 0.48 | .001 | .192 | .174 | +.018 [.012, .023] | .001 |
| Qwen-72B | N16 | 0.94 [0.89, 1.02] | 0.40 / 0.45 | .001 | .252 | .236 | +.015 [.009, .022] | .001 |
| Qwen-72B | N17 | 1.01 [0.96, 1.09] | 0.44 / 0.49 | .001 | .202 | .193 | +.009 [.003, .015] | .006 |
| PURE | N16 | 1.11 [1.05, 1.21] | 0.31 / 0.35 | .001 | .278 | .236 | +.042 [.035, .050] | .001 |
| PURE | N17 | 1.15 [1.08, 1.25] | 0.34 / 0.38 | .001 | .221 | .188 | +.033 [.027, .040] | .001 |
| RLHFlow | N16 | 1.76 [1.66, 1.89] | 0.44 / 0.50 | .001 | .269 | .204 | +.065 [.058, .072] | .001 |
| RLHFlow | N17 | 1.89 [1.80, 2.00] | 0.50 / 0.55 | .001 | .218 | .164 | +.055 [.048, .060] | .001 |
| Math-Shepherd | N16 | 1.37 [1.26, 1.49] | 0.30 / 0.35 | .001 | .269 | .233 | +.036 [.028, .044] | .001 |
| Math-Shepherd | N17 | 1.39 [1.29, 1.52] | 0.35 / 0.40 | .001 | .230 | .199 | +.030 [.024, .037] | .001 |

Note: `frob` is biased upward by sampling noise; the permutation-null mean (0.30-0.50) is the noise floor, so read
the excess over it, not the CI's distance from 0.

Top differing pairs (r_err vs r_ok; N16, all steps; full list in `CLASS_CORRELATION.csv` / `TOP_PAIRS.csv`):
- Qwen-7B: energy_level~realized_z +.13 vs -.08; bocpd_p0~realized_z +.22 vs +.02; energy_innovation~top15_turnover +.27 vs +.45
- Qwen-72B: top15_turnover~realized_z +.15 vs -.05; q15_H1~realized_z +.29 vs +.11; chosen_surprisal~realized_z +.57 vs +.43
- PURE: q15_VE1~realized_z +.34 vs +.13; q15_H1~realized_z +.46 vs +.26; top15_turnover~realized_z +.18 vs -.01
- RLHFlow: true_tail50~realized_drv +.42 vs .00; energy_level~realized_drv +.51 vs +.12; chosen_surprisal~energy_level +.73 vs +.45
- Math-Shepherd: energy_level~realized_drv +.41 vs +.12; true_tail50~realized_drv +.32 vs +.05; q15_H1~realized_z +.21 vs -.03
- N17 adds for Qwen-7B: realized_z~prm_reward -.14 vs +.04.

Pattern: the `realized_*` channels (realized-token z / drift) couple much more strongly with the entropy / energy /
tail channels on error steps than on correct steps, in every PRM backbone.

### Position-tercile check (relative position (k+0.5)/n in [0,1/3), [1/3,2/3), [2/3,1]; z kept from the full answer)

| PRM | view | tercile 1: n_err, frob (null q95), p | tercile 2 | tercile 3 |
|---|---|---|---|---|
| Qwen-7B | N16 | 1579, 1.19 (1.21), .070 | 5412, 1.01 (0.62), .001 | 6156, 1.33 (0.66), .001 |
| Qwen-72B | N16 | 1579, 1.26 (1.27), .067 | 1.22 (0.69), .001 | 1.14 (0.59), .001 |
| PURE | N16 | 1.49 (1.10), .001 | 1.27 (0.76), .001 | 1.21 (0.56), .001 |
| RLHFlow | N16 | 2.01 (1.11), .001 | 1.16 (0.62), .001 | 1.47 (0.65), .001 |
| Math-Shepherd | N16 | 1.55 (0.89), .001 | 1.19 (0.50), .001 | 1.24 (0.60), .001 |

N17 is the same pattern (Qwen-7B tercile 1 p = .28, Qwen-72B tercile 1 p = .35, all else <= .001). The difference
survives position stratification: p <= .001 in 26 of 30 PRM x view x tercile cells; the four exceptions (N16 and N17) are the
first tercile of the two Qwen PRMs, where only 1,579 error steps exist and the noise floor (~1.1) is close to the
observed value. The mean-r difference is not uniform across terciles (often near 0 or negative in tercile 1,
largest in tercile 3), so part of the "agreement strength" gap is tied to late steps.

### Design check and two extra analyses (labelled EXTRA)

Flaw found: answer-z standardization plus a pure mean shift (errors higher on informative channels) already makes
R_err != R_ok, because the class means of the z values vary across answers with the answer's error fraction, and
that common variation correlates the informative channels differently in the two classes. The within-answer
permutation null destroys the mean shift too, so it tests "labels independent of features", not "same covariance
given a mean shift". Two controls:

1. **EXTRA_class_centred**: subtract each answer x class mean before the correlation (removes every per-answer
   mean shift; drops a class's rows in answers where it has one step: 11,184 err / 70,185 ok steps kept), 200
   permutations. frob stays essentially the same (N16: 1.19, 1.21, 1.10, 1.62, 1.34 for 7B, 72B, PURE, RLHFlow,
   MS; null q95 0.39-0.54; p = .005, the minimum for 200 draws). The top pairs are the same.
2. **Equal-covariance Gaussian simulation** with the observed answer structure, labels and per-view mean shift,
   through the same answer-z (50 reps): frob 0.14-0.17 (max 0.23). A pure mean shift produces far less than the
   observed 0.94-1.89.

So the difference is a genuine second-order (dependence) difference, not an artifact of answer-z.

Orientation note: raw mean off-diagonal r depends on each channel's arbitrary sign. CSV also has `mean_abs_*` and
`mean_ori_*` (each view oriented toward error by the label-free sign of its pooled correlation with -reward). The
oriented gap is +.012 to +.056 on all steps, but after class-centring it is ~0 or negative for the two Qwen PRMs
(-.002, -.014) and smaller for the others (+.016 to +.040). So the "errors agree more" reading in the
Curie-Weiss sense is weak and PRM-dependent; the robust finding is the pair-specific pattern (frob).

## Part B: supervised headroom (answer-z N17, 5-fold GroupKFold by source group; within-answer AUC, fold mean +- SD)

| PRM | LDA | QDA | LR linear | LR linear+squares (EXTRA) | LR pairwise | L-SML CHAIN_N16 | -reward |
|---|---|---|---|---|---|---|---|
| Qwen-7B | .8439 +- .0074 | .8364 +- .0058 | .8452 +- .0065 | .8504 +- .0066 | .8537 +- .0077 | .8377 +- .0061 | .8011 |
| Qwen-72B | .8563 +- .0074 | .8500 +- .0025 | .8589 +- .0076 | .8617 +- .0061 | .8632 +- .0062 | .8428 +- .0040 | .7948 |
| PURE | .8335 +- .0084 | .8226 +- .0043 | .8338 +- .0079 | .8402 +- .0095 | .8450 +- .0080 | .8155 +- .0077 | .7628 |
| RLHFlow | .8064 +- .0061 | .8029 +- .0042 | .8086 +- .0056 | .8124 +- .0055 | .8228 +- .0044 | .7748 +- .0071 | .6623 |
| Math-Shepherd | .8075 +- .0093 | .8040 +- .0084 | .8091 +- .0104 | .8133 +- .0098 | .8163 +- .0088 | .7783 +- .0062 | .6844 |

Pooled AUROC is in `QUADRATIC_HEADROOM.csv` (same ordering).

Paired per-fold differences (mean, folds > 0 of 5):

| PRM | QDA - LDA | pairwise - linear | pairwise - squares (cross products only) | LR linear - L-SML |
|---|---|---|---|---|
| Qwen-7B | -.0074 (0/5) | +.0085 (5/5) | +.0034 (4/5) | +.0075 (5/5) |
| Qwen-72B | -.0062 (1/5) | +.0042 (5/5) | +.0014 (4/5) | +.0162 (5/5) |
| PURE | -.0109 (0/5) | +.0111 (5/5) | +.0048 (5/5) | +.0183 (5/5) |
| RLHFlow | -.0035 (2/5) | +.0142 (5/5) | +.0104 (5/5) | +.0338 (5/5) |
| Math-Shepherd | -.0035 (1/5) | +.0072 (5/5) | +.0030 (4/5) | +.0308 (5/5) |

L-SML orientation: in `evaluate_collection.py` every arm is scored as `auc(err, score)` and flagged by top-k of
its answer-z, so higher = error; CHAIN_N16 was used as-is (its within-AUC here, .8377 for Qwen-7B, equals the
evaluator's `within(c, 'CHAIN_N16')`). The PRM reference is `-reward`.

Reading: the class-dependent correlations carry real but small supervised signal (cross products +0.001 to +0.010
within-AUC, largest for RLHFlow, which also has the largest frob). QDA fails to use it: its reg_param was the grid
maximum 0.1 in 23 of 25 folds (grid edge), and the channels are heavy-tailed / partly discrete, so the Gaussian
quadratic model costs more than the dependence gains. The headroom of a supervised linear rule over label-free
L-SML is 2-10x larger than the cross-product headroom.

## Caveats

- Development data (PRMBench already inspected); labels used for every statistic. Supervised numbers are
  headroom references, not candidates.
- Steps inside an answer are not independent; the source-group bootstrap and within-answer permutation account for
  clustering, the Gaussian simulation does not (iid rows), so its 0.15 floor is optimistic.
- Not checked: step length (`log_step_tokens`) and PRMBench category as confounds of the correlation difference;
  error steps may be longer, and longer steps give less noisy channel values (stronger correlations). This is the
  main untested alternative explanation for "errors correlate more".
- QDA grid {0, .01, .1} as specified; the choice at the edge means a larger reg_param might approach LDA, not beat it.
- LR uses C = 1, no class weighting; within-answer AUC is rank-based within answers.
- Channels include the three digit features (kept in this line per the 2026-09-30 decision).

Files: `class_correlation.py`, `CLASS_CORRELATION.csv` (Part A, scopes all / EXTRA_class_centred /
pos_tercile_1..3), `QUADRATIC_HEADROOM.csv`, `TOP_PAIRS.csv`, `RUN_META.json` (N, zero-std counts, orientation
signs, runtime), `run.log`.
