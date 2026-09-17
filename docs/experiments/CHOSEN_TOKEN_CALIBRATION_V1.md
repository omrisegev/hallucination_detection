# Chosen-token calibration v1 (Step 415 [Claude], 2026-09-17)

Development only, frozen 13,769 answers. Module `spectral_utils/chosen_token_calibration.py` (with a
synthetic self-test), extraction `scripts/run_chosen_token_calibration_v1.py`, analysis
`scripts/analyze_chosen_token_calibration_v1.py`, results `results/chosen_token_calibration_v1/RESULTS.json`.

## Question

Omri: can the actually chosen token be quantified by a statistic that does not depend on the
distribution or the entropy? The existing chosen-token channels sit at .56-.58 conditional
correlation with the entropy family, because in a flat distribution every token is surprising.

## Construction

Null hypothesis per token: the provided token was drawn from the model's own top-50 distribution q.
Statistics whose null distribution does not depend on q: PIT from the top (`pit_mid`, mean 1/2 for
every q; `pit_normal` is its Gaussian score), excess surprisal `-log q(x) - H(q)` (mean 0), and
standardized excess `(-log q(x) - H(q)) / sqrt(VE(q) + .01)` (mean 0, variance 1 except in the
near-deterministic band where the floor deflates it). Tokens outside the top 50 are censored below
the smallest listed mass (64,766 of 6,968,779 tokens, 0.9%).

Synthetic check (tokens sampled from their own distribution, entropy spread over two decades):
correlation with entropy is .887 for raw surprisal and .002 / .000 / .001 for PIT / excess /
standardized excess.

## Result on the real data

Token-level correlation with top-50 entropy, per answer, token-weighted mean:

| statistic | corr with entropy |
|---|---|
| raw surprisal | .337 |
| PIT (mid) | .219 |
| PIT (normal score) | .227 |
| excess surprisal | .116 |
| **standardized excess surprisal** | **-.016** |

PIT and plain excess keep part of the entropy dependence on real text: the official answers are not
samples from the scoring model, and the departure from the null itself varies with entropy.
Standardized excess is entropy-free on real data as well.

Step level (Top10), labelled PRMB steps, conditional on the label (labels for measurement only):

| statistic | family AUC | cond. corr with distribution families | with provided_token | with position |
|---|---|---|---|---|
| PIT normal | .676 | .44 to .65 | .96 | -.18 |
| excess surprisal | .658 | .35 to .47 | .96 | -.28 |
| **standardized excess** | .579 | **-.00 to .11** | .60 | **-.43** |

Effective independent signals, conditional, 12 within-answer families (answer length removed, see the
correction below): 2.46; with PIT normal 2.51; with excess 2.62; **with standardized excess 2.78**.

Matched fusion under the Step 413 contract, 10,000-draw paired source-group bootstrap:

| contrast | PB pp [95%] | within [95%] |
|---|---|---|
| 20 streams + PIT, equal, minus 20 equal | +0.01 [-0.41, +0.45] | -.0003 [-.0011, +.0004] |
| 20 + PIT + standardized excess, minus 20 equal | -0.73 [-1.44, -0.01] | -.0038 [-.0048, -.0028] |
| leader six + PIT, equal, minus six equal | +0.48 [-0.24, +1.19] | **+.0035 [+.0021, +.0049]** |
| 20 + PIT: Continuous L-SML minus equal | +0.10 [-0.29, +0.48] | -.0005 [-.0015, +.0006] |
| 20 + PIT: IU-PCR minus equal | +0.31 [-0.37, +0.97] | +.0008 [-.0004, +.0020] |

Six + PIT reaches 40.75 PB / .7623 (frozen BOCPD-corrected innovation5 40.37 / .7632). Standardized
excess alone: 18.77 PB / .6495; PIT alone 31.46 / .7192; H1 alone 36.35 / .7301.

## Reading

1. An entropy-independent chosen-token statistic exists and is verified on real data: the
   standardized excess surprisal. Its conditional correlation with every distribution family is
   0 to .11, so it satisfies the conditional-independence assumption L-SML needs.
2. It is weak (AUC .579) and has its own confound: -.43 with step position. Added to the bank it
   raises the effective independent signal count by the largest margin measured (2.46 to 2.78) but
   lowers fused quality (-0.73pp): at equal weight a weak independent view dilutes the strong blob,
   and no fitted rule recovers the loss.
3. The Top10 readout is part of the problem. A Top10 mean of n null variables grows with n, which
   reintroduces a step-length dependence into a statistic built to have none; a sum/sqrt(n) step
   readout would keep the null N(0,1) at every entropy and every length. Not tested yet.
4. PIT gives a small within-answer gain on the leader six (+.0035, interval excludes zero) but not a
   PB gain, and it is not entropy-free on real text.

## Correction to Step 414

Step 414 reported 2.83 conditional independent signals over 13 families. That count included answer
length in steps as raw (not answer-standardized) values: it is constant within an answer, carries no
within-answer information, and inflated the count. Without it, the 12 within-answer families give
**2.46**.

## Step 416 [Claude]: length-free step readouts

Omri chose to fix the readout before opening a new measurement channel. The Top10 mean of n null
variables grows with n, so it reintroduces step length into a statistic built to be free of it.
Replacement readouts, all from per-step sufficient statistics
(`scripts/run_chosen_token_calibration_steps_v2.py`, `scripts/analyze_chosen_token_step_tests_v2.py`,
`results/chosen_token_calibration_v1/STEP_TESTS_V2.json`):

* `z_std_excess` = sum of per-token standardized excess / sqrt(n)
* `z_pit` = sum of PIT normal scores / sqrt(n)
* `z_pooled` = sum(-log q(x) - H) / sqrt(sum VE + n * .01), the one-sample test "this step is more
  surprising than the model itself expected", with the variance pooled over the step
* `z_pooled_pos` = `z_pooled` with the step-position trend removed; the slope is fitted on the four
  training folds only, so the removal is label-free and out of fold

Synthetic null, steps of 4 / 16 / 64 tokens: `z_pooled` mean -.10 / .05 / -.02 and sd .92 / 1.02 / .97
(small negative bias at 4 tokens from the skew of surprisal); the Top10 mean climbs -.05 / .45 / 1.57.
`z_pit` drifts upward with length (-.02 / .15 / .25) because the mid-PIT normal score is not centred for
discrete distributions; it is reported but not primary. The self-test asserts both facts.

Real data (labels for measurement only on PRMB steps; step length and position correlations over all
steps; "dist. families" excludes the provided-token family, which is built from the same token):

| readout | AUC | corr log step length | corr position | max cond. corr, dist. families | eff. signals with it |
|---|---|---|---|---|---|
| Top10 standardized excess | .579 | .171 | -.490 | .11 | 2.78 |
| Top10 PIT normal | .676 | .409 | -.264 | .65 | 2.51 |
| z standardized excess | .564 | -.039 | -.482 | .06 | 2.79 |
| z PIT | .655 | -.026 | -.240 | .48 | 2.62 |
| z pooled | .618 | .046 | -.430 | .24 | 2.74 |
| **z pooled, position removed** | **.662** | .046 | **-.047** | .26 | 2.74 |

Matched contrasts, 10,000-draw paired source-group bootstrap, * = interval excludes zero:

| contrast | PB pp [95%] | within [95%] |
|---|---|---|
| z pooled alone minus Top10 std. excess alone | +0.70 [+0.08, +1.29]* | +.0034 [+.0000, +.0067]* |
| z pooled pos. alone minus z pooled alone | +0.15 [-0.85, +1.15] | +.0360 [+.0319, +.0401]* |
| six + Top10 std. excess, equal, minus six | -1.31 [-2.24, -0.38]* | -.0065 [-.0080, -.0050]* |
| six + z pooled, equal, minus six | -0.40 [-1.29, +0.48] | +.0012 [-.0003, +.0028] |
| **six + z pooled pos., equal, minus six** | -0.86 [-1.79, +0.06] | **+.0072 [+.0056, +.0089]*** |
| six + z PIT, equal, minus six | +0.39 [-0.43, +1.22] | +.0031 [+.0015, +.0048]* |
| 20 + z pooled pos., equal, minus 20 | -0.33 [-0.87, +0.19] | +.0012 [+.0004, +.0020]* |
| six + z pooled pos.: Continuous L-SML minus equal | -0.69 [-1.30, -0.07]* | +.0011 [+.0000, +.0022]* |
| six + z pooled pos.: IU-PCR minus equal | -3.12 [-4.74, -1.53]* | -.0482 [-.0523, -.0443]* |
| 20 + z pooled pos.: Continuous L-SML minus equal | +0.19 [-0.27, +0.68] | -.0005 [-.0016, +.0006] |

Levels: six + z pooled pos. at equal weight 39.41 PB / **.7661** within; with Continuous L-SML
38.72 / **.7672**, the highest within-answer AUROC in this line (frozen BOCPD-corrected innovation5
40.37 / .7632; six at equal weight 40.27 / .7589).

### Reading

1. The readout fix works on the statistic. The z readouts remove the step-length dependence (.17 to
   .05, and .41 to -.03 for PIT). Removing the position trend without labels lifts the step AUC of
   the pooled test from .618 to .662 and the within-answer AUC of the view alone by +.036.
2. Pooling the variance buys power at a price in independence: the maximum conditional correlation
   with a distribution family rises from .06 (per-token standardized) to .24-.26. It is still the most
   independent view with usable strength that we have.
3. Added to the leader six it gives the largest within-answer gain measured today (+.0072, interval
   excludes zero) and the first within-answer level above the frozen leader (.7661 and .7672 with
   L-SML). On ProcessBench it costs -0.86pp with an interval that includes zero. The two benchmarks
   disagree, so this is not an improvement on both and nothing is promoted.
4. L-SML is now slightly better than equal on within (+.0011*) and worse on PB (-0.69*). IU-PCR fails
   badly on the seven-view set. The fused-weight question is still not answered in L-SML's favour.
5. A plausible reason for the split, not tested: ProcessBench scores the single peak and its misses
   are late-biased, while the new view has its position trend removed; it may pull peaks earlier.
   A peak-displacement diagnostic on the ProcessBench errors would test this directly.
