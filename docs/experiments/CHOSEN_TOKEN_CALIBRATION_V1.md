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
