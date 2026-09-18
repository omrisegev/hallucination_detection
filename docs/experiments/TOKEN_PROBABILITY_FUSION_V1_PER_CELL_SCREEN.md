# Per-cell screen — there is structure, but it is not where the deficit is

Runs `scripts/diagnostics/per_cell_covariance_screen_v1.py`; output
`results/token_probability_fusion_v1/PER_CELL_COVARIANCE_SCREEN.json`.
Amendment 2 of the Stage B pre-registration. Development-only.

## What was compared, and why that object

Per-cell adaptation means fitting the covariance and L-SML separately per cell,
label-free, using only cell identity. **L-SML never sees labels**, so what such a fit
would fit is the per-cell *marginal* correlation of the views. That is the object
compared here — not a conditional matrix — and the choice is the point, not a
compromise.

Distance is the mean absolute difference over the 55 off-diagonal entries of the 11×11
correlation matrix, in correlation units. Between-cell distances are judged against a
within-cell baseline: each cell split in half **by source group**, never by answer, so
the same question cannot land on both sides.

## Result

| cell | split-half baseline | distance to pooled |
|---|---|---|
| GSM8K 4B / 8B | 0.0432 / 0.0382 | 0.1006 / 0.1056 |
| MATH 4B / 8B | 0.0167 / 0.0176 | 0.0838 / 0.0844 |
| OlympiadBench 4B / 8B | 0.0131 / 0.0114 | 0.0962 / 0.0897 |
| Omni-MATH 4B / 8B | 0.0196 / 0.0216 | 0.0873 / 0.0890 |
| PRMBench | 0.0079 | **0.0401** |

| comparison | n pairs | median distance | vs baseline |
|---|---|---|---|
| ProcessBench vs ProcessBench | 28 | 0.0565 | **3.04×** |
| ProcessBench vs PRMBench | 8 | 0.1278 | ~7× |
| all pairs | 36 | 0.0669 | 3.81× |

## Reading — the headline ratio is the least interesting part

**By the screen's own rule the idea is not closed:** 3.04× within ProcessBench is above
sampling noise, so a per-cell fit does have something different to fit. But three
details matter more than the ratio, and two of them argue the variant will not repair
what we care about.

**1. The structure is misaligned with the deficit.** The pair with the *smallest*
between-cell distance in all of ProcessBench is **MATH vs OlympiadBench, 0.0234 and
0.0236** — barely above their 0.013–0.018 baselines. Those two cells have almost the
same covariance to fit, and yet the token arm scores 34.85 / 31.99 on MATH against
30.56 / 31.01 on OlympiadBench, and it is OlympiadBench where we lose to Chen et al. by
12.5 pp. Meanwhile the most *distinctive* cells are GSM8K (baseline 0.038–0.043, the
largest ProcessBench distances at 0.070–0.094 against Omni-MATH) — and GSM8K is where
we are already strongest. **Per-cell covariance fitting has the most to work with
exactly where we need it least, and the least where we need it most.**

**2. PRMBench is the real outlier, and it owns the pooled fit.** Its distance to pooled
is 0.0401 against 0.084–0.106 for every ProcessBench cell — that is, the pooled
correlation matrix is much closer to PRMBench than to any cell it is used to score.
With 37% of the tokens and the opposite shape (13.5 steps per answer, 27 tokens per
step, against ProcessBench's 5–9 steps and 55–93 tokens), PRMBench is not a larger
ProcessBench, and it drags the shared standardizer into a regime no ProcessBench cell
occupies. **This is a live finding independent of per-cell fitting**, and it is a
cheaper thing to act on: simply excluding PRMBench from the ProcessBench fit is a
one-line change with the same access contract.

**3. Model identity barely matters.** The same questions scored by 4B and 8B differ by
0.040–0.045, at the GSM8K baseline and only ~2.5× the baseline elsewhere — much less
than the differences between subsets. Cell identity is mostly *subset* identity.

## Decision

Do not build the per-cell variant yet. The screen permits it, but the structure it
would exploit is concentrated in the cells that do not need help, and the single
largest effect it surfaced — the pooled standardizer being dominated by PRMBench — is
addressable directly and should be tested first.
