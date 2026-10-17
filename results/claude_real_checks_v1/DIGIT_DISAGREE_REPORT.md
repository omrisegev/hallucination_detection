# Digit-disagreement view: full-benchmark development evaluation (Claude, 2026-09-15)

**Development result on the fully exposed 13,769-answer benchmark. Not untouched confirmation.**

## Definition (label-free, zero fitted parameters, same one-pass teacher-forced telemetry)

Token stream `d_t = 1` if the provided token is a single digit (Qwen3 tokenizer ids 15..24) **and** the scorer's
top-1 token is a *different* digit; else 0. Step auxiliary = per-step Top10 mean of `d_t` (the bank's readout).
Candidate = `innovation5 + gamma * std(innovation5) * z(aux)` per answer (`residual_step_score`), gamma=.25 declared
primary before evaluation, gamma=1 secondary. Frozen tail15 gate (q=.33), frozen Top10, same evaluator, same folds.
Scripts: `claude_new_view_ceiling.py` (candidate screening on error cohorts), `claude_digit_disagree_eval.py` (this evaluation).

## Results

| method | PB % | within-AUC | PRMScore (default calibration) |
|---|---:|---:|---:|
| original4 | 37.4749 | 0.753436 | 0.634412 |
| innovation5 (reference) | 39.8314 | 0.760293 | 0.638830 |
| digit aux standalone (diagnostic) | 35.5156 | 0.639385 | 0.136231 |
| **innovation5 + digit, gamma=.25 (primary)** | **41.3300** | **0.776036** | **0.649780** |
| innovation5 + digit, gamma=1 (secondary) | 41.1806 | 0.779274 | 0.649223 |

Paired source-group bootstrap, 10,000 draws:

| contrast | PB delta (pp) | PB CI | within delta | within CI | level |
|---|---:|---|---:|---|---|
| gamma=.25 minus innovation5 | +1.499 | [+0.247, +2.792] | +0.0157 | [+0.0134, +0.0181] | 97.5% |
| gamma=1 minus innovation5 | +1.349 | [−0.212, +2.940] | +0.0190 | [+0.0148, +0.0233] | 95% |
| standalone minus innovation5 | −4.316 | [−6.454, −2.191] | −0.121 | [−0.129, −0.112] | 95% |

First candidate in this research line that improves **both** endpoints with intervals excluding zero.

PB exact hits vs innovation5 (gamma=.25): 1,339 vs 1,268; gained 310, lost 239. By first-error position: early +91/−130,
middle +122/−84, **late +97/−25** (the bucket where innovation had lost).

## Why it works (from the cohort screening, `NEW_VIEW_CEILING.json`)

| stream | corr with fused innovation5 (within answer) | rank-1 rate on the 707 telemetry-silent misses | rank-1 on caught |
|---|---:|---:|---:|
| any q15-shape stream | 1.0 (by construction) | 0.0% | 38–43% |
| provided-token surprisal / gap / rank | 0.22–0.30 | 4–6% | 20–27% |
| tail15 / tail50 mass | −0.24 | 4% | 16% |
| **digit disagreement** | **−0.005** | **20.5%** | 32% |

Participation ratio of the bank rises from 1.34 (five q15 streams) to 3.55 with the new streams. The generic
provided-token streams (Step 315's family) are weak; the restriction to numerals, the semantic locus of math errors,
is what makes the disagreement informative. This is a *different expert* (scorer vs. provided text), not another
summary of the same distribution, which is exactly what the ensemble/U-PCR framing requires.

## Gate-side diagnostic (descriptive only; gate unchanged)

Answer-level disagreement count separates clean from erroneous PB answers: AUC 0.69–0.77 per cell; e.g. GSM8K-q8:
81% of clean answers have zero disagreements vs 32% of erroneous. A candidate second gate feature; not evaluated as a gate here.

## Caveats

- Candidate screening used labelled cohorts (7 candidate streams inspected); gamma was declared before evaluation.
  Development data throughout; the benchmark is fully exposed. Confirmation requires untouched questions (MR-GSM8K
  is the registered external candidate) and the nested PRMScore calibration.
- Math-specific (digits). For self-generated answers, `d_t` means "sampled a digit the model did not rank first";
  its meaning under greedy decoding differs. Historical24 transfer must treat it as a separate view.
- Access is unchanged: one teacher-forced pass, gray-box, no labels, plus ten tokenizer constants.
