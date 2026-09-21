# Stage 1b — the locator kill test. Result: **the locator branch ends.**

Claude, 2026-09-21. Branch `claude/whitebox-layer-views-v1`.
Population: all **13,769** answers / **145,597** steps / 9 cells / **10,477** erroneous answers /
**3,483** source groups. Metric: Mind-the-Gap native Step-level Localization Accuracy,
tolerance 0, **gate-free** — erroneous answers only, within-answer argmax, no no-error decision
anywhere. 10,000 shared paired source-group draws. Development evidence.

Input: `STEP_LEVEL.npz`, 486 MB, sha256 `f25c47f7…`, produced by cluster job 263494 from code
commit `463cd9b41` (clean tree), 296.6 s. Landed after a byte-exact sha256 match.

---

## 1. The pre-registered kill rule, and what it returned

> If no single depth summary from the eligible candidate set beats `final_lens_H` alone on
> per-cell gate-free SLA — with a paired interval excluding zero **and** clearing the
> best-of-N permutation null — the locator branch ends.

**Depth beats `final_lens_H` in 0 of 9 cells. The rule fires.**

Candidate set: 864 columns → 2 constant (`KL(final‖final)` at the last layer) and 16
residual-tap layer-34/35 columns laned out (they restate `final_lens_H` rather than measuring
depth) → **848 eligible**, of which **424** sit on the declared risk orientation.

## 2. Per-cell gate-free SLA

| cell | chance | random_step | step_length | **final_lens_H** | depth_contract | **ct7_locator** |
|---|---:|---:|---:|---:|---:|---:|
| pb_gsm8k_q4 | 20.84 | 23.19 | 41.06 | 45.89 | 39.13 | **48.31** |
| pb_gsm8k_q8 | 20.84 | 21.26 | 41.06 | 41.06 | 39.13 | **47.34** |
| pb_math_q4 | 18.10 | 15.82 | 33.16 | 31.99 | 34.18 | **35.69** |
| pb_math_q8 | 18.10 | 20.88 | 33.16 | 31.65 | 32.83 | **36.87** |
| pb_olympiadbench_q4 | 13.55 | 12.56 | 24.81 | 27.99 | 24.81 | **39.03** |
| pb_olympiadbench_q8 | 13.55 | 14.52 | 24.81 | 29.20 | 27.99 | **37.67** |
| pb_omnimath_q4 | 13.84 | 14.89 | 28.06 | 30.30 | 28.46 | **37.15** |
| pb_omnimath_q8 | 13.84 | 14.10 | 28.06 | 30.83 | 25.96 | **37.02** |
| prmbench_qwen3_8b | 10.14 | 10.42 | 18.13 | 32.38 | 18.57 | **46.40** |
| **mean over 8 PB cells** | 16.58 | 17.15 | 31.78 | **33.61** | 31.56 | **39.89** |

Arms in words:

| arm | what it is |
|---|---|
| `random_step` | a random step. The floor. |
| `step_length` | pick the longest step. One integer, no model involvement. |
| `final_lens_H` | **the kill incumbent** — final-layer lens entropy (top-15 renormalised), Top-10 readout. |
| `depth_contract` | the single best depth column on the **label-free** orientation rule: `lens_logp_tgt.resid.layer_21.lo`, i.e. *a low lens log-probability of the emitted token at residual layer 21 means risk*. |
| `ct7_locator` | **the production locator.** Reproduces its documented 39.89 anchor exactly. |

## 3. The kill contrast

`depth_contract` − `final_lens_H`, 10,000 paired source-group draws:

| cell | delta | 95% interval | verdict |
|---|---:|---|---|
| pb_gsm8k_q4 | −0.068 | [−0.146, +0.010] | tie |
| pb_gsm8k_q8 | −0.019 | [−0.093, +0.054] | tie |
| pb_math_q4 | +0.022 | [−0.027, +0.071] | tie |
| pb_math_q8 | +0.012 | [−0.032, +0.056] | tie |
| pb_olympiadbench_q4 | −0.032 | [−0.072, +0.010] | tie |
| pb_olympiadbench_q8 | −0.012 | [−0.056, +0.031] | tie |
| pb_omnimath_q4 | −0.018 | [−0.061, +0.025] | tie |
| pb_omnimath_q8 | −0.049 | [−0.088, −0.010] | **final_lens_H wins** |
| prmbench_qwen3_8b | −0.138 | [−0.154, −0.123] | **final_lens_H wins** |

Nothing wins anywhere. Two cells lose outright.

## 4. The signal is real — and still not a better locator

This is the finding that keeps the result honest in both directions. The label-selected
**oracle** — best of 848 eligible columns per cell — clears its own permutation null (the
first-error index reassigned uniformly within each answer, 200 times, the same best-of-848
recomputed) in **9 of 9 cells**, by +0.089 to +0.222. So the depth field does carry genuine
step-level error information.

It is simply not *more* information than the final layer already carries:

| | mean over 8 PB cells |
|---|---:|
| `final_lens_H` (one fixed statistic, label-free) | 33.61 |
| depth **oracle** (best of 848, per cell, **with labels**) | **35.46** |
| `ct7_locator` (production, label-free) | **39.89** |

An oracle allowed to pick the best of 848 depth columns per cell, after seeing the answers,
beats a single final-layer statistic by **1.85 points** and still loses to the production
locator by **4.43**. Per cell the ceiling's margin over `final_lens_H` runs +0.48, +2.90, +2.19,
+2.36, +7.41, +2.72, +0.00, **−3.29** — it is negative on one cell outright.

**This confirms the project's strongest negative prior on new data.** Steps 243–245b found
per-layer lens fusion at 0.7253 against 0.7298 for the final-layer statistic alone, on
final-answer detection. Different task, different population, same direction, and now with a
pre-registered rule rather than a post-hoc read.

## 5. The length control — the most uncomfortable number here

Every arm re-scored after within-answer rank-residualisation against step length:

| arm | raw (8 PB) | length-residualised | lost to length |
|---|---:|---:|---:|
| `final_lens_H` | 33.61 | 16.79 | **−16.82** |
| `depth_contract` | 31.56 | 16.99 | −14.57 |
| `ct7_locator` | 39.89 | 20.09 | −19.80 |
| *(chance)* | 16.58 | | |

Residualised, `final_lens_H` and `depth_contract` sit **at chance**; only `ct7_locator` retains
a margin (+3.5). Read carefully: a rank difference is a crude control, not a regression
residual, and it injects distortion of its own, so the absolute residualised values are an upper
bound on the damage rather than a clean estimate. What survives that caveat is the **ordering** —
CT7 keeps signal beyond step length, and the two single-statistic arms largely do not.

This is worth its own follow-up regardless of the white-box line, because it bears on every
step-level readout in the project.

## 6. The four project axes

| axis | what Stage 1b returned |
|---|---|
| **1. A significant feature** | **No.** The best label-free depth column (31.56) falls below both `final_lens_H` (33.61) and `step_length` (31.78). The oracle ceiling reaches 35.46 and still loses to CT7 by 4.43. |
| **2. Number of views** | Useful columns are scattered across all three taps and depths 0–28 (`lens_kl_final.attn.28`, `lens_logp_top1.mlp.17`, `lens_logp_tgt.resid.19`, `lens_kl_final.resid.00`…) — no depth band concentrates the signal. `view_pr` not computed; Stage 3 is now moot for the locator. |
| **3. Conditional independence of errors** | Not measured here — a branch that loses to its incumbent on the primary endpoint does not earn a complementarity analysis. (Contrast Stage 1a, where a *tie* did earn one.) |
| **4. Aggregation / fusion order** | Not applicable — single columns, no fusion, by design of the kill test. |

## 7. Verdict

**The locator branch ends, as pre-registered.** The per-token depth field does not supply a
step-level locator competitive with the final-layer statistic it is derived from, let alone with
the production locator.

What this does **not** kill: Stage 1a's gate result stands unchanged — a tie against the
production gate with genuinely complementary errors (failure phi 0.402, 630 answers rescued).
The two sub-channels were split precisely so that this could happen, and it did.

**What the white-box line is now:** one open experiment, the answer-gate fusion, judged on exact
localizations gained versus lost at a matched opened fraction. Not a locator.

## 8. Artefacts

`KILL_TEST_1B.json` · `STEP_LEVEL.npz` (486 MB, sha256 `f25c47f7…`, job 263494, commit
`463cd9b41`) · `MANIFEST_STEP.json`.
Producers: `cluster/reduce_layer_views_step_level.py`, `scripts/run_whitebox_kill_test_1b.py`.

Reduction integrity, from the manifest: 13,769 answers / 145,597 steps / 864 columns, per-cell
counts exact, `identity_check_max` 0.0 with `axis_control_min` 1.283 (the module-axis assertion
is non-vacuous and passes), token coverage 0.952–1.000 median 1.000, and 3 rows / 10 steps /
3 identical span pairs flagged as overlapping — matching an independent scan of the raw capture
exactly.
