# Stage 1a — the gate kill test. Result: **not killed, not cleared. A tie with complementary errors.**

Claude, 2026-09-19. Branch `claude/whitebox-layer-views-v1`.
Plan: `docs/experiments/WHITEBOX_LAYER_VIEWS_IMPLEMENTATION_PLAN.md`.
Population: all **13,769** answers / 9 cells / **10,477** positives / **3,483** source groups.
Statistics: `weighted_auc` (exactly tie-corrected, sklearn-verified to 1e-13), 10,000 **shared**
paired source-group bootstrap draws. Development evidence only.

---

## 1. What was measured

One binary answer-level question — *does this answer contain an error* — by AUROC, per cell.
Arms, in words:

| arm | what it is |
|---|---|
| **`n_steps`** | the number of reasoning steps in the answer. One integer, no model involvement. |
| **`locator_max`** | the largest CT7 per-step risk inside the answer. |
| **`ct7_gate`** | **the production gate.** CT7's frozen no-error decision (a tail-15 entropy statistic at the .33 within-cell percentile). |
| **`geometry_equal`** | equal-weight mean of all 283 z-scored rotation-invariant white-box geometry columns, oriented by the registered contract. Label-free. |
| **`geometry_oracle`** | the single best geometry column **per cell**, chosen with labels, orientation granted. A **ceiling, never a candidate**. |
| **`geom_L32`** | one **fixed** column across all nine cells: `hidden_dist_adjacent.layer_32` — how far the answer's mean hidden state moves between layers 31 and 32, as a normalized distance. |

## 2. Per-cell AUROC

| cell | n_steps | locator_max | **ct7_gate** | geometry_equal | geom_L32 | geometry_oracle |
|---|---:|---:|---:|---:|---:|---:|
| pb_gsm8k_q4 | 0.531 | 0.583 | 0.667 | 0.585 | **0.717** | 0.745 |
| pb_gsm8k_q8 | 0.531 | 0.581 | 0.657 | 0.614 | **0.689** | 0.701 |
| pb_math_q4 | 0.565 | 0.422 | **0.716** | 0.608 | 0.708 | 0.768 |
| pb_math_q8 | 0.565 | 0.400 | 0.709 | 0.647 | **0.725** | 0.747 |
| pb_olympiadbench_q4 | 0.523 | 0.408 | 0.701 | 0.527 | **0.702** | 0.710 |
| pb_olympiadbench_q8 | 0.523 | 0.392 | 0.710 | 0.560 | **0.734** | 0.734 |
| pb_omnimath_q4 | 0.596 | 0.371 | 0.758 | 0.508 | **0.764** | 0.777 |
| pb_omnimath_q8 | 0.596 | 0.353 | 0.764 | 0.529 | **0.776** | 0.776 |
| prmbench_qwen3_8b | 0.472 | 0.528 | *0.500 (constant)* | 0.565 | **0.660** | 0.684 |
| **mean over the 8 PB cells** | 0.554 | 0.439 | **0.710** | 0.572 | **0.727** | 0.745 |
| POOLED | 0.618 | 0.523 | 0.531 | 0.560 | — | **0.461** |

## 3. Four findings, in order of how much they change the picture

### 3.1 The signal is real — it is not selection noise

`geometry_oracle` is the max of 283 label-selected AUROCs, so it is biased upward by
construction. Measured against **its own** null (labels permuted within cell, 200 times, the
same best-of-283-with-orientation recomputed):

| cell | oracle | null p95 | excess |
|---|---:|---:|---:|
| pb_gsm8k_q4 | 0.745 | 0.595 | **+0.150** |
| pb_math_q4 | 0.768 | 0.561 | **+0.207** |
| pb_omnimath_q4 | 0.777 | 0.567 | **+0.210** |
| prmbench_qwen3_8b | 0.684 | 0.535 | **+0.149** |
| *(all nine cells)* | | | **+0.107 to +0.210** |

Above the null in **9/9** cells. The white-box geometry carries genuine answer-level error
signal. The strong form of H0 — *a noisy reconstruction of the final layer* — is **rejected for
this sub-channel**.

### 3.2 Against the real incumbent it is a **tie**, not a win

`geom_L32` minus `ct7_gate`, 10,000 paired draws:

| cell | delta | 95% interval | verdict |
|---|---:|---|---|
| pb_gsm8k_q4 | +0.050 | [−0.003, +0.101] | tie |
| pb_gsm8k_q8 | +0.032 | [−0.024, +0.086] | tie |
| pb_math_q4 | −0.008 | [−0.040, +0.025] | tie |
| pb_math_q8 | +0.015 | [−0.018, +0.048] | tie |
| pb_olympiadbench_q4 | +0.001 | [−0.035, +0.037] | tie |
| pb_olympiadbench_q8 | +0.024 | [−0.008, +0.056] | tie |
| pb_omnimath_q4 | +0.006 | [−0.026, +0.038] | tie |
| pb_omnimath_q8 | +0.012 | [−0.019, +0.043] | tie |
| **prmbench_qwen3_8b** | **+0.160** | **[+0.150, +0.170]** | **geometry wins** |

**8/8 ProcessBench intervals include zero.** The +0.017 mean PB edge is not a measured win.

The PRMBench win is real but must be stated precisely: `ct7_gate` opens **0 of 6,969** PRMBench
answers — it is constant, so its 0.500 is undefined-by-construction. The honest claim is *the
geometry supplies a working answer-level detector on a cell where the production gate supplies
none*, not that it beat a functioning gate.

### 3.3 **The orientation is label-derived. This is the load-bearing caveat.**

The registered contract declares `distances_and_convergence: +risk`. **Empirically this feature
is −risk at answer level**: with the contract's own orientation `geom_L32` scores **0.273**, and
the 0.727 above depends on a sign flip chosen by looking at labels.

So `geom_L32` is an **oracle-oriented** number, not a label-free candidate. It cannot be
promoted until a label-free orientation rule determines that sign. This is the
`max(auc, 1−auc)` trap the project has already hit three times, and it was one step away from
being reported as a clean win here.

Two things follow. The contract's declared direction is **wrong for this feature on this task** —
worth fixing upstream. And a larger hidden-state step between adjacent late layers predicts a
*correct* answer, which is the opposite of the intuition the contract encodes.

### 3.4 Equal weighting is a bad reader, and that is the arm's weakness — not the channel's

`geometry_equal` (0.572 mean PB) loses to `ct7_gate` on 7/8 cells by up to −0.25, while one
fixed column reaches 0.727 and the per-cell oracle 0.745. Averaging 283 columns dilutes the
signal rather than combining it. Consistent with Step 413: **selection is the gap, not
weighting.**

Top fixed columns are stable, not knife-edge — the top five all sit at 0.721–0.727 and span
three families (`hidden_dist_adjacent`, `hidden_dist_to_final`, `resid_norm_convergence`) at
layers 1, 2, 3, 15 and 32.

---

## 4. The four project axes

| axis | what Stage 1a returned |
|---|---|
| **1. A significant feature** | **Yes.** One fixed column matches a tuned production gate (0.727 vs 0.710 mean PB, intervals overlapping) and is +0.11–0.21 above its selection null in 9/9 cells. Strongest single-feature result this white-box line has produced — with the orientation caveat in §3.3. |
| **2. Number of views** | Depth-decay: layers 0–33 are near-orthogonal to the final-layer entropy (median corr −0.29…+0.11); only layers 34 (0.557) and 35 (0.998) carry it. The curve is non-monotone (dips at ~14, deepens at ~23). Useful columns span three families and many depths. `view_pr` not yet computed — that is Stage 3. |
| **3. Conditional independence of errors** | **The most valuable number here.** Failure phi vs `ct7_gate` at matched opened fraction = **0.402** (range .359–.466). Project scale: .981 near-duplicate · .665 logtail15/base · **.402 ours** · .103 digit/base. Substantially more complementary than the redundant entropy family. **630 answers that the production gate gets wrong, the geometry gets right** (against 1,048 the other way). |
| **4. Aggregation / fusion order** | Not applicable — single columns, no fusion. |

---

## 5. Verdict and what it licenses

**The gate branch is neither killed nor cleared.** Against the rule as written (`beat n_steps`
**and** `locator_max`), `geometry_oracle` passes 9/9 — but I named the wrong incumbents:
`locator_max` is *below chance* on 5/8 cells (0.35–0.42), so that half of the bar was under the
floor. CT7's gate is a tail-15 entropy statistic, **not** a threshold on the locator composite;
that is the token arm's LOCO-5 gate, and I conflated them when writing the rule.

Against the right incumbent the result is a tie on ProcessBench plus a genuine win where the
incumbent is undefined.

**What is licensed:** a fusion experiment at the answer-gate insertion point. A tie in aggregate
with failure phi 0.402 and 630 complementary rescues is exactly the "useful diversity" criterion
the fusion insertion map states for this insertion — *one view catches wrong-but-confident
answers another misses*. That is the next bounded stage, and it needs the gate's real endpoint
(exact localizations gained vs lost at a matched opened fraction), not AUROC.

**What is not licensed:** calling this a white-box win. It is a tie, on a label-derived
orientation, on development data.

**Blocking before any candidate:** a label-free orientation rule. Without it there is no
candidate, only a ceiling.

---

## 6. Artefacts

`KILL_TEST_1A.json` · `KILL_TEST_1A_FOLLOWUP.json` · `FIXED_COLUMN_VS_CT7.json` ·
`AXIS3_ERROR_COMPLEMENTARITY.json` · `ANSWER_LEVEL.npz` (305 MB, sha256 `909ac8b6…`) ·
`MANIFEST.json`.
Producers: `scripts/run_whitebox_kill_test_1a.py`,
`scripts/run_whitebox_kill_test_1a_followup.py`,
`cluster/reduce_layer_views_answer_level.py`, `scripts/land_layer_views_answer_level.py`.

**Stage 1b (the locator branch) has not run.** It needs a step-level reduction of the
token-resolved lens field — a second cluster job — and its kill incumbent is `final_lens_H`.
