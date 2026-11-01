# HANDOFF: re-analysing D1-D3 without the raw data

This file is for the other computer. Everything below can be recomputed from git alone, except where it says "PC only".

## Files

| Path | In git | Contents |
|---|---|---|
| `diag_d1_d3.py` | yes | The script. It needs the raw data and loaders on the PC, but it is readable as the exact definition of every number |
| `D1.csv` | yes | One row per (cell, statistic). Answer level |
| `D2.csv` | yes | One row per (cell, score). Score = `fused`, or `group` with a group index |
| `D3.csv` | yes | One row per (cell, feature). Feature = fused, groupK, or channel name |
| `bundle/<arm>_<mode>_<bench>_<prm>.npz` | yes (force-added, 42 MB total, 68 files) | Per-cell arrays, below |
| `bundle/MANIFEST.json` | yes | Cells, n_answers, n_steps, K |
| `bundle/channels_<mode>_<bench>_<prm>.npz` | **no, PC only** (41 MB; with the cell bundles this would exceed the ~50 MB git budget) | Z (steps × 16, float32, cleaned raw N16 channels, NaN where undefined), names, offsets |
| `../raw_null_gate_v1b/{fix,fix_mono}/labelfree/*.npz` | **no, PC only** (43 + 38 MB) | The v1b per-step / per-answer P(err) arrays. The cell bundles already carry the per-answer statistics from both variants |

**Cell names:** arm ∈ {AUTO, TELE16, SUB5}; mode ∈ {prm (backbone), sa (standalone)}; bench ∈ {processbench,
crv_arith, crv_gsm8k, prmbench}; prm ∈ {qwen_prm_7b, qwen_prm_72b, pure_prm, rlhflow_prm}.

## Cell bundle fields

Let N be the number of steps, A the number of answers, and K the number of channel groups.

| Key | Shape, dtype | Meaning |
|---|---|---|
| `offsets` | (A + 1,) int64 | Answer a holds steps offsets[a]:offsets[a+1] |
| `ans_evaluated` | (A,) bool | The answers scored by v1 (`ev`). On PRMBench, class-`correct` answers are not evaluated |
| `ans_has_err_LABEL` | (A,) bool | **Label.** The answer has a wrong kept step |
| `ans_class_LABEL` | (A,) str or empty | **Label.** PRMBench class, if the loader has it |
| `ans_G1_raw_at_worst` | (A,) f32 | raw_fused at the worst step (the answer-z argmax) |
| `ans_raw_fused_answer_max` | (A,) f32 | Maximum of raw_fused within the answer |
| `ans_G1_p_worst`, `ans_G1_p_worst_mono` | (A,) f32 | G1 P(err) of the worst step (v1b fix / fix_mono) |
| `ans_G1b_1_minus_p` | (A,) f32 | 1 − G1b answer p (higher = more error) |
| `ans_G2_p_worst`, `ans_G2_p_worst_mono` | (A,) f32 | G2 maximum step P(err); NaN if G2 is invalid |
| `ans_G3_posterior`, `ans_G3_posterior_mono` | (A,) f32 | G3 P(answer has an error); NaN if G3 is invalid |
| `ans_group{g}_answer_max` | (A,) f32 | Maximum of u_g over the answer's defined steps |
| `step_raw_fused` | (N,) f32 | Pooled-z fused raw score (G1's input), oriented so that higher = error |
| `step_z_fused` | (N,) f32 | Answer-z fused score (GAP's score) |
| `step_U` | (N, K) f32 | u_g = o_g × pooled z of the group score over its defined steps; NaN where undefined |
| `step_defined` | (N, K) bool | Defined steps per group |
| `step_position` | (N,) f32 | Relative position in the answer (`ds_gate_b.relative_position`) |
| `step_err_LABEL` | (N,) bool | **Label.** The step is wrong |
| `step_keep_eval` | (N,) bool | Labelled steps: c.keep, or keep_gold on ProcessBench (steps after the first error are False) |
| `step_mask` | (N,) bool | keep_eval AND the step's answer is evaluated (v1 `stepmask`) |
| `worst_step` | (A,) int64 | Global index of each answer's worst step |
| `fits_json` | 0-d str | JSON with `g1` (G1 null: ok, reason, pi0, pi0_raw, mu0, sigma0, sd), `groups` (the same per group), `orientation` (o_g), `group_channels` (channel indices per group), `channel_names`, `channel_sign_in_fused`, `sign`, `cross_weights`, `group_weights` |

**Loader:**

```python
import json, numpy as np
B = np.load('results/raw_null_gate_diag_v1/bundle/TELE16_sa_processbench_qwen_prm_7b.npz', allow_pickle=False)
fits = json.loads(str(B['fits_json']))
off, ev, y = B['offsets'], B['ans_evaluated'], B['ans_has_err_LABEL']
from sklearn.metrics import roc_auc_score
s = B['ans_raw_fused_answer_max'][ev]
print(roc_auc_score(y[ev], s))          # should equal D1.csv auc_DIAG for this cell / statistic
```

## How each table was computed (exact)

**D1** (answer level, over evaluated answers ev; label y = has_err[ev]):
- `auc_DIAG`: sklearn `roc_auc_score(y, s)` over answers with a finite s. Ties are averaged. Higher s means more
  error, so AUC < .5 means the ranking is inverted. It is NaN if one class is empty or s is constant.
- `n_answers`: the number of finite s values. The oracle uses the same population.
- `oracle_closed_clean_at_ke85_DIAG`:
  - Sort s over the erroneous answers, ascending, as e.
  - t = e[floor(0.15 · len(e))]. Then at least 85% of erroneous answers have s ≥ t, and t is the largest threshold
    with that property.
  - The result is the share of clean answers with s < t.
  - **This is an oracle:** it places t with labels.

**D2** (step level; one row for the fused score, one per group with o_g ≠ 0):
- **Pool:** the steps the null was fitted on. That is all steps for fused, and the defined steps for a group.
- **Labelled steps:** lab = pool ∧ keep_eval; correct = lab ∧ ¬err.
- **True correct-step distribution:** `true_mean`, `true_sd` (ddof = 0), `true_skew`, and `true_exkurt` (scipy;
  Fisher, so 0 for a Gaussian), all over the correct steps.
- **True shares:**
  - `true_share_correct` = |correct| / |lab|;
  - `true_share_correct_evaluated_only` uses `step_mask` instead of keep_eval.
- **Error shares:**
  - `err_share_labelled` = the mean of err over lab;
  - `err_share_central_band` = the mean of err over lab ∧ q25 ≤ x ≤ q75, with q25 and q75 taken over the whole pool
    (no labels).
- **Fitted null:** `mu0`, `sigma0`, `pi0` and `pi0_raw` come from the v1b null (identical to v1's).
- **Comparisons:**
  - `mu0_minus_true_mean_in_true_sd` = (μ₀ − true_mean) / true_sd;
  - `sigma0_over_true_sd` = σ₀ / true_sd;
  - `pi0_raw_minus_true_share`.
- **Pool coverage:** `n_pool_unlabelled` and `share_pool_labelled`.

**D3** (step level, over step_mask ∧ finite x; label err):
- `step_auc_DIAG`: `roc_auc_score(err, x)`.
- `cohen_d_DIAG`: (mean_err − mean_cor) / pooled sd, with ddof = 1.
- **Features:**
  - raw_fused;
  - each u_g (already oriented);
  - each cleaned raw channel, **not oriented**.
- `fused_sign`: the sign of the channel inside the fused score, sign × cw_g × w_j. Multiply d by it, and use 1 − AUC
  for a negative sign, to orient a channel as the fused score does.

## Labels

**Label-using columns:** only the columns ending in `_DIAG` and the D2 `true_*`, `err_share_*` and `*_minus_true*` /
`*_over_true*` columns use labels. Everything else is label-free and identical to the v1b `fix` gates.

**Recomputation check:** in every cell, the script asserted that its recomputed raw_fused and z_fused (max |Δ| ≤ 1e-9)
and G1 and G2 per-step P(err) (atol 1e-9, rtol 0, NaN equal) match the saved v1b `fix` arrays.

## Judgment calls to check

1. **D2 "labelled" set:** it is keep_eval, without the evaluated-answer filter. The PRMBench class-`correct` answers
   sit in the null pool, so they count as labelled correct steps. The evaluated-only share is also given.
2. **ProcessBench steps after the first error are unlabelled** (keep_gold = False) but are in the null pool. D2's true
   distribution excludes them, and `share_pool_labelled` shows how many there are (median .64).
3. **The D2 central band** is the .25-.75 quantiles of the score over the whole pool, the same band that central
   matching uses.
4. **The oracle closed_clean** uses s ≥ t as "error" and keeps at least .85 of errors, the largest such t (ties
   included). A reviewer checked this against a brute-force search on 299 tie-heavy synthetic cases.
5. **Group answer maxima** ignore undefined steps. In these banks all steps are defined (digit_active is 1 everywhere),
   so this changes nothing.
6. **Channels in D3 are the cleaned raw channels** (pooled z with the positional cubic removed, as `votes_e1b.clean`
   produces). For AUTO, the PRM column is the cleaned PRM risk.
7. **Medians in the README** are taken over cells (arms × modes × PRMs) per benchmark. The arms share telemetry, so
   these cells are not independent.
8. **The "best single group" in D1** is the maximum over groups within a cell. It is a selection, so it is optimistic.
