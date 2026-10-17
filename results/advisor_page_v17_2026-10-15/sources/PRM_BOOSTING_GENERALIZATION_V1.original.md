# PRM boosting generalization v1: implementation spec for Codex

Requested by Omri on 2026-10-13. Claude wrote the plan; Codex implements the code. Claude runs the
jobs on AIRCC and does the analysis. Base branch: `claude/prm-boosting-v1`. Work branch:
`claude/prm-boosting-generalization-v1`.

## 0. What has already been shown (Qwen2.5-Math-PRM-7B, PRMBench)

Source: `results/prm_boosting_v1/README.md`, `SMALL_BANK.csv`, `COMPOSITIONS.csv`.

**Telemetry source.** One forward of the PRM returns both outputs:
- the step rewards;
- token-probability telemetry, computed by applying the checkpoint's own `lm_head` to the fine-tuned
  backbone's last hidden state.

PRM-7B never uses its `lm_head`, which is bit-identical to Qwen2.5-Math-7B-Instruct's. The bank is
built from that telemetry with the frozen per-answer feature code.

**Results.** Within-answer AUC / official PRMScore:

| Method | Within-AUC | PRMScore |
|---|---|---|
| PRM reward alone | .801 | 65.46 |
| N16 from the PRM backbone, label-free | .813 | 67.20 |
| ½ z(PRM) + ½ z(N16 backbone) | .841 | 68.53 |
| ½ z(PRM) + ½ mean of 3 channels, out of fold | .851 | 69.38 |

**The 3 channels.** They were picked with labels by greedy selection inside the 5 source-group folds.
All 5 folds chose the same 3, in the same order:
- `energy_level`
- `hist_entropy_cusum_abs_series`
- `realized_drv`

**The question for this round.** Does this generalize to other PRMs? The channels and the recipe are
**frozen** here and are not re-selected.

## 1. Frozen recipe (do not tune on the new models)

For each PRM `M`, on PRMBench (the same 6,969 rows, read from
`results/prmbench_qwen3_8b_telemetry_thinkfix/prmbench_telemetry.pkl`, the same as in v1):

**PRM risk.** `prm_risk = logit(1 - v)`, where `v` is M's step validity probability in [0, 1],
clipped to [1e-6, 1 - 1e-6]. The model-specific mapping to `v` is in section 2.

**Frozen small bank `S3`.**
- `score = 0.5 * az(prm_risk) + 0.5 * mean(az(energy_level), az(hist_entropy_cusum_abs_series), az(realized_drv))`.
- `az` = per-answer z-score, as computed by `Population.cell_bank` in
  `results/monday_external_panel_v1/score.py`.
- Every channel is oriented +1 (higher = riskier), fixed from v1. Do not re-orient.
- The channels come from **M's own backbone telemetry**.

**Arms to score.** All arms use the deployed `ANSWER_Z_Q80` rule unless marked otherwise.

| Arm | Definition |
|---|---|
| `PRM_native` | M's official PRMBench decision rule (section 2) |
| `PRM_Q80` | `prm_risk` under the same rule as our methods |
| `N16_own` | MEAN_MOM_ga on N16 from M's backbone telemetry (label-free, unchanged `fuse.py` path) |
| `EQ2_own` | ½ z(PRM) + ½ z(N16_own) |
| `S3_own` | the frozen small bank above (primary) |
| `S3_q3` | the same 3 channels taken from the Qwen3-8B mask1 bank instead of M (a cross-model control) |
| `S6_own` | the v1 all-data k=6 set, as a secondary check. The two Qwen3 channels in it are read from the Qwen3-8B mask1 bank |

The v1 all-data k=6 set: `bb:energy_level, bb:hist_entropy_cusum_abs_series, bb:realized_drv,
q3:energy_level, q3:hist_energy_cusum_abs_series, bb:hist_topk_varentropy_series`.

**Metrics and uncertainty.**
- Within-answer AUC (v3 labels).
- Official PRMScore via `spectral_utils.prmbench.prmbench_evaluate`, both total and per classification.
- Paired source-group bootstrap: 2,000 draws, seed 20261013, the same code path as
  `results/prm_boosting_v1/small_bank_bootstrap.py`.
- Contrasts: `S3_own − PRM_native`, `S3_own − PRM_Q80`, `S3_own − N16_own`, and `EQ2_own − PRM_native`.

**Gates before any fused number counts.**
- **G1 reward parity.** `PRM_native` PRMScore within 1.0 of the leaderboard Overall (table in section 2).
  If it misses, stop and report. Do not fuse a misreproduced PRM.
- **G2 telemetry sanity.** Median token surprisal is below 1 nat, and there are 0 step/text mismatches
  (see `n_step_text_mismatch` in the existing driver manifest).
- **G3 bank parity.** `build_bank.py --expect-equal` on the Qwen3 cell still gives 0.0. This is a
  regression check, since build_bank must stay unchanged.

## 2. Models (in run order) and the official scoring for each

The official wrappers are in `github.com/ssmisya/PRMBench`, `mr_eval/models/*.py`. Port each scoring
path **verbatim** (input template, separator, score position, threshold). PRMBench `steps` are passed
through `remove_step_prefix` (in `mr_eval/utils/model_utils.py`) where the wrapper does that.

| # | Leaderboard name (Overall) | HF id (verify) | Official wrapper | Reward → v | Native rule |
|---|---|---|---|---|---|
| 1 | Qwen2.5-Math-PRM-72B (68.2) | `Qwen/Qwen2.5-Math-PRM-72B` | `qwen_prm.py` (same as the 7B: `<extra_0>`, 2-way head) | softmax[:,1] | v > 0.5 |
| 2 | Skywork-PRM-7B (65.1) | `Skywork/Skywork-o1-Open-PRM-Qwen-2.5-7B` | `skywork_prm.py`; needs `skywork-o1-prm-inference` (`PRM_MODEL`, `prepare_input`, `derive_step_rewards`); steps rendered as `"Step {i}: {step} ки\n"` | step reward (probability) | first difference of rewards > −0.05 (first step: the reward itself) |
| 3 | RLHFlow-PRM-Mistral-8B (54.4) | `RLHFlow/Llama3.1-8B-PRM-Mistral-Data` | `llama3_1_8b_prm.py`: multi-turn chat, user = step (question prefixed to step 1), assistant = "+"; score at pattern position + 3 | softmax(logits)[token 10 = "+"] at the score position (full-vocab softmax, as official) | v > 0.5 |
| 4 | Pure-PRM-7B (65.3) | `jinachris/PURE-PRM-7B` (the wrapper default is `jinachris/Qwen2.5-Math-7B-PRM800K`; confirm which one the leaderboard used) | `pure_prm.py`: `AutoModelForTokenClassification`, question + steps each followed by `\n\n`, score at each `\n\n` | r = p1 − p0, v = (r + 1)/2 | r > 0 |
| 5 | Skywork-PRM-1.5B (61.1) | `Skywork/Skywork-o1-Open-PRM-Qwen-2.5-1.5B` | as #2 | as #2 | as #2 |
| 6 | MathShepherd-Mistral-7B (47.0) | `peiyi9979/math-shepherd-mistral-7b-prm` | `math_shepherd.py`: `"{question} Step i: {step} ки\n"`, candidate tokens "+"/"-" at the `ки` positions | softmax over [+,−][0] | v > 0.5 |
| 7 | ReasonEval-7B (60.1), optional | `GAIR/ReasonEval-7B` | `reasoneval.py` (custom `score_head`; checkpoint has **no lm_head**) | per the wrapper | per the wrapper |

Omri's priority for the first round: #1, #2 and #3, which cover a stronger PRM, a different training
recipe in the same family, and a different family whose LM head is actually trained. #4 to #6 follow
if time allows. #7 is only possible with a borrowed base-model head; list it as blocked unless that
head is available.

## 3. Head inventory: where the token probabilities come from (Omri's requirement)

The telemetry must come from **the PRM's own backbone hidden states in the same forward that produces
the reward**, not from a standalone base model. For each model, record in the manifest:
- `lm_head_source`: `own_used` (RLHFlow and MathShepherd are causal LMs whose head is trained and used),
  `own_unused` (Qwen PRM: `lm_head.weight` is in the checkpoint but the class ignores it), or
  `borrowed_base:<hf id>` (when the checkpoint has no head, e.g. a value-head or token-classification
  model; take the base model's `lm_head.weight`, or the tied embedding if `tie_word_embeddings`, and
  apply it to the PRM's final hidden state after the final norm).
- `lm_head_vs_base_max_abs_diff` against the stated base model.
- Whether the hidden state fed to the head is after the final norm. It must be the same tensor the
  base LM head would see.

Expected per model (verify; do not trust this table):
- Qwen-72B: `own_unused`.
- Skywork: check `PRM_MODEL` (likely a value head on Qwen2.5-Math; the base `lm_head` may be present).
- RLHFlow and MathShepherd: `own_used`.
- PURE: likely `borrowed_base:Qwen/Qwen2.5-Math-7B`.
- ReasonEval: `borrowed_base` (WizardMath/Llemma); hence optional.

Scored positions: the **step tokens only**. Separator/tag tokens (`<extra_0>`, `ки`, `\n\n`) and the
injected assistant "+" turns are not scored. The logit row for step token t comes from position t − 1
of the same forward, as in `run_prmbench_prm_telemetry.py`. Where the template injects text between
steps ("Step i:", the RLHFlow assistant turns), that text conditions the next step but is excluded
from the per-step token spans. Record this.

## 4. Code to write (Codex)

1. **`spectral_utils/prm_adapters.py`**: one class per model with a common interface:
   - `build(tok, question, steps) -> dict(input_ids, step_token_positions: list[list[int]], score_positions: list[int])`.
     `step_token_positions[k]` are the absolute positions of step k's text tokens, and
     `len(score_positions) == len(steps)`.
   - `rewards(model_outputs, batch) -> list[float]`: the official reward (v-scale per section 2).
   - `native_labels(rewards) -> list[int]`: 1 = valid, the official rule.
   - `vocab_logits(hidden, positions) -> Tensor[T, V]`: implements the head inventory.
   - `load(model_id) -> (model, tok, head_meta)`.

   Reuse `spectral_utils.prm_scorer` for the Qwen family. The current Qwen path in
   `cluster/run_prmbench_prm_telemetry.py --mode prm` must become the `qwen_prm` adapter and stay
   bit-identical (section 5, test T1).
2. **`cluster/run_prmbench_prm_telemetry.py`**: add `--adapter <name>` (keep `--mode prm|instruct`
   working for the Qwen-7B path). Batch size 1 is fine. Keep the same output schema (`gen_token_ids`,
   `step_token_spans` in compressed step-token coordinates, `token_entropies`, `token_spilled_energies`,
   `token_logsumexp`, `top_k_logprobs`, `rewards`, `native_labels`, `n_input_tokens`,
   `n_step_text_mismatch`), the SIGTERM/checkpoint/exit-85 contract and the manifest fields. Add
   `lm_head_source`, `lm_head_vs_base_max_abs_diff`, `adapter`, `official_wrapper_commit`.
   - Long inputs: rows over the model's context are an explicit, counted failure, never silently
     truncated.
   - For the 72B model: `device_map="auto"` on one B200 in bf16. Compute the vocab logits for step
     positions in chunks of 1,024 positions to bound memory.
3. **`cluster/submit_prmbench_prm_telemetry.sbatch`**: unchanged, except that `--time` is chosen per
   model by the caller. The 72B needs a chained resume (see the CLAUDE.md AIRCC rules: linear
   `--dependency=afterany` chain).
4. **Skywork dependency.** Vendor the needed functions from `SkyworkAI/skywork-o1-prm-inference`
   (`model_utils/prm_model.py`, `io_utils.py`) into `third_party/skywork_prm/` with the upstream commit
   hash and license, and make no edits besides imports.
5. **`results/prm_boosting_v1/build_bank.py`**: do not change it. It is model-agnostic (the step count
   is asserted and the token count is not).
6. **`results/prm_boosting_generalization_v1/evaluate.py`** (new). For each model tag:
   - read its FEATURES.npz and its telemetry rewards;
   - compute every arm in section 1 with the frozen definitions, plus the N16 fit via the
     `fuse.py`/`score.py` adapter (MEAN_MOM_ga), and the gates G1 to G3;
   - write `RESULTS_<tag>.csv` (arm × {within_auc, prmscore, per-classification prmscore}),
     `BOOTSTRAP_<tag>.csv`, `GATES_<tag>.json`, and one cross-model summary
     `SUMMARY.csv` (rows = model, columns = arms).

   Generalize `fuse.py` to take the PRM reward source as an argument; it currently hard-codes the
   Step 242 pkl. No new selection or tuning happens anywhere in this file.

## 5. Tests (CPU, before any GPU job; Claude runs `/preflight` before submitting)

- **T1 regression.** The `qwen_prm` adapter reproduces, for the first 40 rows,
  `results/prm_boost_v1/full_prm/prmbench_telemetry.pkl` on AIRCC exactly (rewards, gen_token_ids,
  spans, and telemetry to 1e-5). This needs a GPU, so Claude runs it as the first pilot. Write it as a
  `--compare-to` option of the driver.
- **T2 template unit tests** (tokenizers only, no weights, `tests/test_prm_adapters.py`). For each
  adapter and 3 fixture rows (including one with an empty step and one with a `Step 3:` prefix):
  - the number of score positions equals the number of steps;
  - step token positions decode back to the cleaned step text;
  - no step position overlaps a separator/tag/template position;
  - the official input string is reproduced character-for-character against a snapshot taken from the
    upstream wrapper logic.
- **T3 head inventory.** A unit test for `vocab_logits` on a tiny random config shows that `own_unused`
  and `borrowed_base` give `hidden @ W.T` after the final norm.
- **T4 evaluator.** On the existing v1 outputs, `evaluate.py` reproduces `SMALL_BANK.csv` all-data k=3
  (0.8508 / 69.3846) and `COMPOSITIONS.csv` EQ2(PRM, BB) (0.8405 / 68.5338) for tag `qwen_prm_7b`.

## 6. Run plan (Claude)

For each model: pilot of 40 rows (gates G1 partial, G2, T1 where it applies), then the full 6,969 rows,
then the bank (CPU, the `build_bank.py` + `native_job.sbatch` pattern from v1), then `evaluate.py`.

Large files stay on AIRCC under `results/prm_boost_gen_v1/<tag>/`. Locally only FEATURES.npz
(about 54 MB each, pulled one at a time because C: is short) and the CSVs. Order: 72B, Skywork-7B,
RLHFlow-8B, then #4 to #6.

Estimated GPU time: about 15 min for a 7-8B model; for the 72B, about 2-3 h (chain 2 links of 2 h).

Known AIRCC pitfalls from v1:
- **Never source `pyenv_native_312_20261011/activate.sh` in the shell that runs `sbatch` for a GPU
  job.** Its PYTHONPATH leaks into the container and breaks torch.
- `SLURM_SUBMIT_DIR` is the directory sbatch was called from. The sbatch `cd`s to `$CODE_DIR`.
- The NGC container's pip pulls a pyarrow that is incompatible with numpy 1.26. Read rows from the
  telemetry pkl (`--meta-pkl`) instead of importing `datasets`.
- The mask1 rule module on AIRCC `code_thinkfix_rescore_v1` is the pre-fix version. Use
  `code_prm_boost_v1/mask_fixed`, which keeps already-empty steps empty.

## 7. What not to do

- No re-selection of channels, weights, signs or thresholds on the new models. This is a frozen-
  transfer test. Any new selection is a separate, labelled experiment.
- Do not report a fused number for a model that fails G1.
- The fused scores inherit each PRM's label supervision. Only the fusion step is label-free, and the
  3-channel set was label-selected on Qwen-PRM-7B. Report it that way.
- PRMBench is already-inspected development data. This is a cross-model generalization check, not an
  untouched test. Socratic-PRMBench and Hard2Verify transfer is a possible phase 2 and needs a separate
  plan.
