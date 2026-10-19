# Handoff: Math-Shepherd digit-id fix and constant-column guard (2026-10-19)

Written on the Mac, which has no AIRCC access. Nothing below has been run on AIRCC yet. Run it from the
machine that has AIRCC access, after `/preflight` passes.

## Why

Math-Shepherd is the only PRM where L-SML loses to plain averaging in the five-arm ablation. Two separate causes:

1. **Digit-id bug (fixed in code here).** `external_banks_v4.DIGIT_IDS = 15..24` is '0'..'9' only in the
   Qwen2.5/Qwen3 and Llama-3 vocabularies. Math-Shepherd's backbone is Mistral-7B-v0.1 (SentencePiece), where
   15..24 are byte-fallback tokens. So all three digit channels are exactly 0 on Math-Shepherd
   (`ABL5_DIAG_math_shepherd.json`, `zero_var_channels`). This was already recorded as Amendment A2 in
   `results/prm_boosting_generalization_v1/run_v1/RESULTS_README.md`, but never fixed, and ablation5 ran N16 on
   Math-Shepherd anyway. The zero columns take K slots in L-SML's spectral clustering and change the partition:
   4 telemetry groups become 5 plus a zero group (`SWEEP_DIAG_math_shepherd.json`).
2. **PRM weight (not fixed).** Even in N13 (no digits), L-SML gives the Math-Shepherd PRM a cross share of about
   0.05, against about 0.13 under equal weights, because the PRM barely correlates with its telemetry
   (corr 0.07, versus 0.42-0.48 for the other PRMs). That is a separate research question.

## What changed in code

| File | Change |
|---|---|
| `spectral_utils/digit_token_ids.py` (new) | `resolve_digit_ids(tokenizer.json)` reads the ids of '0'..'9' and fails loudly if a digit is missing or ids mismatch `expected`; `check_digit_channels_vary` raises when a digit channel is constant over a cell. Self-contained (json + numpy) so it can be loaded by path. |
| `spectral_utils/external_banks_v4.py` | `digit_step_features(row, spans, digit_ids=None)`; the default stays 15..24, bit-exact with before. |
| `results/prm_boosting_v1/build_bank.py`, `results/prm_boost_external_v1/build_bank_external.py` | New required `--tokenizer <tokenizer.json>`. Digit channels are recomputed with that tokenizer's ids when they differ from 15..24 (unchanged for Qwen/Llama-3). The build stops if a digit channel is constant. MANIFEST gains `digit_ids` (ids, tokenizer path and sha256). `digit_token_ids.py` is looked up in `$DIGIT_TOKEN_IDS_PY`, next to the builder, `<repo>/spectral_utils/`, or `cwd/spectral_utils/`. |
| `spectral_utils/fusion_utils.py` | `lsml_continuous_guarded`: drops views with sd <= 1e-9, fits L-SML on the rest, reinserts the dropped ones with weight 0 (`c = -1`). With no constant view it is exactly `lsml_continuous`. Raises if fewer than 3 views remain. |
| `results/prm_boost_ablation5_v1/ablation5.py`, `ablation5_socratic.py` | Use the guarded L-SML. A rerun changes only Math-Shepherd N16/S5. The bank sweep script is unchanged (historical record). |

Tests: `tests/test_digit_token_ids.py`, `tests/test_lsml_constant_guard.py` (plus `tests/test_external_banks_v4.py`):
40 pass locally. The local Qwen2.5, Qwen3-8B and Llama-3.1 tokenizers all resolve to 15..24. The Mistral
tokenizer is not on the Mac, so **its real digit ids are not verified yet**. That is step 2 below.

## Steps on the AIRCC machine

Paths as in `run_ablation5.sh` and `ext_down.sh`: `S=/shared/cycle2_tau_averbuch_prj/omrisegev1`,
`R=$S/results`, `O=$R/prm_boost_gen_v1`, `M=$S/eval_mirror_prm_gen_v1`, snapshot `$S/code_thinkfix_bank_553de79`.

1. `git fetch origin claude/prm-boosting && git switch claude/prm-boosting` (or pull), then `/preflight`.
2. **Verify tokenizers.** Find the `tokenizer.json` of each backbone on AIRCC (HF cache of
   `peiyi9979/math-shepherd-mistral-7b-prm` or `mistralai/Mistral-7B-v0.1`, and the four others), then:
   `python3 -c "from spectral_utils.digit_token_ids import resolve_digit_ids as r; print(r('<path>'))"`.
   Expected: Math-Shepherd does NOT give 15..24; Qwen-7B/72B, PURE and RLHFlow give 15..24. Stop if any
   expectation fails.
3. **Ship code without overwriting the old copies.** Copy the two builders into the snapshot as new names,
   with the helper next to them:
   `scripts/prm_boost_build_bank_v2.py`, `scripts/prm_boost_build_bank_external_v2.py`, `scripts/digit_token_ids.py`.
   For the mirror `$M`: diff its `spectral_utils/fusion_utils.py` and `external_banks_v4.py` against this commit
   first. Sync only if the diff is just the new code; otherwise stop and report.
4. **Parity (G3).** Rerun the Qwen3 cell with `prm_boost_build_bank_v2.py --expect-equal --tokenizer <Qwen3-8B tokenizer.json>`.
   It must print `PARITY 0.0 0`, the same as before the change.
5. **Rebuild Math-Shepherd banks (CPU only, no GPU).** The saved telemetry keeps top-50 ids/logprobs, so no
   inference is needed.
   - PRMBench: read `$O/bank_math_shepherd/MANIFEST.json` for the original `raw` and `base`, and build into a
     NEW dir, e.g. `$O/bank_math_shepherd_digitfix` (the builder refuses an existing dir).
   - Socratic: `$R/prm_boost_external_v1/full_math_shepherd_socratic/external_telemetry.pkl` into `bank_v3`.
   - Each MANIFEST must show the Mistral digit ids, and the build must not stop on a constant digit channel.
6. **Rerun the ablation for Math-Shepherd only**, into new output dirs (keep `ablation5_v1` untouched):
   `ablation5.py math_shepherd $O/bank_math_shepherd_digitfix/FEATURES.npz $O/full_math_shepherd/prmbench_telemetry.pkl $O/ablation5_digitfix_v1`,
   then `ablation5_socratic.py` on the new Socratic bank and `eval_socratic_ablation5.py`. Check that
   `ABL5_DIAG_math_shepherd.json` now has empty `zero_var_channels` for N16 and S5.
7. Fetch the outputs, compare Math-Shepherd N16 against N13 and against the committed results, and record:
   a HISTORY.md step, PROGRESS.md, and a LESSONS.md line (a bug recorded as an amendment was never fixed in code
   and the affected arm was run again).

Expected outcome, stated as a hypothesis: real digit channels on Math-Shepherd, and N16 no longer worse than N13
because of the partition. The low PRM weight (cause 2) should remain. Fixing that is a separate, label-free design
question, still open.
