# thinkfix_rescore_v1: PRMBench / ProcessBench re-score after the Qwen3 think-block fix

**What happened.** Qwen3 PRMBench + ProcessBench (8B and 4B) telemetry was regenerated with the empty
think block in the teacher-forcing prefix (`*_thinkfix`, AIRCC jobs 280425-280430, 2026-10-10). This
directory re-runs the frozen Monday scorers on that telemetry, unchanged except for the input. The
primary input is `thinkfix_mask1`: forced token 0 removed, per Omri's 2026-10-10 decision. The
unmasked `thinkfix` cells are a sensitivity check. An `old` replay on the pre-fix cells validates
the whole chain.

**Key number.** PRMBench within-answer AUC goes **down slightly**. Mean change (mask1 minus old) over
the 10 banks x 5 headline scorers: **-0.0088** (range -0.023 to +0.009, n = 50). On the highlight
bank N16, MEAN_MOM_ga goes 0.7909 -> 0.7767 and HEM_MOM_cc goes 0.7961 -> 0.7773. N16 remains the
highlight bank in all three variants. Source: `PRMBENCH_COMPARISON.csv`, table `panel_a`.

**Supervised LR** (monday_lr_reference_v1 rerun; table `lr_reference` / `lr_gaps`). On N16 the
nested-CV LR within-answer AUC goes 0.8224 -> 0.8015 (mask1). Its lead over the best label-free
scorer narrows from 0.0263 to 0.0226; over MEAN_MOM_ga it narrows from 0.0315 to 0.0248. The LR
ProcessBench transfer hit goes 0.4018 -> 0.3727. The old replay reproduces the frozen N16 LR AUC
exactly (0.8223662; the lr_weights gate is enforced for `old`). The replay differs from frozen
only in N53/N56, by at most 8.3e-4.

**External, locked L-SML evaluation (step 3 part 1)** (`external/`, `EXTERNAL_LSML_COMPARISON.csv`).
The unchanged METHOD_FREEZE/BUNDLE scorer was run on the think-fix Hard2Verify and Socratic Qwen3-8B
records. QwQ is not re-collected. The new analysis freeze was written before labels, with
amendment A2 for Windows path separators.
- mask1 frozen L-SML metric: Hard2Verify 0.4367 -> 0.4210, Socratic 0.6322 -> 0.6278.
- Frozen L-SML vs CT7 on Socratic: +0.061 [0.053, 0.070] (Bonferroni).
- The BUNDLE was fitted on pre-fix source telemetry; its inputs are gone, so it was not refit.

**External, Monday panel methods (step 3 part 2)** (`external/<variant>/panel_chain/`,
`EXTERNAL_PANEL_COMPARISON.csv`).
- Pipeline: family features -> v4 external banks -> label-free panel / cont / nested scoring and
  sealing -> evaluation. QwQ is removed from the cell lists. The tokenizer check also searches the
  AIRCC HF cache. The family freeze was re-hashed before scoring.
- The panel adapter's dev parity gate passes to about 1e-15. The official-evaluator replays pass.
- N16, ANSWER_Z_Q80, official metric, old -> mask1:
  - Socratic MEAN_MOM_ga 0.633 -> 0.616; ALL_EQ 0.627 -> 0.611.
  - Hard2Verify (200 answers) moves both ways: MEAN_MOM_ga 0.404 -> 0.418, TCI_ONLY_ga 0.391 -> 0.433.
- Nested-bank within-AUC, mean change -0.006.

**Self-generated experiment (step 4)** (`selfgen/`). B16 BASE pooled within-AUC, self-generated vs
teacher-forced ProcessBench (FINAL890 labels):
- Old 0.7566 vs 0.7630; mask1 0.7566 vs 0.7415. The difference is +0.015, CI [-0.021, 0.052], so
  still equivalent at the 0.03 margin.
- GRP reference falls from 0.720 to 0.606 (MATH cells), so GRP is no longer equivalent.
- The ProcessBench reference was refitted per dataset with the unchanged b16_fit code on the
  rebuilt banks. Its replay of Step 462 is within 2.7e-9.
- Not re-run: evaluate_selfgen's teacher-forced panel, which needs the frozen bundle's
  VALIDATION.npz (cannot be refit).
- The Mind the Gap answer-level evdrop scores include the 4 generated think tokens. The expected
  effect is small, and they were not re-scored.

**Mask rule fix (2026-10-11).** The first mask1 build rejected 3 Socratic records whose source
already had empty steps; external records keep empty steps as (k, k). An already-empty step now
stays empty. The rebuilt cell has 2995/2995 records. The first build is kept as
`*_mask1_superseded_emptystep` (MASK.json in `mask1_cells/`). The PRMBench, ProcessBench and
Hard2Verify mask1 cells contain no already-empty spans, so they are unchanged.

## Variants

| dir | input telemetry | notes |
|---|---|---|
| `old/` | pre-fix cells (`prmbench_qwen3_8b_telemetry_full`, `pb_qwen3_{8b,4b}`) | replay; must equal the frozen results |
| `thinkfix/` | `*_thinkfix` | empty think block in the prefix |
| `thinkfix_mask1/` | `*_thinkfix_mask1` | **primary**; forced token 0 removed (`spectral_utils/first_token_mask.py`, MASK.json per cell: 0 failures in every cell) |

Each variant has `nested_banks/` (monday_nested_banks_v2/dev_run.py), `cont_baselines/`
(monday_cont_baselines_v1/run.py: L-SML continuous, IU-PCR), `lr_reference/` and `lr_weights/`
(supervised LR), and `prmscore_inputs/`. Patched copies of each frozen script and a unified `DIFF.txt`
sit beside its outputs.

## Chain and gates

1. **Feature bank.** `scripts/rebuild_source_bank_v4.py` rebuilds external_banks_v4
   `source/FEATURES.npz` (48 + 5 + 3 channels, 13,769 answers, 145,597 steps, JOINED order) from raw
   telemetry. It uses the exact per-answer functions of the frozen gates (claude/ssl-pseudolabel-residual-v1
   @ 553de79c6, code hashes equal to the frozen bank manifest).
   **Gate:** the old raw cells reproduce the frozen bank with max abs error 6.1e-7 (limit 1e-6) and 0
   filled values (`source_bank_old_rebuild/MANIFEST.json`, AIRCC job 280517). This gate was required
   before any new variant was scored.
2. **Step spans** (only used for dev_run's STEP_LENGTH reference and ProcessBench "longest step"
   strata): rebuilt per variant (`STEP_TOKEN_SPANS_*.npz`). `SPANS_CHECK.json` asserts:
   - thinkfix == old on all steps;
   - mask1 == old shifted by the mask rule (step 0 shorter by exactly 1 token in all 13,769 answers).
3. **Scorers.** `run_prmbench_rescore.py` (dev_run + cont) and `run_lr_rescore.py` (LR reference +
   weights). Labels/folds (v3 labels, v2 source groups via step_evidence_v1), method roster, seeds
   and bootstrap draws are taken unchanged from the frozen scripts.
   - **Old replay:** dev_run's hard B13/PRMBench recheck = 3.7e-11 (limit 1e-9). Of 3,633 compared
     numbers, 27 differ by more than 1e-9, all PRMBench rows in the largest banks N40/N53/N56: max
     3.3e-4 (one category), otherwise <= 1.7e-4. Every ProcessBench number and every row for
     N11-N25 is exact. The source is the 6e-7 bank-level difference propagating through the
     iterative fits on the widest banks.
   - **LR:** the N16 'OOF within-AUC = 0.822366' assert is enforced for `old` and recorded for the
     new variants.
4. `compare.py` writes:
   - `PRMBENCH_COMPARISON.csv`: within-AUC, top-1-on-error, top-1 PRMScore, by bank and scorer;
     per-category from PANEL_B_PRMBENCH_TYPES; cont baselines; LR.
   - `PROCESSBENCH_COMPARISON.csv`: argmax hit, strata, official F1.
   - `COMPARISON_SUMMARY.json`.

## Deviations (accepted by the coordinator, 2026-10-10)

- **Mind the Gap reference rows were not recomputed.** Its input `EVIDENCE_DROP.npz` no longer exists:
  the token-probability-fusion-v1 worktree was removed. Rows are stripped (`nested_banks/NOT_RECOMPUTED.json`).
- **Could not be re-run** because their inputs are gone or because they score another pipeline:
  - `monday_candidate_panel_v1` (PRM_CATEGORY_METRICS): 1,440 sealed fits built on the missing
    DERIVATIVE_CHANNELS/TOKEN_MATRICES/EVIDENCE_DROP files.
  - `prmscore_adapter_v1`: scores old JOINED arms of a different pipeline, not this bank.
  - `decision_rule_v1` R1_global: reads DERIVATIVE_CHANNELS + ct7 profiles.

  Per-category numbers come from PANEL_B_PRMBENCH_TYPES instead.
- **`G` in `prmscore_inputs`** is `G_def_equal_DS_survivors_global_z (not decision_rule_v1 replay)`:
  the mean over each bank's DS-filter survivors of every raw channel, standardized once over all
  PRMBench steps. `DS_SIGNED` is not a scorer in these scripts; `S__DS_FILTER_EQ` is the DS filter +
  equal mean.
- **Compute.** Ran on AIRCC CPU nodes. The NGC container fails to start there (pyxis "spank plugin
  stack"), so the jobs use system python3.12 with the local frozen-run library versions unpacked from
  pinned wheels: numpy 2.2.4, scipy 1.17.1, pandas 3.0.3, scikit-learn 1.8.0, torch 2.6.0+cpu. The
  frozen scripts' Windows path constants resolve to a mirror directory of the same name.

## prmscore_inputs

`prmscore_inputs/bank_<BANK>.npz` (here: thinkfix_mask1; `old/` and `thinkfix/` hold their own) contains:
- `offsets` (A+1), `idx` (A PRMBench ids, control rows included);
- `S__<method>`: answer-z step scores for ALL_EQ, DS_FILTER_EQ, HEM_MOM_cc, MEAN_MOM_cc, MEAN_MOM_ga,
  TCI_ONLY_ga, HEM_MOM_ga, MEAN_EQ, HEM_EQ, LSML_CONT(+_LEVELSIGN), IU_PCR(+_LEVELSIGN), ALL_EQ_ORIENTED;
- `G` (definition above).

`prmbench_meta.json` holds idx, classification, category, error_steps (1-based) and n_steps.
