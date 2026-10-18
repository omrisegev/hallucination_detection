# Handoff: L-SML of PRM + small telemetry banks, and the oracle-count ceiling (2026-10-18)

For the agent on the Windows machine (it has the `aircc` alias). Omri asked for this. The Mac session
could not reach AIRCC because the VPN was down, so nothing has run yet.

## The question

Omri rejects the fixed ½·PRM + ½·mean(N16) recipe (HALF_N16 in `results/prm_boosting_controls_v1`).
It is a hand-set average, which goes against the project's fusion approach. He wants to know how
L-SML performs on the PRM output together with our signals, for every bank of at most 16 channels.
Do not report or headline any averaged arm.

Second question: how much PRMScore a perfect per-answer error count adds on top of each fused
ordering (the 65.4 -> 71.7 ceiling was measured on telemetry alone, never on PRM + telemetry).

## What the script does

`lsml_bank_sweep.py`. Same inputs, orientation, Q80 rule and source-group bootstrap as
`results/prm_boosting_controls_v1/lsml_controls.py`.

For each of the 5 PRMs and each bank (N11, N13, N14, N16, plus label-selected S5/S6, reported as such)
it computes. `ext.BN` is loaded from `results/monday_nested_banks_v2/BANKS.json`, which has no B-banks, so
the B-bank branch of the script finds nothing. S5/S6 are skipped if a channel is missing from a PRM's
FEATURES.npz; check `skipped_banks` in SWEEP_DIAG.

- `PRMG_<bank>`: L-SML with the PRM in its own group, and telemetry groups detected on the bank
  alone. This is already the hierarchical form, so no separate hierarchical arm exists.
- `AUTO_<bank>`: L-SML with automatic grouping over PRM + bank.
- `TELE_<bank>`: L-SML on the bank without the PRM, as a reference.
- `CHAIN_<bank>` (Omri, 2026-10-18 18:55): take the groups that L-SML finds on the bank alone (the
  TELE fit), reduce each group to its within-group SML score, answer-z each group score, concatenate
  with z(PRM), and fuse the K+1 group scores with one SML. The PRM is one group of K+1 on the same
  scale as each telemetry group, not 1 channel of 17.
  - Relation to PRMG: PRMG is the same chain without the standardization. A synthetic check gave
    max |CHAIN_raw - PRMG| = 6e-15. So the only change in CHAIN is that each group score has unit
    variance before the cross stage; in PRMG a group with a larger score SD gets more cross weight.
  - `CHAIN_undetermined` is true when K+1 <= 3 (SML has no spare degrees of freedom).
- `PRM_native` and `LSML_CONT` (the 3 label-chosen channels), as references.

Every arm is scored under Q80 and under ORACLE (top-k per answer, with k = true error count).
Diagnostics: groups, the PRM's weight share in the fused score, and whether AUTO absorbs the PRM.
For PRMG and CHAIN also the cross-stage weights and the PRM's cross-stage share
(`PRMG_prm_cross_share`, `CHAIN_prm_cross_share`), plus the SD of each PRMG group score.

## Steps

1. `git fetch origin claude/prm-bank-sweep-2026-10-18` and check out
   `results/prm_boosting_bank_sweep_v1/`.
2. Copy `lsml_bank_sweep.py` to `/shared/cycle2_tau_averbuch_prj/omrisegev1/eval_mirror_prm_gen_v1/`.
   Do **not** use `sync_code.sh`, which overwrites the shared code dir that other jobs use.
3. Run `/preflight` (CPU job). Then submit `run_sweep.sh` as a CPU job the same way as controls job
   287777. It needs about 4x the controls runtime: up to 6 banks x 4 L-SML fits per PRM.
4. **Parity gate, check this before reading anything else.** In the log, `PRMG_N16 Q80` and
   `AUTO_N16 Q80` must match the controls' `LSML_N16_PRMG` / `LSML_N16_AUTO`, and `LSML_CONT Q80` must
   match `LSML_CONT`, for all 5 PRMs (`CONTROLS_RESULTS_*.csv`). Qwen-7B for example: 0.8298/67.80,
   0.8243/67.49, 0.8495/69.48. If they differ, stop and report.
5. Fetch `$R/prm_boost_gen_v1/bank_sweep_v1/` (SWEEP_RESULTS/BOOTSTRAP/DIAG per PRM, plus the log)
   into this directory. Write a README with these tables:
   - within-AUC and PRMScore Q80 for PRMG/AUTO per bank x PRM, against PRM_native and LSML_CONT;
   - PRM weight share per bank (does it fall as the bank grows?);
   - CHAIN vs PRMG per bank x PRM (contrast `CHAIN_<bank>[Q80] - PRMG_<bank>[Q80]`), with the PRM's
     cross-stage share in each;
   - ORACLE minus Q80 per arm (count headroom on the fused score).
6. Commit on this branch and push. Report back in plain Hebrew, numbers first, with the files they
   came from.

## Boundaries

- CPU only. No GPU and no new inference.
- PRMBench development data, not external evidence. ORACLE uses labels and is a ceiling, not a method.
- N14 and N16 contain digit channels. N11 and N13 are digit-free; say so when comparing.
