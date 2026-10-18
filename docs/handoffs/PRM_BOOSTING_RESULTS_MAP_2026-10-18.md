# PRM boosting: where every result lives (2026-10-18)

Written for the agent on Omri's other computer that will build a separate PRM boosting presentation.
Everything below is PRMBench **development** data unless it says Socratic. All fusion is label-free; labels
are used only for scoring. The PRMs themselves were trained with labels, so the fused score inherits that.
Read each directory's README before quoting a number; the numbers here are pointers, not a substitute.

State at 2026-10-18 ~22:30 IDT: no AIRCC job is running or pending. Nothing from today has been uploaded
to Google Drive yet (no rclone run found).

## Branches (all pushed to origin)

| branch | what it holds | merged into integration? |
|---|---|---|
| `claude/integration-2026-10-17` | dev branch; contains stage 1, generalization, controls, the evidence plan and the 2026-10-18 PRMBench-citing-papers slide | it is the integration branch |
| `claude/prm-boost-controls-2026-10-18` | CPU controls (job 287777) | yes (commit bb7472ae2, a1d2dee36) |
| `claude/prm-bank-sweep-2026-10-18` | L-SML of PRM + every bank of at most 16 channels, CHAIN arm, oracle-count ceiling (job 288319) | **no**, needs Omri's OK |
| `claude/prm-boost-external-2026-10-18` | external transfer to Socratic-PRMBench (5 PRMs), timing, parity, this note | **no**, needs Omri's OK |
| `claude/voters-analysis-2026-10-18` | PRM voters fusion (no-go) | **no**, needs Omri's OK |
| `claude/lr-backup-cluster-reps-2026-10-18` | per-cluster representations vs LR (backup for the LR slide, not boosting) | **no** |
| `claude/prm-boosting-generalization-results-v1`, `codex/prm-boosting-generalization-v1` | the 2026-10-13/14 cross-PRM runs; already reflected in integration's `results/prm_boosting_generalization_v1` | historical |

Note: on `claude/prm-bank-sweep-2026-10-18`, commit `7a22eed36` carries the wrong message ("Refactor stage 0:
pin the active PRMBench line's headline numbers"). It actually only adds `SUMMARY.png` and `plot_summary.py`.
It was already pushed, so the history was not rewritten.

## Result directories, in story order

1. **Stage 1: Qwen2.5-Math-PRM-7B + its own backbone telemetry** — `results/prm_boosting_v1/` (integration).
   `COMPOSITIONS.csv`, `BOOTSTRAP_*.csv`, channel atlas, small-bank (label-selected) stage 2.
2. **Generalization to 5 PRMs** (Qwen-7B, Qwen-72B, PURE, RLHFlow, MathShepherd) —
   `results/prm_boosting_generalization_v1/` (integration); frozen 3-channel `LSML_CONT` recipe under `run_v1/eval_lsml4/`.
   Experiment spec `docs/experiments/PRM_BOOSTING_GENERALIZATION_V1.md`.
3. **CPU controls** — `results/prm_boosting_controls_v1/` (integration). Label-free channel choice (HALF_N16,
   LSML_N16_PRMG), wrong-answer permutation null, position/length residualization, ablations, native count.
   Note: Omri rejects headlining the fixed ½·PRM + ½·N16 average (HALF_N16); show the L-SML arms.
4. **Bank sweep + CHAIN + oracle count** — `results/prm_boosting_bank_sweep_v1/` (bank-sweep branch).
   `README.md`, `TABLES.md` (5 tables), `SUMMARY.png`, `SWEEP_{RESULTS,BOOTSTRAP,DIAG}_<prm>.*`. Parity with the
   controls passes exactly. N11/N13 are digit-free; N14/N16 contain digit channels; S5/S6 are label-selected.
5. **External transfer, Socratic-PRMBench** — `results/prm_boost_external_v1/` (external branch).
   `README.md` (parity T1, timing table, Socratic result), `eval_socratic/{RESULTS,BOOTSTRAP}_socratic.csv`,
   `scores_socratic/SCORES_<prm>.json` (the `.npz` fused scores are on this machine but git-ignored; copies on AIRCC),
   `full_<prm>_socratic/{manifest.json,bank_v2/MANIFEST.json}` (all 5 COMPLETE, 2,995 rows each).
   Verdict: does **not** transfer consistently; the pre-registered rule fails, so Hard2Verify was not run.
6. **PRM voters (alternative: fuse several PRMs/judges)** — `results/prm_voters_prmbench_v1/` (voters branch).
   No-go: fusion is below the best single PRM and below boosting.
7. **Literature positioning** — `docs/meetings/prmbench_citing_papers_2026-10-18/SLIDE_HE.{html,png}` (integration).
8. **Plan and evidence axes** — `docs/post_advisor_2026-10-17/05_PRM_BOOSTING_EVIDENCE_PLAN_HE.md` (integration);
   2026-10-15 findings `docs/reports/ADVISOR_MEETING_EXTERNAL_AND_PRM_BOOSTING_FINDINGS_20261015.md`.

## Published pages

- Advisor meeting 15 Oct story and talking points: https://claude.ai/artifact/VwfJJe8Zjn5ryC3pY1CBeZ
- PRMBench advisor review page: https://claude.ai/artifact/NQNTYpFqVyx6xRnRS4NENV
- No published page exists yet for the controls, bank sweep or Socratic results; their source of truth is the READMEs above.

## Large raw outputs (not in git)

AIRCC base `/shared/cycle2_tau_averbuch_prj/omrisegev1/results/`:
- `prm_boost_v1/` — stage-1 raw telemetry (~1 GB per path) and FEATURES.npz.
- `prm_boost_gen_v1/` — 5-PRM telemetry and features; `controls_v1/` and `bank_sweep_v1/` (small, already copied).
- `prm_boost_external_v1/full_<prm>_socratic/external_telemetry.pkl` — about 1.0–1.1 GB each, 5 files;
  `bank_v2/FEATURES.npz` (~9 MB each), `REWARDS.npz`; `scores_socratic/SCORES_<prm>.npz`.
- `prm_boost_external_v1/timing_*` and `t1_qwen_prm/` — timing/parity pickles (4–10 MB each).

Job outcomes worth knowing: the four Hard2Verify timing jobs (287916-18, 287920) ended FAILED 2 because 2 of 12
answers exceed the 4,096-token context (counted failures, fusion blocked by design); job 288316 stopped on three
empty steps after the first-token mask (`full_qwen_prm_socratic/bank/FAILURE.json`) and was replaced by 288317,
which wrote `bank_v2/`.
