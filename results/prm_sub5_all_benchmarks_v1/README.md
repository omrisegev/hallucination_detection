# prm_sub5_all_benchmarks_v1: SUB5 on every benchmark we have data for (2026-10-28, CPU only)

**What this is.** SUB5 is the 6-channel subset that `results/prm_view_selection_v1` picked as
"JOINT 8 cells (no Math-Shepherd)": `q15_H1 + chosen_surprisal + top15_turnover + dominant_freq16 + bocpd_p0 + digit_spread`.
All six are channels of the 16-channel bank (N16) behind TELE_N16, so no new inference was needed. This directory
re-fuses the stored channels on every benchmark/PRM cell and puts SUB5 beside the PRM, TELE_N16, MEAN_N16 and CHAIN_N16.
The Math-Shepherd PRM is excluded. The four PRMs are Qwen2.5-Math-PRM-7B (`qwen_prm_7b`), Qwen2.5-Math-PRM-72B,
PURE-PRM-7B and RLHFlow-PRM-Mistral-8B.

**Important caveat:** SUB5 was **selected with PRMBench and Socratic labels** (backbone telemetry). Every PRMBench and
Socratic number is in-sample for the choice of channels, including the standalone cells, which use another model's
telemetry but were scored after the choice. Hard2Verify, ProcessBench, CRV-Arith, CRV-GSM8K, PRM800K and DeltaBench are
out of sample for the subset. All of this is development data.

Two arms:
- **SUB5 alone**: telemetry only (`stage2.TeleCell.evaluate`). It uses the stored frozen-bank orientation, per-answer
  GAP votes, L-SML groups (K >= 3 required), DS-EM weights between groups, and those weights applied to the answer-z
  continuous values.
- **SUB5 +PRM**: the PRM enters as one more channel and as the DS-EM anchor (`stage1.CellData.evaluate`). Backbone
  cells only.

- **SUB5 mean** / **SUB5 +PRM mean**: the plain average of the same six oriented answer-z channels (with the PRM's
  answer-z as a seventh for +PRM). These controls show how much L-SML itself adds over equal weights.

Both classes run unchanged and are refitted per cell without labels. The only redirection is of
`E.load` / `E.channels` / `stage1.prm_arm` to the new cells.

## Results

All tables (rows = arms, columns = the 4 PRMs) are in [TABLES.md](TABLES.md). The long-form data is in
[SUB5_ALL.csv](SUB5_ALL.csv) (benchmark, mode, prm, arm, metric, value).

Metrics:

| benchmark | metrics |
|---|---|
| PRMBench | PRMScore GAP, PRMScore at the same TPR 87.3 (`competitors.prism_rows`), within-AUC, CROP kept% at alpha 0.05 and pooled AUROC (`competitors.crop_rows`) |
| Socratic, DeltaBench | macro-F1 GAP, within-AUC (`evaluate_cab.row`) |
| CRV-Arith, CRV-GSM8K, PRM800K | the same, plus CROP Table 6 kept% at alpha 0.05 and AUROC (`crop_cab.py` protocol, q201 grid, 10 seeds) |
| Hard2Verify | official step balanced F1 GAP on the 177-answer common set, within-AUC |
| ProcessBench | average F1 under the NC rule (PRM native gate and count, the arm's order; `eval_processbench.py`), within-AUC (prefix domain) |

Populations follow the existing tables: evaluate_cab's `common` set (DeltaBench 447 answers, Hard2Verify 177; every
other benchmark uses all of its answers), all 6,969 PRMBench answers, and all 3,400 ProcessBench answers.

Two cells have no SUB5 score because the six channels form fewer than 3 L-SML groups (the method requires K >= 3):
- CRV-Arith backbone, Qwen-72B, SUB5 +PRM (K = 2);
- CRV-Arith standalone, Qwen-7B, SUB5 alone (K = 2).

## Cells

64 cells: 8 benchmarks x 4 PRMs x (backbone, standalone). The backbone cells of PRMBench and Socratic, which SUB5 was
selected on, and of Hard2Verify were added on 2026-10-28 (second run), together with the two average arms.

## Parity ([PARITY.json](PARITY.json)): 191 checks, all pass, largest difference 1.4e-14

- **In-sample backbone cells**: SUB5 alone, SUB5 mean and SUB5 +PRM equal `stage2/NAMED_EXACT.csv` (s2_fused, s2_mean,
  s1_fused) on all 8 PRMBench/Socratic backbone cells. On Hard2Verify backbone, all four SUB5 arms equal
  `h2v_holdout/H2V_SUB4_SUB5.csv` for all 4 PRMs.

- **Gate 2** (reproduces earlier SUB5 numbers):
  - PRMBench Qwen-7B: SUB5 alone 68.7782 and SUB5 +PRM 69.9031 PRMScore GAP (`stage2/COMPETITORS.csv`);
  - Hard2Verify Qwen-7B: SUB5 alone 49.3454 (`h2v_holdout/H2V_SUB4_SUB5.csv`). All are exact.
- **Gate 1**: TELE_N16 is recomputed in every cell and matches the existing tables:
  - `CAB_EVAL_ALL.csv`: macro/balanced F1 and within-AUC;
  - `RESULTS_processbench.csv`: avg F1 NC and within-AUC;
  - `STANDALONE_MAIN.csv`: PRMScore GAP and within-AUC;
  - `CROP_TABLE6_CAB.csv`: CROP kept%.
- **Channel alignment**: in all 52 cells, MEAN_N16 equals the mean of the loaded channels to at most 5.5e-8 (float32
  storage).

## Data (not in git; local `hd_collection_data/`)

- CROP/ARES backbone and standalone cells: `crop_ares_v1/scores/` and, new here, `crop_ares_v1/channels/`. Both come from
  `gdrive:hallucination_detection/cluster_results/prm_boost_crop_ares_benchmarks_v1/scores/*_CHANNELS.npz`, written by the
  AIRCC downstream job 293712 with `ABL5_EXPORT_SCORES=1`.
- Standalone PRMBench channels: `prmbench_standalone/ABL5_CHANNELS_*__standalone.npz`, from
  `gdrive:.../prm_standalone_telemetry_v1/abl5/`.
- ProcessBench backbone: no channel file existed. The bank features
  (`gdrive:.../prm_boost_processbench_v1/full_<prm>/bank_v3/{FEATURES,REWARDS}.npz`) were re-exported locally with the
  unchanged `results/prm_boost_ablation5_v1/ablation5_socratic.py` (`ABL5_EXPORT_SCORES=1`). The output is in
  `sub5_all_v1/pb_prm/abl5/` and is copied to `sub5_all_v1/scores/SCORES_prm_<prm>_processbench*.npz`.
  - The re-exported PRM, TELE_N16, CHAIN_N16, PRMG_N16 and HALF_N16 equal the frozen Drive score files to <= 4e-15.
  - Those Drive files match the sha256 values in the original `EVAL_MANIFEST.json`.

## Method notes

- **Fit pool.** The stage-1 and stage-2 code takes its fit pool from the cell's `keep` mask. That mask is every step on
  Socratic, Hard2Verify, CRV, PRM800K and DeltaBench, and the non-control answers on PRMBench, as before. ProcessBench's
  gold `include` drops the steps after the first error, which would leak labels, so for ProcessBench the fit pool is
  set to every step. The metrics still use the official ProcessBench evaluator.
- **PRM scale.** The PRM scale inside SUB5 +PRM and in the `PRM` (GAP) row follows STATUS_META's choice on
  PRMBench, Socratic and Hard2Verify, as in the existing tables. That choice is post hoc. Every other benchmark uses the
  PRMBench scale of the same PRM, fixed in advance: 1-reward for the Qwen and PURE PRMs, logit for RLHFlow.
  `PRM native` is the PRM's own 0.5 threshold.
- **Standalone cells.** In the standalone cells, CHAIN_N16 combines the PRM with the standalone model's telemetry.
  The PRM rows equal the backbone cell's PRM rows, because the rewards are grafted onto the same answers.

## Dashboard

`dashboard/` holds the interactive scorecard: `scores.csv` (made by `make_data.py` from SUB5_ALL.csv),
`published.csv` (= PUBLISHED_COMPETITORS.csv) and the page `index.html`. Published claude.ai artifact:
https://claude.ai/artifact/3bknWqmrPsYNycXfjmWpMb (private until shared).

[PUBLISHED_COMPETITORS.csv](PUBLISHED_COMPETITORS.csv) lists published scores with their repo source:
- PRMBench, Socratic and Hard2Verify leaderboards;
- ProcessBench from Qwen PRM paper Table 7 (native threshold);
- CROP Table 6 kept % and AUROC for PRMBench, CRV-Arith, CRV-GSM8K and PRM800K. Direct PRM and Trace+PRM use
  Qwen-PRM-7B, so they are compared with that PRM only.

DeltaBench has no published number on our metric. The ARES and DeltaBench papers report other metrics on other subsets.

## Files and reproduction

- `sub5_all.py`: the scoring script. Run as `gates`, `job <prm|sa> <bench> <prm>`, `list` or `assemble`.
- `run_all.sh`: runs one process per cell, so memory stays bounded on the 16 GB machine, then assembles. Per-cell
  outputs go to `parts/`.
- `tables.py`: writes TABLES.md.
- `vendor/`: unchanged copies of
  - `evaluate_cab.py`, `crop_cab.py` and `CAB_EVAL_ALL.csv` / `CROP_TABLE6_CAB.csv`, from
    `claude/prm-boost-missing-benchmarks-2026-10-26` @ 109bf203c;
  - `eval_processbench.py`, `pb_metrics.py` and `RESULTS_processbench.csv`, from `claude/integration-2026-10-17` @ 46b3a6318.

To reproduce:

```
bash results/prm_sub5_all_benchmarks_v1/run_all.sh && python results/prm_sub5_all_benchmarks_v1/tables.py
```
