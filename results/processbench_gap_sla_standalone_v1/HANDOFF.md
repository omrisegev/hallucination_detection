# HANDOFF: standalone GAP SLA vs Mind the Gap (2026-11-02)

Branch `claude/standalone-gap-vs-mind-the-gap-2026-11-02`, cut from `claude/prm-boosting` @ a76e0ba (Step 483). Not merged.

## What
Standalone counterpart of Step 483 (`results/processbench_gap_sla_v1/`). Same metric, same rule, same published rows. Only the score source changes: base-model telemetry (SCORES_sa_*) instead of PRM-backbone telemetry (SCORES_CHAIN_*).

## Formulas
- SLA = mean over erroneous answers (gold label != -1, n = 2,221) of 1[pred == gold first-error step]. Macro = mean of GSM8K (207), MATH (594), OlympiadBench (661), Omni-MATH (759).
- GAP rule per answer with m scored steps: sort risk descending; km = min(ceil(m/2), m-1); k = 1 + argmax over the first km sorted gaps; flag top-k; pred = min index of flagged steps. m = 1 -> step 0.
- Index map: scores cover non-empty steps only; the predicted position is mapped back to the full-step index via the `nonempty` mask. In ProcessBench there are no empty steps (DATA_CHECK: 25,697 rows = gold step count in all 8 files), so the raw Step 483 convention gives identical numbers.
- Bootstrap: 2,000 resamples of answers inside each subset, seed 0, percentile 95% CI of Macro SLA.

## Assumptions
- TELE_N16 = L-SML over the 16 standalone channels (3 digit channels included); MEAN_N16 = equal-weight average. Both fit label-free on all 3,400 answers (as stored by AIRCC; no refit here).
- Mind the Gap numbers are copied from their Table 3 (ICML 2026), Shannon Drop, Qwen3-4B/8B.

## Inputs (not in git)
- Standalone: `C:\Users\omris\TAU\hd_collection_data\crop_ares_v1\scores\SCORES_sa_<prm>_processbench.npz`; Drive copy `gdrive:hallucination_detection/cluster_results/prm_boost_crop_ares_benchmarks_v1/`.
- Backbone (parity only): `C:\Users\omris\TAU\hd_collection_data\sub5_all_v1\pb_prm\scores_drive\SCORES_CHAIN_<prm>.npz`.
- Gold: `results/prm_boost_processbench_v1/local/processbench_gold.json` (in git).

## Reproduce (CPU, seconds)
```
set SA_SCORES_DIR=<standalone dir>
set PB_SCORES_DIR=<backbone dir>   # optional, for parity
python results/processbench_gap_sla_standalone_v1/evaluate_sla_standalone.py
```
Outputs: RESULTS.csv, BOOTSTRAP.csv, PARITY.json (12/12 PASS vs Step 483), DATA_CHECK.json. Summary: TABLES.md.
