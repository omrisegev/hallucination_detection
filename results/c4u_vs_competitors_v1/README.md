# C4U vs PRISM, CROP, SLIDER, Mind the Gap (2026-11-08)

Thread "Compare C4U against PRISM, CROP, SLIDER, Mind the Gap". Frozen recipe `C4U-parisi-within-jaffe-cross-v1`
(`results/rigorous_recipe_v1/c4u_standalone/`, copied unchanged from the PC; sha256 of c4u_new.py
961096828cc3ebaf...). Nothing in the recipe was changed. ARES not run (dropped from deck v2).

| file | what |
|---|---|
| `c4u_competitors.py` | runs C4U per (benchmark, PRM) and applies the competitor protocols from `telemetry_only_vs_competitors_v1/compute.py` |
| `tables.py` | `RESULTS.csv` + AUTO rows of `RESULTS_BACKBONE.csv` + published numbers -> `TABLES.md`, `SUMMARY.csv` |
| `RESULTS.csv` | every C4U row and the PRM rows re-computed through the same code |
| `PARITY.json` | 476 checks; `FIT.json`: K, group sizes, prevalence, flag rate per cell |

## Reproduction check (before any protocol)
`run_c4u.py` cells, on this machine: input hash, flag hash, PRMScore and within-AUC identical in 8/8. The rank hash
rounded to 9 decimals differs in 5/8: the rank moves by ~1e-14 (BLAS), which flips the 9th decimal of a few steps.
Flags and all metrics are bitwise equal, so the C4U decisions are the frozen ones.

## Inputs (R1 everywhere)
- PRMBench: `c4u_new.load('prmbench', prm, 'R1')`; INPUTS_<prm>.npz copied from AIRCC
  `/shared/cycle2_tau_averbuch_prj/omrisegev1/auto_raw_plus_z_v1/bundle/inputs/` (sha256 = bundle SHA256SUMS.txt).
- ProcessBench: raw bank `hd_collection_data/sub5_all_v1/pb_prm/bank_<prm>/FEATURES.npz` (local).
- CRV-Arith, CRV-GSM8K, PRM800K: raw bank `results/prm_boost_crop_ares_benchmarks_v1/full_prm_<prm>_<bench>/bank_v3/FEATURES.npz`
  from AIRCC (copied to `hd_collection_data/c4u_r1_inputs/crop_ares/`, sha256 checked, COPY_REPORT.md there).
- R1 for a non-PRMBench cell = `build_R1` applied to that cell: [answer-z PRM risk, raw PRM error prob standardized,
  answer-z of the global rank->normal of the 16 raw N16 channels]. Digit channels ranked over active steps only.
  Rank pool = all scored steps of the cell (the CROP/ARES 'common' answers). Gate: answer-z of the raw bank values
  reproduces the stored N16 channels (max abs diff < 3e-7 in all 16 cells).

## Decision rules
- Flag metrics (PRMScore, SLIDER label-free, ProcessBench F1, SLA): C4U's own flags (top k_a, k_a = round(sum of DS
  posterior)). ProcessBench prediction = first flagged step, -1 if none.
- Threshold metrics (matched TPR, SLIDER LR, CROP T6/T17): C4U's continuous rank under the same label-calibrated
  threshold procedure as every other row.

## Reproduce
```
COLLECTION_DATA=<hd_collection_data with c4u_r1_inputs/, auto_raw_plus_z_bundle -> c4u_r1_inputs/auto_raw_plus_z_bundle>
python c4u_competitors.py && python tables.py
```
Needs the CROP/ARES score cells (prm + sa, all 5 PRMs, for the 'common' population), h2v SCORES (imported by sub5_all),
socratic/prmbench collection files. Runs on CPU in ~25 min.
