# C4U and TELE16 on every competitor benchmark, both modes (2026-11-08)

Omri, 2026-11-08: every benchmark of PRISM, CROP, SLIDER, ARES and Mind the Gap must have all four PRMs (Qwen-PRM-7B,
Qwen-PRM-72B, PURE, RLHFlow; Math-Shepherd PRM excluded) evaluated with C4U and TELE16 in both modes, from saved raw data.
CPU only, stored raw banks, no refit of any frozen recipe. Development data.

| file | what |
|---|---|
| `run_all.py` | builds C4U inputs per (mode, benchmark, PRM) and runs every protocol; `python run_all.py prm` / `sa` |
| `tables.py` | `RESULTS_prm.csv` + `RESULTS_sa.csv` -> `TABLES.md` (headline metric per benchmark) |
| `RESULTS_{prm,sa}.csv` | every row (1,276 per mode); `PARITY_{prm,sa}.json`, `FIT_{prm,sa}.json` (K, groups, prevalence, R1 gate) |
| `run_{prm,sa}.log` | console output |

## Which benchmark each paper uses (coverage, 4 PRMs x 2 modes)

| benchmark | papers | raw data (both modes) | here |
|---|---|---|---|
| PRMBench | PRISM T2/T7, SLIDER (redundancy), CROP T6, ARES | complete | yes |
| ProcessBench | PRISM T6, Mind the Gap T3 (SLA), CROP T6 | complete | yes |
| CRV-Arith, CRV-GSM8K | CROP T6 (T17 cross-domain in c4u_vs_competitors_v1) | complete | yes |
| PRM800K | CROP T6 | complete | yes (PURE is trained on PRM800K: in-training) |
| Math-Shepherd | CROP T6 | complete | yes (labels look degenerate: every arm < .5 AUROC, as CROP's Direct PRM .454) |
| DeltaBench | ARES | RLHFlow complete; Qwen-7B/72B/PURE 508 of 1,236 (4,096-token context) | 447 common answers now; long-context cells pending (branch `claude/deltabench-long-context-2026-11-08`) |
| ClaimTrees, CaptainCook (ARES); GSM8K/AIME RRI (SLIDER); BoN/beam/GRPO (PRISM) | | not collected | out of scope: no math step labels, or needs generation/training |

## Inputs
- PRM backbone (`prm`): PRMBench R1 from the frozen bundle (`c4u_new.load`), ProcessBench `sub5_all_v1/pb_prm/bank_<prm>`, all CROP/ARES
  sets from `c4u_r1_inputs/crop_ares_all/full_prm_<prm>_<bench>/bank_v3` (copied from AIRCC 2026-11-08, 130/130 score md5 OK).
- Standalone (`sa`): same build_R1 columns [answer-z PRM risk, standardized raw PRM error prob, answer-z of the global rank->normal of
  the 16 raw channels], with the 16 raw channels from the standalone model's bank (`prm_standalone_banks/bank_<prm>` for PRMBench,
  `crop_ares_all/full_sa_<prm>_<bench>/bank_v3` elsewhere). The PRM columns are the same PRM's rewards grafted onto the same answers.
  This is our reading of "C4U standalone": the recipe is unchanged, only the telemetry source differs. TELE16 contains no PRM input.
- R1 gate (every cell, both modes): answer-z of the raw bank values reproduces the stored N16 channels (< 1e-4, asserted).

## Parity
- `prm` mode: 664 checks; 280 C4U rows reproduce `c4u_vs_competitors_v1/RESULTS.csv` exactly (PRMBench, ProcessBench, CROP).
  One check fails: the Qwen-PRM-7B PRMBench **input** hash differs from `expected_results.json` on this Windows machine, while the
  C4U flags hash, PRMScore and within-AUC are bitwise identical (the other three PRMs pass). Treated as a float-formatting/BLAS
  difference in the inputs that does not reach any output; reported, not hidden.
- `sa` mode: no stored C4U reference exists (new cells); the TELE16 and PRM rows come from the same stored scores as
  telemetry_only_vs_competitors_v1.

## Rules
- Flag metrics (PRMScore, SLIDER label-free, ProcessBench F1 and SLA, DeltaBench macro-F1): own label-free flags (C4U: DS count;
  TELE16: GAP; PRM: native threshold).
- Threshold metrics (PRISM matched TPR, SLIDER LR, CROP T6): label-calibrated threshold, identical procedure for every row.

    COLLECTION_DATA=C:/Users/omris/TAU/hd_collection_data python run_all.py prm && python run_all.py sa && python tables.py
