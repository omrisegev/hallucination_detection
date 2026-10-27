# prm_boost_crop_ares_benchmarks_v1: the CROP / ARES benchmarks we did not have (2026-10-26)

Plan and gap table: [docs/experiments/PRM_BOOST_MISSING_BENCHMARKS_2026-10-26_HE.md](../../docs/experiments/PRM_BOOST_MISSING_BENCHMARKS_2026-10-26_HE.md).

**Status (2026-10-26):** COLLECTION COMPLETE. 65 label-free score files (25 PRM-mode, 40 standalone) are in
`$O/scores/` on AIRCC (`MD5_scores.txt`) and are uploaded to
`gdrive:hallucination_detection/cluster_results/prm_boost_crop_ares_benchmarks_v1/`. Evaluation with labels has not run.

## Runs (AIRCC, 2026-10-26)

Code: `code_standalone_v1` (SYNC_COMMIT 41b63ac) for the telemetry. The downstream used `code_thinkfix_bank_553de79`
(`prm_boost_build_bank_external_v2.py`, digits from the standalone tokenizer) and `eval_mirror_prm_gen_v1`
(`ablation5_socratic.py`). Preflight: every row passed except local disk (6.7 GB free, below the 10 GB limit). Omri
waived that row because all outputs stay on AIRCC.

| step | jobs | result |
|---|---|---|
| census (CPU) | 293520-24 | FAILED: the container does not start on CPU nodes (`nvidia-container-cli` missing). Not needed, because the driver counts context overflow itself |
| parity, prm mode, PRMBench first 40 rows vs the frozen pkls | 293525,27,29,31,33 | **PASS ×5, max gap 0** for rewards and all telemetry |
| parity, standalone vs the frozen standalone pkls | 293526-34 (even) | the 40-row run differs on ~2.2% of tokens (entropy p99 0.007; 97% bit-identical) |
| standalone determinism (same 40 rows, twice) | 293616 | **PASS, gap 0**. The difference above comes from bf16 batch composition (a 40-row run vs the full run), not from the code |
| downstream parity P0: rebuild the Socratic qwen_prm bank + ablation5 scores | 293600 | **PASS**: 7/7 bank keys and 26/26 score keys identical to `full_qwen_prm_socratic/bank_v2` and `collect_socratic_v1/SCORES_qwen_prm.npz` |
| timing, 12 rows per cell | 293535-99 | 65/65 manifests; the only failures are counted context overflow |
| full runs | 293622-86, 293707 | 65/65 COMPLETE or COMPLETE_WITH_FAILURES (counted context overflow only) |
| downstream: bank + scores | 293712 | COMPLETED, 65 SCORES files |

Notes:
- **Math-Shepherd PRM on Math-Shepherd (293644)** crashed. Row `math_shepherd:926` contains Cyrillic text with the
  PRM's own step tag `ки`, so the PRM finds 9 scores for 3 steps. The run was redone on 9,999 rows
  (`answers_excl_tag926.json`, job 293707). The crashed output is kept in `crashed_293644_full_prm_math_shepherd_math_shepherd/`.
- **rlhflow_prm step-text mismatches** (prm mode): 1 row on crv_gsm8k, 175 on math_shepherd, 153 on prm800k and 434 on
  deltabench. These are boundary-token decode differences of the Llama template. The frozen runs have the same
  behaviour (PRMBench 152/6969, Socratic 44, ProcessBench 87), and it is not counted as a failure.
- **Standalone cells get the PRM rewards** of the same answers (`graft_rewards.py`). An answer the PRM could not read is
  therefore also missing from the standalone scores, even when the base model read it. Example: standalone Mistral read
  all of DeltaBench but is scored on 447 rows.

Coverage (answers with scores / all):

| benchmark | qwen_prm | qwen_prm_72b | pure_prm | rlhflow_prm | math_shepherd |
|---|---|---|---|---|---|
| crv_arith, crv_gsm8k, prm800k, processbench, socratic | all | all | all | all | all |
| math_shepherd | 10000 | 10000 | 10000 | 10000 | 9999 |
| deltabench (prm and sa) | 508 | 508 | 510-511 | 1236 | 447 |
| hard2verify (sa) | 181 | 181 | 182 | 200 | 177 |

The DeltaBench and Hard2Verify losses are the 4,096-token context of the Qwen2.5-Math family and of the Math-Shepherd
PRM. Nothing is truncated.

Training-data overlap, to be marked in every table:
- Math-Shepherd train is the training data of the Math-Shepherd PRM (and RLHFlow is trained on data of the same type).
- `jinachris/Qwen2.5-Math-7B-PRM800K` (pure_prm) is trained on PRM800K.

## Inputs (`make_inputs.py`, run locally 2026-10-26)

The rules re-implement CROP's code (read, not run): `crop/cheap_baselines.py::select_target_traces` and
`crop/scripts/import_external_process.py`, at commit `HEAD` of github.com/matthewyccheung/crop on 2026-10-26.

| benchmark | answers | steps | step error rate | CROP Table 4 | median chars |
|---|---|---|---|---|---|
| crv_arith | 1,500 | 10,316 | 3.38% | 1,500 / 10,316 / 3.4%: **match** | 451 |
| crv_gsm8k | 1,319 | 8,995 | 7.30% | 1,319 / 8,995 / 7.3%: **match** | 898 |
| math_shepherd | 10,000 | 37,514 | 47.57% | 10,000 / 37,514 / 47.6%: **match** | 526 |
| prm800k | 8,000 | 70,698 | 4.15% | 8,000 / 70,698 / 4.2%: **match** | 864 |
| deltabench | 1,236 | 20,324 sections | 8.0% | ARES gives no subset size | 14,237 |

Sources (fetched by `fetch_sources.sh`; the sha256 of each file is in `INPUT_MANIFEST.json`):
- `facebook/crv`: the annotated arithmetic and GSM8K files.
- `peiyi9979/Math-Shepherd`: the first 10,000 lines of `math-shepherd.jsonl`. Its single file streams in line order, so
  these are the first 10,000 streamed rows.
- `openai/prm800k`: `phase2_test.jsonl` (2,762 rows) and the first 12,000 lines of `phase2_train.jsonl`. The first
  8,000 traces with at least one parsed step come from these.
- `OpenStellarTeam/DeltaBench`: `Deltabench_v1.csv`. One step per section; an error is a section listed in
  `reason_error_section_numbers`.

`answers.json` carries no labels. Gold (`correct[i]`, `category`) is written to a separate directory and is never
read by a scoring step.

## AIRCC layout (`aircc/`)

| file | what |
|---|---|
| `env.sh` | paths, account, the 5 PRM adapters, the standalone model and prompt style of each PRM, and the frozen PRMBench pkls for parity |
| `submit_census.sh` | CPU tokenizer census per benchmark (reuses the ProcessBench `length_census.py`) |
| `submit_gpu.sh parity` | PRMBench first 40 rows, both modes. `--compare-to` for prm mode; `parity_compare.py` for standalone |
| `submit_gpu.sh timing prm\|sa` | 12 rows per benchmark, 1 h wall |
| `submit_gpu.sh full prm\|sa` | all rows; set `WALL_7B` / `WALL_72B` from the timing |
| `upload_drive.sh` | rclone to `gdrive:hallucination_detection/cluster_results/prm_boost_crop_ares_benchmarks_v1/` |

The prm mode runs on the 5 new benchmarks. The standalone mode runs on the 5 new benchmarks plus ProcessBench,
Socratic and Hard2Verify, whose PRM telemetry already exists.

## Label evaluation (local, 2026-10-27; development data)

`evaluate_cab.py` (metric functions imported from `prm_boost_collection_v1/evaluate_collection.py`) and `crop_cab.py`
(CROP Table 6 protocol, functions copied from `prmbench_paper_metric_parity_v1/crop_metrics.py`). Gold lives only in
`hd_collection_data/crop_ares_v1/gold/` on the local machine. Outputs: `eval/CAB_EVAL_ALL.csv`, `eval/CROP_TABLE6_CAB.csv`,
`eval/SUMMARY_TABLES.md`.

Parity:
- the evaluator reproduces the frozen Socratic qwen_prm numbers exactly (PRM native 68.0917, HALF_N16 Q80 67.3720);
- Direct PRM (Qwen-7B) under CROP's protocol vs CROP Table 6: CRV-Arith AUROC .943 / kept 97.4% (CROP .9421 / 97.5),
  CRV-GSM8K .849 / 79.8 (.8659 / 81.7), Math-Shepherd .455 / 1.0 / risk 0.9 (.4537 / 1.0 / 0.9), PRM800K .930 / 68.3
  (.9327 / 67.2).
