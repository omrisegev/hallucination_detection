# prm_boost_crop_ares_benchmarks_v1: the CROP / ARES benchmarks we did not have (2026-10-26)

Plan and gap table: [docs/experiments/PRM_BOOST_MISSING_BENCHMARKS_2026-10-26_HE.md](../../docs/experiments/PRM_BOOST_MISSING_BENCHMARKS_2026-10-26_HE.md).

**Status:** the input builder ran locally. The four CROP sets match CROP Table 4 exactly. No AIRCC job has run yet:
three ssh probes timed out on 2026-10-26.

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
