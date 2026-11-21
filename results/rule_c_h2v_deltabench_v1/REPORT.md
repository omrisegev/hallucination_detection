# Rule C on Hard2Verify and DeltaBench: interim status (2026-11-21)

Status: **the CPU parts are done; the GPU part is blocked.** I submitted no job on AIRCC and deleted no file. I did
not change rule C (agentS2: beta grid, tie rule, quantile, sg, GAP). I only added benchmark adapters.

## 1. Blockers

| Blocker | Facts | Needed |
|---|---|---|
| AIRCC workspace | `/shared/cycle2_tau_averbuch_prj` does not exist any more. The 11-18 jobs used `$S=/shared/cycle2_tau_averbuch_prj/omrisegev1` for code, results, pyenv, wheelhouse and hf_cache. HOME is now `/shared/cycle4_tau_averbuch_spot_prj/users/omrisegev1` (empty). The cycle4 `hf_cache/hub` has no Qwen2.5-Math-7B-Instruct | A new cycle4 setup (section 5) and Omri's yes for sbatch |
| GPU authorization | The GPU request came through the coordinator note. Omri's own message to this session asked only to verify rule C | Omri: yes or no |
| PC disk | C: is the only drive and has 0.75 GB free. The DeltaBench ext raw pkl is 2.33 GB. Omri asked for no import to the PC until the cleanup session finishes | Build the DeltaBench step table on AIRCC, or after the cleanup |
| Guard hook | `guard_git.py` blocked a sparse-checkout change in this worktree. I did not retry it. I extracted the needed files with `git archive 4acda6f6` into the scratch HD_ROOT instead | none |

## 2. Hard2Verify C step table and parity (done)

| Item | Value |
|---|---|
| Source | Drive `prm_boost_crop_ares_benchmarks_v1/full_sa_qwen_prm_hard2verify/grafted.pkl` (123,717,617 B) + bank_v3 (byte-equal to the PC copy) |
| extract_steps.py (unchanged, 4acda6f6) | GATE PASS (max abs diff 4.8e-7). 181 answers, 1,738 steps, 0 empty steps. 19 rows have no telemetry (18 are longer than 4,096 tokens, 1 other failed row); I did not rerun them |
| Method of record on the new table (hard top10 + 5 step channels, R0, L-SML, GAP) | Official H2V step Balanced F1 on the 177-answer common set = **47.2247** |
| Stored reference (SUB5_ALL.csv, TELE_N16 standalone qwen_prm_7b) | 47.2247. Difference 0.0000: **PASS** (gate 0.1) |
| Note | The brief called 47.2 "PRMScore". It is the official H2V step Balanced F1. `hd_collection_data/h2v/SCORES_qwen_prm.npz` gives 46.71, but it comes from an older bank_v2 run, not the stored cell |

## 3. Adapters and variant inputs (done, label-free)

- `code/bench_adapters.py`: builds the C pool from an extract_steps table. It attaches labels only when a gold file is
  passed (`crop_ares_v1/gold/<bench>.json`: err = not correct, keep = include, y_ans = any err).
- `code/make_groups.py`: a question group is the sha1 of the whitespace-normalized question text. H2V: 200 answers in
  79 groups. DeltaBench: 1,236 answers in 1,217 groups.
- `code/make_variants_ext.py`: imports `make_v3` and `make_v4` from `make_variants_v34.py` without changes. There is
  no dev/test split (side 'all'). The RNG seed index is 1, the PRMBench index, because external benchmarks are not
  'processbench'.

| H2V children (181 parents) | Rows | Skipped |
|---|---|---|
| V3 numeric edit | 174 | 7 (no editable number in steps 2..n) |
| V4 on-topic donor | 141 | 10 without a sibling answer, 30 without a donor in the group. Relax levels 0/1/2 = 109/21/11. Same-generator donor: 0.7% |

The driver inputs are in `h2v_inputs/hard2verify_{V3,V4,preC_V3,preC_V4}_answers.json` (format for `--answers`).
I have not built the DeltaBench children yet. DeltaBench will get V3 only; this is a planned deviation.

## 4. DeltaBench official evaluation (implemented, checked)

Source: `github.com/OpenStellarTeam/DeltaBench`, commit `6b67c64ad177553ec73f85ff98b18b0e6627341f` (2025-03-04),
`evaluation.py`. That script evaluates LLM critics, not PRMs. I took these rules from it:
- Target = set(reason_unuseful + reason_error section numbers).
- Per-sample P/R/F1 on section-number sets. Predictions above max(target) are dropped.
- Macro = mean over samples.
- A sample with an empty target crashes in `max()` and pandas then skips it. All 1,236 samples have a non-empty union
  target, so none is skipped.

The PRM rule comes from the paper (arXiv 2502.19361v3, sec. 2.5): t = mu - sigma of the rewards over the whole
dataset, and a section with reward < t is an error. App. A.8 gives the 5% quantile rule.

Checks on our data:
- Our gold (`crop_ares_v1/gold/deltabench.json`) holds the reason_error sections only. It is equal to them on all
  1,236 samples. 63 samples have no reason_error section.
- The stored `score__PRMPROB` is risk-high (corr with native_valid -0.83), so reward = 1 - PRMPROB.
- The section count matches the gold on 100% of rows: the 20,324 steps are the labelled sections.

| PRM (stored PRM-mode cell) | Our mu-sigma macro-F1 (union / error-only target) | Paper (Table 2) | Gap |
|---|---|---|---|
| Qwen2.5-Math-PRM-7B (ext) | **29.34** / 28.01 | 29.22 | +0.12 |
| Qwen2.5-Math-PRM-72B (ext) | 29.95 / 29.97 | 26.38 | +3.57 |
| PURE-PRM-7B (ext) | 22.40 / 23.32 | not in paper | |
| RLHFlow-PRM-Mistral-8B | 14.46 / 13.90 | Llama3.1-8B-PRM-Mistral-Data 9.45 (Table 2 column mapping not certain) | |
| Math-Shepherd (447 answers only) | 11.56 / 10.49 | not in paper | |

Not reproduced:
- **5% quantile rule.** Paper 7B: 46.48 at threshold 0.2168. Ours: 11.63 with our own 5% quantile, 13.45 with their
  threshold 0.2168.
- **HitRate@1.** Paper 7B 49.15 / 72B 41.13. Ours: 27.51 / 27.75.

Likely cause: our cells give ONE reward per section, but the paper's PRM inputs are probably finer (several steps
inside a section). A recall of 73.86 at a 5% quantile is only possible if each section has many scored units. The
paper text does not say how section scores are aggregated. So the mu-sigma rule matches only for the 7B. Treat the
72B and appendix numbers as unmatched.

### Arms that need no new telemetry (stored scores, point values, union target)

| Arm | DeltaBench official macro-F1 |
|---|---|
| Method of record s0 (TELE_N16 standalone 7B-Instruct, ext cell), GAP flags | **31.19** (error-only 31.82) |
| Method of record s0 under mu-sigma (s0 > mu + sigma) | 26.23 |
| Method of record HitRate@1 | 29.53 |
| Qwen2.5-Math-PRM-7B mu-sigma / GAP | 29.34 / 32.69 |
| Qwen2.5-Math-PRM-72B mu-sigma / GAP | 29.95 / 32.99 |
| PURE-PRM-7B mu-sigma / GAP | 22.40 / 25.48 |
| RLHFlow-PRM-Mistral-8B mu-sigma / GAP | 14.46 / 10.60 |

These numbers use labels but come from existing arms: no rule C and no selection step. Rule C needs the V3 telemetry.
The group-bootstrap CIs will come with the final evaluation.

## 5. Drive inventory for a cycle4 rebuild (read-only, no download)

| Item (11-18 job) | Drive path | Size | Missing |
|---|---|---|---|
| Model Qwen2.5-Math-7B-Instruct | `gdrive:hf_cache_flat/Qwen__Qwen2.5-Math-7B-Instruct/` | 14.20 GiB (30 files) | no. The revision is not verified: check tokenizer sha256 c0382117... after the copy |
| BANK_CODE `code_thinkfix_bank_553de79` (external builder v2) | `hallucination_detection/aircc_workspace_final/code_thinkfix_bank_553de79/` | 62.4 MiB | no |
| PB_MASK `code_thinkfix_rescore_v1` | `aircc_workspace_final/code_thinkfix_rescore_v1/` | 17.2 MiB | no |
| `code_prm_boost_v1/mask_fixed` | `aircc_workspace_final/code_prm_boost_v1/mask_fixed` | 4.8 KiB | no |
| PB_NAMES `results/prm_boost_v1/bank_prm/FEATURES.npz` | `cluster_results/prm_boost_v1/bank_prm/FEATURES.npz` | 53.9 MB | no |
| RoPE x3 code (DeltaBench) | `aircc_workspace_final/code_longctx_v1/` (6.25 GiB with data); also branch `claude/deltabench-long-context-2026-11-08` (`spectral_utils/rope_extension.py`) | | no |
| SSL driver and job scripts | git `4acda6f6` (`results/ssl_one_experiment_v1/aircc/`, `cluster/run_prmbench_prm_telemetry.py`) | small | no |
| Wheelhouse `$S/wheelhouse/tf_thinkfix_20261010/*.whl` (the compute nodes are offline) | not found (I searched `aircc_workspace_final` to depth 4) | | **yes** |
| pyenv `pyenv_native_312_20261011` | not found | | **yes**, but the job runs in the nvcr pytorch 25.01 container, so only the wheels are needed |
| H2V raw C telemetry + bank | `cluster_results/prm_boost_crop_ares_benchmarks_v1/full_sa_qwen_prm_hard2verify/` | 124 MB + 0.65 MB | no |
| DeltaBench ext raw C telemetry + bank | `.../full_sa_qwen_prm_deltabench_ext/` | 2.33 GB + 7.4 MB | no |
| H2V V3/V4 driver inputs | this folder, `h2v_inputs/` | 1.5 MB | no |
| DeltaBench answers (for the V3 build) | PC `hd_collection_data/crop_ares_v1/inputs_text/deltabench/answers.json`; Drive `.../inputs/deltabench/answers.json` | 20.5 MB | no |

Setup estimate on cycle4, from the login node:
1. Copy the model with rclone: 14.2 GiB, about 3-5 min.
2. Rebuild the wheelhouse (`pip download` of the tf_thinkfix pins on the login node): about 10 min.
3. Clone the code at 4acda6f6 and copy the bank code: about 5 min.
4. Change the paths in `common.sh` and `job.sbatch` and the container mount to cycle4: about 15 min. This changes the
   job scripts (paths only), because the 11-18 paths are hard-coded to cycle2.
5. Run /preflight and the 12-row smoke: one short GPU job, about 10 min plus the queue wait.

Total: about 1 h of work before the first real GPU job, plus the queue waits.

## 6. Files

- `code/`: adapters, groups, variant builder, parity, DeltaBench evaluation
- `outputs/`: PARITY_H2V.json, DELTABENCH_PRM_CHECK.json, DELTABENCH_BASELINES.json
- `h2v_inputs/`: groups, sidecars, driver inputs, variant report
- `logs/h2v_extract_steps.log`
