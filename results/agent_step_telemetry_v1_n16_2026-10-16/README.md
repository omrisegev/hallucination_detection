# N16 Continuous L-SML and IU-PCR on agent step telemetry (Qwen3-8B), 2026-10-16

This run is label-free. Fits use only the unlabeled N16 feature matrices. Evaluator labels are opened afterwards, for scoring only.
Nothing was committed and no cluster was used. Raw records: `scratchpad/ast/full/{tprm,apb,aft}`.

## What was mirrored
- **Fusion.** `results/monday_cont_baselines_v1/run.py`: `fit_lsml` and `fit_iu`, copied verbatim. This is the script rerun as `thinkfix_rescore_v1/thinkfix_mask1/cont_baselines/run_patched.py` (commit `080877eb`, reviewed in `f05f10b8`), and it produced PRMBench N16 L-SML 0.779193 and IU-PCR 0.780459.
  - L-SML is `fusion_utils.lsml_continuous` with default arguments. It fails if K<3.
  - IU-PCR is `upcr.upcr_fit(F, **laplacian_upcr.IU_FIT_DEFAULTS)` (two-component L2), with F = V z-scored over the cell's rows.
  - The primary result uses each method's native label-free sign. `_LEVELSIGN` is the check, and it was identical everywhere.
  - Libraries come from the `codex/agent-step-telemetry-v1` worktree, whose `fusion_utils` matches `080877eb`.
- **Bank.**
  - Roster: N16 from `monday_nested_banks_v2/BANKS.json`.
  - V: `monday_external_panel_v1/score.py:build_bank`, i.e. `answer_standardize` on the 13 plain channels and `masked_std` on the 3 digit channels.
- **Features.** `scripts/rebuild_source_bank_v4.py:one()`, which runs `step_features`, `new_step_features` and `digit_step_features`, then `fill_missing`. The six modules come from `results/monday_fusion_analysis_v1/code`. Their SHA256 equal the `code_sha256` in `thinkfix_rescore_v1/source_bank_thinkfix_mask1/MANIFEST.json`.
- **Fit scope.** One cell per benchmark, as PRMBench is fitted. Per-subset fits (the ProcessBench analogue, one cell per dataset) are secondary.

## Adaptations (labelled)
1. **Trace.** The trace is the concatenation of an item's nonempty step spans. `top_k_logprobs` exist only at `top_k_token_indices`, and the frozen code needs a [T,50] matrix. The indices equal the step tokens for 100% of items. Four APB items have no nonempty step.
2. **Mask.** The mask1 rule (drop trace token 0) was applied. No step was emptied by it.
3. **Normalization.** Canonical per-answer z (`ansz`) makes every channel identically zero on ToolPRMBench, which has one step per variant. Both methods then fail by their own rules, with no fallback. It also makes any per-trajectory mean exactly 0.
   - So this run adds `cellz`: the same transforms, but z-scored over the fitted cell instead of per answer.
   - `ansz` trajectory-mean numbers are float noise; ignore them, including the 0.630 for `ansz/subset/IU_PCR`.
   - `ansz` trajectory-max numbers are confounded by trajectory length.
4. **Features dropped.** No N16 feature is missing from this telemetry. `realized_z` is answer-standardized by definition, so it is constant (zero) on single-step ToolPRMBench items.

## Files
| File | Contents |
|---|---|
| `extract_bank.py` | Feature extraction to `FEATURES_<b>.npz` (all 56 channels) and `MANIFEST_<b>.json` |
| `score_n16.py` | Fits and metrics, written to `SCORES_<b>.npz`, `FIT_DIAGNOSTICS.json` and `METRICS.json` |
| `aft_meanent_pooled.py` | Pooled injected/diagnosed AUROC for the baseline, written to `AFT_MEANENT_POOLED.json` |

Metric definitions follow `../agent_step_telemetry_v1_baselines_2026-10-16/agent_baselines.py`. Bootstrap: 1000 draws, seed 0.

## APB judge reference
The same-backbone reference is the Qwen3-8B Non-Thinking LLM judge.
- **Matching.** APB uid `apb:<ds>:<n>` uses `total_index` n. Judge records map through `compare.py:_record_key`, offset + 5·query_index + sample_index. The evaluator labels equal the reference labels on 1000 of 1000 items.
- **Official result.** `first_neg1_idx_acc` is 0.407 on all 1000 items and 0.409 on our 986.
  - This metric also credits correct "no error" predictions. Our continuous scores have no abstention, so it is not computed for them.
- **Comparable result.** On the 601 error trajectories, the judge's first -1 matches the gold first -1 in 0.111 of cases. This is comparable to our argmax hit.
- **Step AUROC.** Using the judge's -1/0/+1 label as a score gives 0.628.

## Results (primary = per-benchmark fit, native sign; `_LEVELSIGN` identical)
**ToolPRMBench pairwise acc** (cellz only; ansz = all-tie / fit failure)
| row | all [95% CI] | GTA | ToolSandbox | bfcl | tooltalk |
|---|---|---|---|---|---|
| best simple baseline (len) | 0.625 [0.582,0.667] | .487 | .838 | .514 | .634 |
| mean_ent | 0.584 | .703 | .738 | .486 | .314 |
| N16 L-SML (bench) | 0.593 [0.551,0.641] | .475 | .862 | .432 | .558 |
| N16 IU-PCR (bench) | 0.600 [0.557,0.643] | .492 | .854 | .441 | .570 |
| N16 L-SML / IU-PCR (per-subset fit) | 0.634 / 0.658 [0.613,0.703] | .534/.525 | .877/.885 | .441/.450 | .651/.767 |
| Qwen3-Coder-Next (best published) | — | .771 | .762 | .378 | .826 |
| xLAM-7b-r (weakest published) | — | .424 | .446 | .396 | .477 |

**AgentProcessBench**
| row | step AUROC [CI] | bfcl | gaia | hotpotqa | tau2 | first-err loc (rnd .216) | traj AUROC (max) |
|---|---|---|---|---|---|---|---|
| best baseline (mean_ent) | 0.582 [0.569,0.595] | .675 | .445 | .680 | .525 | .316 | .596 |
| N16 L-SML ansz | 0.584 [0.572,0.597] | .679 | .463 | .659 | .575 | .324 | .531 |
| N16 IU-PCR ansz | 0.590 [0.578,0.603] | .681 | .470 | .661 | .586 | .324 | .530 |
| N16 L-SML / IU-PCR cellz | 0.587 / 0.592 | .676/.678 | .379/.379 | .611/.617 | .563/.560 | .341/.341 | .521/.521 |
| N16 ALL_EQ ansz (plain mean) | 0.591 | .683 | .467 | .652 | .591 | .326 | .525 |
| Qwen3-8B LLM judge | 0.628 (3-level) | | | | | .111 (official first_neg1_idx_acc .407) | |

**AFTraj-2K** (trajectory AUROC, mean over steps, cellz; ansz mean is undefined)
| row | all | agentic | coding | math | injected (pooled) | diagnosed agentic | diagnosed math | loc (rnd .128) |
|---|---|---|---|---|---|---|---|---|
| best baseline (mean_ent) | 0.692 | .511 | .951 | .829 | .692 | .472 | .448 | .195 (best: mean_nll .243) |
| N16 L-SML cellz | 0.706 | .467 | .955 | .811 | .711 | .478 | .553 | .174 |
| N16 IU-PCR cellz | 0.705 | .466 | .955 | .812 | .708 | .476 | .549 | .184 |
| N16 IU-PCR ansz (loc only) | — | | | | | | | .175 |

Pooled "diagnosed vs safe of same domains" is 0.645 for N16 and 0.633 for mean_ent. That pooled figure is confounded, because it mixes agentic and math base rates. The per-domain diagnosed figures are the honest numbers.
