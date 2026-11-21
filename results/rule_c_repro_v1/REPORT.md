# Rule C reproduction on the PC, and H2V / DeltaBench data inventory (2026-11-21)

Rule C (agentS2): keep a GAP flag only where the global step score sg > tau. Tau is the (1 - beta) quantile of sg
over the GAP-flagged pseudo-clean steps of a synthetic V3/V4 matched-prefix pseudo-benchmark. Beta is chosen by the
pseudo-metric only.

Scope: CPU only. No GPU or AIRCC job was started. No file was deleted. The bundle originals are untouched.

## 1. Result in one line

Rule C reproduces on the PC. Every headline number (dev and test, PB and PRMBench) is equal to the Mac value.
The flags of rule C (dev and test) are bit-equal. The only difference is a small numeric drift in the answer score sC
(max 0.037, Spearman 0.9999999), which does not change any reported metric of rule C or of the pi-hat gate.

## 2. Setup

| Item | Value |
|---|---|
| Worktree | `C:\Users\omris\TAU\wt_rule_c_repro`, branch `claude/rule-c-repro-2026-11-21` from `4acda6f6`. Sparse checkout (C: had < 1 GB free): `spectral_utils/`, `results/ssl_one_experiment_v1/`, `results/prm_boost_processbench_v1/local/`, `results/rule_c_repro_v1/` |
| Bundle | `gdrive:hallucination_detection/cluster_results/ssl_step_threshold_v1_review/` (tar.gz 20,955,196 B + README_HE.md) |
| Originals (untouched) | `C:\Users\omris\TAU\rule_c_ws\originals\review_bundle\` |
| $WS (working copy) | `C:\Users\omris\TAU\rule_c_ws\run\`. Bundle outputs moved to `out_bundle/` inside the copy; runs write to fresh `out/` folders |
| $WS/hd | `spectral_utils/` + `results/ssl_one_experiment_v1/` of `4acda6f6` |
| $WS/W | `steps/`, `inputs/` = NTFS junctions to `hd_collection_data\ssl_one_experiment_v1\`; `processbench_gold.json` copied (byte-equal to the repo copy) |
| Path replace | Mac path -> `C:/Users/omris/TAU/rule_c_ws/run` in the working copy only (`scripts/setup_ws.sh`) |
| Python | numpy 2.2.4, scipy 1.17.1, scikit-learn 1.8.0 (Windows) |

### 2.1 Hash check (originals)

| File | sha256 (prefix) | FROZEN.md / FREEZE_HASHES.txt |
|---|---|---|
| agentS2/pseudo.py | 25a90edf...a64c | match |
| agentS2/out/RULES_processbench.npz | 4934635...1d4f | match |
| agentS2/out/RULES_prmbench.npz | 3347557c...8e32258 | match |
| agentS2/out/PSEUDO_LABELFREE.json | 8dd6437d...2d62 | match |
| phase1_labelfree.py | 3a90b89b... | match |
| phase2_evaluate.py | 4d6e8b64... | match |
| prmbench_phase1_labelfree.py | 9dc257b0... | match |
| prmbench_phase2_evaluate.py | 47916df8... | match |
| hd/spectral_utils/{fusion_utils, binary_view_estimation, error_count}.py (4acda6f6) | 82a29630 / 20be28ff / eae70e61 | match CODE_IDENTITY.txt |

Full list: `logs/00_hashes.txt`.

### 2.2 Code diff: bundle `ssl_ws/code/` vs worktree `results/ssl_one_experiment_v1/` (4acda6f6)

21 of 21 files are byte-identical (`ssl_data.py`, `ssl_lib.py`, `ssl_stages.py`, `run_analysis.py`, `make_variants*.py`,
`extract_steps.py`, ... ; `gate_dev.py`, `pair_*.py`, `verify_*.py` match their copies under `analysis_v1/`). No difference.

## 3. Reproduction: expected vs PC

| Step | Expected (Mac) | PC | Status |
|---|---|---|---|
| parity.py | PB 41.653, PRMBench 68.3277 | PB F1 41.65304, PRMB no gate 68.32767 (PASS) | match |
| pseudo.py: tau PB / PRMB | 1.40236 / 2.03531 | 1.4023637527655424 / 2.0353067886374325 (Mac ...426 / ...433, last digit) | match |
| pseudo.py: beta PB / PRMB | .51 / .46 | .51 / .46 | match |
| pseudo.py: flagsC (dev) | RULES_*.npz | bit-equal, both benchmarks | match |
| pseudo.py: openA, openB, alphas, pseudo_flags, pseudo_kerr | RULES_*.npz | bit-equal | match |
| pseudo.py: pseudo_sg | RULES_*.npz | max abs diff 7.5e-15 | match (float) |
| pseudo.py: pseudo_sA (rules A/B only) | RULES_*.npz | max abs diff 1.1e-3 | small drift (see 3.1) |
| eval.py dev rule C | PB 43.64, PRMB 68.91 | PB F1 43.636, PRMB 68.910 | match |
| eval.py all outputs | EVAL.json (470 leaves) | 467 equal; 3 differ, all in the label-using rule-A real curve diagnostic | match on headline |
| phase1 PB (test flags) | PHASE1_TEST_FLAGS.npz | flagsC, f0, flags_ref, open_pihat, half bit-equal; float scores differ in last digits | match |
| phase2 PB test | F1 43.34, +4.36 vs pi-hat | 43.3392, +4.3644 [+1.72, +6.88]; 129/130 leaves equal (only the npz hash differs) | match |
| PRMBench phase1/phase2 test | +0.39 [+0.0004, +0.79] vs no gate | +0.3924 [+0.0004, +0.7921]; 132/133 leaves equal (only the npz hash) | match |
| size_curve.py | SIZE_CURVE.json | 761/761 leaves equal | match |

### 3.1 Mismatches

1. **eval.py stops on a bitwise check.** Line 23 asserts `np.array_equal(LN['sC'], sC)` against the Mac's
   `agentX1/out/LN_SCORES_<bench>.npz`. On the PC, sC differs: max abs diff 0.0369 PB / 0.0142 PRMB (sd of sC 2.77 / 2.63),
   Spearman 0.99999986 / 0.99999995, rank order not identical (`logs/08_sC_drift.txt`). I ran `scripts/eval_pc.py`: the
   same file with only that check changed to print the difference, and the output renamed to `EVAL_pc.json`. The
   drift comes from the answer-level L-SML fit (platform floating point); s0, W, sg and the GAP flags are equal to 1e-9.
   Effect: none on rule C (it does not use sC). The pi-hat gate open sets give the same metrics on dev and on test.
2. **PSEUDO_LABELFREE.json:** 46 of 726 leaves differ, all in rules A / B (thresholds at the 4th-5th decimal, pseudo
   curve values at the 2nd decimal) and `pseudo_auc_answer_score_B` (1e-5). All come from the same sA drift. Rule C keys
   and the chosen alpha values of A and B are equal.
3. **EVAL diagnostics:** 3 values of the label-using post-evaluation curve of rule A differ (PB alpha 0: 0.288 vs 0.457;
   PRMB alpha .10: 57.739 vs 57.741). Same sA cause. Not a headline number.
4. **parity.py** writes `out/PARITY.json` relative to the current folder; from $WS this needs `$WS/out/` (I made it).

Diff listings: `logs/07_json_diffs.txt`.

## 4. Metric tables

Group-bootstrap 95% CI (2,000 draws, seed 20261118 dev; the phase-2 bootstrap on test). pi-hat gate = sC rank gate at the
label-free pi-hat (.580 PB dev / .5859 PB test; .704 PRMB dev / .6959 PRMB test). True-pi gate = sC rank gate at the
true wrong-answer share: a context arm that uses labels.

### 4.1 ProcessBench dev (1,695 answers; `logs/06_dev_table.json`)

| Arm | F1 [CI] | SLA [CI] | err acc | clean acc | 0-flag share |
|---|---|---|---|---|---|
| Rule C | **43.64** [40.58, 46.49] | 34.38 [31.17, 37.81] | 34.38 | 60.38 | .326 |
| No gate | 0.00 | 38.23 [34.86, 41.60] | 38.23 | 0.00 | 0 |
| pi-hat gate | 40.98 [37.38, 44.26] | 28.27 [25.17, 31.45] | 28.27 | 75.63 | .420 |
| True-pi gate (.646) | 41.65 [38.31, 44.75] | 30.30 [27.24, 33.56] | 30.30 | 67.68 | .354 |

| Difference (rule C minus) | F1 [CI] | SLA [CI] |
|---|---|---|
| No gate | +43.64 [+40.58, +46.49] | -3.84 [-5.92, -1.75] |
| pi-hat gate | +2.66 [+0.46, +4.98] | +6.11 [+4.12, +8.19] |
| True-pi gate | +1.98 [+0.11, +4.01] | +4.08 [+2.21, +6.08] |

### 4.2 ProcessBench test (1,705 answers, 1,419 groups; `outputs_json/PHASE2_TEST_RESULT.json`)

| Arm | F1 [CI] | SLA | err acc | clean acc | 0-flag share |
|---|---|---|---|---|---|
| Rule C | **43.34** [40.21, 45.87] | 34.49 | 34.49 | 58.64 | .306 |
| No gate | 0.00 | 38.21 | 38.21 | 0.00 | 0 |
| pi-hat gate | 38.97 [35.37, 42.15] | 26.66 | 26.66 | 73.62 | .414 |
| True-pi gate (.660) | 41.06 [37.56, 43.99] | 30.27 | 30.27 | 64.62 | .340 |

| Difference (rule C minus) | F1 [CI] |
|---|---|
| pi-hat gate (primary) | **+4.36 [+1.72, +6.88]** |
| No gate | +43.34 [+40.21, +45.87] |
| True-pi gate | +2.28 [+0.27, +4.26] |

Halves: A +3.64 [+0.19, +7.10], B +5.97 [+1.70, +10.24] vs pi-hat. Phase 2 does not compute SLA CIs on test.

### 4.3 PRMBench PRMScore

| Arm | Dev [CI] (3,499 answers) | Test [CI] (3,470 answers, 353 groups) |
|---|---|---|
| Rule C | **68.91** [68.01, 69.90] | **68.56** [67.61, 69.46] |
| No gate | 68.33 [67.47, 69.26] | 68.17 [67.32, 69.00] |
| pi-hat gate | 65.81 [64.94, 66.73] | 65.62 [64.79, 66.45] |
| True-pi gate | 67.76 [66.89, 68.65] (.865) | 67.75 [66.88, 68.59] (.867) |
| 0-flag share, rule C | .176 | .176 |

| Difference (rule C minus) | Dev [CI] | Test [CI] |
|---|---|---|
| No gate | +0.58 [+0.13, +1.03] | **+0.39 [+0.0004, +0.79]** |
| pi-hat gate | +3.11 [+2.57, +3.68] | +2.94 [+2.36, +3.50] |
| True-pi gate | +1.15 [+0.71, +1.56] | +0.81 [+0.43, +1.19] |

Test halves vs no gate: A +0.65 [+0.09, +1.28], B +0.14 [-0.39, +0.69].

Note: dev numbers carry the known selection optimism (rule C was picked among 15 rules on dev labels, README_HE.md 2).

## 5. Label-use audit

Search terms: `lab`, `err`, `y_ans`, `keep`, `cls`, `wrong_step`, `clean_step`, `gold`, `labels` in pseudo.py, eval.py,
supp.py, size_curve.py, phase1_labelfree.py, prmbench_phase1_labelfree.py, and the shared loader agentX1/common.py, then
the functions on the tau path.

| File:line | Access | On the tau / beta / flag path? |
|---|---|---|
| pseudo.py:3 | docstring | no |
| pseudo.py:61-62 | `err` = pseudo error at k of the child (synthetic), `keep = np.ones` | yes, but label-free (synthetic k from the sidecar) |
| pseudo.py:99 | `D.load_pool(WDIR, bench, V, split)` for V3/V4: no gold, no labels passed | label-free |
| pseudo.py:47, 92 | reads only `B['ev0']['W']` from step_eval | label-free |
| common.py:30-32 (load_dev) | loads gold + PRMBench labels into the pool object | loaded, see next row |
| ssl_stages.py:151-156 (step_eval, called by load_dev) | computes dev F1/SLA (PB) or PRMScore with `C.keep` (PRMB) and per-channel AUROC from `wrong_step`/`clean_step` into `ev0` | **computed but not read**: W, s0, f0 come from `lsml_linear(X)` and `gap_flags` before any label use; no rule script reads `ev0` keys other than `W` |
| ssl_stages.py:220-226 (AnswerModel), run_analysis.py:136 (feats) | sign by corr with s0, ECDF, RankGauss, L-SML | label-free |
| pseudo.py:93-94 | sg orientation = sign corr(sgC, s0) | label-free |
| eval.py:61-81 | `Cd.lab`, `Cd.err`, `Cd.y_ans` | evaluation only (post-evaluation diagnostics) |
| eval.py:28 (`RA.score_flags`) | labels inside the scorer | evaluation only |
| supp.py:44-46 | `keep` = local variable (flag keep share), not the label field | label-free |
| size_curve.py:67-68, 113-114 | `err` = pseudo error (synthetic k) | label-free |
| size_curve.py:93-99 | loads gold + labels for the test pool | evaluation only (dev/test curves) |
| phase1 (PB):47-49, prmbench_phase1:34-36 | loads gold + labels with the test pool (`load_pool`) | loaded; the flag code reads only telemetry (`rawT`, `act`, `off`) |

Conclusion: no label enters tau, beta, sg orientation or the flags. Labels are loaded into memory by `load_dev` and by
the phase-1 test pool, and `step_eval` computes labelled dev metrics into `ev0`; these values are not used by the rule.
The known open points stay: rule C was selected on dev labels among 15 rules, and phase 1 / phase 2 ran in the same
session without an external timestamp (README_HE.md 3).

## 6. Part B: Hard2Verify and DeltaBench inventory (no runs)

### 6.1 What exists (standalone telemetry, mode `sa`)

Stored standalone cells: `prm_boost_crop_ares_benchmarks_v1/full_sa_<prm>_<bench>`. The `<prm>` name gives the base
model of the PRM. For Qwen2.5-Math-7B the cell is `qwen_prm` (model `Qwen/Qwen2.5-Math-7B-Instruct`, rev ef9926d7,
prompt `qwen_cot_chat`, top-50 logprobs): the same model and driver as the ssl_one_experiment C runs.

| Benchmark | Cell | Base model | Answers in bank / total | Steps | Failed rows |
|---|---|---|---|---|---|
| Hard2Verify | full_sa_qwen_prm_hard2verify | Qwen2.5-Math-7B-Instruct | 181 / 200 | 1,738 | 19 (18 context > 4,096 tokens + 1) |
| Hard2Verify | full_sa_pure_prm_hard2verify | Qwen2.5-Math-7B (base) | 182 / 200 | 1,744 | 18 |
| Hard2Verify | full_sa_qwen_prm_72b_hard2verify | Qwen2.5-Math-72B-Instruct | 181 / 200 | 1,738 | 19 |
| Hard2Verify | full_sa_rlhflow_prm_hard2verify | Llama-3.1-8B-Instruct | 200 / 200 | 1,860 | 0 |
| Hard2Verify | full_sa_math_shepherd_hard2verify | Mistral-7B-v0.1 | 177 / 200 | 1,703 | 23 |
| DeltaBench | full_sa_qwen_prm_deltabench (native 4,096) | Qwen2.5-Math-7B-Instruct | 508 / 1,236 | 4,886 | 728 (context) |
| DeltaBench | **full_sa_qwen_prm_deltabench_ext** (native + RoPE dynamic NTK x3) | Qwen2.5-Math-7B-Instruct | **1,236 / 1,236** | 20,324 | 0 |
| DeltaBench | full_sa_pure_prm_deltabench_ext | Qwen2.5-Math-7B (base) | 1,236 / 1,236 | 20,324 | 0 |
| DeltaBench | full_sa_qwen_prm_72b_deltabench_ext | Qwen2.5-Math-72B-Instruct | 1,236 / 1,236 | 20,324 | 0 |
| DeltaBench | full_sa_rlhflow_prm_deltabench(_ext) | Llama-3.1-8B-Instruct | 1,236 / 1,236 | 20,324 | 0 |
| DeltaBench | full_sa_math_shepherd_deltabench | Mistral-7B-v0.1 | 447 / 1,236 | 3,924 | 789 (no ext cell) |

Formats and places:

| Data | Format | Where |
|---|---|---|
| Raw per-token telemetry (top-50 ids/logprobs, entropies, spilled energy, logsumexp, gen ids, step spans) | `external_telemetry.pkl`, `grafted.pkl` (H2V qwen 124 MB; DeltaBench native 432 MB; DeltaBench ext 2.33 GB) | Drive `cluster_results/prm_boost_crop_ares_benchmarks_v1/full_sa_<prm>_<bench>/` (and AIRCC `$R/prm_boost_crop_ares_benchmarks_v1/`) |
| Step bank (56 columns incl. TELE16 inputs, digit_active, offsets, uids) | `bank_v3/FEATURES.npz` + REWARDS.npz + MANIFEST.json (`labels_read: false`) | PC `hd_collection_data/c4u_r1_inputs/crop_ares_all/full_sa_*`; Drive same folders; H2V also `c4u_r1_external_inputs_v1/hard2verify/<prm>/` |
| Standalone readouts (H2V only) | `data_export/readout_steps_sa_v1/hard2verify__<prm>.npz` | repo (4acda6f6). No DeltaBench file there |
| Step texts | DeltaBench `inputs/deltabench/answers.json` (1,236 rows: uid, question, steps, generator); H2V `run_identity.json` (`row_ids`, `questions_steps`, 200 rows) | Drive `prm_boost_crop_ares_benchmarks_v1/inputs/` and `full_sa_qwen_prm_hard2verify/`; PC `hd_collection_data/crop_ares_v1/inputs_text/deltabench/` |
| Labels | `{uid, correct[], include[], category}` per answer (H2V also `out_of_range_error_indices`) | PC `hd_collection_data/crop_ares_v1/gold/hard2verify.json` (200), `deltabench.json` / `deltabench_ext.json` (1,236); repo `results/remote_claims_replay_2026-10-15/inputs/hard2verify_gold.json` |
| Scores | SCORES_*.npz / _CHANNELS.npz | PC `hd_collection_data/h2v/`, `frozen_h2v/`, `crop_ares_all/scores/` |

Label facts: H2V 200 answers, 79 questions (1-4 answers each), mean 9.3 steps, 158 answers with an error. DeltaBench
1,236 answers, 1,217 questions (only 19 questions have 2 answers), mean 16.4 steps, 1,173 answers with an error (95%);
generators: Omni-MATH 448, code (Codeforces 117, FullStackBench 97), GPQA 93, OlympiadBench 68, KOR-Bench 60, others.

Other H2V telemetry (different model, not standalone Qwen2.5-Math): Qwen3-8B thinkfix cell
`dataset_cache/lsml_external_thinkfix/hard2verify_qwen3_8b_full_thinkfix_153589e/` (metadata only on the PC).

### 6.2 Can the ssl_one_experiment pipeline run on H2V and DeltaBench?

| Step | Needs | H2V | DeltaBench |
|---|---|---|---|
| C run (telemetry) | the stored standalone run | exists (181/200; the 19 long rows need the RoPE x3 run, which was done only for DeltaBench) | exists (`_ext`, 1,236/1,236) |
| C bank | PB-style build (graft_rewards.py + prm_boost_build_bank_external_v2.py) | exists (bank_v3) | exists (bank_v3 of `_ext`) |
| extract_steps.py | raw pkl + bank FEATURES; gate top10 == bank | can run on CPU now (pkl on Drive, 124 MB) | can run on CPU (pkl 2.33 GB; local disk too small now: C: < 1 GB free) |
| Split (SPLIT.json) | question groups, dev/test | **missing**: new split by question needed (79 groups) | **missing**: 1,217 groups |
| ssl_data.load_pool labels | PB gold (first error, 0-based) or PRMBench labels (error_steps, classification) | **missing adapter** from `{correct, include}`; out_of_range_error_indices handling | **missing adapter**; `include` mask; no 'correct' class concept as in PRMBench keep |
| make_variants_v34.py | `--pb-answers` / `--prmb-meta` inputs; hard-coded to PB and PRMBench (`bench` names, rng seeds, file names) | **code change needed** (new bench branch) | **code change needed** |
| V3 (numeric edit) | editable number in steps 2..n | likely most answers (math) | math rows yes; code rows (Codeforces, FullStackBench) edit code digits: check realism |
| V4 (on-topic donor) | another answer to the SAME question | possible: 195/200 answers have a sibling | **not possible for 1,198/1,236 answers** (no sibling); about 19 pairs only |
| GPU driver | run_prmbench_prm_telemetry.py `--answers` (external label-free answers.json) | supports H2V format | supports; needs `--rope-extension dynamic_ntk x3` (branch claude/deltabench-long-context-2026-11-08) for 726 rows > 4,096 tokens |
| Bank for variants | same PB-style builder | ok | ok |
| Official metric | H2V step Balanced F1 (in evaluate_cab.py) | ok | no official DeltaBench protocol (see 6.4) |

### 6.3 Size and GPU time estimate (C + V3 + V4)

Reference: PB/PRMBench SSL jobs on B200: PB C 580 s wall (forward 249 s for 3,400 rows, bank 199 s, extract 84 s);
PRMBench C 931 s (forward 398 s, bank 237 s, extract 108 s). Standalone forward on B200 (stored cells): H2V 17.7 s for
200 rows (182 run), DeltaBench native 64 s (510 rows), DeltaBench RoPE x3 294 s (726 long rows).

| Benchmark | C | V3 rows (dev only, as PB/PRMB) | V4 rows | GPU jobs | Estimate per job | Total |
|---|---|---|---|---|---|---|
| Hard2Verify | reuse stored (0 GPU); 19 long rows need 1 short RoPE job if wanted | <= about 100 | <= about 97 | 2 (V3, V4) (+1 for long C rows) | < 10 min (forward < 1 min; bank + extract dominate) | about 20-30 min GPU, 1 queue wait |
| DeltaBench | reuse `_ext` (0 GPU) | <= about 618 (fewer after skips) | about 10-20 (no siblings) | 1-2 | 12-15 min (forward 2-5 min with RoPE x3; bank about 3 min for about 10k steps; extract about 2 min) | about 15-30 min GPU |

These are estimates from the stored timings, not measurements. The split counts assume a 50/50 question split.

Risks for rule C on these benchmarks: (1) H2V is small: about 100 dev answers give about 100-200 pseudo pairs (PB had
2,111); size_curve.py shows tau is unstable below about 400 pairs. (2) DeltaBench has 95% wrong answers: the
pseudo-clean prefixes are heavily contaminated, and V4 is almost empty, so the pseudo-benchmark is V3-only
(size_curve.py: PRMBench V3-only gives a worse tau than V4-only). (3) Both benchmarks need the label adapters and a split
before any evaluation.

### 6.4 Official DeltaBench evaluation (mu-sigma threshold)

None in the repo. I searched branch `4acda6f6`, `claude/deltabench-long-context-2026-11-08`,
`claude/c4u-tele16-all-benchmarks-2026-11-08`, `claude/prm-boosting` and all local refs for DeltaBench code with a
sigma / z-score / mean-std threshold. DeltaBench is scored with the project's own rules in
`results/prm_sub5_all_benchmarks_v1/vendor/evaluate_cab.py` (= `results/prm_boost_crop_ares_benchmarks_v1/evaluate_cab.py`:
within-AUC, pooled AUROC, Macro-F1 with native <= 0.5 / Q80 / GAP / ORACLE rules) and in
`results/ares_protocol_v1/ares_protocol.py` (ARES readout, 447-answer common set). The official mu - sigma protocol must
be written new.

## 7. Files in this folder

| Path | Content |
|---|---|
| `logs/00_hashes.txt` | sha256 of originals and of the PC outputs |
| `logs/01_parity.txt` ... `05_size_curve.txt` | console output of every run |
| `logs/03_eval.txt` | the failed original eval.py (bitwise sC assert) |
| `logs/03_eval_pc.txt` | eval_pc.py output |
| `logs/06_dev_table.json` | dev table (section 4) |
| `logs/07_json_diffs.txt` | leaf-by-leaf diffs of all JSON outputs vs the bundle |
| `logs/08_sC_drift.txt` | PC vs Mac sC |
| `outputs_json/` | JSON outputs of the PC runs (npz not committed) |
| `scripts/setup_ws.sh` | how $WS was built |
| `scripts/eval_pc.py` | eval.py with the bitwise sC check relaxed (PC path) |
| `scripts/dev_table.py` | dev arms incl. the true-pi gate |
| `scripts/jsondiff.py`, `scripts/sc_check.py` | comparison helpers |
