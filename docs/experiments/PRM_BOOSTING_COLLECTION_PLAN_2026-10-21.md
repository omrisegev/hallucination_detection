# PRM boosting: data collection plan before the advisor documents (2026-10-21)

**Who runs it:** the machine with AIRCC access and the external gold labels. Written on the Mac, which has
neither. Omri asked for this as the step that comes before writing two English documents for the advisors:
1. **Status snapshot:** what we measured and what is clear. It builds on the five-arm ablation page and its caveats.
2. **Competitor positioning:** PRISM, CROP, the PRMBench table of trained PRMs and critics; ARES, SLIDER and
   FUSE/Weaver as background.

The collection gives both documents the same per-step scores and the same metrics as each competitor's paper.
Nothing here needs a GPU or new inference. All runs are CPU, on telemetry that already exists.

---

## 0. Read first (context, in this order)

| What | Where | Why |
|---|---|---|
| This plan | here | |
| Decision: **GAP replaces Q80** as the per-answer flag count (Omri, 2026-10-21) | `results/error_count_label_free_v1/README.md`, `DECLARATION.md`, `gap_test/`, `gap_plots/` | New PRMScore numbers use GAP. Q80 stays as the historical rule; k=2 and ORACLE are references. |
| Decision: **ProcessBench is out of the thesis focus** | memory note; CLAUDE.md 2026-10-08 | Do not compute ProcessBench rows below. Existing PB results stay on record, marked "out of focus". |
| Decision: **pre-fix Qwen3 template data are not used** | `results/thinkfix_rescore_v1/README.md` | `results/sml_weight_estimator_v1/` sections 3 and 5 ran on the old template; ignore them. |
| Audit of which reported numbers still hold (8 wrong statements) | `docs/reviews/number_audit_2026-10-21/AUDIT.md`, `review/REVIEW.md` | Do not copy the wrong ones. |
| Competitor registry: tiers, protocols, metrics | `docs/research_notes/competitor_registry_2026-10-21/REGISTRY.md` | Section 2 below follows it. |
| Your own comparison mapping | branch `claude/prm-boost-comparison-papers-2026-10-21`, `docs/research_notes/prm_boost_comparison_papers_2026-10-21/` | Its figure list (PRISM Fig 5b, CROP Fig 3/4/8) is used below. ProcessBench items are dropped. Add ARES, VerifySteer, the trained-PRM/critic tables and the GAP numbers. |
| Paper-metric parity already done for Qwen-7B on PRMBench | `results/prmbench_paper_metric_parity_v1/README.md` + `common.py`, `prism_metrics.py`, `crop_metrics.py`, `slider_metrics.py` | Extend these scripts; do not rewrite them. |
| Sign-rule inversion and the PRM-weight mechanism | `results/sml_weight_estimator_v1/README.md` sections 1, 2, 4 | The ablation page needs the corrected wording. |
| Five-arm ablation and the Math-Shepherd digit fix | `results/prm_boost_ablation5_v1/` (`ablation5.py`, `ablation5_socratic.py`, `digitfix_v1/README.md`, `page/`) | Stage 1 reruns these with score export. |

**Rules that apply to everything below**
- **Label isolation.**
  - Scoring is label-free.
  - PRMBench labels may be read on AIRCC (the existing ablation5 evaluation does).
  - Socratic and Hard2Verify gold stays local, so their evaluation runs locally.
- **Count rules, defined exactly:**
  - GAP: sort the answer's answer-z scores in descending order. Over k = 1..min(ceil(n/2), n-1), flag the top
    k* = argmax(s_k − s_{k+1}). For n = 1, flag 1. The code is in `results/error_count_label_free_v1/banks_robustness.py:gap_counts`.
  - Q80: flag the steps above the pooled 80th percentile of answer-z scores.
  - k=2 and ORACLE (flag the true count) are references only.
- **The PRM reference uses the same rule as the fused arm.** Report the PRM on both scales, logit(1−reward) and
  (1−reward), because Q80 and GAP depend on the scale. Use the better one as the reference.
  - Qwen-7B PRMBench: logit Q80 65.82 / GAP 66.37; prob Q80 67.32 / GAP 68.01.
- **Every table row carries an access label:**
  - label-free fusion;
  - label-selected channels (LSML_CONT, S5);
  - supervised (LR analogs);
  - trained by its authors (the PRMs);
  - fixed control (HALF; it is a control, never a method).
- **Gates.** Run `/preflight` before any sbatch. Chain jobs linearly. Every output goes to a NEW directory, so
  existing results are never overwritten. Development data must be labelled as such.

---

## 1. Stage 1 on AIRCC (CPU): export per-step scores for every arm, label-free

Paths as in `results/prm_boost_ablation5_v1/digitfix_v1/run_digitfix.sh`:
`S=/shared/cycle2_tau_averbuch_prj/omrisegev1`, `R=$S/results`, `O=$R/prm_boost_gen_v1`, `M=$S/eval_mirror_prm_gen_v1`,
`X=$R/prm_boost_external_v1`, `C=$S/code_thinkfix_bank_553de79`. A draft job script is in
`results/prm_boost_collection_v1/aircc/run_collect.sh`; check every path during `/preflight`.

### 1.0 Ship and parity
- **Ship to the mirror `$M`** (copy files; do not sync the whole repo):
  - `results/prm_boost_ablation5_v1/ablation5.py`. Its new `ABL5_EXPORT_SCORES=1` export is off by default and
    changes no output.
  - `ablation5_socratic.py`.
  - `results/sml_weight_estimator_v1/socratic_sign/{nonneg_sml.py,run_prmbench.py,run_socratic.py}`.
  - `results/sml_weight_estimator_v1/socratic_chain_flip/diag_chain_socratic.py`.
- **The mirror's `fusion_utils.py`** already has `lsml_continuous_guarded` (digit-fix run, `guarded_append.py`).
  `rank1_offdiag_fit` is not needed here.
- **Parity gate:** rerun `ablation5.py qwen_prm_7b` with the export on, into a new directory. Its `ABL5_RESULTS_*.csv`
  must be byte-identical to `$O/ablation5_v1`. Stop if not.

### 1.1 PRMBench: all arms, 5 PRMs (main collection)
Run `ABL5_EXPORT_SCORES=1 python3 -B $M/ablation5.py <tag> <bank> <telemetry> $O/ablation5_export_v1` for:
- qwen_prm_7b: `$R/prm_boost_v1/bank_prm/FEATURES.npz`, `$R/prm_boost_v1/full_prm/prmbench_telemetry.pkl`
- qwen_prm_72b, pure_prm, rlhflow_prm: `$O/bank_<prm>/FEATURES.npz`, `$O/full_<prm>/prmbench_telemetry.pkl`
- math_shepherd: **`$O/bank_math_shepherd_digitfix/FEATURES.npz`**, which has the digit fix; `$O/full_math_shepherd/prmbench_telemetry.pkl`

**Outputs per PRM:** `ABL5_SCORES_<tag>.npz`. It holds row_ids, offsets, source_group, reward, native_error_flag,
position, log_step_tokens, and score__<arm> for every arm: PRM_native, PRM_Q80_PROB, TELE/MEAN/EQ/HALF/CHAIN/PRMG/P
variants for N13/N16/S5, the RES and PERM controls, POS_ONLY, LEN_ONLY, PRMPOS(LEN) and LSML_CONT.

**Parity:** each `ABL5_RESULTS` must equal the committed `ablation5_v1` file, or `ablation5_digitfix_v1` for Math-Shepherd.

### 1.2 PRMBench: sign-safe variant (diagnostic, post-registration)
- **Run:** `ABL5_EXPORT_SCORES=1 python3 -B run_prmbench.py $M abs <tag> <bank> <telemetry> $O/ablation5_signabs_v1` for the same 5 PRMs.
- **What it does:** |v| SML weights and no anchor flip.
- **Expectation:** this should change nothing where no weight was negative.
- **Purpose:** it is a control for the sign rule. It does not replace the recipe.

### 1.3 Socratic: all arms, 5 PRMs
- **Scoring:** `python3 -B $M/ablation5_socratic.py <bank> $X/collect_socratic_v1/SCORES_<prm>.npz`.
  - Banks: `$X/full_<prm>_socratic/bank_v2` for 4 PRMs, and **`bank_v3` for math_shepherd**, which has the digit fix.
  - Each run writes a label-free scores file.
- **Sign-safe variant:** `run_socratic.py <repo=$M> abs <bank> $X/collect_socratic_signabs_v1/SCORES_<prm>.npz`.
- **Sign diagnosis** (the label-free part only), on the fixed Math-Shepherd bank:
  `diag_chain_socratic.py --repo $M --bank $X/full_math_shepherd_socratic/bank_v3 --out $X/collect_socratic_v1/DIAG_ms_v3.json`.
  It answers whether the current N16 value, .427, is also the sign flip.

### 1.4 Hard2Verify: rebuild Math-Shepherd, then all arms for 5 PRMs
- **Rebuild:** `$C/scripts/prm_boost_build_bank_external_v2.py --raw $X/full_math_shepherd_hard2verify/external_telemetry.pkl`
  with `--tokenizer <Mistral tokenizer.json>` and `--out $X/full_math_shepherd_hard2verify/bank_v3`.
  - The audit found that the current Hard2Verify Math-Shepherd bank has zero digit channels.
  - The other 4 PRMs use 15..24 correctly, so their bank_v2 stays.
  - Record the digit ids in the MANIFEST.
- **Scoring:** `ablation5_socratic.py <bank> $X/collect_h2v_v1/SCORES_<prm>.npz`. Hard2Verify banks use the same
  external layout: FEATURES.npz + REWARDS.npz.
- **Parity:** on the overlapping arms (PRM, HALF_N16, PRMG_N16, CHAIN_N16) the scores must match the existing
  `$X/scores_hard2verify/SCORES_<prm>.npz` for the 4 unchanged PRMs.
- **Sign-safe variant:** as in 1.3.

### 1.5 Channel matrices for the supervised analogs (CROP's Trace+PRM row)
These are built in and written only when `ABL5_EXPORT_SCORES=1`. Each holds the N13 and N16 answer-z channel matrices
and their channel names, aligned to the same offsets as the scores, with no labels:
- `ablation5.py` writes `ABL5_CHANNELS_<tag>.npz` next to the scores.
- `ablation5_socratic.py` writes `<SCORES stem>_CHANNELS.npz`; set the variable in 1.3 and 1.4 too.

Smoke test on the Mac (Socratic qwen_prm bank_v2): the scores file is identical to the run without export
(26 arrays), and the channel file holds N13 (26,052 × 13) and N16 (26,052 × 16).

**Layout note:** `ablation5_socratic.py` and `run_socratic.py` resolve the repo as `parents[2]` of the script.
Copy `ablation5_socratic.py` to `$M/results/prm_boost_ablation5_v1/`, so it finds
`$M/results/prm_boosting_generalization_v1/` and `$M/results/prm_boosting_v1/fuse.py`, and pass `$M` as `<repo>`.

### 1.6 Return
- **Drive:** rclone everything to `gdrive:hallucination_detection/cluster_results/prm_boost_collection_v1/`.
- **Git:** commit a compact copy under `results/prm_boost_collection_v1/scores/`, float32:
  - key arms: PRM logit, PRM prob, TELE_N16, MEAN_N16, EQ_N16, HALF_N16, PRMG_N16, CHAIN_N16, CHAIN_N13, LSML_CONT,
    CHAIN_N16 sign-abs, POS_ONLY and LEN_ONLY;
  - ids, offsets, position, log_step_tokens and source_group.
  - Commit it only if the total stays under about 40 MB; otherwise use Drive only and list the paths.
- **Also commit:** the job scripts, logs, MANIFESTs and the parity table.

---

## 2. Stage 2, local (labels; external gold stays local): metrics and figures

Put one evaluator in `results/prm_boost_collection_v1/evaluate_collection.py`. It reads the stage-1 scores and
computes, for any arm and any count or threshold rule:
- the official metric: PRMBench via `spectral_utils.prmbench.prmbench_evaluate`; Socratic and Hard2Verify via
  `spectral_utils.family_external_metrics.summary`;
- within-answer AUC and pooled AUROC;
- FPR/FNR/TPR/TNR/Precision and Pos/Neg accuracy;
- per-category results and S/S/S (PRMBench);
- paired bootstrap CIs: by source group on PRMBench (2,000 draws), by answer on the external sets.

**Parity anchors (must reproduce before any table):**
- Qwen-7B PRMBench:
  - PRM native 65.49;
  - CHAIN_N16 Q80 68.25, GAP 70.40, k=2 70.80, ORACLE 75.40;
  - PRM prob Q80 67.32, GAP 68.01.
- All 5 PRMs, PRMBench: every `ABL5_RESULTS` Q80 PRMScore and within-AUC.
- Socratic: `results/prm_boost_ablation5_v1/socratic/RESULTS_socratic.csv` for 4 PRMs, and
  `digitfix_v1/socratic/RESULTS_socratic.csv` for Math-Shepherd. Also the fork's GAP − Q80 numbers for Qwen-7B and
  72B: CHAIN_N16 +1.37 and +1.01.
- Hard2Verify: `HARD2VERIFY_RESULT_2026-10-19.md` on branch `claude/prm-boost-external-2026-10-18`, on the
  177-answer common set.

### 2.1 For the status snapshot (document 1)
A table per benchmark (PRMBench, Socratic, Hard2Verify) × 5 PRMs:
- **Rows:**
  - PRM native;
  - PRM same rule (GAP), on the better scale;
  - TELE_N16, MEAN_N16, EQ_N16, CHAIN_N16, CHAIN_N13;
  - LSML_CONT, marked label-selected;
  - HALF_N16, marked as a control.
- **Columns:**
  - within-AUC;
  - PRMScore with GAP;
  - PRMScore with Q80 (historical), k=2 and ORACLE (references).
- **Paired contrasts with CIs:**
  - CHAIN − PRM, both under GAP;
  - CHAIN − EQ;
  - TELE − MEAN;
  - EQ − PRM (adding the telemetry at equal weight);
  - GAP − Q80 for each arm.
- **Ablation decomposition under GAP.** This is the content of the ablation page, recomputed:
  - the steps PRM → EQ → CHAIN and MEAN → TELE → CHAIN;
  - the rank table;
  - the controls RES, PERM, POS_ONLY, LEN_ONLY, PRMPOSLEN.
- **PRMBench categories and S/S/S** for PRM (GAP) vs CHAIN_N16 (GAP).
- **PRM weight share in CHAIN/PRMG fits**, from `ABL5_DIAG` and the Socratic json. Also the sign-flip count, and
  the sign-abs variant beside the shipped one.
- **The rebuilt ablation page:** extend `results/prm_boost_ablation5_v1/page/build_data.py` so PRMScore can be shown
  under GAP, with Q80 as an option. Fix its caveat cards from the audit: Hard2Verify did run, the Math-Shepherd
  numbers after the fix, and "5 of 20 (3 distinct)" sign flips.

### 2.2 For the competitor document (document 2)
For each item below: the metric exactly as the paper defines it, our rows for 5 PRMs × (PRM, PRM same rule, TELE,
CHAIN_N16, CHAIN_N13, LSML_CONT), the paper's rows copied from REGISTRY.md with the access label, and a CSV of the
data behind every figure.

**PRISM** (arXiv 2606.09078; same PRM, same metric)
- **Table 1:** Pos-Acc / Neg-Acc / PRMScore.
- **Table 2:** FPR / FNR / Precision / TPR / TNR / PRMScore / FPR reduction. Use both pools: the 6,211 non-control
  answers, and also with the 758 control rows. Their FNR and Precision only reproduce on the second pool.
- **Table 7:** 9 categories and S/S/S.
- **Matched-TPR rows at TPR 87.30.** This threshold is chosen with labels, so mark it as a reading.
- **Fig 5b, the most important figure:**
  - an FPR–FNR curve over a global threshold sweep for PRM, CHAIN_N16 and LSML_CONT;
  - PRISM's point (FPR 47.13 / TPR 87.30) marked on it;
  - our label-free operating points (Q80, GAP) marked on it.
- **Fig 7:** score histograms on error vs correct steps.
- **Fig 4:** within-answer pairwise accuracy by score margin.

**CROP** (arXiv 2605.30085; the closest idea, supervised; EMNLP 2026 Oral)
- **Table 6, PRMBench:** extend `crop_metrics.py`.
  - Their protocol: the first 5,000 Hub rows; seeds 2806-2815; 60/20/20 splits stratified by has-error; a fixed
    101-point grid plus the train-quantile-201 bridge.
  - Proxies: Random; position/length/format LR (supervised); Direct PRM; TELE_N16; CHAIN_N16; CHAIN_N13; LSML_CONT;
    Telemetry13+PRM LR (supervised, the analog of their Trace+PRM row, built from the 1.5 channel matrices).
  - Metrics: pooled AUROC, kept %, risk %, mean ± sd over seeds.
  - Run it for all 5 PRMs. CROP itself used Qwen-7B only, so the other 4 are our extension.
- **Fig 3:** kept vs α, for α in {.01, .02, .05, .10, .15, .20}.
- **Fig 4:** AUROC vs kept scatter, one point per proxy × PRM.
- **Fig 5:** risk and retention by trace-length bin.
- **Fig 8:** contamination by first-error position (early / mid / late).
- **Table 8:** α sensitivity.
- **Table 17-style cross-domain check:** calibrate on PRMBench, test on Socratic. This replaces ProcessBench.

**SLIDER** (arXiv 2610.00571; secondary)
- Extend `slider_metrics.py` to 5 PRMs: redundancy-class detection accuracy under the three readings already
  defined there, plus GAP.
- The paper's protocol is ambiguous. Say so.

**ARES** (arXiv 2507.12948, EMNLP 2025; training-free; secondary)
- Macro-F1 on PRMBench, with the threshold chosen by 5-fold CV on labels, as in their protocol. Do this for PRM and
  for our fused arms.
- Their subset size is not stated, so use the full set and mark the row approximate.
- In their own appendix the plain PRM (.749) is within one SD of ARES (.736). Do not write "beats".

**Leaderboard tables.** No computation; copy the numbers from REGISTRY.md and add our GAP rows:
- PRMBench: trained PRMs (Qwen-72B 68.2, PathFinder 67.7, R-PRM 66.8, ActPRM-X 66.7); critics (Gemini-2-thinking
  68.8, o1-mini 68.8 on 394 samples, GPT-4o 66.8); humans 83.8.
- Socratic: o3-mini 75.7; the best PRM is Qwen-7B at 68.0.
- Hard2Verify: step Balanced F1 (GPT-5 85.83; Qwen-PRM-7B 42.37), plus first-error F1 as in VerifySteer.
  Compute ours on the 177 common answers.

**Not computed:**
- Optimal Aggregation, which needs Best-of-N generation. Before it is cited, check which arXiv version gave 62.1
  vs 63.3.
- FUSE, Weaver, ScalePRM and uPRM (response-level or ProcessBench-only). Applying FUSE's aggregation rule to our
  step scores is optional, labelled as an adaptation.

### 2.3 Figures
PNG plus a CSV of the data for every figure, in `results/prm_boost_collection_v1/figures/`. Every figure caption
names the benchmark, the N, the count/threshold rule and the access of each series.

---

## 3. Optional, lower priority (only if time allows; not needed for the two documents)
- Rerun Step 467's weight-estimator bank study (`results/sml_weight_estimator_v1/processbench_banks/score_banks.py`)
  on the thinkfix_mask1 source bank (`$R/thinkfix_rescore_v1/source_bank_thinkfix_mask1/FEATURES.npz`), PRMBench
  only. It replaces the removed old-template sections 3 and 5.
- Math-Shepherd bank-sweep cells N14/S6 and the controls' N16 on the digit-fixed bank (the audit found them stale).

## 4. Deliverables and acceptance
- **Branch:** push to `claude/prm-boost-collection-v1`, cut from `claude/prm-boosting`. Do not merge; the Mac reviews first.
- **`results/prm_boost_collection_v1/README.md`** contains:
  - the parity table (every anchor in section 2: PASS or FAIL with the difference);
  - the job ids;
  - the file list;
  - for every number: its source file and its access label.
- **Acceptance:**
  - every parity anchor passes;
  - no label is used in scoring or in choosing a rule (except rows marked as label-chosen readings);
  - no ProcessBench rows;
  - no old-template data;
  - all scores carry their PRM and benchmark.
- **After it lands:** the Mac builds the two English documents from `results/prm_boost_collection_v1/` only.
