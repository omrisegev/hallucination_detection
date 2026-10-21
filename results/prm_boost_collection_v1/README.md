# PRM boosting collection v1 (2026-10-21)

**What this is.** The data collection that comes before the two English advisor documents (status snapshot,
competitor positioning). Plan: `docs/experiments/PRM_BOOSTING_COLLECTION_PLAN_2026-10-21.md`.
- Stage 1 ran on AIRCC (CPU only, label-free scoring). It re-exported the per-step scores of every arm for 5 PRMs on
  PRMBench, Socratic-PRMBench and Hard2Verify, plus a sign-safe variant.
- Stage 2 ran locally, where the external gold labels stay. It computes the metrics, tables and figures.
- No GPU, no new inference, no ProcessBench rows, no pre-fix Qwen3-template data. Labels are never used in scoring
  or to choose a rule. The exceptions are rows marked as label-chosen readings: the matched TPR point, CROP/SLIDER/ARES
  thresholds, the supervised LR rows, and the better PRM scale (see "Access labels").
- **Development data.** PRMBench has been inspected many times, and so have the external sets used here.
  Nothing in this directory is untouched confirmation.

**Bottom line**
1. **Every parity anchor passes.**
   - Stage 1: every `ABL5_RESULTS` file is byte-identical, and every external score array that was required to
     match does match.
   - Stage 2: 825 checked numbers, each with difference 0.0.
   - Competitor package: 57 numbers pass. The 6 LR numbers are informational, because the position/length
     features differ by construction.
2. **PRMBench (GAP rule).** CHAIN_N16 beats the PRM under the same rule for all 5 PRMs:
   - Qwen-7B +2.39 [+1.90, +2.90];
   - Qwen-72B +1.60 [+1.06, +2.12];
   - PURE +3.27;
   - RLHFlow +6.86;
   - Math-Shepherd +6.18.
   GAP adds about +2 PRMScore over Q80 for every arm.
3. **Socratic.** CHAIN_N16 loses to the PRM under the same rule for 4 of 5 PRMs. The exception is RLHFlow,
   +6.67 [+5.79, +7.61]. Math-Shepherd −16.09 is an inversion caused by the anchor flip (5 of 20 Socratic fits,
   see section 4). Qwen-72B −14.29 is not: its CHAIN_N16 fit is not flipped (corr(fit, TELE) +0.98) and the PRM
   gets 4% of the weight. Only its CHAIN_N13 fit is inverted. (Corrected 2026-10-21 by the Mac review.)
4. **Hard2Verify (177 answers).**
   - CHAIN_N16 helps the weak PRMs: RLHFlow +27.95, Qwen-72B +7.85.
   - It hurts PURE: −10.79 [−17.18, −4.62].
   - Qwen-7B (−1.55) and Math-Shepherd (+1.65) are within noise.

---

## 1. Stage 1 (AIRCC)

| item | value |
|---|---|
| shared path | `/shared/cycle2_tau_averbuch_prj/omrisegev1` (every input is here; `cycle3_tau_averbuch_prj/omrisegev1` holds only an unrelated dir). Account `cycle3_tau_averbuch_prj`, QoS `owner_940`, partition `cpu` |
| Python | `pyenv_native_312_20261011/activate.sh` (bare `python3` on the cpu nodes has no numpy: first submission 290474 failed on import, wrote nothing; 290475/290476 auto-cancelled) |
| jobs | **290477** (A: 1.0 + 1.1), **290478** (B: 1.2 sign-abs), **290479** (C: 1.3 + 1.4 + external parity); linear `afterok` chain |
| scripts | `aircc/common.sh`, `aircc/job_a_prmbench.sh`, `aircc/job_b_signabs.sh`, `aircc/job_c_external.sh`, `aircc/submit_chain.sh`, `aircc/parity_npz.py`, `aircc/upload_drive.sh`; `aircc/run_collect.sh` is the superseded Mac draft (its `cmp ... && echo` gate would not stop under `set -e`; fixed in job A) |
| logs | `aircc/logs/pbc_{a,b,c}_*.out`, `aircc/logs/MD5_*.txt`, `aircc/logs/parity/*.json` |
| mirror `$M` | `eval_mirror_prm_gen_v1`. Shipped (md5 verified): `ablation5.py` (old copy kept as `ablation5.py.bak_db0549266_2026-10-21`; the only diff is the off-by-default export), `results/prm_boost_ablation5_v1/ablation5_socratic.py`, `results/sml_weight_estimator_v1/socratic_sign/{nonneg_sml,run_prmbench,run_socratic}.py`, `results/sml_weight_estimator_v1/socratic_chain_flip/diag_chain_socratic.py`, and `results/prm_boosting_generalization_v1/run_v1/evaluate_amended.py` (byte-identical to the mirror's copy one level up; `ablation5_socratic.py` executes the `run_v1/` path, which was missing) |
| Mistral tokenizer | `$S/hf_cache/hub/models--mistralai--Mistral-7B-v0.1/snapshots/27d67f1b…/tokenizer.json` (exists, sha256 `11c08db2…`) |

**New output directories (nothing existing was overwritten):**
- `$R/prm_boost_gen_v1/ablation5_export_v1`
- `$R/prm_boost_gen_v1/ablation5_signabs_v1`
- `$X/collect_socratic_v1` and `$X/collect_socratic_signabs_v1`
- `$X/collect_h2v_v1` and `$X/collect_h2v_signabs_v1`
- `$X/full_math_shepherd_hard2verify/bank_v3`
- `$R/prm_boost_collection_v1` (the job directory)

### Stage-1 parity table

| gate | result | detail |
|---|---|---|
| 1.0 Qwen-7B export on vs `ablation5_v1` | **PASS** | `ABL5_RESULTS_qwen_prm_7b.csv` byte-identical (`cmp`) |
| 1.1 Qwen-72B, PURE, RLHFlow vs `ablation5_v1` | **PASS** | byte-identical; BOOTSTRAP and DIAG also identical |
| 1.1 Math-Shepherd (digit-fixed bank) vs `ablation5_digitfix_v1` | **PASS** | byte-identical |
| 1.3 Socratic, 4 PRMs with unchanged banks, vs frozen `scores_socratic` | **PASS** | PRM, HALF_N16, PRMG_N16 (= LSML_N16_PRMG): max abs diff 0 |
| 1.4 Hard2Verify, 4 PRMs with unchanged banks, vs frozen `scores_hard2verify` | **PASS** | PRM, HALF_N16, PRMG_N16, CHAIN_N16: max abs diff 0 |
| Math-Shepherd external (reported, not gated) | expected change | PRM and every N13 arm are identical. The N16 arms changed (the digit channels are no longer constant): H2V CHAIN_N16 max diff 1.44, Socratic PRMG_N16 1.91 |
| 1.4 Math-Shepherd Hard2Verify `bank_v3` | built | 177 records, 23 context-overflow rows skipped (as before). Mistral digit ids `[28734, 28740, 28750, 28770, 28781, 28782, 28784, 28787, 28783, 28774]` recorded in its MANIFEST (`matches_qwen_15_24: false`) |
| transfer AIRCC → local | **PASS** | md5 of all 25 + 31 + 74 files equal |

**Sign diagnosis (1.3).** Plan question: is Math-Shepherd's Socratic N16 value of .427 also the sign flip?
**Yes.**
- After the digit fix, Math-Shepherd CHAIN_N16 on Socratic correlates −0.99 with its own telemetry score (TELE_N16).
- The anchor flip toward z(PRM) inverted it. Details: `tables/PRM_WEIGHT_SHARE.csv`, and
  `DIAG_ms_v3.json` (Drive: `external/collect_socratic_v1/`).

---

## 2. Stage 2 (local): evaluator and parity

- **Code.** `evaluate_collection.py` holds the core: loaders, rules, the official metrics, rates, categories, and
  the paired bootstrap. It is driven by:
  - `collection_parity.py`
  - `collection_status.py` (2.1)
  - `collection_competitors.py` (2.2)
  - `collection_figures.py` (2.3)
  - `page_rule_data.py` (ablation page)
  - `make_compact_scores.py` (git copy)
- **Data root.** Set it with `COLLECTION_DATA`. On this PC it is `C:\Users\omris\TAU\hd_collection_data`, outside
  the repo.
- **Labels.**
  - PRMBench labels come from `results/advisor_completion_2026-10-11/inputs/prmbench_meta.json` (raw 1-based
    error_steps).
  - Socratic and Hard2Verify gold is `results/remote_claims_replay_2026-10-15/inputs/*_gold.json`.
- **Rules.**
  - **GAP** is copied verbatim from `banks_robustness.py:gap_counts`.
  - **Q80:** PRMBench uses `>`; the external sets use `>=`. Each keeps its own official code.
  - **K2 and ORACLE** are references only.
  - Hard2Verify uses the 177 answers that every PRM can read.
- **Do not vectorize `answer_z`.** On quantized rewards the GAP argmax hits exactly tied gaps, so a 1e-16 change
  flips k*. A `reduceat` version moved Qwen-7B PRM-prob GAP from 68.01 to 68.00. The code keeps the reference
  per-answer loop.

### Stage-2 parity anchors (`PARITY_STAGE2.csv`, every row; `COMPETITOR_PARITY.csv`)

| anchor | checked | result |
|---|---|---|
| A1 Qwen-7B PRMBench plan values: native 65.49; CHAIN_N16 Q80 68.25 / GAP 70.40 / k=2 70.80 / ORACLE 75.40; PRM prob Q80 67.32 / GAP 68.01; PRM logit Q80 65.82 / GAP 66.37 | 9 | **9 PASS** (exact to 2 decimals) |
| A2 every row of the committed `ABL5_RESULTS` for all 5 PRMs (PRMScore + within-AUC, 49 arms; Math-Shepherd from `digitfix_v1/digitfix/`); also exported == committed bytes | 495 | **495 PASS**, max diff 0.0 |
| A3 Socratic `RESULTS_socratic.csv` (4 PRMs; Math-Shepherd from `digitfix_v1/socratic/`), official + within-AUC; gap_test CHAIN_N16 Q80/GAP and GAP − Q80 (Qwen-7B +1.3705, Qwen-72B +1.0099) | 236 | **236 PASS**, max diff 0.0 |
| A4 Hard2Verify 177-answer common set vs `refs/hard2verify_common_177_RESULTS.csv` (branch `claude/prm-boost-external-2026-10-18`); n_common = 177 | 85 gated | **85 PASS**, max diff 0.0. Not gated: 30 skipped (N11/AUTO arms, not part of this collection), 16 Math-Shepherd N16 rows reported as INFO (8 unchanged, 8 changed by the digit fix: CHAIN_N16 .385 → .361, TELE_N16 .383 → .361) |
| PRISM Qwen-7B rates (FPR/FNR/TPR/TNR/Precision/PRMScore) vs `prmbench_paper_metric_parity_v1/PRISM_RESULTS.json` | 24 | **24 PASS** |
| CROP Qwen-7B AUROC/kept/risk vs `CROP_RESULTS.json` | 18 + 6 LR | **18 PASS**. The Telemetry13+PRM LR reproduces exactly too. The position/length LR differs (kept 11.43 vs 11.79), because its step lengths now come from each PRM's own tokenizer, not the Qwen3 counts in `population.npz` (marked INFO) |
| SLIDER Qwen-7B readings vs `SLIDER_RESULTS.json` | 15 | **15 PASS** |

The CROP check caught a real issue. My first subset used the alphabetical order of the ids, which gave Direct PRM
AUROC .8042 instead of .7999. `crop_metrics.py`'s `sorted(d)` sorts **integer** pkl keys, which is the Hub order.
`prmbench_meta.json` keeps that order (checked equal), and it is now used.

---

## 3. Tables for the status snapshot (2.1), `tables/`

| file | content |
|---|---|
| `STATUS_MAIN.csv` | per benchmark × PRM: PRM native, PRM on both scales and "PRM same rule (better scale under GAP)", TELE/MEAN/EQ/CHAIN_N16, CHAIN_N13, LSML_CONT, HALF_N16; within-AUC, pooled AUROC, official metric under GAP / Q80 / K2 / ORACLE, flag rates |
| `STATUS_CONTRASTS.csv` | paired bootstrap (2,000 draws; PRMBench by source group, external by answer): CHAIN_N16/CHAIN_N13/EQ/MEAN/TELE/HALF/LSML_CONT − PRM (GAP), CHAIN − EQ, TELE − MEAN, CHAIN − TELE, CHAIN − PRM native, GAP − Q80 for every arm |
| `ABLATION_GAP.csv` | PRM → EQ → CHAIN and MEAN → TELE → CHAIN under GAP; controls RES/PERM vs their base, POS_ONLY/LEN_ONLY/PRMPOS/PRMPOSLEN vs PRM |
| `ABLATION_RANKS.csv` | every arm ranked per cell under GAP and Q80 |
| `PRMBENCH_CATEGORIES.csv` | 9 categories + S/S/S: PRM (GAP, better scale), CHAIN_N16/N13 (GAP), PRM native |
| `PRM_WEIGHT_SHARE.csv` | PRM weight share in 60 CHAIN/PRMG fits (least-squares weights on [z(PRM), channels], R² ≥ .99), shipped vs sign-abs, with corr(fit, TELE) |
| `STATUS_META.json` | N per cell, the PRM reference scale chosen per cell, anchor flips |

**Cross-check.** The tables were computed twice: once without caches, once with per-arm caches. The six output CSVs are byte-identical. `STATUS_MAIN.csv` was then regenerated (`collection_status.py main_only`) after a row-copy fix. In the external "PRM same rule" rows, the native row shares the arm name `PRM`; contrasts were not affected.
| `PAGE_RULE_DATA.json` | ablation-page data under GAP (3 benchmarks) and Q80 (Hard2Verify) |

**Headline numbers.**
- Source: `tables/STATUS_MAIN.csv` and `tables/STATUS_CONTRASTS.csv`.
- Official metric on a 0-100 scale. Socratic and Hard2Verify are ×100.
- N: 6,211 PRMBench non-control answers, 2,995 Socratic answers, 177 Hard2Verify answers.

**CHAIN_N16 (GAP) minus the PRM under the same rule (GAP, better scale), 95% CI:**

| benchmark | Qwen-7B | Qwen-72B | PURE | RLHFlow | Math-Shepherd |
|---|---|---|---|---|---|
| PRMBench | +2.39 [+1.90, +2.90] | +1.60 [+1.06, +2.12] | +3.27 [+2.74, +3.80] | +6.86 [+6.26, +7.46] | +6.18 [+5.45, +6.96] |
| Socratic | −3.21 [−3.88, −2.49] | −14.29 [−15.18, −13.42] | −5.59 [−6.49, −4.72] | +6.67 [+5.79, +7.61] | −16.09 [−16.91, −15.31] |
| Hard2Verify | −1.55 [−6.97, +3.71] | +7.85 [+3.32, +12.58] | −10.79 [−17.18, −4.62] | +27.95 [+23.66, +32.76] | +1.65 [−3.75, +7.54] |

**Ablation steps under GAP**, as point estimates with 95% CI in `ABLATION_GAP.csv`.
- **PRMBench.**
  - EQ − PRM (adding the telemetry at equal weight) is positive for 4 of 5 PRMs; Qwen-72B is −0.49 [−1.08, +0.10].
  - CHAIN − EQ is positive for all 5 (+0.44 to +2.08).
  - TELE − MEAN is positive for all 5 (+0.61 to +1.58).
- **Socratic.** CHAIN − EQ is negative for Qwen-72B (−4.81), RLHFlow (−1.57) and Math-Shepherd (−8.62).
- **Hard2Verify.** Every CHAIN − EQ interval contains 0.

**GAP − Q80 for CHAIN_N16:**
- PRMBench: +2.02 to +2.33 for all 5 PRMs.
- Socratic: +1.37 / +1.01 / +0.52 / +0.09 / −0.35.
- Hard2Verify: about +5.7 for 4 PRMs, +0.98 for RLHFlow.

**PRM reference scale.** The better PRM scale under GAP is chosen after scoring, which favours the PRM. It is
1−reward for the 3 Qwen-family PRMs on PRMBench, and logit for RLHFlow and Math-Shepherd. The choice for every cell
is in `STATUS_META.json`.

## 4. Weight share and sign flips (2.1)
- **PRM weight share in CHAIN_N16** (`PRM_WEIGHT_SHARE.csv`). On Socratic it is .21 (Qwen-7B), .04 (Qwen-72B),
  .09 (PURE), .03 (RLHFlow) and .04 (Math-Shepherd). The same estimator on the same arms reproduces the 0-21% range
  of `sml_weight_estimator_v1` section 4.
- **The anchor flip fires in 5 of 20 Socratic fits.** These cover 3 distinct PRM × bank cells:
  - Math-Shepherd CHAIN_N13, PRMG_N13, CHAIN_N16 and PRMG_N16;
  - Qwen-72B CHAIN_N13.
  - In each one, corr(fit, TELE) is −0.97 to −1.00.
- **No flips elsewhere:** 0 of 20 on PRMBench and 0 of 20 on Hard2Verify.
- **Sign-abs removes all 5** (0 of 60).
- **Detector.** A negative PRM weight is not the right test: the flip toward z(PRM) makes the PRM weight positive
  by construction. The detector is corr(fit, TELE) < 0.

## 5. Competitor tables (2.2), `tables/COMP_*.csv`, data in `figures/data/`
Every row carries `access` (and `threshold_access` where the threshold uses labels).

- **PRISM** (`COMP_PRISM_T1_T2.csv`, `COMP_PRISM_T7_CATEGORIES.csv`, `COMP_PRISM_MATCHED_TPR.csv`, `COMP_PRISM_PAPER_ROWS.csv`).
  - Covers Tables 1, 2 and 7 for 5 PRMs, on both pools (non-control; with the 758 control rows).
  - Qwen-7B under GAP: CHAIN_N16 has FPR 45.06 and PRMScore 70.40, vs PRISM's 47.13 / 68.0.
  - Matched TPR 87.30 (a label-chosen reading): the untouched PRM gets FPR 47.70 / 68.26, and CHAIN_N16 gets
    47.78 / 68.20. **At matched TPR, label-free CHAIN is not better than the moved-threshold PRM on Qwen-7B or
    Qwen-72B.** It is better on PURE, RLHFlow and Math-Shepherd: 49.82 vs 52.24, 57.77 vs 77.14, and 57.28 vs 81.34.
- **CROP** (`COMP_CROP_T6.csv`, `COMP_CROP_ALPHA.csv` (Table 8 / Fig 3), `COMP_CROP_CROSS_DOMAIN_SOCRATIC.csv`, `COMP_CROP_PAPER_ROWS.csv`).
  - Covers 5 PRMs × 9 proxies × 7 alphas.
  - Qwen-7B at α = .05:
    - LSML_CONT keeps 25.1% and CHAIN_N16 keeps 23.3%.
    - The supervised Telemetry13+PRM LR keeps 24.8%. CROP's supervised Trace+PRM keeps 21.8%.
    - Direct PRM keeps 4.8% on the fixed grid and 17.2% on the quantile grid, so that number is a grid artifact.
  - **Cross-domain check (calibrate on PRMBench, test on Socratic).** The fused arms break the guarantee:
    - Qwen-7B CHAIN_N16 has risk 6.7% at α = .05 (keeps 19.1%);
    - Math-Shepherd CHAIN_N13 has 53.5%.
    - The PRM stays inside it but keeps little: Qwen-7B keeps 12.4% at risk 2.1%.
- **SLIDER** (`COMP_SLIDER.csv`).
  - Qwen-7B balanced accuracy with the label-free GAP rule: CHAIN_N16 71.2, TELE_N16 72.3, CHAIN_N13 72.4. The
    PRM under the same rule gets 54.7. SLIDER's Step-RRI is 61.91, but its protocol is ambiguous (APPROXIMATE).
  - Under the labelled LR-threshold plain-accuracy reading, the PRM on the 1−reward scale is **higher** than the
    fused arms on the Qwen PRMs: 78.8 vs 74.1 on Qwen-7B.
- **ARES** (`COMP_ARES.csv`).
  - Macro-F1 with a 5-fold CV threshold on labels, on the full PRMBench (APPROXIMATE).
  - Qwen-7B: PRM 68.17, CHAIN_N16 68.10, LSML_CONT 69.70.
  - With a label-tuned global threshold, CHAIN does not beat the PRM on the Qwen PRMs. It does on RLHFlow (64.13
    vs 55.79), on Math-Shepherd (64.34 vs 53.33) and on PURE (67.47 vs 66.24).
  - ARES's own appendix: PRM .749 vs ARES .736. Do not write "beats".
- **Hard2Verify** (`COMP_HARD2VERIFY.csv`).
  - Step Balanced F1 on the 177 common answers, plus the VerifySteer-style ErrorID F1.
  - **Under GAP, ErrorID F1 is 0 for every arm.** GAP always flags at least one step, so it can never answer "no
    error" on the 41 fully correct answers. ErrorID is meaningful only for rules that can abstain (native).
- **Leaderboards** (`COMP_LEADERBOARDS.csv`). The numbers are copied from `REGISTRY.md`, with our native and GAP
  rows added.
- **Not computed:**
  - Optimal Aggregation (needs Best-of-N generation);
  - FUSE, Weaver, ScalePRM and uPRM;
  - CROP's Token/format and Trace-feature proxies (need the step text and Llama-3.1-8B features).

## 6. Figures (2.3), `figures/` (PNG) + `figures/data/` (CSV) + `figures/CAPTIONS.md`
- PRISM:
  - `prism_fig5b_fpr_fnr` (5 PRMs);
  - `prism_fig7_score_histograms`;
  - `prism_fig4_pairwise_by_margin`.
- CROP:
  - `crop_fig3_kept_vs_alpha`;
  - `crop_fig4_auroc_vs_kept`;
  - `crop_fig5_by_length`;
  - `crop_fig8_by_first_error`.
- Status:
  - `status_official_by_prm_gap`;
  - `status_contrasts_gap`.

## 7. Ablation page
`results/prm_boost_ablation5_v1/page/` has two new controls:
- a **rule toggle**: GAP (the default), or the historical Q80;
- a **Hard2Verify tab**.

Data sources:
- GAP data and Hard2Verify come from `tables/PAGE_RULE_DATA.json`.
- Q80 for PRMBench and Socratic stays on the frozen ablation5 files.

The caveat cards were corrected per `docs/reviews/number_audit_2026-10-21/AUDIT.md`:
- Hard2Verify did run;
- the Math-Shepherd numbers are now the post-digit-fix ones;
- the anchor-flip count is now 5 of 20 Socratic fits, in 3 distinct PRM × bank cells.

Rebuild:

```
python build_data.py && python build_page.py
```

## 8. Scores
- **Git copy (correction 2026-10-21, Mac review).** The npz files below were NOT committed: `.gitignore` excluded
  `results/**/*.npz`, so only `scores/MANIFEST.json` reached git. An exception for `results/prm_boost_collection_v1/scores/`
  now exists. Re-add the files from the PC that has them (`git add -f` is no longer needed) and push. Until then they
  are on Drive.
- `scores/` is meant to hold a float32 copy of the key arms, 27.4 MB (under the 40 MB limit). It covers
  PRMBench / Socratic / Hard2Verify × 5 PRMs. Each file has ids/uids, offsets, the non-empty mask, position,
  log_step_tokens, source_group and native decisions, and no labels. The arms are PRM logit, PRM prob, TELE / MEAN /
  EQ / HALF / PRMG / CHAIN_N16, CHAIN_N13, LSML_CONT, CHAIN_N16 sign-abs, POS_ONLY and LEN_ONLY.
  `scores/MANIFEST.json` has the sha256 of every source file.
- **Full outputs.** The full stage-1 outputs, including the channel matrices, are on AIRCC (paths in section 1) and
  on Drive at `gdrive:hallucination_detection/cluster_results/prm_boost_collection_v1/`:
  - `prmbench/ablation5_export_v1` and `prmbench/ablation5_signabs_v1`;
  - `external/collect_*` and `external/full_math_shepherd_hard2verify/bank_v3`;
  - `jobs/`.
- **Upload problem.** The first two upload attempts found no `gdrive:` remote, because the account's HOME moved to
  the cycle3 path. `upload_drive.sh` now passes the cycle2 config with `--config`. See section 9 for the upload
  status.

## 9. Open points
- **Drive upload: done.** One transient upload error was retried and succeeded. `rclone check` passed for the
  PRMBench export. Log: `aircc/logs/upload_drive.log`.
- **Section 3 of the plan:** first left out as optional, then approved and run later the same day. See section 11.

---

## 10. CROP controls C1–C7 (plan §2.2 addendum, commit 20df0ecd; run 2026-10-21)
Code is in `crop_controls/`: `crop_controls.py` (AIRCC), `run_crop_controls.sh`, `submit_v3.sh` and
`crop_controls_report.py` (local). Raw per-seed rows: `crop_controls/raw_v3/`.

**Runs.**
- The preflight smoke jobs were 290559 and 290562. The vectorized version's CSV is byte-identical to the loop version's.
- **v1 (290564–290568) and v2 (290573–290577) failed parity and are not used.**
  - v1 built the earlier subset in FEATURES row order instead of Hub order.
  - v2 stratified the split by "an in-range step is labelled an error" instead of "error_steps is non-empty".
- **v3 (290585–290589) is the run of record.**

**Parity.**
- `CROP_CONTROLS_PARITY.csv`: the earlier-subset rows with the controls off reproduce `tables/COMP_CROP_T6.csv` for
  all 5 PRMs. 134 of 135 numbers agree within 1e-9. The remaining one is RLHFlow position/length LR AUROC, off by
  4e-9 (LR solver numerics); its kept and risk are exact.
- C4: the inductive machinery fitted on 100% of the steps reproduces the transductive scores to ≤ 3.6e-15, for all
  5 PRMs (`crop_controls/logs/CROP_CONTROLS_PARITY_*.json`).

**C1, exact subset.**
- The first 5,000 raw Hub rows hold 67,452 steps with an error rate of 15.86%. CROP reports 67,445 / 15.9%.
- CROP's 5 duplicate `classification_idx` rows are all multi_solutions: 85 steps, no errors, raw positions 226–340.
  Each has the same question as its twin but a different process. No telemetry or PRM score exists for them, and
  scoring them would need new inference.
- C1 therefore uses **4,995 scored traces** (67,367 steps, error rate 15.88%) and marks the 5 rows as unscored.

**Tables** (`tables/`):
- `CROP_C1_MAIN.csv`: C1 subset at α = .05, every proxy × grid.
- `CROP_C4_INDUCTIVE.csv`: inductive vs transductive fits.
- `CROP_C5_SHUFFLES.csv`: label and order shuffles.
- `CROP_C6_ALPHA.csv`: the α sweep, with the empty-prefix rate, infeasible seeds and the 1/(n+1) > α mark. The mark
  never fires (n_cal ≈ 1,000).
- `CROP_C7_BOUNDARY.csv`: boundary deviation.

Figure: `figures/crop_controls_c6_kept_vs_alpha.png`.

**C2 map.** u = clip((s − min_train)/(max_train − min_train), 0, 1), from the training-split steps. It is fixed in
advance and identical for every proxy. The train-quantile-201 grid is run as well, and so is CROP's raw fixed grid
for every proxy already in [0,1].

### Headline: kept % (risk %) at α = .05, quantile-201 grid, C1, mean of 10 seeds

| proxy | access | Qwen-7B | Qwen-72B | PURE | RLHFlow | Math-Shepherd |
|---|---|---|---|---|---|---|
| Direct PRM (1−reward) | trained PRM | 17.3 (3.9) | 20.6 (7.4)\* | 17.2 (4.8) | 8.3 (4.4) | 11.5 (4.2) |
| C3 Format/position LR | supervised | 16.7 (4.9) | (same) | (same) | (same) | (same) |
| C3 Format/position + PRM LR | supervised | 21.7 (4.7) | 21.3 (4.9) | 22.1 (4.8) | 17.8 (4.6) | 18.6 (5.0) |
| Telemetry13 + PRM LR | supervised | 25.1 (4.6) | 24.9 (5.1) | 25.2 (5.1) | 22.1 (4.7) | 20.6 (4.5) |
| CHAIN_N16, transductive | label-free fusion | 23.3 (4.8) | 22.0 (4.7) | 20.7 (4.8) | 16.3 (4.2) | 17.3 (4.9) |
| CHAIN_N16, inductive (C4) | label-free fusion | 23.2 (5.0) | 22.2 (4.7) | 20.7 (4.8) | 16.4 (4.2) | 17.4 (4.9) |
| LSML_CONT, inductive (C4) | label-selected channels | 25.3 (4.6) | 26.2 (4.8) | 21.3 (4.8) | 18.2 (4.2) | 16.8 (4.9) |

\* **Qwen-72B Direct PRM breaks the α guarantee: 7.4% risk at α = .05.**
- 58% of its rewards are exactly 1.0, so its risk score has a large tie at 0.
- Even the smallest λ keeps those steps, with 7.4% contamination.
- No λ is feasible for α ≤ .05 in any of the 10 seeds, and CROP's fallback (the smallest λ) does not abstain.
- The fused scores have no such ties.

**Findings**
- **C3 explains CROP's Trace+PRM row.** Our format/position + PRM LR keeps 21.7% on Qwen-7B. CROP's Trace+PRM keeps
  21.8%, and the format/position LR alone keeps 16.7% (their Token/format: 16.2%). CROP's gain over the PRM is
  matched by surface/position cues plus the PRM, without their Llama trace features.
- **C4: inductive ≈ transductive** for every PRM, within 0.4 points. Fitting our fusion on all answers including the
  test traces does not inflate the result.
- **C2: the fused lead is not a grid effect.**
  - On the fixed min-max grid, CHAIN_N16 drops by only 0.5–1.4 points (Qwen-7B 23.3 → 21.9).
  - Direct PRM drops to 4.9 on that grid. Its 4.8% in CROP's paper is a grid artifact.
- **The supervised Telemetry + PRM LR keeps the most** on every PRM (20.6–25.2). Label-free CHAIN_N16 reaches
  82–93% of it on the Qwen and PURE PRMs.
- **C5, label shuffle.**
  - With shuffled calibration labels, the label-free arms become *more* conservative: risk about 1.4% and kept
    roughly halved.
  - Supervised LRs refitted on shuffled labels lose kept (Telemetry LR 25.1 → 15.9) but stay near α.
  - So the certificate depends on real labels, but risk did not rise above α as the plan expected.
- **C5, order shuffle.** Kept falls to 5–11% for every proxy. Prefix utility depends on where in the trace the errors
  sit. The scores were computed in the original order and permuted together with their labels; re-scoring permuted
  text would need inference.
- **C7, boundary deviation |M−O|/T on Qwen-7B.** Direct PRM 36.4, Telemetry LR 28.4, LSML_CONT 28.3, CHAIN_N16 30.3.
  Almost all of it is over-withholding; unsafe overshoot is about 1 for every proxy.

## 11. Optional section 3 (approved by Omri 2026-10-21; job 290578)
Code: `section3/run_section3.sh` and `section3/score_banks_prmbench.py`. The latter is a PRMBench-only copy of
`sml_weight_estimator_v1/processbench_banks/score_banks.py`. That original ran on the old template and was
removed on 2026-10-21 (HISTORY Step 468); it is in git history up to commit b5c5c203. Next to it are
`rank1_fit.py` and `prm_label_contract.py`, both verbatim copies, because the AIRCC mirror lacks them.
Report: `section3/section3_report.py`. Raw outputs: `section3/results/`.

**Parity** (`SECTION3_PARITY.csv`): 82 of 82 pass, max diff 5.6e-16.
- 3b: every arm without digit channels (N11, N13, PRM_native, LSML_CONT) is unchanged. That is 62 numbers.
- 3a: LSML_CONT and ALL_EQ_ORIENTED reproduce `thinkfix_rescore_v1/thinkfix_mask1/cont_baselines/ABSOLUTE.csv` on all
  10 banks (20 numbers).

**3b, Math-Shepherd on the digit-fixed bank** (`tables/SECTION3_3B_SWEEP.csv`, `SECTION3_3B_CONTROLS.csv`; Q80).
Within-AUC and PRMScore, old → new:

| arm | within-AUC old → new | PRMScore old → new |
|---|---|---|
| CHAIN_N14 | .7019 → .7162 | 59.48 → 60.36 |
| TELE_N14 | .6872 → .6947 | 58.46 → 59.21 |
| PRMG_N14 | .6921 → .7077 | 58.70 → 60.11 |
| CHAIN_S6 | .7586 → .7648 | 63.57 → 63.86 |
| TELE_S6 | .7208 → .7239 | 61.46 → 61.68 |
| PRMG_S6 | .7389 → .7505 | 62.36 → 63.16 |
| controls HALF_N16 | .7640 → .7698 | 62.52 → 62.82 |
| controls LSML_N16_PRMG | .7133 → .7608 | 60.58 → 63.52 |
| controls LSML_N16_AUTO | .7285 → .7458 | 61.70 → 62.71 |

**3a, weight estimator on thinkfix_mask1, PRMBench within-AUC** (`tables/SECTION3_3A_BANKS.csv`). This replaces the
removed old-template sections 3 and 5. Values are old template → mask1.

| bank | LSML_CONT (shipped) | LSML_ABS_ORI | LSML_RANK1_ORI | ALL_EQ_ORIENTED (plain mean) |
|---|---|---|---|---|
| N13 | .7803 → .7661 | .7770 → .7655 | .7677 → .7557 | .7757 → .7641 |
| N16 | .7958 → **.7792** | .7919 → .7793 | .7762 → .7704 | .7906 → **.7811** |
| N25 | .7821 → .7740 | .7853 → .7733 | .7637 → .7711 | .7823 → .7757 |
| N40 | .7243 → .7393 | .7731 → .7666 | .7745 → .7609 | .7729 → .7606 |
| N43 | .7583 → .7616 | .7877 → .7751 | .7779 → .7638 | .7836 → .7724 |
| N56 | .7775 → .7705 | .7682 → .7687 | .7600 → .7500 | .7757 → .7731 |

- **On mask1, the plain oriented mean matches or beats shipped L-SML on 7 of 10 banks**, N16 included (.7811 vs
  .7792). On the old template, L-SML led on N16.
- The rank-one fit is below shipped L-SML on 7 of 10 banks. It is above on N22, N40 and N43, the banks where
  shipped L-SML is weakest.
- |v| weights help clearly only on N40 (+.027) and N43 (+.014), where shipped L-SML had negative weights. Elsewhere
  they are within ±.002.
