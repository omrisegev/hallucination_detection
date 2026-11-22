# Rule C on Hard2Verify and DeltaBench (2026-11-22)

**Result: rule C does not transfer.**
- On Hard2Verify, rule C is worse than the method of record (GAP, no gate): **-18.07** official step Balanced F1
  [-23.17, -13.51].
- On DeltaBench, the frozen label-free fit selects beta = 0, so rule C flags no section and scores **0.00** official
  macro-F1. The method of record scores 31.09.

The method itself is unchanged: the agentS2 code, the beta grid, the tie rule, the quantile, sg and GAP. Only benchmark
adapters were added. The protocol is `PROTOCOL.md` (frozen before any child telemetry; amendment 1 was added before any
rule C output). There was one labelled evaluation per benchmark. I deleted no file; failed job folders are kept under
`*.failed_<job>` names.

## 1. Rule C fit (label-free, `code/pseudo_ext.py`)

| | Hard2Verify | DeltaBench |
|---|---|---|
| C pool (CR = C re-run in the children's container) | 181 answers, 1,738 steps, 78 question groups | 1,236 answers, 20,324 steps, 1,217 groups |
| Children | V3 174, V4 141 | V3 1,208 (no V4: only 19 questions have 2 answers) |
| Pseudo pairs | **315** | **1,208** |
| Pseudo-metric | PRMScore, keep = ones (pseudo.py) | official macro-F1, pseudo-clean scored "empty=1" |
| beta / tau | **.46 / 1.9390** | **0.00 / 6.1890** (tau = max: flag nothing) |
| Pseudo-metric at the choice / with all GAP flags kept | 55.50 / 50.92 | 50.00 / 11.69 |
| Real share of flags kept | .495 | 0 |
| Real 0-flag share | **.348** | **1.000** |
| sg vs s0 correlation (orientation) | +.70 | +.80 |
| Pseudo-wrong exact GAP hit at k | .241 | .142 |

Why DeltaBench degenerates: under "empty=1", every flag on a pseudo-clean prefix costs a full point. The GAP hit rate
at the corrupted step is only .142, so no threshold gains more on pseudo-wrong prefixes than it loses on pseudo-clean
ones. The maximum is "flag nothing" (50 = all clean prefixes correct, all wrong prefixes 0). On the real data 95% of
answers have an error section, so 0 flags gives F1 = 0.

H2V tau stability (label-free subsamples of pseudo pairs, 200 draws, `outputs/SIZE_CURVE_hard2verify.json`):

| pairs | beta 5/50/95% | tau 5/50/95% |
|---|---|---|
| 25 | .08 / .38 / .71 | 0.30 / 2.27 / 4.25 |
| 50 | .18 / .41 / .64 | 0.84 / 2.04 / 3.35 |
| 100 | .21 / .43 / .61 | 1.22 / 1.97 / 3.33 |
| 200 | .34 / .46 / .55 | 1.41 / 1.93 / 2.65 |
| 315 (all) | .46 | 1.939 |

## 2. Labelled evaluation (once, `code/eval_ext.py`; 95% CI by question-group bootstrap, 2,000 draws)

### 2.1 Hard2Verify, own population (181 answers, 78 groups)

| Arm | Official step Balanced F1 | PRMScore form | 0-flag share |
|---|---|---|---|
| **Rule C** | **29.32** [24.68, 33.85] | 49.74 [46.72, 52.84] | .348 |
| Method of record (GAP, no gate) | **47.39** [43.92, 50.73] | 55.69 [53.42, 57.95] | 0 |
| Qwen2.5-Math-PRM-7B native / GAP | 16.27 / 45.94 | 44.55 / 55.64 | |
| Qwen2.5-Math-PRM-72B native / GAP | 31.58 / 35.92 | 50.49 / 49.53 | |
| PURE-PRM-7B native / GAP | 51.37 / 50.72 | 58.55 / 58.39 | |
| RLHFlow-PRM-Mistral-8B native / GAP | 5.99 / 4.93 | 35.80 / 35.15 | |

| Rule C minus | Balanced F1 [CI] | PRMScore [CI] |
|---|---|---|
| **Method of record (primary)** | **-18.07 [-23.17, -13.51]** | -5.95 [-8.95, -3.12] |
| PRM-7B native | +13.05 [+7.32, +18.65] | +5.19 [+1.96, +8.20] |
| PRM-7B GAP | -16.61 [-22.39, -10.67] | -5.89 [-9.37, -2.38] |
| PRM-72B native | -2.26 [-7.91, +3.22] | -0.75 [-3.57, +2.19] |
| PURE native | -22.05 [-29.01, -15.06] | -8.80 [-13.05, -4.38] |
| RLHFlow native | +23.33 [+18.01, +28.75] | +13.94 [+10.88, +16.96] |

On the common population (177 answers, all 5 PRMs, Math-Shepherd included):
- Rule C 29.20 vs method of record 47.89. Difference -18.69 [-23.83, -13.67].
- Math-Shepherd native / GAP: 7.87 / 40.12.

### 2.2 DeltaBench (1,236 answers, 1,217 groups), official section macro-F1

| Arm | Union target (official script) [CI] | Error-only target | 0-flag share |
|---|---|---|---|
| **Rule C** | **0.00** [0.00, 0.00] | 0.00 | 1.000 |
| Method of record, GAP flags | **31.09** [29.14, 33.11] | 31.68 | 0 |
| Method of record s0 under mu-sigma (s0 > mu + sigma) | 26.06 [24.17, 28.03] | 26.74 | .089 |
| Qwen2.5-Math-PRM-7B mu-sigma / GAP | 29.34 / 32.69 | 28.01 / 32.61 | .201 / 0 |
| Qwen2.5-Math-PRM-72B mu-sigma / GAP | 29.95 / 32.99 | 29.97 / 33.30 | .195 / 0 |
| PURE-PRM-7B mu-sigma / GAP | 22.40 / 25.48 | 23.32 / 26.38 | |
| RLHFlow-PRM-Mistral-8B mu-sigma / GAP | 14.46 / 10.60 | 13.90 / 10.77 | |

Rule C minus the method of record (GAP): **-31.09 [-33.11, -29.14]**. Rule C is below every arm.

Paper check (sec. 4 of the interim report, unchanged): with this evaluator, PRM-7B mu-sigma = 29.34 vs the paper's
29.22. The 72B gives 29.95 vs the paper's 26.38, and the appendix 5% quantile rule and HitRate@1 do not reproduce.

## 3. Parity and checks

| Check | Result |
|---|---|
| Model snapshot on cycle4 | every file equals the Hugging Face tree of ef9926d7 (sha256 shards, sha1 others); tokenizer sha256 c0382117 = stored runs |
| Builder environment: H2V bank rebuilt from the stored grafted.pkl vs the stored bank_v3 | PASS: the 11 token channels bit-identical, 5 step channels <= 1.6e-15, structure exact. My first check was too strict on one unused column, see section 4 |
| Method of record on the stored H2V C table (PC) | 47.2247 = stored SUB5_ALL 47.2247 (exact) |
| Method of record on CR (container re-run), H2V common | 47.89 (stored 47.22; CR is a different environment) |
| PRM-7B GAP on H2V common (stored PRM scores) | 45.82 = stored 45.8199 |
| CR vs stored C, 16 raw channels, Spearman (`outputs/PARITY_CR_vs_stored_*.json`) | H2V: min .960 (top15_turnover), dominant_freq16 .974, others >= .996. DeltaBench: min .993 |
| Smoke S1 / S2 (stop rules) | PASS in all 5 child/CR jobs. S2 equal prefix: 12 of 12 pairs, max diff 0.0 (H2V V3, V4, DeltaBench V3) |
| Smoke S3 (report only, 12 rows vs stored top10) | H2V min Spearman .91 / .86 (top15_turnover); expected >= .99. This is the reason for amendment 1 |
| extract_steps GATE | PASS on all 7 tables (C, CR, V3, V4 for H2V; C, CR, V3 for DeltaBench) |
| Full runs | H2V CR 181/181, V3 174/174, V4 141/141; DeltaBench CR 1,236/1,236, V3 1,208/1,208; 0 failed rows after the RoPE merge |

## 4. Deviations and incidents (in order)

1. The cycle2 workspace was gone, so I set up cycle4 from scratch (Omri's approval, 2026-11-21 22:10):
   - the model from Hugging Face at the pinned revision;
   - the wheelhouse rebuilt with the exact pins;
   - the bank code from Drive;
   - frozen code as `git archive` of this branch (v5 = 24dd1a217).
   - The jobs = the 11-18 job.sbatch with cycle4 paths, plus a RoPE merge (DeltaBench) and a builder-parity step.
2. Job 372094 failed: the stored pickles are numpy 2, the container numpy 1.26. Fix: a loader shim
   (`np2_alias_run.py`), used only for the stored-cell steps.
3. Job 372108 failed my own bank-parity check (absolute 1e-9). The difference was 1.05e-9 on an unused column with
   values up to 8.7e4 (relative 1e-14). Fix: a scale-aware check (structure exact; TELE16 <= 1e-12). The smoke rules
   are unchanged.
4. Job 372126 (DeltaBench V3) failed: the driver exits 2 on the native pass when rows exceed 4,096 tokens. Fix:
   accept exit 2 on that pass only; the merged run must still have 0 failures.
5. Amendment 1 (before any rule C output): the C pool is CR, the C answers re-run in the children's container.
   - Reason: smoke S3 showed the stored C cell (native env) and the container differ on top15_turnover.
   - The 11-18 runs also re-ran C.
6. DeltaBench has no V4 (no siblings). Its pseudo-metric scores pseudo-clean prefixes with "empty=1", because the
   official script drops samples with an empty target. Both points were pre-declared in PROTOCOL.md.
7. H2V: the 19 answers with no stored telemetry are not scored. 18 of them are longer than 4,096 tokens, and no RoPE
   run was made for H2V.
8. `eval_ext.py` crashed once on DeltaBench (a duplicate dict key) before it wrote any output. I fixed the key and
   reran. This is still the single evaluation.
9. The DeltaBench size curve was not run. Beta is 0 on the full set.

## 5. Files

- `code/`: pseudo_ext.py, eval_ext.py, size_curve_ext.py, parity_cr_vs_stored.py, adapters, DeltaBench evaluator
- `aircc/`: job scripts
- `outputs/`: EVAL_*, PSEUDO_LABELFREE_*, SIZE_CURVE_hard2verify, PARITY_*, DELTABENCH_*, NPZ_SHA256.txt (the npz files are not committed)
- `outputs/aircc_records/`: JOB_*.json and SMOKE_*.json
- `logs/`
- Raw AIRCC outputs (runs, banks, steps, pre; about 15 GB): `gdrive:hallucination_detection/cluster_results/ssl_step_threshold_h2v_deltabench_v1/aircc/`. The upload is md5-checked per file; status in section 6.

## 6. Upload status

See the final message for the upload result (it ran after this report was written).
