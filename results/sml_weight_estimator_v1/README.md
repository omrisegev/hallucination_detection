# SML sign rule and weight estimator (Step 467, 2026-10-19/20)

CPU only, no inference, label-free fitting. Labels are read only after scores exist. All numbers are point
estimates on data already inspected (development PRMBench/ProcessBench, and external Socratic used post hoc).

## Bottom line
1. **Sign rule bug (method, not data).** `sml_fuse_signed` lets components be negative and picks the global sign
   by majority. Callers then flip the fused score toward an anchor. On pre-oriented views that is outside the
   L-SML paper's assumption (Jaffe et al. 2016, Lemma 4: every view beats random). When a confounder makes two
   good views negatively correlated, the PRM gets the negative weight and the anchor flip inverts the telemetry.
2. **The estimator is biased toward equal weights, and that bias helps.** The zero-diagonal eigenvector is not
   the rank-one fit of the off-diagonal covariance. The unbiased fit (`fusion_utils.rank1_offdiag_fit`) scores
   worse on the development banks. Do not adopt it.
3. **Non-negative weights help only where L-SML had negative weights:** the large banks N40, N43 and N53.
   N16 with the shipped L-SML remains the best bank on both benchmarks.

## 1. Socratic CHAIN below chance (`socratic_chain_flip/`)
Math-Shepherd CHAIN within-AUC was .468 (N13; N16 before the digit fix). Its inputs: PRM alone .718, TELE .562.
- **Mechanism** (`DIAG_*.json`, `diag_chain_socratic.py`):
  - SML gives the PRM w = -.209 against +.614/+.580/+.493 for the 3 telemetry groups.
  - The anchor flip toward z(PRM) then makes every telemetry weight negative: corr(CHAIN, TELE) = -.988.
  - The flip fired in 3 of 20 Socratic fits: Math-Shepherd CHAIN and PRMG N13/N16, and Qwen-72B CHAIN N13 (.460).
  - In the PRMBench bank sweep, the PRM weight is positive in all 60 recorded CHAIN/PRMG fits
    (`../prm_boosting_bank_sweep_v1/SWEEP_DIAG_*.json`).
- **Why two good views are negatively correlated** (`COV_DECOMPOSITION.txt`, `cov_decomp.py`; Socratic gold):
  - Cov = E[Cov|y] + p(1-p)·ΔPRM·ΔTELE. On Math-Shepherd that is -.176 = -.186 (within class) + .010 (signal).
  - The signal term is tiny because the telemetry barely separates the classes (ΔTELE = .06 SD).
  - Step position drives the within-class term. PRM correlates with position at +.43 within class, TELE at
    -.53 to -.58. After removing position, the within-class correlation is +.06.
  - Qwen-7B is the familiar case for comparison: within-class +.313, no flip.
- **Synthetic reproduction:** `synthetic_chain_flip.py`.

## 2. Non-negative SML weights on Socratic (`socratic_sign/`, post hoc, 5 PRMs)
`run_socratic.py <repo> <mode> <bank> <out>` runs the shipped `ablation5_socratic.py` with one SML mode.
Mode `signed` reproduces the shipped scores exactly (max |diff| 0.0). The non-negative modes drop the anchor
flip. CHAIN N16 within-AUC, PRM order Qwen-7B / Qwen-72B / PURE / RLHFlow / Math-Shepherd:

| mode | | | | | |
|---|---|---|---|---|---|
| signed (shipped) | .776 | .613 | .698 | .590 | .468 |
| abs (\|v\|) | .776 | .648 | .698 | .593 | .599 |
| clip (v<0 -> 0) | .776 | .639 | .698 | .591 | .567 |
| nnr1 (rank-one fit, v>=0) | .790 | .634 | .691 | .586 | .564 |
| PRM alone | .824 | .814 | .796 | .549 | .718 |
| 1/2 PRM + 1/2 mean | .813 | .809 | .812 | .572 | .735 |

The inversion is gone, but CHAIN still loses to the PRM alone on 4 of 5 PRMs. Socratic banks are the pre-digit-fix
`bank_v2` from Drive (`cluster_results/prm_boost_external_v1/`).

## 3. Weight estimator on the nested banks (`processbench_banks/`, development data)
- **Bank:** the old-template 56-channel source bank, rebuilt locally with `scripts/rebuild_source_bank_v4.py
  --variant old`. All 9 raw inputs are hash-identical to `../thinkfix_rescore_v1/source_bank_old_rebuild/MANIFEST.json`.
- **Scorer:** `score_banks.py`, the same per-cell protocol as `../monday_cont_baselines_v1/run.py`.
- **Parity:** LSML_CONT and ALL_EQ_ORIENTED reproduce `../thinkfix_rescore_v1/old/cont_baselines/ABSOLUTE.csv`
  exactly on all 10 banks, for both metrics.
- **The current line, thinkfix_mask1, is not tested here.** Its banks are on AIRCC only.

ProcessBench argmax first-error hit (%, cell-equal mean of 8 cells) / PRMBench within-AUC:

| bank | LSML_CONT (shipped) | LSML_ABS_ORI | LSML_RANK1_ORI | ALL_EQ_ORIENTED |
|---|---|---|---|---|
| N13 | 37.48 / .7803 | 37.32 / .7770 | 35.99 / .7677 | 36.96 / .7757 |
| N16 | **41.61 / .7958** | 41.47 / .7919 | 38.61 / .7762 | 41.16 / .7906 |
| N25 | 40.69 / .7821 | 40.76 / .7853 | 34.86 / .7637 | 40.79 / .7823 |
| N40 | 35.67 / .7243 | 37.21 / .7731 | 36.65 / .7745 | 36.19 / .7729 |
| N43 | 37.98 / .7583 | 38.69 / .7877 | 37.39 / .7779 | 38.78 / .7836 |
| N53 | 33.52 / .7709 | 36.07 / .7661 | 35.29 / .7510 | 35.48 / .7680 |
| N56 | 36.58 / .7775 | 37.27 / .7682 | 35.73 / .7600 | 37.56 / .7757 |

All 10 banks are in `RESULTS.csv`.
- **Orientation:** pre-orienting channels by the LEVEL sign changes nothing (LSML_CONT_ORI = LSML_CONT; gauge invariance).
- **Rank-one fit:** it concentrates weight. In N56 PRMBench one group gets .89 of the weight, an effective number of
  groups of 1.3 against 6.2 shipped (`FIT_DIAGNOSTICS.json`).
- **Negative weights:** 8 of 90 shipped fits had a negative cross-group weight (N22 and N25: 2 each; N40, N43, N53, N56: 1 each).
- **Agreement with the LR-gap analysis:** the oracle-voting ceiling (`../lr_gap_explanation_draft_2026-10-15/analysis/ladder.py`)
  allows only +0.3 to +0.6 points on most banks and about +2 on N43. Weighting beyond equal has little room here.

## Reproduce
```
python results/sml_weight_estimator_v1/processbench_banks/score_banks.py . <rebuilt FEATURES.npz> \
    dataset_cache/four_localization/prmbench_qwen3_8b_telemetry_full/prmbench_telemetry.pkl <out>
python results/sml_weight_estimator_v1/socratic_sign/run_socratic.py . abs <socratic bank_v2> <out SCORES npz>
python results/sml_weight_estimator_v1/socratic_chain_flip/cov_decomp.py <socratic_gold.json> <SCORES npz ...>
```
`run_prmbench.py` is the same sign-mode wrapper for `ablation5.py` on AIRCC. It has not been run yet.

## 4. Where the PRM enters, and whether nnr1 gives it its weight (`socratic_sign/prm_placement.py`, 2026-10-20)
Question (Omri): does the rank-one fit give the PRM its proper weight when the PRM is not forced into its own group?
- **Placements:** AUTO = L-SML over [z(PRM), channels] with its own grouping. FLAT = one SML stage over all views.
  CHAIN = shipped.
- **Weight share:** the effective PRM weight share is the least-squares weight of the fused score on its input views,
  normalized (fit R² ≥ .99). Files: `RESULTS_socratic_placement.csv`, `PRM_WEIGHT_SHARE_placement.json`.

N16, within-AUC (signed PRM weight share), PRM order Qwen-7B / Qwen-72B / PURE / RLHFlow / Math-Shepherd:

| arm | | | | | |
|---|---|---|---|---|---|
| PRM alone | .824 | .814 | .796 | .549 | .718 |
| CHAIN nnr1 | .790 (.18) | .634 (.04) | .691 (.07) | .586 (.01) | .564 (.00) |
| AUTO nnr1 | .732 (.05) | .689 (.04) | .667 (.00) | .598 (.04) | .553 (.00) |
| FLAT nnr1 | .749 (.06) | .677 (.03) | .673 (.02) | .605 (.02) | .559 (.00) |
| AUTO signed | .722 (.06) | .661 (.04) | .660 (-.03) | .595 (.03) | .458 (.05) |

**No.** In every placement and mode the PRM gets 0-8% of the weight (CHAIN up to 21% on Qwen-7B). Every arm is far below the
PRM alone on the strong PRMs.
- **Mechanism (model reading, not separately tested):** under the rank-one model the PRM's loading is
  v_p = sqrt(r_pa·r_pb / r_ab).
  - A PRM that correlates weakly with the telemetry (r_pa, r_pb small) gets a small loading.
  - Telemetry channels that correlate strongly with each other (r_ab large, inflated by shared nuisance such as
    step position) shrink it further.
  - No covariance-only estimator can see that the PRM is the strongest view in that case.
- **The external benchmarks show the same picture** (ProcessBench, job 289297, `../prm_boost_processbench_v1/`;
  Hard2Verify, branch `claude/prm-boost-external-2026-10-18`): CHAIN lands near the telemetry alone. It helps the weak
  PRMs (RLHFlow) and hurts the strong ones.

## 5. Answer-local SML: what bank size and step count would make it work? (`answer_local/`, 2026-10-20)
Question (Omri): how large must the bank (m) be, and how many steps (n), for SML or L-SML fitted on ONE answer's steps
to be stable? And does shrinkage help?
- **Setup:** the same old-template bank and adapter as section 3, with channels oriented by the LEVEL sign.
- **Parity:** EQ reproduces ALL_EQ_ORIENTED exactly (N16 41.16 / .7906).
- **Scope:** all PRMBench answers (6,030 with both classes) and all ProcessBench answers.
- **Method:** SML weights are |leading eigvec| of offdiag(R); labels are read only after scoring (`run.py`).

PRMBench within-AUC / ProcessBench argmax hit (%):

| method | N13 | N16 | N25 | N56 |
|---|---|---|---|---|
| EQ (plain mean) | .7757 / 36.96 | .7906 / 41.16 | .7823 / 40.79 | .7757 / 37.56 |
| POOLED SML (cell covariance, other answers' unlabeled steps) | .7783 / 36.51 | .7911 / 40.68 | .7786 / 39.38 | .7651 / 36.86 |
| LOCAL SML (one answer) | .7723 / 36.65 | .7853 / 41.13 | .7723 / 39.49 | .7614 / 36.94 |
| local shrunk to pooled, Schäfer-Strimmer λ | .7774 / 36.33 | .7908 / 40.61 | .7774 / 39.00 | .7637 / 36.43 |
| local shrunk to equal weights, λ=.9 | .7759 / 36.99 | .7905 / 41.51 | .7811 / 40.61 | .7743 / 37.42 |

- **No (m, n) makes answer-local SML beat the plain mean.** LOCAL minus EQ in AUC is -.003 / -.005 / -.010 / -.014 for
  m = 13 / 16 / 25 / 56, so a larger bank makes it worse.
  - Effective rank of the pooled correlation is only 4.6 / 5.4 / 5.9 / 7.7: extra channels add entries, not independent
    information.
  - The rank of an answer's covariance is min(n-1, m), with a median of 4 to 11.
- **Signal versus noise** (`SIGNAL_NOISE.csv`, PRMBench N16):
  - The label-driven covariance term p(1-p)·dX_i·dX_j is .10 / .09 / .06 / .03 / .015 for the step bins
    ≤6 / 7-10 / 11-15 / 16-24 / ≥25.
  - The sampling SE 1/√(n-3) is .58 / .45 / .32 / .25 / .18, so signal/SE stays at about .1 to .2.
  - Longer answers do not help: the number of error steps does not grow with n, so p(1-p) falls about like 1/n while
    the noise falls like 1/√n.
- **Shrinkage helps only by returning to pooled or EQ quality.**
  - The data-driven Schäfer-Strimmer intensity toward the pooled target has a median of 1.00.
  - Shrinkage toward equal weights at λ=.9 reproduces EQ. N16 ProcessBench +0.35 pp is about 12 answers, picked from a
    grid, and is not established.
- **Grouped L-SML on one answer runs:** 0-5 failures in 1,000 answers, K mostly 4-5. It is clearly worse than EQ on the
  same answers. N16 PRMBench .766 vs .799; ProcessBench 33.1% vs 39.6%. The cell-level L-SML fit gives 41.61 / .7958.
- **Inferred:** the weights SML needs cannot be estimated from one answer's steps. Any useful weighting has to borrow
  strength across answers (unlabeled), and even pooled SML only matches the plain mean on these banks.
