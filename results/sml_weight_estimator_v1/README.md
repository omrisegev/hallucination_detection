# SML sign rule and weight estimator (Step 467, 2026-10-19/20)

CPU only, no inference, label-free fitting. Labels are read only after scores exist. All numbers are point
estimates on data already inspected (development PRMBench/ProcessBench, and external Socratic used post hoc).

## Bottom line
1. **Sign rule bug (method, not data).** `sml_fuse_signed` lets components be negative and picks the global sign
   by majority. Callers then flip the fused score toward an anchor. On pre-oriented views that is outside the
   L-SML paper's assumption (Jaffe et al. 2016, Lemma 4: every view beats random). When a confounder makes two
   good views negatively correlated, the PRM gets the negative weight and the anchor flip inverts the telemetry.
2. **The estimator is biased toward equal weights, and that bias helps.** The zero-diagonal eigenvector is not
   the rank-one fit of the off-diagonal covariance. The unbiased fit (`fusion_utils.rank1_offdiag_fit`) is below
   the shipped L-SML on 7 of 10 corrected-template PRMBench banks (section 3). Do not adopt it.
3. **Non-negative weights help clearly only on N40 and N43**, where shipped L-SML had negative weights. On the
   corrected template, the plain oriented mean matches or beats shipped L-SML on 7 of 10 banks, N16 included.

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

## 3. Weight estimator on the nested banks (corrected Qwen3 template, PRMBench only)
Re-run on the thinkfix_mask1 banks as collection section 3a (job 290578): code
`../prm_boost_collection_v1/section3/score_banks_prmbench.py`, results
`../prm_boost_collection_v1/section3/results/score_banks_thinkfix_mask1/` (`RESULTS.csv`, `FIT_DIAGNOSTICS.json`).
Parity: LSML_CONT and ALL_EQ_ORIENTED reproduce `../thinkfix_rescore_v1/thinkfix_mask1/cont_baselines/ABSOLUTE.csv`
on all 10 banks. ProcessBench was not re-run (out of the thesis focus).

PRMBench within-AUC:

| bank | LSML_CONT (shipped) | LSML_ABS_ORI | LSML_RANK1_ORI | ALL_EQ_ORIENTED |
|---|---|---|---|---|
| N13 | .7661 | .7655 | .7557 | .7641 |
| N16 | .7792 | .7793 | .7704 | **.7811** |
| N25 | .7740 | .7733 | .7711 | .7757 |
| N40 | .7393 | .7666 | .7609 | .7606 |
| N43 | .7616 | .7751 | .7638 | .7724 |
| N53 | .7596 | .7608 | .7493 | .7647 |
| N56 | .7705 | .7687 | .7500 | .7731 |

All 10 banks are in that `RESULTS.csv`.
- **Rank-one fit:** below shipped L-SML on 7 of 10 banks; above on N22, N40 and N43, where shipped L-SML is weakest.
  It concentrates weight: one group gets .88 of it on N53 and .84 on N56 (effective number of groups 1.3-1.4,
  against 5.0-5.7 shipped).
- **Negative weights:** 4 of 10 shipped fits have a negative cross-group weight. |v| helps clearly only on N40 (+.027)
  and N43 (+.014); elsewhere it is within ±.002.
- The earlier old-template version of this section, and its ProcessBench numbers, were removed on 2026-10-21 (Omri):
  pre-thinkfix data are not used for analysis.

## Reproduce
```
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

## 5. Answer-local SML (removed)
The answer-local study (how large the bank and how many steps one answer needs for a stable SML fit) ran on the
pre-thinkfix template. Its folder and numbers were removed on 2026-10-21 (Omri). It has not been re-run on the
corrected template, so the question is open.
