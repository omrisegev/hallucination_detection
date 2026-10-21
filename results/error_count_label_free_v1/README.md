# Label-free error count per answer (2026-10-21, PRMBench, development data; not committed results until reviewed by Omri)

**Question (Omri):** can nnr1, whole-answer metrics or other literature methods estimate how many steps to flag in
an answer better than the fixed Q80 rule? The rule matters because PRMScore needs a binary flag per step.
- **Scope:** CPU only, thinkfix_mask1 telemetry (`../thinkfix_rescore_v1/prmscore_inputs/`). Qwen-7B PRM boosting was
  refit exactly with `../prmbench_paper_metric_parity_v1/common.py`.
- **Labels:** used only for evaluation.
- **Review:** an independent check in `../../docs/reviews/number_audit_2026-10-21/review/REVIEW.md` re-implemented GAP
  from the declaration text and got identical counts on all 6,969 answers.

## Results (PRMScore, 6,211 non-control answers)
| rule | CHAIN_N16 (PRM boosting, Qwen-7B) | LSML_CONT (mask1 N16) |
|---|---|---|
| Q80 (current) | 68.25 | 65.38 |
| GAP: largest gap in the answer's sorted scores, k <= ceil(n/2) | 70.40 | 67.57 |
| constant k=2 (PRMBench construction prior, `../count_structure_check_v1/`) | 70.80 | 67.92 |
| ORACLE true count (uses labels, ceiling only) | 75.40 | 71.73 |

- **GAP vs Q80:** +2.19 [1.73, 2.63] (source-group bootstrap) on LSML_CONT, +2.15 on CHAIN_N16. It holds in all 140
  telemetry scorer × bank cells and all 4 boosting arms.
- **GAP vs k=2:** -0.35 [-0.72, +0.00], so GAP does not beat the constant.
- **Caveats on GAP:**
  - The cap was chosen knowing PRMBench's count structure. Without the cap GAP scores 57.8 (-7.6 vs Q80); with a cap
    of 4 it scores 68.33.
  - There is no external check yet (Socratic per-step data are on AIRCC).
- **Why Q80 leaves headroom:** Q80's count tracks answer length (Spearman .78-.80) while the true count does not (.09).
  This was already reported in `../count_structure_check_v1/README.md`.
- **nnr1:** it cannot estimate the count. Second moments identify only v_i = sqrt(1-b²)(2π_i-1), which confounds
  prevalence with view quality. The pooled third-moment prevalence is 0.42 against a true 0.14-0.16, and the
  per-answer nnr1 scale has Spearman -0.06 with the true count.
- **The other declared estimators all lost to Q80:** ESD outlier test, raw level, third-moment share, nnr1 scale,
  PRM-native count, and answer-level Poisson matching. See `RESULTS.csv` and `DECLARATION.md`.

Files: `DECLARATION.md` (declared before results; see the review for timing caveats), `RESULTS.csv`,
`GAP_ALL_BANKS.csv`, `CONTRASTS.csv`, `ROBUSTNESS.json`, `PARITY.csv`, `COUNTS.npz`, `run.py`, `common_ec.py`,
`parity.py`, `analysis.py`, `banks_robustness.py`. The scripts read inputs from session scratch paths; adjust them
before rerunning.
