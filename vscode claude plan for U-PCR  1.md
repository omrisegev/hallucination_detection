# Plan: U-PCR integration ladder (incremental, measured, reversible)

Scope this phase to the U-PCR track only. Every change is one **rung** you can measure and keep/drop independently. Integration is **additive and flag-gated** — L-SML stays the default, U-PCR is opt-in, nothing is ripped out. That's how "no hard decisions" is honored: the data decides, and every step is reversible.

Two U-PCR-specific facts that shape the ladder:
- **Pruning must be off.** `upcr_fuse`'s default `min_frac`/`exclude_frac` drops weak experts — that's subset selection, which you've ruled out. All 30 features stay in every rung.
- **U-PCR is *not* sign-invariant** (unlike L-SML). Its additive model $C_{ij}=g^2+a_i+a_j$ presumes each feature is positively oriented, so U-PCR may lean on the orientation prior *more* than L-SML. We **measure** that (R5) rather than assume it.

## Steps (rungs)

1. **R0 — Harness + baselines (no method change).** Wire U-PCR into the 25-cell in-scope bench alongside the anchors: GOOD_6 = 0.7594, **L-SML-all-30** (the honest label-free, no-subset baseline U-PCR must beat), simple mean, and LR@30 = 0.7810 (ceiling). Must reproduce these before trusting any delta. This is the measurement backbone.
2. **R1 — U-PCR minimal-honest.** All 30 features, **pruning disabled** (`min_frac=0`, `exclude_frac=inf`), 1-component, `var_y=0.25`, pre-oriented input. *Keep-criterion:* ≥ L-SML-all-30 within CI (apples-to-apples).
3. **R2 — 2-component decorrelation** (`auto_components`, $\lambda_2>0.1\,\mathrm{Tr}$). *depends on R1.* Directly tests the rank-1→rank-2 fix (your 0.31× second factor). Report trigger-rate and $\lambda_2/\mathrm{Tr}$ per cell.
4. **R3 — $g^2$ / var_y sensitivity.** *depends on R1.* Sweep `var_y` ∈ {0.10, 0.25, per-cell $p(1-p)$}; compare $g^2$ via projection-residual (Eq. 20) vs $\lambda_1/m$ (Eq. 18). *Keep the most **robust** setting* (lowest cross-cell AUROC variance), not the peak. Check $\hat g^2/\mathrm{Var}(Y)$ tracks cell difficulty.
5. **R4 — Robust additive loss** (absolute vs squared for the $a$-solve, Remark 1). *depends on R1.* Insurance against your redundant/dependent features. Keep if it helps or is neutral.
6. **R5 — Orientation-cost probe** (audit, not an integration). *parallel with R2–R4.* Run U-PCR on oriented vs unoriented vs randomly-flipped features to quantify how much it relies on the orientation prior vs L-SML (which is invariant). Decision-relevant for your "no specific feature" goal.
7. **R6 — Consensus global-sign anchor.** Resolve U-PCR's output sign against the mean-of-oriented-features consensus (drops the epr dependence). Keep if 0 disagreements on known-good cells.
8. **R7 — Cross-cell pooling of U-PCR's $\hat\rho$** (stretch; bridges to the broader plan). James–Stein/EB shrink of $\hat\rho$ across the 25 cells, evaluated **LOCO**. The payoff rung for small-$n$ cells ($n<40$). Keep if the LOCO-honest gain > 0.

## Evaluation backbone (every rung reports the same row)

- Primary: **macro AUROC** over 25 cells. Secondary: QA-macro, math-macro, #cells improved, median per-cell ΔAUROC.
- **Paired bootstrap 95% CI** (clustered by question id) vs **GOOD_6** and vs **L-SML-all-30**; plus gap to the 0.7810 ceiling.
- Free U-PCR diagnostics: $\hat g^2/\mathrm{Var}(Y)$, projection residual, $\lambda_2/\mathrm{Tr}$, #components used, $\hat\rho$ top/bottom-weighted features, and `n_kept` (must = 30 — proves pruning is off).
- LOCO only applies to R7; R1–R6 are per-cell label-free, so no selection optimism.

## Relevant code (external repo `omrisegev/hallucination_detection`, `master`)

- `spectral_utils/fusion_utils.py` — `upcr_fuse`, `upcr_pipeline`, `upcr_proj_residual`, `_upcr_g2_grid` (the U-PCR surface); compared against `sml_fuse_signed`/`lsml_continuous` (incumbent), `simple_average_fusion` (mean), `nadler_fuse`.
- `scripts/inscope_bench_common.py` — the 25-cell harness (source the baseline rows).
- **new** `scripts/upcr_ladder.py` — runs the ladder, emits the decision table (keeps the incumbent bench untouched → reversible).
- `results/selector_bench/inscope_feature_orientation_summary.csv` — sanity-check that $\hat\rho$ up-weights the strong logprob family.

## Verification

1. R0 reproduces GOOD_6 = 0.7594 and LR@30 = 0.7810 before any rung is trusted.
2. Per rung: paired-bootstrap CI vs incumbent (report intervals, not point estimates); `n_kept == 30` every cell; $\hat\rho$ direction matches the orientation CSV.
3. R7 only: LOCO — held-out cell's $\bar\rho$/shrinkage come from the other 24; report the $n<40$ split separately.

## Decisions / scope

- **In:** U-PCR only, constraint-aligned (all features, pruning off), additive/flag-gated (`method='upcr'` beside the L-SML path). L-SML remains default until the decision table justifies promotion.
- **Out (this phase):** general cross-cell pooling beyond U-PCR's $\hat\rho$, Ledoit-Wolf on the L-SML path, triplet, self-consistency, and any Step-203 factorial re-run.
- **No hard decision:** the output is a rung → ΔAUROC + CI + diagnostic → keep / drop / park table; nothing is replaced or removed.

## Further Considerations

1. **Separate script vs extend `gap_ladder.py`?** Recommend a new `scripts/upcr_ladder.py` — keeps the incumbent bench pristine and the change reversible.
2. **var_y when per-cell prevalence is unknown:** recommend fixing 0.25 for R1–R2 to isolate other levers, then revisit in R3.
3. **Promotion threshold:** recommend strict (non-overlapping paired-bootstrap CI vs L-SML-all-30) to promote to default, lenient (point improvement) to "park and keep exploring."

## Citations (weight-estimation lineage)

- Parisi, Strino, Nadler, Kluger (2014), *Ranking and combining multiple predictors without labeled data*, PNAS 111(4):1253–8, arXiv:1303.3257 — SML, rank-1 covariance, leading eigenvector ∝ balanced accuracy.
- Jaffe, Nadler, Kluger (2015), *Estimating the accuracies of multiple classifiers without labeled data*, AISTATS — triplet method (excluded by request).
- Fetaya, Nadler, Jaffe, Kluger, Jiang (2016), *Unsupervised ensemble learning with dependent classifiers*, AISTATS 351–360, arXiv:1510.05830 — L-SML, dependent groups, low-rank covariance.
- Dror, Nadler, Bilal, Kluger (2017), *Unsupervised Ensemble Regression*, arXiv:1703.02965 — U-PCR: optimal weights $w^*=C^{-1}\rho$ (Lemma 1), additive model $C_{ij}=g^2+a_i+a_j$, $\rho$-sign + all-ones global orientation, expert pruning, 1-/2-component weights. (Repo attributes the later Tenzer et al. AISTATS 2022 version.)
- Merz & Pazzani (1999), *A principal components approach to combining regression estimates*, Machine Learning 36:9–32 — PCR*, multi-component eigenvector weights (supervised ancestor of U-PCR).
- James & Stein (1961), *Estimation with quadratic loss*, Berkeley Symp. — shrinkage dominates the MLE for $m\ge3$ (basis for R7 pooling).
- Efron & Morris (1973), *Stein's estimation rule and its competitors — an empirical Bayes approach*, JASA 68:117–130 — empirical-Bayes reading of partial pooling.