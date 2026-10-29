# Where each idea of the raw-null clean-answer gate comes from (2026-10-30)

This note documents the sources of every idea used in `spectral_utils/raw_null_gate.py` and in its
pre-registration, `docs/experiments/RAW_NULL_GATE_PREREG_2026-10-30.md`.

**How to read the source column.**
- **"Literature"** means the idea is taken from a published method.
- **"Our construction"** means we combined published pieces in a way that, as far as we know, no paper does.
  These parts carry no external guarantee beyond the assumptions stated in the row.

**How the citations were checked.** Every bibliographic entry was verified on 2026-10-30 against Crossref
(api.crossref.org, by DOI or bibliographic query) or against the publisher's or proceedings' own page. Descriptions
of content are paraphrases, unless they are in quotation marks and taken from a local copy in `papers/extracted/`.

## 1. The ideas, one by one

| # | Idea used | Source | What we take from it, and what we change |
|---|---|---|---|
| 1 | **Two-groups model and local false discovery rate.** Each score comes from a mixture f = π₀f₀ + (1 − π₀)f₁ of "null" (here: a correct step) and "non-null" (an erroneous step). The probability that a case is null, given its score, is lfdr(z) = π₀f₀(z)/f(z). | Literature: Efron, Tibshirani, Storey & Tusher 2001 [R1]; Efron 2008 [R4]. | The step's P(error) is 1 − lfdr, one-sided (§3.3 of the pre-registration). They used it for genes in microarrays; we use it for reasoning steps. |
| 2 | **Empirical null.** Estimate f₀ from the data instead of assuming a theoretical N(0, 1), because in large-scale data the null itself is often shifted or wider. | Literature: Efron 2004 [R2]; Efron 2007 [R3]; Efron 2010 [R5]. Reference implementation: the R package `locfdr` [R6]. | Our raw step scores have no theoretical null, so the empirical null is the only option. This is the core reason the gate does not need the prevalence that FRAMING_THEORY P6 shows is unidentified. |
| 3 | **Central matching and the "zero assumption".** The centre of the distribution consists almost only of null cases, so a quadratic fitted to log f near the centre gives log(π₀f₀), and hence μ₀, σ₀ and π₀. | Literature: Efron 2004 [R2]; Efron 2010 [R5]; `locfdr`'s central-matching option [R6]. | We use the central 50% (quantiles .25 to .75), the default fraction of `locfdr` as we understand it. **The assumption matters:** most steps are correct (PRMBench about 85–90%; ProcessBench and CRV more), so the centre should be correct steps. |
| 4 | **Lindsey's method for f.** Bin the scores, then fit a Poisson regression of the counts on a smooth basis. | Literature: Efron & Tibshirani 1996 [R7]; used by `locfdr` [R6]. | We use a degree-7 Legendre polynomial instead of `locfdr`'s natural spline with 7 degrees of freedom. **This is our simplification;** the synthetic tests check recovery of π₀, μ₀ and σ₀. |
| 5 | **Conformal p-values against a reference set of inliers**, giving a finite-sample valid p-value under exchangeability. | Literature: Bates, Candès, Lei, Romano & Sesia 2023 [R8]. | Used in G1b. **Our reference set is the steps GAP does not flag.** They are mostly, not all, correct, and they are not exchangeable with the tested step. So the formal guarantee does not transfer; the p-value is a heuristic here. |
| 6 | **Šidák correction** for the minimum of n tests: p_answer = 1 − (1 − p)ⁿ. | Literature: Šidák 1967 [R9]. | Applied to the answer's worst step over its n steps (G1b). Exact under independence between steps, which we do not have. |
| 7 | **False discovery rate and the null proportion π₀.** | Literature (context): Benjamini & Hochberg 1995 [R10]; Storey 2002 [R11]. | Not used as a procedure. Storey's π₀ estimator is the standard alternative to central matching, and could be a sensitivity check. |
| 8 | **Dawid–Skene model:** several noisy raters, each with its own sensitivity and specificity, conditionally independent given the true label, fitted by EM. | Literature: Dawid & Skene 1979 [R12]. | Used twice. (a) Between the telemetry groups, as in the shipped L-SML stage. (b) **New:** at the answer level (G3), over the groups' answer-level "abnormal step" votes. |
| 9 | **SML: ranking and weighting classifiers without labels** through the leading eigenvector of their covariance. | Literature: Parisi, Strino, Nadler & Kluger 2014 [R13]. | Reported for G3 as the SML ranking of the groups at the answer level. |
| 10 | **L-SML model: groups of dependent classifiers, conditionally independent between groups given the label.** | Literature: Jaffe, Fetaya, Nadler, Jiang & Kluger 2016 [R14]. Quote, from the local copy, §2: "Classifiers that depend on different latent variables are thus conditionally independent given Y". | **G2 relies on this.** If the groups are conditionally independent given the step's label, the step posterior odds are the prior odds times the product of the groups' likelihood ratios. Each ratio comes from that group's own two-groups fit (row 1). **Our construction:** two-groups fits per group, combined by the L-SML independence structure. It is exact only to the extent that the groups really are independent; Step 481 found within-class correlations of .07–.16 after position removal. |
| 11 | **Estimating classifier accuracies without labels**, and when they are identifiable. | Literature: Jaffe, Nadler & Kluger 2015 [R15]. | Background for G3's DS fit at the answer level: with at least 3 conditionally independent voters, accuracies and prevalence are identifiable. |
| 12 | **Spectral start for Dawid–Skene EM.** | Literature: Zhang, Chen, Zhou & Jordan 2014 [R16]. | Not used here. Step 482 found that it changes nothing on our data. G3 uses the majority start of `ds_em_votes`. |

## 2. What is ours, and which project findings motivate it

**Raw scores instead of per-answer scores.** Per-answer z-scores and GAP votes remove each answer's level, so a clean
answer looks like an erroneous one.
- Evidence: the DS-gate experiments E1, E1 fixed and E1b (branch `claude/ds-gate-2026-10-29`, `results/ds_gate_*`).
- The raw scores used here are pooled over all steps, with a pooled cubic in position removed. This is Step 481's
  procedure (`votes_e1b.clean`), and it keeps the level.

**Not estimating the share of erroneous answers directly.** FRAMING_THEORY P6
(`docs/research_notes/math_rigor_2026-10-24/FRAMING_THEORY.md`) shows that this share is not identified from second
moments. An answer-level two-component mixture invents a second group on heavy-tailed data
(`results/answer_mixture_gate_v1`, Step 484). Rows 2–3 above replace that estimate with an estimate of what a correct
step looks like.

**Worst step, then GAP.** Omri's framing:
- **The worst step** is the step GAP would flag first.
- **If that step is plausible as a correct step on the raw scale,** the answer is called clean.
- **Otherwise** GAP flags as usual.

GAP itself (largest gap, at most ⌈n/2⌉ steps) is the project's decision rule since 2026-10-21.

**SML in the answer-level decision (G2, G3).** Omri asked that the SML / L-SML theory enter the answer score, not
only the fusion:
- G2 enters through row 10;
- G3 enters through rows 8 and 9.

**Related work that needs labels, for contrast.** CROP (`papers/digests/crop-conformal-clean-prefix.md`) certifies a
clean prefix with conformal risk control, but its threshold is calibrated on labelled traces. The gate here uses no
labels at any point. In exchange, it has no formal guarantee.

## 3. References (verified 2026-10-30)

- [R1] Efron, B., Tibshirani, R., Storey, J. D., Tusher, V. (2001). Empirical Bayes analysis of a microarray
  experiment. *Journal of the American Statistical Association* 96(456), 1151–1160. doi:10.1198/016214501753382129.
- [R2] Efron, B. (2004). Large-scale simultaneous hypothesis testing: the choice of a null hypothesis. *JASA* 99(465),
  96–104. doi:10.1198/016214504000000089.
- [R3] Efron, B. (2007). Size, power and false discovery rates. *Annals of Statistics* 35(4), 1351–1377.
  doi:10.1214/009053606000001460.
- [R4] Efron, B. (2008). Microarrays, empirical Bayes and the two-groups model. *Statistical Science* 23(1), 1–22.
  doi:10.1214/07-STS236.
- [R5] Efron, B. (2010). *Large-Scale Inference: Empirical Bayes Methods for Estimation, Testing, and Prediction.*
  Cambridge University Press, IMS Monographs. doi:10.1017/CBO9780511761362.
- [R6] Efron, B., Turnbull, B. B., Narasimhan, B. `locfdr`: computes local false discovery rates. R package, CRAN.
  doi:10.32614/CRAN.package.locfdr.
- [R7] Efron, B., Tibshirani, R. (1996). Using specially designed exponential families for density estimation.
  *Annals of Statistics* 24(6), 2431–2461. doi:10.1214/aos/1032181161.
- [R8] Bates, S., Candès, E., Lei, L., Romano, Y., Sesia, M. (2023). Testing for outliers with conformal p-values.
  *Annals of Statistics* 51(1), 149–178. doi:10.1214/22-AOS2244.
- [R9] Šidák, Z. (1967). Rectangular confidence regions for the means of multivariate normal distributions. *JASA*
  62(318), 626–633. doi:10.2307/2283989.
- [R10] Benjamini, Y., Hochberg, Y. (1995). Controlling the false discovery rate: a practical and powerful approach to
  multiple testing. *Journal of the Royal Statistical Society B* 57(1), 289–300. doi:10.1111/j.2517-6161.1995.tb02031.x.
- [R11] Storey, J. D. (2002). A direct approach to false discovery rates. *JRSS B* 64(3), 479–498.
  doi:10.1111/1467-9868.00346.
- [R12] Dawid, A. P., Skene, A. M. (1979). Maximum likelihood estimation of observer error-rates using the EM
  algorithm. *Applied Statistics (JRSS C)* 28(1), 20–28. doi:10.2307/2346806.
- [R13] Parisi, F., Strino, F., Nadler, B., Kluger, Y. (2014). Ranking and combining multiple predictors without
  labeled data. *PNAS* 111(4), 1253–1258. doi:10.1073/pnas.1219097111.
- [R14] Jaffe, A., Fetaya, E., Nadler, B., Jiang, T., Kluger, Y. (2016). Unsupervised ensemble learning with dependent
  classifiers. *AISTATS 2016*, PMLR 51, 351–360. Local copy: `papers/Unsupervised Ensemble Learning with Dependent
  Classifiers.pdf`.
- [R15] Jaffe, A., Nadler, B., Kluger, Y. (2015). Estimating the accuracies of multiple classifiers without labeled
  data. *AISTATS 2015*, PMLR 38, 407–415. Local copy in `papers/`.
- [R16] Zhang, Y., Chen, X., Zhou, D., Jordan, M. I. (2014). Spectral methods meet EM: a provably optimal algorithm for
  crowdsourcing. *NeurIPS 2014*; journal version *JMLR* 17(102), 1–44, 2016.

## 4. What could not be checked here

- **The content of R2–R5 and R7.** The full texts were not accessible (publisher pages blocked automated access), and
  there are no local copies. The descriptions in rows 2–4 reflect the standard account of these methods, not a
  re-reading. Before writing them into a paper, read §§ of R2 and the `locfdr` vignette to confirm the
  central-matching details and the default central fraction.
- **No local PDFs** of R1–R13 or R16, except R14 and R15 (and FUSE/CROP for context). Adding them to `papers/` and
  running `/paper-digest` would make this note checkable offline.
