# PRM-anchored view selection: stage 0 pre-registration (2026-10-25)

Omri, 2026-10-25: "כן, תכתוב רישום מוקדם ותריץ את שלב 0 על CPU (בינאריזציה עם GAP לכל ערוץ, אמידת דיוק לכל ערוץ,
מודל שני גורמים) על PRMBench ועל Socratic".

Translation: write a pre-registration, then run stage 0 on CPU (per-channel GAP binarization, per-channel accuracy
estimation, two-factor model) on PRMBench and on Socratic.

Context:
- Long-term goal: a small, principled set of telemetry views that complements the PRM, in place of the fixed 16.
- Why now: the honest-review handoff (`docs/handoffs/HONEST_PROJECT_REVIEW_HANDOFF_2026-10-25.md`) shows the
  PRMBench gain does not transfer to Socratic.

**Stage 0 is descriptive. It selects no subset, proposes no method, and promotes nothing.** Its outputs decide how
stage 1 (selection, with its own pre-registration and held-out test) is designed.

## Data and access

- **Benchmarks and PRMs.** PRMBench (6,969 answers; metric pool = the 6,211 non-control answers) and Socratic-PRMBench
  (2,995 answers). Five PRMs: qwen_prm_7b, qwen_prm_72b, pure_prm, rlhflow_prm, math_shepherd.
- **Loading.** Through `results/prm_boost_collection_v1/evaluate_collection.py` (`load`, `channels`). Data in
  `COLLECTION_DATA` (`/Users/osegev/Desktop/hd_collection_data`).
- **Views (17).**
  - The PRM error score, on the reference scale of `tables/STATUS_META.json`. That scale was chosen post hoc earlier;
    it is disclosed and kept for continuity with the reported tables.
  - The 16 backbone telemetry channels N16 (the same as CHAIN_N16, including the three digit channels).
- **Representation.** Answer-z (per-answer z-score), as in CHAIN.
- **Label-free orientation.** Each telemetry channel is multiplied by the sign of its pooled Pearson correlation with
  the PRM error score, on the same benchmark × PRM. That makes the PRM the anchor.
- **Estimation pool.** Steps in `keep`: PRMBench non-control answers; Socratic included steps. Answers with a single
  step are excluded from estimation.
- **Fitting scope.** Pooled across answers within one benchmark × PRM (transductive, label-free). This is not
  answer-local fitting.

## Labels

- Labels are used only in the outputs marked VALIDATION.
- Those outputs are computed after the label-free estimates, which are written first and not changed afterwards.
- No label enters any orientation, threshold, binarization or estimate.

## A. Per-view GAP binarization (label-free)

- **Rule.** For each view and each answer, GAP (`evaluate_collection.gap_counts`, the frozen rule) is applied to the
  answer-z values. Its top-k steps get vote +1 (error), the others −1.
- **Abstention.** A view that is constant within an answer (std < 1e-8) abstains on that answer: all −1. Abstentions
  are counted.
- **Ties.** Broken by position, exactly as `topk_flags`.
- **VALIDATION: the view's own PRMScore under its GAP flags** (official evaluator). This gives a per-channel
  "classifier" table.

## B. Per-view accuracy without labels (binary votes, 17 voters)

- **B1. Spectral, Parisi 2014 + Jaffe et al. 2015.**
  - Rank-one fit of the off-diagonal vote covariance: u_i = √(1−b²)(ψ_i+η_i−1).
  - b from the third-moment ratio T_ijk / (u_i u_j u_k) = −2b/√(1−b²), median over distinct triples.
  - Then ψ_i (P(flag | error)), η_i (P(no flag | correct)) and the prevalence p = (1+b)/2.
  - The sign of u is fixed by the anchor (u_PRM > 0).
- **B2. Dawid-Skene EM** (two-coin, binary), initialized from the majority vote, 200 iterations or until
  |Δloglik| < 1e-8.
- **B3. Conditional-independence diagnostics, label-free.**
  - Share of the off-diagonal sum of squares not explained by the rank-one fit, overall and per view.
  - The FUSE triplet statistic: for each view k, the relative RMS residual of T_ijk regressed through the origin on
    Σ_ij over pairs i<j not containing k. Zero under triplet conditional independence.
- **VALIDATION.**
  - True ψ, η, balanced accuracy and prevalence from the labels.
  - Agreement of B1 and B2 with the truth: Spearman correlation over views, MAE.
  - True class-conditional vote correlations: the size of the off-diagonal entries of R_err and R_ok.

## C. Two-factor model on the continuous answer-z views (label-free)

- **Model.** A latent error factor u, plus one nuisance factor λ with λ_PRM = 0 (the anchor assumption).
  `spectral_utils.fusion_theory.anchored_loadings(X, anchor=0, method='fa')`, written and tested by the PRM math rigor
  session.
- **Reported per view:** u_k, λ_k, the uniqueness ψ_k = C_kk − u_k² − λ_k², and the one-factor triad estimate
  λ1_k² = median over g of C_kP C_kg / C_Pg, with its interquartile range over g. The spread is the copy/dependence
  signal.
- **Information on the error factor** (Bartlett score precision, nuisance projected out): I = 1/[(ΛᵀΨ⁻¹Λ)⁻¹]_11 with
  Λ = [u, λ].
  - Reported for the PRM alone (u_P²/ψ_P), for all 17 views, and for each view's drop-one change.
  - Label-free headline: I_all / I_PRM, the estimated factor by which telemetry adds information beyond the PRM after
    nuisance removal.
- **VALIDATION.**
  - Within-answer AUC and pooled AUROC of the Bartlett error-factor score and of the nuisance-factor score.
  - The PRM alone as reference.
  - The point-biserial correlation of the nuisance score with the labels, conditional on the error score (should be
    near 0 if the anchor assumption holds).

## D. Descriptive fused scores (VALIDATION only; not candidates)

The following go through GAP and the official metric next to the PRM (same rule) and CHAIN_N16:
- the B2 Dawid-Skene posterior log-odds over the 17 binary votes;
- the B1 SML log-odds weights applied to the votes;
- the C Bartlett error-factor score.

These are not selected and not promoted. Stage 1 has its own pre-registration.

## Predictions, written before any result

- **P1.** On PRMBench, the label-free per-view balanced-accuracy estimates (B1 and B2) rank the views close to the
  truth: Spearman ≥ .7 for at least 4 of 5 PRMs. If not, binary estimation is not reliable enough to select with.
- **P2 (perturbation-artifact hypothesis).** If the PRMBench telemetry signal is largely a "rewritten text" artifact:
  - the telemetry views' true balanced accuracy drops more from PRMBench to Socratic than the PRM's does;
  - I_all / I_PRM is smaller on Socratic than on PRMBench, for at least 4 of 5 PRMs.

  A drop does not prove the hypothesis (the benchmarks differ in other ways), and its absence weakens it.
- **P3.** Rank-one misfit is concentrated in telemetry–telemetry pairs, not in PRM–telemetry pairs.
- **P4.** The nuisance factor carries little label signal once the error factor is controlled (|partial r| < .05).

## Outputs

`results/prm_view_selection_v1/stage0/`:
- `stage0.py`, `run.log`, `RUN_META.json`
- `BINARY_VIEWS.csv`, `CI_DIAGNOSTICS.csv`, `TWO_FACTOR.csv`, `FUSED_DESCRIPTIVE.csv`
- `README.md` with a predictions check

Helpers go in `spectral_utils/binary_view_estimation.py`, with synthetic-recovery tests in
`tests/test_binary_view_estimation.py`. These must pass before the real run.

## Not in scope

- subset selection;
- Hard2Verify, ProcessBench, and the 958 self-generated Qwen3 answers (they need PRM inference, i.e. a GPU and a
  separate approval);
- any GPU job;
- any change to frozen results.

## Amendment 1, before the first real-data run (2026-10-25)

`fusion_theory.anchored_loadings(method='fa')` failed the synthetic recovery test.
- **Test case.** A clean two-factor case: 6 views, n = 80,000, known u and λ, λ_anchor = 0.
- **What happened.** Its alternating fit diverged (one nuisance loading reached about −2×10⁵), and the function
  abstained with "a loading exceeds its view variance".

Part C therefore uses a new fit, `binary_view_estimation.anchored_two_factor`:
- the same model and the same anchor constraint;
- unweighted least squares on the off-diagonal entries;
- Levenberg-Marquardt with 8 starts.

It recovers u, λ and ψ within .03 on the same synthetic case (`tests/test_binary_view_estimation.py`, 5/5 pass).
Nothing else changes. The divergence in the existing helper is reported to the PRM math rigor session; it is not
fixed here.
