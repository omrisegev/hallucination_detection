# History audit of the math peer review (2026-10-24)

**What this is.** For each of the 12 weaknesses (W1–W12) in
`docs/research_notes/math_peer_review_2026-10-24/MATH_PEER_REVIEW_HE.md`, this file records what the project
already tested, what it concluded at the time, and what is still missing. It also checks the review's cited numbers.

**How it was made.**
- Two read-only agents searched `HISTORY.md`, `PROGRESS.md`, `Research_Directions.md`, `LESSONS.md`,
  `results/*/README*`, `docs/**` and the code. Each searched six of the weaknesses.
- I re-checked the cross-cutting numbers myself from the saved result files (section 3).
- Branch `claude/prm-boost-math-rigor-2026-10-24` (= `claude/prm-boosting` @976845d5 + the review).
  No experiment was run for this file.

## 1. Bottom line

1. **No weakness is new to the project, and none is fully closed.**
   - Three were already answered, mostly in older lines:
     - the sign mechanism (W9);
     - posterior and count rules, which failed (W10);
     - "SML ≈ oriented mean" for telemetry-only fusion (W5).
   - The improvement condition and gate (W3) and the K-dependence of the PRM's weight (W8) were never tested
     as experiments.
   - The rest are partial.
2. **One mechanism underlies W1–W5 and W8: the method cannot estimate, without labels, how good the PRM is
   relative to the telemetry.**
   - The PRM's cross-stage share is ≈ 1/(K+1).
   - CHAIN ≈ ½ PRM + ½ mean on the strong PRMs.
   - The PRM's loading is never the largest (section 3).
3. **With labels, the sign of CHAIN − PRM is predictable in 15/15 cells** by "telemetry-alone within-AUC ≥ PRM
   within-AUC". The fusion helps exactly where it should. The missing piece is a label-free estimate of that
   comparison.
4. **The review's corrected weight u/(1−u²) (task 2) attacks the wrong part.** The weight formula is not the
   problem; the loading estimate is. With the PRM never the top loading, u/(1−u²) amplifies a telemetry group.
   The unbiased rank-one fit (the same loadings) already lost on 7 of 10 banks (Steps 467–468).
5. **Several cited numbers come from a different arm or benchmark than the review implies:** telemetry-only
   fits, Socratic-only fits, or the PRMBench bank sweep (section 4).

## 2. Weakness by weakness

| Weakness | Prior tests (step, file) | Verdict | Missing → where it is handled now |
|---|---|---|---|
| **W1** The latent may be position, not error | Step 466 / `results/prm_boosting_controls_v1/README.md`: CHAINRES (channels residualized on position + log length) still beats the same-rule PRM on 5/5 PRMBench PRMs; PERM (telemetry moved to another answer of equal length) drops CHAIN to .52–.59. Steps 245–246 (third-moment factor removal: "not identifiable as nuisance from unlabeled moments"), Step 263 (selector trusted a planted nuisance), Steps 461/464 (latent class ≠ error label; prevalence .23 vs .14), Step 459 (position alone .7307 Socratic / .8649 H2V; rule: every external protocol has a step-index row) | PARTIAL | External RES/PERM controls; corr(fused, position/length) per cell; rank-2 of the cross stage → Stage 1 |
| **W2** Shared nuisance breaks CI across groups | Steps 397, 414, 418 (CI atlas; 2.83 effective CI signals), 450–452 (shared level factor), `violation4` third-moment test used as a selector (`results/monday_external_panel_v1/score.py:112`: "TCI compatibility and verifier quality are separate axes"), LR-gap §6/§9 (`results/lr_gap_explanation_draft_2026-10-15/EXPERIMENTS_SUMMARY.md`: label share of between-cluster correlation 0.8–1.5%; removing position/length lowers it and drops the N56 oracle .810→.762) | PARTIAL | Within-class PRM-vs-group covariance on every cell (only Socratic exists, `sml_weight_estimator_v1` §1) → Stage 1 |
| **W3** No improvement condition, no gate | Mechanism v_p = √(r_pa r_pb / r_ab) (`results/sml_weight_estimator_v1/README.md` §4). "None of the weighting rules we tested fixes this" (advisor status 2026-10-21). Step 458: label-free switch, 0/80 with the family held out. Steps 450–452, 459: label-free sensitivities/prevalence biased. `results/prm_boost_socratic_diagnosis_v1`: supervised ceiling on Socratic only +.009 to +.024 for strong PRMs | PARTIAL (pattern documented, no proposition, no gate) | Proposition (P2) and gate → `FRAMING_THEORY.md`, `GATE_PREREG.md`, Stage 2 |
| **W4** SML weights are not Fisher/Bayes | Steps 451–452 (Dawid-Skene log-odds weights: no gain), 461 (label-free LDA slope 2–3.6× too large), 464 LR-gap (oracle LDA .7959 vs oracle vote .7834; label-free whitened SML ≤ .738), Codex two-component Gaussian latent model (no winner), Steps 467–468 (`rank1_offdiag_fit` loses on 7/10 banks) | PARTIAL | The exact u/(1−u²) on CHAIN's cross stage was never run. Expected to fail (section 1.4) → reference arm in Stage 3 |
| **W5** Shrinkage toward equal weights helps | Steps 450–457 (L-SML ≤ plain mean on 8/8 banks, Step 457), 464 (tensor MoM negative), 467–468 (oriented mean ≥ L-SML on 7/10 banks). With the PRM: CHAIN ≈ HALF (section 3) | ANSWERED for telemetry-only; PARTIAL for CHAIN | No λ sweep. EQ arm uses unflipped channel signs (energy_innovation, top50_js point the wrong way on PRMBench), so CHAIN − EQ mixes orientation with weighting → Stage 4 |
| **W6** Per-answer z-scoring | Steps 302 (centering: pooled AUC changes, within-answer ranking unchanged), 374 (centering, not scaling, causes the calibration loss), 453 (one-step answers: z ≡ 0), 454 (answer-z flags ~99% of clean answers), 459 (global-level per-answer count +.0070) | PARTIAL | Inside CHAIN: global-z / centering-only / rank arms, equal-answer-weight covariance, length bins, absolute-scale PRM → Stage 4. Theory: centering preserves the one-factor structure (P1) |
| **W7** Grouping heuristic, no stability | Steps 204 (loading scale drives K), 205 (knife-edge at m=4), 397–399/413 (Joint L-SML ARI 1.0 at every K; forced K spans 1.1 points), 447 (exhaustive partition ceiling), 451/455 (fold stability on B13 only). AUTO absorbs the PRM in 15/20 bank-sweep cells | PARTIAL | Refit ARI and K variance for CHAIN; forced-K sweep → Stage 4 |
| **W8** PRM weight depends on K | No experiment. The saved diagnostics confirm it (section 3) | CONFIRMED from saved data | Proposition (P3) → `FRAMING_THEORY.md` |
| **W9** Sign rule and anchor flip | Step 467: majority sign + anchor flip inverts CHAIN under a position confounder (fired in 5/20 Socratic arm-fits; `docs/reviews/number_audit_2026-10-21/AUDIT.md` I11), 0/60 on PRMBench; abs/clip/nnr1 remove the inversion but CHAIN still loses on 4/5 Socratic PRMs. Orientation provenance: N banks are raw channels "fixed by construction", no labels (`results/monday_nested_banks_v2/`); 2/16 N16 channels are natively inverted on PRMBench; signed SML ignores channel orientation (Steps 200, 439) | ANSWERED (mechanism); PARTIAL (fix) | Non-negative weights + abstain on violation become part of the new estimator's contract |
| **W10** Decision rules not model-derived | Step 459 (DS posterior > .5, DS expected-PRMScore threshold, DS expected count: all lose to Q80), Step 464 (third-moment prevalence .23 vs .14), Step 292 (count family A–D; ECE .176–.192), `results/error_count_label_free_v1` (third-moment prevalence .42 vs .14–.16; nnr1 count Spearman −.06; GAP − k=2 = −0.35) | ANSWERED (negative) | Not repeated. GAP is stated as a nonparametric split with a minority cap (P6). A label-calibrated count rule is in the calibrated track (`ADVISOR_TOOLS.md`) |
| **W11** Bootstrap does not refit | Code read: `results/prm_boosting_bank_sweep_v1/lsml_bank_sweep.py:164–178` reweights source groups on fixed scores. Closest refit: CROP C4 (`results/prm_boost_collection_v1/crop_controls/`) refits CHAIN on 60% of traces over 10 seeds (AUROC within ±.003) | PARTIAL | Full-refit bootstrap for the headline contrasts → Stage 4 |
| **W12** Transductive fit, forking paths | Steps 291–294 (source-disjoint folds), Step 462 (transductive vs pooled folds within ±.004; LESSONS rule: transductive label-free fitting is legitimate), CROP C4 (inductive ≈ transductive by trace), pre-registered external tests failed (Socratic 10-18, ProcessBench 10-20), disclosures on the ablation page; untouched test never done (`docs/research_notes/DIRECTIONS_INVENTORY_2026-09_10.md`) | PARTIAL | Source-group split-half, Holm, and the gate's own pre-registration (`GATE_PREREG.md`). Pareto Testing over the design grid (calibrated track). An untouched holdout needs GPU inference and Omri's approval |

## 3. Cross-cutting numbers I re-checked

| Claim | Value | Source |
|---|---|---|
| PRM cross-stage share × (K+1) | median 1.03, range .89–1.32 (24 non-Math-Shepherd cells) | `results/prm_boosting_bank_sweep_v1/SWEEP_DIAG_*.json`, `CHAIN_prm_cross_share`, `CHAIN_n_views` |
| PRM rank among CHAIN cross weights (N11/N13/N14/N16) | never first; rank 2–6 of K+1 in 20/20 fits | same files, `CHAIN_cross_weights` |
| CHAIN_N16 vs HALF_N16 within-AUC (Q80 rows) | Qwen-7B .8377/.8373, Qwen-72B .8428/.8424, PURE .8156/.8128, RLHFlow .7749/.7303, MS (digit-fixed) .7783/.7698 | `results/prm_boost_ablation5_v1/ABL5_RESULTS_*.csv`, `digitfix_v1/digitfix/` |
| Sign of CHAIN − PRM predicted by TELE ≥ PRM (within-AUC) | 15/15 cells; MS Hard2Verify is +.002 | `results/prm_boost_collection_v1/tables/STATUS_MAIN.csv` |
| Observed CHAIN > PRM | 9/15 (PRMBench 5/5, Socratic 1/5, Hard2Verify 3/5) | same |
| External labels propagate after the first error | Hard2Verify 80 of 158 erroneous answers mark every later step wrong; Socratic 896 of 2,989 | `results/remote_claims_replay_2026-10-15/inputs/*_gold.json` |
| The review's Gaussian toy, population values | CI: PRM .856, Bayes .881 (share .45), equal .811, SML eigenvector .845 (share .24). Nuisance: Bayes .867, equal .720, SML .653 (share .046) | recomputed in pure Python from the review's setup |

## 4. Corrections to the review

1. **Toy table, CI row.** The shipped SML eigenvector is below the PRM alone (.845 vs .856), not above it
   (.858). The nuisance row is .653 vs the review's .664. The conclusion gets stronger.
2. **W6.** Centering per answer does not break the one-factor structure. The signal becomes y − ȳ_j, still one
   latent times Δ. A simulation with a shared answer-level factor shows centering is what points the rank-one
   loading at the error (cos with Δ: 1.000 centered vs .639 raw). Only the per-answer SD scaling and very short
   answers are real issues.
3. **W4 / task 2.** u/(1−u²) on unanchored 'complete'-scale loadings gives .672 vs the PRM's .856 in a
   Socratic-like simulation. The PRM's loading is estimated at .12 instead of .47. The estimate is the
   problem, not the formula.
4. **W3.** "Weak" and "strong" hold per cell, not per model. Qwen-72B is weak on Hard2Verify (within-AUC .513),
   and fusion helps it there (+7.85 PRMScore).
5. **W7.** "AUTO absorbs the PRM in 15/20 cells" refers to 4 PRMBench banks × 5 PRMs, not 4 benchmarks × 5 PRMs.
   The Math-Shepherd N14/N16 rows predate the digit fix (AUDIT X1).
6. **W9.** ".468" is CHAIN_N13 on Socratic (= pre-fix N16). After the digit fix N16 is .427.
   "4 of 10 banks with a negative cross weight" is telemetry-only L-SML; CHAIN on PRMBench has 0/30.
7. **W10.** "GAP without cap 57.8" and "GAP − K2 = −0.35" are telemetry-only L-SML. CHAIN is 58.80 and
   −0.40 [−0.77, +0.00].
8. **W12.** "~208 diffs" matches the project's own page. The file has 184 contrasts × 2 metrics, 210 with
   controls (AUDIT M7). The digit bug was recorded on 10-14 (before Math-Shepherd lost) and fixed on 10-19.

## 5. Repository corrections found on the way

- `results/sml_weight_estimator_v1/README.md` §1 still says the anchor flip fired in 3/20 Socratic fits.
  The 10-21 audit counts 5/20 arm-fits (AUDIT I11).
- That README cites `COV_DECOMPOSITION.txt`, which is not in the repository. `cov_decomp.py` is.
- `docs/research_notes/Bracha Laufer's Research_ LLMs and Anomaly Detection.md` credits her with Learn-then-Test,
  CATs, early-exit and COIN papers by other authors.
- `docs/research_notes/Ofir Lindenbaum's Research Areas.md` credits the RADAR benchmark (Gu et al.) to his lab.
  Both notes came from Gemini. Details in `ADVISOR_TOOLS.md`.
