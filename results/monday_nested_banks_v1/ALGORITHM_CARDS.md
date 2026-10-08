# Algorithm cards (research agent, 2026-10-06; read-only synthesis, numbers copied from the cited files)

All estimator-vs-truth numbers are PRMBench only (prmbench_qwen3_8b, 94,203 steps, true prevalence 0.1396); ProcessBench has no full step labels. Development data. `C/` = `results/monday_fusion_analysis_v1/code/`, `RN` = `results/mom_filter_within_diagnostic_20261004/RESEARCH_NOTE_HE.md`, `EA` = `results/estimator_input_v1/run_20261004_resume/ESTIMATOR_ACCURACY.csv`.

## 1. Per-answer z-score and binary tail marks
- z: x̃ = (x − mean_answer)/sd_answer (constant column -> 0) — `C/spectral_utils/lsml_gate_locator_research.py:45-55`.
- marks: f = +1 on the top k = max(1, ceil(0.2·n)) steps of the answer, else −1; exact ties broken by a fixed random key — `C/scripts/experiments/er_stage_b.py:16-30`; keys rng(20260928) channels, rng(20260929) groups.
- tie-aware centred marks (partition only) — `C/scripts/experiments/calfix_common.py:204-223`; per-cell quota `quota_marks` — `C/spectral_utils/monday_fusion_core.py:59-69`.
- origin: project-internal; ±1 input required by Dawid–Skene / Parisi et al. 2014.
- marked share is 0.230 (not 0.20) because of ceil; every clean answer also gets marks (false votes by construction).

## 2. Dawid–Skene EM filter (keep π̂ > 0.5)
- P(f) = Σ_y P(y) Π_i P(f_i|y); EM with 5 starts; ψ_i = P(+1|err), η_i = P(−1|clean), π_i = (ψ_i+η_i)/2; keep π̂_i > 0.5 — `C/scripts/experiments/er_stage_a.py:118-139`, `monday_fusion_core.py:108`.
- origin: Dawid & Skene, J. R. Stat. Soc. C 28(1), 1979.
- vs truth (EA, channel votes, answer quota, mean of 8 banks): prevalence 0.313 vs true 0.140 (error 0.174); ψ-MAE 0.101; η-MAE 0.081; Spearman(π̂, π) 0.741 (0.407 B16 – 0.911 B20).
- does right: drops exactly the anti-oriented channels (5/5 folds; all 103 channels that would get a negative weight in 72 bank-cells). Does wrong: inflated sensitivity from shared level factor (q15_H1 ψ true 0.456 vs DS 0.757); top-3 pruning by π̂ does not help (−0.0001 [−0.0008, 0.0007]).

## 3. SML and L-SML
- SML: off-diagonal covariance rank-one q_ij = t_i t_j, t from leading eigenvector (`er_stage_a.py:55-77`, `C/spectral_utils/fusion_utils.py:342-419`); 3-unit guard = equal weights.
- L-SML: score s_ij = Σ |r_ij r_kl − r_il r_kj|, spectral clustering, K by residual (`fusion_utils.py:422,737,777,973`); anchor q15_H1 (72/72), orientation by Spearman with anchor.
- origin: Parisi, Strino, Nadler, Kluger, PNAS 2014 (SML); Jaffe, Fetaya, Nadler, Jiang, Kluger, "Unsupervised Ensemble Learning with Dependent Classifiers", 2016 (L-SML).
- quality: ranking only; Spearman equals MoM's (0.799 channel votes, 0.730 group votes). Full L-SML below plain average 8/8 banks (0.7532 vs 0.7656); between-group class-conditional |r| 0.14–0.31 (worst 0.72–0.86); adding 21 weak channels 0.7645 -> 0.6724.

## 4. Tensor method of moments (MoM)
- E[z_i z_j] = t_i t_j; E[z_i z_j z_k] = α t_i t_j t_k, α = −2b/√(1−b²); ψ, η closed form — `er_stage_a.py:80-115`.
- origin: Jaffe, Nadler, Kluger, "Estimating the Accuracies of Multiple Classifiers Without Labeled Data", AISTATS 2015 (Alg. 1).
- vs truth (EA): channel votes prevalence 0.244 (err 0.104), ψ-MAE 0.165, η-MAE 0.076, Spearman 0.799; group votes prevalence 0.211 (err 0.071), ψ-MAE 0.159, η-MAE 0.062, Spearman 0.738 (0.429–0.964).
- caveat: MoM prevalence reproduces the marked share (0.2306 vs 0.230) — not a measurement. As a filter instead of DS: −0.0014 PRMB, keeps an anti-oriented channel on B35.

## 5. HEM (latent-group EM, within-group weights)
- P(a_g|Y), P(f_i|a_g); w_i = max(0, log(e1(1−e0)/(e0(1−e1)))) — `C/scripts/experiments/ds_group_weights.py:24-73`.
- origin: dependent-classifier model of Jaffe et al. 2016, estimated by full EM; project-internal estimator (cvf_v2/em.py, Step 457).
- quality: Step 450 HEM over channels prevalence error 0.1435, ψ-MAE 0.1101, η-MAE 0.0717, Spearman 0.7791; learned within-group weights cost ~1 pt on ProcessBench (HEM_MOM 0.3552 vs MEAN_MOM 0.3656).

## 6. TCI_ONLY ("Prop 2.4" triplet selection)
- groups ≥ 4: for each triplet + anchor, V = mean_s[1 − (t·c)²/(|t|²|c|²)] with t_pq = E[z_p z_q z_s], c_pq = C_pq; keep argmin V; MoM between groups — `results/mom_filter_within_diagnostic_20261004/tci_wide_stage1.py:44-51,90-102`.
- origin: Proposition 2.4 of Lee et al., "FUSE: Ensembling Verifiers with Zero Labeled Data", arXiv:2604.18547 (2026) (a diagnostic there); the selection rule is Omri's (2026-10-05).
- quality: marks violate CI 2,080× (B13) / 3,228× (B32) vs a CI null; detects groups; 8 banks PRMB +0.0026 [0.0019, 0.0033] vs MEAN_MOM, unseen 6 banks +0.0001; ProcessBench −0.010 [−0.014, −0.005].

## 7. Between-group weights
- w_g = max(0, log(ψ_g η_g /((1−ψ_g)(1−η_g)))), Σ|w| = 1, ≥ 3 groups — `er_stage_b.py:38-41`, `monday_fusion_core.py:79-101`; ML vote weight (Dawid–Skene / Parisi 2014).
- quality: near-optimal between groups (MoM 0.7817/0.7537 vs supervised LR 0.7839/0.7613 on B13/B32); but LR on 30 random equal-size partitions 0.7874/0.7621; loss is equal within-group averaging (nonneg LR on all channels recovers 97%/86% of plain->LR gap); true group accuracies 0.7667 vs estimated 0.7670.

## 8. Partition and merge
- L-SML Alg. 1 on tie-aware marks, then absorb_merge: ρ(A,B) = λ2(R[A∪B])/min(λ1(R[A]), λ1(R[B])), merge while ρ < 0.5, never below 3 groups — `C/scripts/experiments/lsml_merge_step.py:24-46`.
- Monday: 0 grouping failures in 72 bank-cells, K 3–8; merged-partition equal weights below plain average every bank (0.7566 vs 0.7656); beats 2,000 random partitions on dependence but far from CI.

## Why binary tails
For: DS/SML/MoM/HEM theory needs ±1 votes; class-conditional between-group |r| lower on marks (0.08–0.21) than continuous (0.14–0.31); TCI violation ~10× closer to null on marks and positive controls more sensitive (z 13 vs 0.8); tie-aware marks give stable 4 groups 5/5 folds; mark partition beats L-SML's own on B13/B32.
Against/limits: loses on B20/B51; L-SML weights learned on marks lose to equal weights (−0.0022 [−0.0045, −0.0001]) — marks are estimator inputs only, continuous values are combined; prevalence not estimable from quota marks; quota fraction barely matters (14 thresholds: PRMScore 63.77–64.25); calibrated tails cut TCI violation 139->24 without improving localization; no head-to-head with continuous DS/MoM (undefined on continuous).

Reusable: `results/estimator_input_v1/run_20261004_resume/figures/F6_estimator_accuracy.*`, F1/F4/F5/F7/F10; `results/mom_filter_within_diagnostic_20261004/` tables; `.worktrees/decision-rule-v1/results/lsml_merge_step_v1/REPORT_HE.html`; `docs/meetings/advisor_prep_2026-10-05/sources/provenance.md`, `tensor.md`.
Unverified by the agent: Parisi 2014 and Dawid–Skene 1979 citations (from provenance.md); Step 451 5/5 partition stability (quoted from RN).
