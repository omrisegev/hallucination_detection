# Record index — Stages 1–3b of the 2026-09-12 plan (Claude, 2026-09-13)

One entry point to every experiment, method, result file, review and code path produced under Omri's staged
mandate of 2026-09-12/13. All runs use the frozen localization benchmark (13,769 answers: ProcessBench 6,800 in
8 cells + PRMBench 6,969; v3 labels; FOLDS_V2 source groups; top-10 token-mean step readout; external mean-entropy
gate q = 0.3; PRMScore q = 0.8 on held folds; 10,000-draw paired source-group bootstrap). All numbers are
development evidence. Framing: no consistent overall advantage from learned fusion has been demonstrated;
representation, optimization, normalization and readout remain partly entangled.

| Stage | Question | Method(s) | Result files | Note / review | Code | HISTORY |
|---|---|---|---|---|---|---|
| 1 sampling | Does choosing which windows fit the fusion weights help? (Omri's DUFS-window idea; Codex's window/GMM contract) | 8 selectors × 7 cores, 56 arms | `results/localization_full_sampling_v3/evaluation/{METRICS,INTERVALS,REVIEW}.json` (root) | `docs/reviews/research_consolidation_2026-09-08.html` (root); Stage-1 account §6 | `scripts/run_full_sampling_v3.py`, `complete_research_consolidation_v1.py` (root, Codex) | 332 completion, 356 |
| 1 RBM stability | Is the Gaussian RBM start-dependent? | H1/H4, 3 exact starts, min-NLL | `.worktrees/rbm-literature-completion-v1/results/rbm_literature_completion_v1/stability/` | `RBM_PROGRAM_STAGE1_ACCOUNT_2026-09-12.md` §4 | `scripts/run_rbm_literature_program.py` (Codex) | 356 |
| 1 RBM capacity | Why does exact H4 not converge? | convergence/unit-condition analysis + maxiter probe | `.../capacity/CAPACITY_CONVERGENCE.{csv,json}`, `CAPACITY_INTERPRETATION.md` | account §2 | `scripts/analyze_rbm_capacity_convergence.py` | 356 |
| 1 RBM depth | Second layer on H4 | original (declared failures) + logit amendment | `.../depth_amended/` | `docs/experiments/RBM_DEPTH_AMENDMENT_20260912.md`; account §3; `STAGE1_REVIEW_FIGURES/` | `scripts/run_rbm_depth_amended.py`, `review_rbm_depth_amended.py` | 356, 357 |
| 1 account | One table across all Codex RBM suites + controls | — | `.../STAGE1_COMPARISON_TABLE.{md,csv}`, `STAGE1_CONTRASTS.md` | `docs/research_notes/RBM_PROGRAM_STAGE1_ACCOUNT_2026-09-12.md` | `scripts/build_stage1_account_tables.py` | 356 |
| 2 cross-rank | Do the cross-rank products of the varentropy expansion help? (B2_sel vs B2d_sel) | equal / IU / shrink IU / Joint L-SML; supervised step-level diagnostic | `results/varentropy_expansion_fusion_v1/{fast_pass,joint_pass,supervised}/` | `docs/experiments/VARENTROPY_EXPANSION_FUSION_V1.md`; `docs/research_notes/VARENTROPY_EXPANSION_STAGE2_INTERIM_2026-09-12.md`; `*/REVIEW_FIGURES/` | `spectral_utils/varentropy_expansion_fusion.py`, `run_varentropy_expansion_*.py`, `review_varentropy_expansion_fusion_v1.py` | 356b, 357 |
| 3 Rényi combination | Does combining several Rényi orders beat one entropy? | α {0.1, 0.25, 0.5, 1, 2, ∞}; R6 / R6+SEL; equal / IU / shrink / Joint | `results/renyi_view_fusion_v2/{fast_pass,joint_pass}/` | `docs/experiments/RENYI_VIEW_FUSION_V2.md`; `docs/research_notes/RENYI_VIEW_FUSION_STAGE3_2026-09-13.md`; `fast_pass/REVIEW_FIGURES/` | `spectral_utils/renyi_view_fusion_v2.py`, `run_renyi_view_fusion_v2.py`, `review_renyi_view_fusion_v2.py`, `test_renyi_view_fusion_v2.py` | 358 |
| 3b α sweep | Which α is best? Is there a varentropy with escort weights? | 20 H_α views incl. the α→0 limit; 11 escort-varentropy VE_α views | `results/renyi_alpha_sweep_v1/` (+ `SELECTION.md`, `REVIEW_FIGURES/`) | `docs/research_notes/RENYI_ALPHA_SWEEP_STAGE3B_2026-09-13.md` | `spectral_utils/renyi_alpha_sweep.py`, `run_renyi_alpha_sweep_v1.py`, `review_renyi_alpha_sweep_v1.py`, `renyi_alpha_sweep_selection.py` | 359 |

Worktrees / branches: Stage 1 RBM work in `.worktrees/rbm-literature-completion-v1` (branch
`codex/rbm-literature-completion-v1`); Stages 2–3b in `.worktrees/varentropy-expansion-fusion-v1` (branch
`claude/varentropy-expansion-fusion-v1`, sparse code-only checkout; large sqlite checkpoints stay untracked).
Root logs (`HISTORY.md`, `PROGRESS.md`, `Research_Directions.md`) carry the same Step 356–359 entries.

## Headline numbers to remember (PB all-8 % / PRMB within-AUC / PRMScore)

| Score | PB | within | PRMScore | Where |
|---|---:|---:|---:|---|
| token entropy (reference) | 35.44 | 0.730 | 0.625 | frozen |
| varentropy15 (reference) | 35.96 | 0.738 | 0.626 | Step 339 |
| best learned RBM row (low-corr-6, not a registered primary) | 36.99 | 0.740 | 0.631 | Stage 1 |
| cross-rank products under IU (B2_sel) | 33.95 | 0.727 | 0.616 | Stage 2 (products hurt: −0.80 pp vs B2d_sel) |
| six Rényi orders fused, IU (R6) | 35.62 | 0.731 | 0.598 | Stage 3 (below the best single order) |
| Rényi α→0 limit (mean log q over top-15) | 35.53 | **0.744** | 0.633 | Stage 3b |
| escort varentropy α = 0 (equal weights, minus sign) | 35.57 | **0.753** | **0.636** | Stage 3b (best PRMB single stream) |
| escort varentropy α = 0.75 | **36.76** | 0.732 | 0.619 | Stage 3b (highest PB point; interval includes 0) |

## Method definitions introduced in these stages (all label-free, answer-local)

* **Varentropy expansion columns** (Stage 2): D_i = q_i s_i², P_ii = (q_i s_i)², P_ij = q_i q_j s_i s_j (i < j);
  identity Σ D − Σ P_ii − 2 Σ P_ij = varentropy15 (checked to 1e-9).
* **Rényi views** (Stage 3): H_α = log(Σ q^α)/(1 − α) on the renormalized top-15; H_1 = entropy15; H_∞ = −log q_1;
  H_0 = log 15 (constant, excluded).
* **α→0 limit view** (Stage 3b): mean_i log q_i over the top-15 (ordering limit of H_α as α → 0).
* **Escort varentropy** (Stage 3b): VE_α = Σ w_i s_i² − (Σ w_i s_i)², w ∝ q^α; VE_1 = varentropy15; for α ≤ 0.25 the
  natural sign is reversed (anchor flip on 100 % of answers).
* **Roster passes** (Stages 2–3): fast (non-Joint) and joint passes scored separately on the same population; the joint
  evaluation appends the fast-pass step scores with an identity check.
* **Reviewer pattern**: independent replay from raw log-probabilities with a python-sorted top-10 mean, separate metric
  arithmetic, manifest-hash binding, and a PNG figure review with arithmetic checks by a separate agent.
