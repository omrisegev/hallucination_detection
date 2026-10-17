# Joint L-SML v1 — Agent B structural report

This run is label-free structural development only. It does not compute benchmark efficacy, open a scoring arm, or support promotion.

## Task 1 — ridge target diagnostic

18/18 frozen C-v2 lanes have minimum pairwise donor-score Spearman >= 0.99 across target conditions 1e2, 1e3 and 1e4.

- Observation: the plot reports rank stability of the actual donor fused scores Xw, replacing coefficient-space cosine.
- Inference: a passing lane is insensitive in ranking to this frozen regularization range on its donor rows.
- Limitation: this is neither outcome performance nor out-of-population stability.

## Orientation and global pruning

The global V2 roster retains 23/28 active raw streams. Removed streams: entropy_rolling_hl_ratio, entropy_pe_series, spilled_cusum_abs_series, spilled_rolling_min, energy_cusum_abs_series.

Weak: entropy_rolling_hl_ratio, entropy_pe_series, spilled_rolling_min. Sign-unstable: entropy_pe_series, energy_cusum_abs_series. Degree-rejected: entropy_rolling_hl_ratio, entropy_pe_series, spilled_cusum_abs_series, spilled_rolling_min, energy_cusum_abs_series.

- Observation: sign is estimated independently in nine cells and opacity in the heatmap tracks |v|.
- Inference: streams without stable/meaningful orientation are excluded from geometry and fusion rather than repaired ad hoc.
- Limitation: signs are gauge-fixed by entropy_series and remain donor-population estimates.

## Joint disjoint-group estimator

Joint L-SML produced an admissible fit in 16/18 lanes; 2 lanes had no K whose consensus and every LOAO fold kept all groups at size >=3. Among all 18 lanes, joint misfit is lower in 16 and 16 pass convergence, multistart, profiled-Jacobian and finite-weight checks.

- Observation: every fitted lane uses a K>=3 LOAO-consensus partition chosen by ARI stability, never by residual fit; blocked lanes have no fitted model.
- Inference: the joint estimator directly tests whether a shared factor plus disjoint group factors explains donor covariance more faithfully than the historical two-stage factorization.
- Limitation: lower covariance misfit does not imply better localization; no labels or benchmark scores were accessed.

## Weight-map comparison

The per-lane minimum pairwise donor-score Spearman among hierarchical-joint, model-inverse, sample-inverse and existing continuous L-SML spans 0.708240 to 0.976135.

- Observation: the comparison is made in score/ranking space on identical donor rows.
- Inference: disagreement localizes the practical consequence of changing the loading estimator or covariance map.
- Limitation: these are retrospective donor diagnostics and no fused score arrays were saved.

## Agent A handoff

- Absolute orientation registry: `/Users/osegev/Desktop/hallucination_detection/local_cache/worktrees/og_sml_agent_b_v1/results/joint_lsml_v1_r2/V2_ABSOLUTE_ORIENTATION_REGISTRY.json`
- Global pruned roster: `/Users/osegev/Desktop/hallucination_detection/local_cache/worktrees/og_sml_agent_b_v1/results/joint_lsml_v1_r2/V2_GLOBAL_PRUNED_ROSTER.json`
- trace_length is nuisance-only and excluded from the active roster.
- This handoff does not authorize overlap, LAG, T1/T2, or scoring.

## Plots

- `task1_ridge_score_stability.svg`
- `task1_ridge_score_stability.png`
- `orientation_sign_stability.svg`
- `orientation_sign_stability.png`
- `joint_lsml_structural_overview.svg`
- `joint_lsml_structural_overview.png`
- `loao_consensus_stability.svg`
- `loao_consensus_stability.png`
