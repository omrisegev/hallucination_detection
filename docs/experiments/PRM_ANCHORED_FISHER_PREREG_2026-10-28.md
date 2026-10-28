# Pre-registration: PRM-anchored Fisher fusion (P5) on all local data (CPU, 2026-10-28)

Written and committed before the first run on real data. Development data; labels only score.
Branch: `claude/prm-anchored-fisher-2026-10-28`.

## Why

Omri, 2026-10-28, asked to implement the algorithm of FRAMING_THEORY P5 and run it on all the data we have. Under
P5's assumptions, the algorithm does three things:
- it weights each telemetry group by its covariance with the PRM, instead of by agreement among the views;
- it identifies the PRM's own loading without labels;
- it uses Fisher weights, which do not assume conditional independence.

**What exists already.**
- `spectral_utils/fusion_theory.anchored_loadings` and `gate_decision` were used in Step 469, but only as a
  fuse-or-PRM gate. That gate abstained in 15/15 cells.
- The fusion with anchored Fisher weights has never been scored on real data.
- Step 476's AUTO-F used Fisher weights without the anchor, and it failed.

## Algorithm (`spectral_utils/prm_anchored_fisher.py`; synthetic tests `tests/test_prm_anchored_fisher.py`, 5 pass)

1. **Views.** CHAIN_N16's cross-stage views: z(PRM risk) and the answer-z telemetry group scores, from L-SML
   Algorithm 1 on the 16 channels. Parity: they must equal `prm_boost_math_rigor_v1/common.py` `chain_inputs` VC to
   1e-9 in every cell; otherwise stop.
2. **Estimation matrix.** The views residualized within answers on cubic relative position, plus log step tokens
   where available (PRMBench). This is the same basis as Step 469.
3. **Exclusion.** Groups with covariance ≤ 0 with the PRM, on the estimation matrix, get weight 0.
4. **Loadings u.** From `anchored_loadings` on the remaining views.
5. **Weights.** w = C⁻¹u on the estimation covariance, not clipped. The score is views · w, with the raw (not
   residualized) views.
6. **Fallback.** If u is not identified, or fewer than 2 groups remain, the score is the PRM view.

## Arms

| Arm | Route for u | Fallback |
|---|---|---|
| **ANCH_AUTO (primary)** | `rank` if at least 4 groups remain, else `cumulant`; `cumulant` also when `rank` is not identified | PRM |
| ANCH_RANK | `rank` only | PRM |
| ANCH_CUM | `cumulant` | PRM |
| ANCH_CUMW | `cumulant_w01` (winsorized at 1%) | PRM |
| ANCH_CCA | `cca` | PRM |
| ANCH_AUTO_G | as ANCH_AUTO, then fuse only if the point estimate G(w) > 0 | PRM |
| ANCH_AUTO_FBC | as ANCH_AUTO | CHAIN_N16 |

**References.**
- PRM (STATUS_MAIN "PRM same rule");
- CHAIN_N16 (recomputed; must equal STATUS_MAIN within 1e-6);
- TELE_N16 (STATUS_MAIN);
- **ORACLE_FISHER**, labels used and in-sample: LDA weights on the same views, Σ_within⁻¹(μ_err − μ_ok). This is
  headroom, not a method.

## Cells

PRMBench, Socratic-PRMBench and Hard2Verify × qwen_prm_7b, qwen_prm_72b, pure_prm, rlhflow_prm and math_shepherd:
15 cells, which is all the local data.
- The main reading uses the 4-PRM line (12 cells). Math-Shepherd is reported separately.
- Hard2Verify follows CHAIN's recipe: fit on all loaded answers, score on the 177-answer common set.
- ProcessBench is not run: it is out of focus, and the collection evaluator has no loader for it.

## Endpoints

- **Primary:** the official metric under GAP (PRMBench PRMScore; external 0–100).
- **Secondary:** within-answer AUC.
- **Paired bootstrap:** 2,000 draws, seed 20261028; PRMBench resampled by source group, external sets by answer.
- **Contrasts:** ANCH_* − CHAIN_N16 and ANCH_* − PRM.
- **Diagnostics per cell:**
  - whether u was identified, and by which route;
  - the excluded groups;
  - the PRM's weight share;
  - the number of negative weights;
  - G.

## Reading rule (fixed now)

- **R1.** ANCH_AUTO is better than CHAIN_N16 if it gains (95% interval above 0) in at least 4 of the 12 line cells
  and loses (interval below 0) in none.
- **R2.** On the strong-PRM Socratic cells (Qwen-7B, Qwen-72B, PURE), ANCH_AUTO "reaches the PRM" in a cell if its
  interval against the PRM is not entirely below 0. R2 holds if this happens in at least 2 of 3.
- **R3.** Report the identification rate. If ANCH_AUTO falls back to the PRM in more than half of the 12 cells,
  the reading is "not identified on this data", whatever R1 and R2 say.
- **Nothing is tuned on these outputs.**

## Limits stated in advance

- The anchor assumption is partly violated: Step 469 measured within-class |corr| .13–.35 between the residualized
  PRM and the group scores.
- Step 469's cumulant route abstained in 15/15 cells, mostly because "a loading exceeds its view variance". ANCH_CUM
  is expected to fall back often.
- All three benchmarks were inspected before. This is development evidence.
