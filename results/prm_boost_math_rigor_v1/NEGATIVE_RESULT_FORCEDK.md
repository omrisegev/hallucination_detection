# Negative result — CHAIN K robustness, forced K = 2..8 (prm-boost-auto-forcedk-v1, part B2)

**Hypothesis (one sentence):** CHAIN's automatic K gives a GAP PRMScore within 0.5 of the best forced K = 2..8, in
≥ 4/5 PRMBench PRMs.

**Date / step:** 2026-10-25, Step 475 [PRM math rigor]. Pre-registration committed first: `fa72e7ec`.

**Benchmark / population:**
- PRMBench, all 6,969 answers, 5 PRMs. Socratic (2,995) and Hard2Verify (177 common answers) are reported.
- Access: label-free fitting; labels only score.

## Arms compared
Source: `stage8/AUTO_FORCEDK_RESULTS.json`, PRMBench GAP PRMScore (within-AUC).

| arm | what it is | Qwen-7B | Qwen-72B | PURE | RLHFlow | MS |
|---|---|---|---|---|---|---|
| reference (incumbent) | CHAIN_N16, automatic K | 70.40 (.838), K4 | 71.09 (.843), K6 | 69.51 (.816), K4 | 66.25 (.775), K4 | 66.39 (.778), K4 |
| best forced K (label-chosen, not a method) | max over K = 2..8 | 70.70, K3 | 71.77, K5 | 69.52, K5 | 66.36, K5 | 67.48, K2 |
| auto − best forced | B2 margin −0.5 | −0.30 ✓ | **−0.68 ✗** | −0.02 ✓ | −0.11 ✓ | **−1.09 ✗** |
| simple control | AUTO_N16 (free grouping incl. PRM) | 69.87 (.824) | 70.18 (.828) | 69.06 (.799) | 65.92 (.765) | 64.78 (.746) |

**Win/loss record:**
- **B2: met in 3/5 PRMs**, and 4 were needed. **Not met.**
- **A, AUTO vs CHAIN: CHAIN preferred.**
  - Within-AUC Δ(AUTO − CHAIN) is −.010 to −.033, every CI below 0 (5/5).
  - The PRMScore Δ is −0.33 to −1.61. The CI is below 0 in 4/5 PRMs; for RLHFlow it is [−0.73, +0.08].
- **B1 (the P3 prediction): met.** The Spearman correlation between K and the cross-stage PRM share is −1.00 in 5/5.
  Share × (K+1) is .93–1.37 for four PRMs, and .23–.89 for MS.

## Three plausible reasons it failed
1. **K acts as a knob on the PRM's share.**
   - P3 predicts this, and B1 shows it: the share goes from .25–.37 at K = 2 down to .03–.13 at K = 8.
   - So the PRMScore depends on K wherever the PRM's share matters.
   - Example, MS: K = 2, 3 (share ≥ .22) score 67.3–67.5, and K ≥ 5 (share ≤ .07) score 62–64.5.
2. **The residual criterion that chooses K does not target fusion quality.** Steps 203–204 already noted that the
   criterion is a poor guide. It picks a reasonable K for three PRMs, but misses by about 1 point for MS and
   Qwen-72B.
3. **The PRMScore of a single K is noisy.** Qwen-72B's curve is not smooth: K4 69.59, K5 71.77, K6 71.09. Part of
   the gap to the "best K" is maximization over noise. The best K is chosen with labels.

## Verdict
- [ ] closes the DIRECTION (the idea cannot work under this access/contract)
- [x] closes this IMPLEMENTATION only (roster, lambda, readout, bank); the idea stays open

K is not a neutral choice. It sets the PRM's share, and the automatic criterion does not choose it for quality. A
better design would decide the PRM's share explicitly (the improvement condition, P2) rather than through K.

## What would reopen it
A label-free estimate of the PRM's share that does not depend on K. For example, an estimated cross stage whose
PRM weight is invariant to how the telemetry is split, like the block-CCA/Fisher weights in `ADVISOR_TOOLS.md`. A
fixed weight does not count: fixed-weight averaging is ruled out as a method.

**Source files:**
- Results: `stage8/AUTO_FORCEDK_RESULTS.json`, `stage8/AUTO_FORCEDK_VERDICT.json`.
- Code: `stage8_auto_forced_k.py`. Pre-registration: `AUTO_FORCEDK_PREREG.json`.
- Command: `COLLECTION_DATA=<data> python stage8_auto_forced_k.py stage8`.
- n_checked / n_total: 15 / 15 cells; 9 arms per cell (CHAIN, AUTO, 7 forced K).
