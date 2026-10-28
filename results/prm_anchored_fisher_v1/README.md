# PRM-anchored Fisher fusion (P5) on all local cells (CPU, 2026-10-28)

**Pre-registration:** `docs/experiments/PRM_ANCHORED_FISHER_PREREG_2026-10-28.md` (7203cb67, committed before the
run).

**Algorithm:** `spectral_utils/prm_anchored_fisher.py`; synthetic tests in `tests/test_prm_anchored_fisher.py`
(5 pass).

**Run:** `run_anchored.py` → `RESULTS.json`, `RESULTS.csv`, `PARITY.json`, `run.log`.
- **Cells:** 15, PRMBench, Socratic and Hard2Verify × 5 PRMs.
- **Data:** development data only; labels only score.

**Parity before reading:**
- the module's cross-stage views equal `chain_inputs`: max diff 0.0, 15/15;
- the recomputed CHAIN_N16 equals STATUS_MAIN: max diff 1.4e-14, 15/15.

## Verdict: not identified on this data

The pre-registered rule R3 decides it. The anchored weights were identified in only **3 of the 12** line cells
(4 PRMs × 3 benchmarks), and in 2 of the 3 Math-Shepherd cells. Everywhere else the arm falls back to the PRM view.

| Rule | Result | Met? |
|---|---|---|
| R1: gains over CHAIN_N16 in ≥ 4 of 12 cells, no losses | 4 gains, 7 losses | no |
| R2: does not fall below the PRM in ≥ 2 of the 3 strong-PRM Socratic cells | below in 3/3 (−0.6, −3.6, −0.6) | no |
| R3: identified in more than half of the cells | 3/12 | no, so the verdict is "not identified" |

**Why it falls back.** These are the reasons `anchored_loadings` gives, per cell, in `RESULTS.csv` (`reason`).

| Route | Reason it fails | Meaning |
|---|---|---|
| Second order (`rank`) | "flat rank profile", or "not identified" | The telemetry block cannot separate the error factor from the nuisance factor. The nuisance is much larger than the error signal in the group scores. |
| Third order (`cumulant`) | "a loading exceeds its view variance" | u₀ comes out too small, so u_k = C₀k/u₀ is too large. This is the same failure as Step 469's gate. |

**Where it was identified (line cells).**
- **Socratic RLHFlow:** 53.05, against the PRM's 50.38 and CHAIN's 57.06. The PRM share is .52.
- **Hard2Verify PURE:** 47.59, against the PRM's 50.76 and CHAIN's 39.96. The PRM share is .10.
- **Hard2Verify RLHFlow:** 6.47, against the PRM's 5.11 and CHAIN's 33.06. Here the near-useless PRM got .55 of the
  weight.

**The 4 "gains" over CHAIN are fallbacks, not the algorithm.**
- **Where:** Socratic Qwen-7B (+2.6), Qwen-72B (+10.7) and PURE (+5.0), and Hard2Verify PURE.
- **What happened:** in the three Socratic cells, the arm returned the PRM view. Returning the PRM avoids CHAIN's
  collapse there.
- **A detail that matters:** the fallback is CHAIN's own PRM view, z(logit(1 − reward)). On PRMBench that view
  scores below the "PRM same rule" comparator, which uses the better of two scales chosen after scoring. For
  example, on Qwen-7B it gives 66.37 against 68.01.

## The label-using reference (ORACLE_FISHER: LDA weights on the same views, in-sample; headroom, not a method)

- **PRMBench:** above CHAIN in 4/5 cells (+0.7 to +1.1). Level on Qwen-7B (−0.08).
- **Socratic, within-AUC:** above the PRM in all 3 strong-PRM cells (.828/.828/.818, against .824/.814/.796).
- **Socratic, PRMScore under GAP:** below the PRM on Qwen-72B (−2.2) and PURE (−1.0); above it on Qwen-7B (+1.3).

So even the best linear weights with labels gain little over a strong PRM on Socratic, and GAP's binarization can
erase that small gain.

## Reading

**The P5 identification needs signal that the views do not carry.** The second-order route needs the error part of
the telemetry covariance to stand out from the nuisance. On this data it does not: Step 469 found that 82–100% of the
covariance is structure within the classes. The third-order route returns implausible loadings.

**What it rules out.** The anchored algorithm is not a fix for the strong-PRM Socratic gap *on this telemetry*. When it
does help there, it is only by falling back to the PRM.

**What it does not rule out.** The anchor idea with views whose noise is not shared with the PRM. These are the
counterfactual PRM readings proposed in `docs/research_notes/PRM_MODEL_NEW_FEATURES_2026-10-26.md`
(branch `claude/prm-new-features-2026-10-26`).
