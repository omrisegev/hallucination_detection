# Joint L-SML optimization v2 — Amendment R2 (pre-label)

Date: 2026-09-06, registered while the structure stage is still running
(41/45 folds; all 40 ProcessBench lanes frozen, PRMBench at 1/5) and BEFORE any
label access. Both items below were forced by observed label-free structure, and
neither may be decided after labels.

## 1. Coverage completeness rule for tuned selection (new)

**Observed**: across the 40 frozen ProcessBench lanes, exactly one roster row has
incomplete coverage — `dufs_pf_lsml` (R16) produced weights in 32/40 lanes and
**failed closed in 8** (`DUFS_PF_FAIL_CLOSED`: fewer than the registered 9
surviving features under the historical signed-mu>0 rule). Every other row,
including all four hook families and both successors, has complete 40/40
coverage. Missing lanes: gsm8k_q8/outer3, math_q4/outer0, math_q4/outer4,
math_q8/outer0, math_q8/outer2, olympiadbench_q4/outer1, omnimath_q4/outer3,
omnimath_q8/outer4.

**Rule (registered now)**: a roster row is eligible to be *selected* as a
panel's tuned configuration only if it has complete coverage across every
(cell x fold) lane of that panel, outer and inner. A row failing this is
`COVERAGE_INCOMPLETE`: it is excluded from inner selection and from the
tuned-vs-tuned readout, and reported descriptively on the lanes where it exists,
with its fail-closed count stated.

**Why this and not an imputation**: a panel-level macro-F1 or AUROC needs a
score on every lane; substituting a fallback for the missing lanes would make
R16 a spliced coverage policy — precisely the Step-348 `Joint-or-flat` construct
whose splice manufactured part of the failure being re-examined. Fail-closed
plus honest exclusion is the registered alternative. R16 keeps its role as the
DUFS hard-selector baseline wherever it is defined.

## 2. Correction to protocol Section 3.3 (factual)

The registered disclosure of 2026-09-05 stated the active-23 provenance family
sizes as "1/3/8/2/3/6". That figure was carried over from stale v1 planning text
and is **wrong**. The verified sizes are:

`entropy_level 1, entropy_dynamics 11, sampled_token_energy 2, partition_energy 3, topk_distribution 6` (sum 23).

The substance of the disclosure is unchanged and still holds: there is exactly
one size-3 family (`partition_energy`), so the Step-205 small-m guard is **not**
a no-op on provenance CONT arms — it fires once per provenance lane (observed:
exactly 40 guarded stages across the 40 ProcessBench lanes), and the v2
fixed-family arm is therefore the guarded variant, not bit-identical to the
Step-347 control. The `fixed_family_cont_unguarded` continuity row registered in
Amendment R1 remains the instrument that isolates this. Two families are
undersized (<3) and are merged by the registered deterministic
`provenance_merged_labels` rule for the joint arms only.

## 3. Label-free structure observed so far (recorded, not a result)

- INTERNAL grouping selected K=3 in 35/40 lanes and found no admissible
  partition in 5/40 (12.5%), which is **within** the Amendment-R1 fragility cap
  (>10 of 40 would flag `STRUCTURALLY_FRAGILE`); S1/S2 remain eligible. Those 5
  lanes fell back to provenance with the same map type, as registered.
- Hook 1 (gated-affinity grouping) was BLOCKED in 24/40 lanes (60%) and fell
  back to provenance. Recorded as a pre-label structural fragility of that hook.
- DUFS soft-gate seed stability: max mean_seed_std 0.0757 across all lanes,
  under the registered 0.15 cap.
- Pre-normalization fused-score scale spans 0.25 to 10.71 across arms and lanes
  (43x). This is the direct measurement of the Step-348/349 scale mechanism and
  the reason SD=1 is an invariant here.

None of the above uses labels.
