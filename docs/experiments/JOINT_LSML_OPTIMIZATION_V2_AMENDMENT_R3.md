# Joint L-SML optimization v2 — Amendment R3 (pre-label)

Date: 2026-09-06, registered BEFORE any label access, while the last two PRMBench structure
folds (outer3/outer4) were still running. Triggered by the pre-label structural review
(`scripts/joint_lsml_optimization_v2/prelabel_structure_review.py`, this amendment's companion),
not by any label-derived quantity.

## 1. Gap found by the pre-label review

Protocol Section 7.4 makes gate inertness a mechanical guard: a gated row whose weight map has
cosine >= 0.995 with its ungated (lambda = 0) map on every lane is `MECHANISM_INERT` and cannot be
frozen for fresh data. For the Hook 2 rows and the Hook 1 rows the lambda = 0 reference is a
frozen roster row (`prov5_cont`, `internal_cont`, `internal_joint`), so the guard is computable
from the frozen artifacts. For the Hook 3a/3b rows (`internal_joint_liu010/050`,
`internal_joint_diag010/050`) the lambda = 0 reference is the ungated model-inverse map
`(Psi(C_model) + gamma I)^-1 v` — verified as the exact identity target in the audit suite, but
never written as a frozen row. Without it the inertness guard cannot be applied to four of the
sixteen Joint-family rows, and the dose pairs already show that the two lambda values barely
move those maps (median cosine 0.988 / 0.989 between lambda = 0.1 and 0.5 on the 43 frozen
lanes), so inertness is a live possibility that must be decidable before labels.

## 2. Repair (additive, descriptive, non-selectable)

- One new row id `internal_joint_modelinv_lam0` in `fit_v2_arms`: the shared INTERNAL joint fit
  (`_internal_joint_fit()`, identical seed and grouping to R4/R10–R13) followed by
  `regularized_joint_map_weights(mode="liu", lam=0.0)` — i.e. the exact lambda = 0 identity
  branch, then the shared SD = 1 + orientation boundary. It is NOT added to `LSML_ROSTER`
  (budget stays 16 vs 16), is not a selection candidate, is excluded from the coverage census,
  and is reported only as the inertness reference for Hook 3a/3b.
- Third additive pass `scripts/joint_lsml_optimization_v2/third_pass_amendment_r3.py`, run per
  (cell x OUTER fold) only (the inertness and cross-fold guards are outer-lane guards), writing
  `scores_amend_r3.npz` (`internal_joint_modelinv_lam0__{w,top10,spanmax,detector}`) and
  `MANIFEST_AMEND_R3.json` with exact relative paths. Frozen `MANIFEST.json`,
  `scores_outer.npz`, and the R1 files are never rewritten.
- The pre-label review then evaluates the Hook 3 inertness guard against this reference and
  reports the outcome per row (`MECHANISM_INERT` or not) before the evaluator is invoked.

## 3. Ordering constraint

The module edit adding the row branch is applied only after every structure/R1 process has
ended (the 2026-09-06 late integrity amendment forbids replacing files or reloading modules
mid-run). The third pass therefore runs after the R1 second pass on all 45 folds and before
`repair_centered_diagnostic.py` / `audit_prelabel.py` / `freeze_integrity.py`, so the late
integrity snapshot binds the R3 source (the producer `spectral_utils` + script snapshot).
The late integrity record binds an explicit artifact set that predates R3 and is left
untouched (Codex's contract is not edited); the R3 outputs and the structural-review outputs
are bound instead by `AMENDMENT_R3_FREEZE.json` at the results root — sha256 of every
`scores_amend_r3.npz` / `meta_amend_r3.json` / `MANIFEST_AMEND_R3.json`,
`prelabel_structure_review.{json,md}`, and the sha256 of the late integrity record itself,
written after `freeze_integrity.py` and before the evaluator.

Module patch applied 2026-09-06 16:20 after confirming zero live structure/R1 processes
(commit `7c7207ef`, 15 additive lines). Smoke fold `pb_gsm8k_q4/outer0`: reference row
reproduces the frozen R4 grouping (`internal`), lambda = 0, condition-matched ridge only.

## 4. Other pre-label findings recorded at registration (label-free, 43/45 folds)

- INTERNAL grouping blocked on 5/40 ProcessBench lanes and 0/3 PRMBench lanes — under the
  Section-3.4 cap; K = 3 on every selected lane.
- Gated-affinity grouping (Hook 1) blocked on 24/40 ProcessBench lanes. The Section-3.4 cap is
  written for the INTERNAL family; this review applies the same cap to the Hook 1 family and
  flags `internal_gaff_cont` / `internal_gaff_joint` `STRUCTURALLY_FRAGILE[processbench]` —
  development readouts still reported, fresh-data eligibility barred on that panel.
- Gate seed std max 0.081 (cap 0.15). All arms pass map agreement vs `prov5_cont` (min Spearman
  0.651). Cross-fold map cosine below 0.5 only for the permutation control
  `permctl_gate_prov5_cont` on one cell (0.491) — a control, not a candidate.
- Hook 2 rows are not inert (no lane at cosine >= 0.995); `internal_joint_gate100` moves the
  map strongly (median cosine 0.31 vs `internal_joint`).
- Small-m census: `internal_cont` fitted-but-flagged (m = 4) on 5 lanes; `dufs_pf_lsml` flagged
  on 24 lanes and fail-closed on 43/240 ProcessBench lanes (Amendment R2 exclusion stands).

Nothing in this amendment is decided after label access.
