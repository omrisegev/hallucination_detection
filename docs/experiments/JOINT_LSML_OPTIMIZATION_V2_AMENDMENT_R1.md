# Joint L-SML optimization v2 — Amendment R1 (pre-label)

Date: 2026-09-05, registered while the structure stage is running and BEFORE any label access
(structure progress at registration: 5/45 outer folds; the amendment adds only a second
additive pass over frozen artifacts and evaluator rows — no frozen artifact is modified).
Authorized by Omri's three-point directive of 2026-09-05.

## 1. Canonical-implementation verification (requested check)

Verified by direct code inspection:

| v2 arm family | entry point used here | canonical? |
|---|---|---|
| All L-SML / fixed-family rows | `spectral_utils.fusion_utils.lsml_continuous` (maintained continuous arm, `groups=` seam) | YES — the same function behind the advisor pages' "lsml" rows (`repgrid_scoring.score_subset` → `lsml_continuous_pipeline` → `lsml_continuous`) |
| All IU / U-PCR rows | `spectral_utils.upcr.upcr_fit` with the deployed token-channel `IU_CONFIG` (2 components, scale 0.25, l2, gates off) + the exclusion+refit port row | YES — the same entry point as the deployed-arm advisor chain (`scripts/labelfree_standing_report.py:71,283` `upcr_rho_oriented` → `upcr_fit`). The legacy `fusion_utils.upcr_pipeline` is NOT called anywhere in v2 |
| Joint rows | `spectral_utils.joint_lsml` (the Step-347 frozen estimator + the registered v2 wrappers) | YES |

**What `results/action_items/item4_benchmarking.html` actually used** (requested report): that
page is the response-level published-detector scoreboard. Its rows read
`results/reasoning_benchmark.csv`, produced by `repgrid_scoring.score_subset`, which dispatches
`method="lsml"` → `lsml_continuous_pipeline` (canonical maintained L-SML) and
`method="upcr"` → **the legacy `upcr_pipeline`** (`spectral_utils/repgrid_scoring.py:27,219`).
So item4's L-SML columns are canonical; its U-PCR columns predate the deployed arm and went
through the legacy path. **v2 is therefore stricter than item4 on the U-PCR axis, not weaker**
— v2 uses the deployed `upcr_fit` estimator that the label-free standing chain uses.
Channel caveat, stated plainly: item4 is response-level detection on the GOOD_5 reference
subset; v2 is token-level localization on the frozen active-23 roster. The like-for-like
historical anchors for v2 are the Step-347/348 frozen numbers, already registered as context.

## 2. Continuity row (new, registered)

`fixed_family_cont_unguarded`: the maintained `lsml_continuous` on the frozen provenance
families with `small_m_guard=False` — the exact historical estimator (the Step-347 control's
fit path) — followed by the v2 SD=1 + unified-orientation boundary. Fitted in the second pass
per (cell x outer fold); reported on both panels beside the guarded fixed-family arm.

Equivalence note (why one row suffices): step AUROC is scale- and (given the shared
orientation rule) sign-invariant, so on PRMBench this row IS the historically-presented
convention evaluated under the v2 folds; on ProcessBench it differs from history only through
the SD=1 scale repair, which is the registered fix, never a silent change. The guarded-vs-
unguarded delta per panel is the isolated cost/benefit of the Step-205 guard on our own
method, side by side, as requested.

## 3. Module B 3x3 trajectory-fusion grid (new, registered)

Grid: feature-axis substrate x trajectory-axis fuser, all label-free from frozen artifacts
(frozen per-fold weights + deterministic preparation rebuild + frozen fold maps):

- Substrates: `iu_c2_s25_l2_exoff` (deployed IU), `internal_cont` (S2, L-SML),
  `internal_joint` (S1, Joint).
- Trajectory fusers over the k=10 order-statistic views (z-scored per slot on outer-train
  full-length steps; renormalized-truncation replay on the transformed matrix):
  - `sml`: `sml_fuse_signed` (the existing B1 mechanism);
  - `iu`: deployed-config `upcr_fit` on the 10 views;
  - `joint`: LOAO-consensus partition over the 10 units (K=3 is the only admissible
    candidate at min group size 3), `fit_joint_lsml`, guarded hierarchical head. A lane where
    LOAO finds no admissible partition records that grid cell BLOCKED for that fold — no
    fallback (descriptive grid cell).
- Every fused reducer weight vector passes sign orientation + SD=1 on the train-full step
  scores before replay.

**Pre-registered mechanistic prediction (Omri, 2026-09-05, recorded before labels)**: the
order statistics are strongly mutually correlated (near-singular covariance across the 10
slots), which structurally favors IU-PCR's PSD-projection + analytic-ridge solve on the
trajectory axis. A trajectory-IU win over trajectory-SML is confirmatory evidence for this
mechanism; the grid records it either way.

**Multiplicity**: exactly ONE primary Module-B contrast replaces the previous B1-vs-B0
primary — the inner-CV-selected best of the 9 combos vs the frozen top-10-mean control (B0)
on the SAME substrate, PRMBench primary, ProcessBench secondary. All other grid cells, plus
the B1/B2a/B2b/B3 rows of the original module, are descriptive. Combo selection uses
outer-train-fitted scores evaluated on inner-validation steps (the same registered
approximation as the B2a alpha selection).

## 4. Small-group rule confirmation (requested check)

Confirmed as registered (protocol Section 3.3): every SML-family eigen-stage at m=3 units —
within-group, cross-group, any axis — is replaced by pre-registered equal weights over
SD-standardized units, then passes the shared boundary (sign orientation, SD=1). m=2 needs no
guard: the rank-1 off-diagonal system is exactly determined and yields the equal-magnitude
solution by construction. m=4 is fitted but flagged `small_m_flag`.

On the 10 order-statistic units this guard is NOT binding for the `sml` and `iu` trajectory
fusers (m=10). Inside the `joint` trajectory fuser it IS load-bearing and disclosed in the
Module-B notes: only K=3 is admissible on 10 units, so the cross stage is guarded
(equal-weight over SD-standardized virtual classifiers), size-3 groups are guarded within,
and a size-4 group is fitted-but-flagged.

## 5. Execution

Second pass `scripts/joint_lsml_optimization_v2/second_pass_amendments.py` runs after each
fold's structure freeze (or after the full structure stage), writes
`scores_continuity.npz`, `moduleb_grid.npz`, and `MANIFEST_AMEND_R1.json` per
(cell x outer fold) — additive files only; the frozen `MANIFEST.json` is never rewritten.
The evaluator gains the continuity row and the 3x3 section with the single primary contrast.
Nothing in this amendment is decided after label access.
