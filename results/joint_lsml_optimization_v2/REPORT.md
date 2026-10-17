# Joint L-SML optimization v2 — results report

Protocol: `docs/experiments/JOINT_LSML_OPTIMIZATION_PLAN_V2.md` with pre-label amendments R1, R2, R3 and the
2026-09-06 late integrity amendment (`docs/experiments/JOINT_LSML_V2_LATE_INTEGRITY_AMENDMENT_20260906.md`).
Populations: 8 ProcessBench cells (Qwen3-4B / Qwen3-8B x gsm8k / math / olympiadbench / omnimath) and PRMBench
Qwen3-8B; nested 5-outer / 5-inner label-free grouped folds; every learned arm fitted on outer-train only.

## Late-freeze limitation (retained verbatim)

EXECUTION_REGISTRY.json was not written at launch; source/config hashes are bound by freeze_integrity.py AFTER the structure stage (dated 2026-09-06). Lineage rests on the committed producer revision plus the per-fold manifests, not on a launch-time freeze. This limitation is retained verbatim in the results report.
Integrity record: `INTEGRITY_RECORD_V2.json` (`LATE_PRELABEL_SNAPSHOT_NOT_LAUNCH_PROOF`, created 2026-09-06T15:15:44.317810+00:00),
Amendment R3 artifacts bound by `AMENDMENT_R3_FREEZE.json`. Labels were decoded only by the patched evaluator
after `verify_prelabel_record` passed; the pre-label structural review reported zero aborts.

## 1. Headline (tuned-vs-tuned, 16 vs 16) and label-free successors

| Contrast | PRMBench step AUROC | ProcessBench macro-F1 |
|---|---|---|
| tuned Joint/L-SML family | 0.6724 | 0.3437 |
| tuned IU family | 0.6665 | 0.3493 |
| **tuned Joint − tuned IU** (paired grouped bootstrap) | +0.0059 [+0.0027, +0.0091] → **SUPPORT** | -0.0056 [-0.0137, +0.0029] → **NULL** |
| deployed IU-PCR (`iu_c2_s25_l2_exoff`) | 0.6665 | 0.3394 |
| deployed U-PCR port (`iu_c2_s25_l2_exon`) | 0.6523 | 0.3431 |
| S1 `internal_joint` (label-free, the repaired contribution) | 0.6110; vs deployed IU -0.0556 [-0.0590, -0.0522] | 0.1289; vs deployed IU -0.2160 [-0.2379, -0.1960]; activation guard **CATASTROPHE** |
| S2 `internal_cont` (label-free) | 0.5885; vs deployed IU -0.0780 [-0.0819, -0.0742] | 0.1343; vs deployed IU -0.2048 [-0.2241, -0.1868]; activation guard **CATASTROPHE** |
| fixed-family CONT control `prov5_cont` | 0.6506 | 0.3444 |
| continuity row `fixed_family_cont_unguarded` (historical estimator) | 0.6661 | 0.3489 |

Gates: development PRMB **SUPPORT**, development PB **NULL**, S1 NOT_PROMOTED, S2 NOT_PROMOTED.

### Selection frequency (inner 5-fold, tuned-vs-tuned)

| Panel | Family | Selected per outer fold | Most selected | Qualifier |
|---|---|---|---|---|
| prmbench | lsml | {'internal_joint_liu010': 5} | `internal_joint_liu010` (5/5) | stable (>=3/5) |
| prmbench | iu | {'iu_c2_s25_l2_exoff': 5} | `iu_c2_s25_l2_exoff` (5/5) | stable (>=3/5) |
| processbench | lsml | {'prov5_cont_gate050': 1, 'prov5_cont_gate100': 3, 'prov5_joint': 1} | `prov5_cont_gate100` (3/5) | stable (>=3/5) |
| processbench | iu | {'iu_c2_s25_l1_exon': 2, 'iu_c2_s25_l2_exon': 1, 'iu_c1_s25_l1_exon': 2} | `iu_c2_s25_l1_exon` (2/5) | UNSTABLE_SELECTION |

### Mechanism attribution (Section 7.3) and the R3 lambda=0 reference

- PRMB winner `internal_joint_liu010` vs tuned IU: +0.0059 [+0.0027, +0.0091].
- graph-permutation control `permctl_graph_internal_joint_liu010` (AUROC 0.6732) vs tuned IU: +0.0067 [+0.0036, +0.0097].
- ungated lambda=0 model-inverse reference `internal_joint_modelinv_lam0` (AUROC 0.6734) vs tuned IU: +0.0069 [+0.0037, +0.0100]; `internal_joint_liu010` vs that reference: -0.0010 [-0.0015, -0.0005].
- weight-map cosine liu010 vs lambda=0 reference: median 0.9949, min 0.9904, 21/45 lanes at >= 0.995 (not MECHANISM_INERT under the every-lane rule; near-inert).
- **Verdict: MECHANISM_UNATTRIBUTED (graph-permutation control passes the same gate).**

### Fresh-data freeze (Section 7.2)

- PRMBench: `internal_joint_liu010` (5/5); non-inferior vs deployed IU: True; flags none.
- ProcessBench: `prov5_cont_gate100` (3/5); non-inferior vs deployed IU: False (+0.0030 [-0.0084, +0.0143]); flags none.
- PB per-fold tuned Joint vs deployed IU: +0.0019 [-0.0100, +0.0135].
- S1 / S2 are carried as registered but fail promotion on both panels (HARM, and CATASTROPHE on ProcessBench).

### Guard cost (Amendment R1 continuity row) and named-control contrasts

- prmbench continuity_vs_deployed_iu: -0.0005 [-0.0018, +0.0008]
- prmbench guarded_vs_unguarded_fixed_family: -0.0154 [-0.0165, -0.0144]
- prmbench deployed_upcr_port_vs_deployed_iu: -0.0142 [-0.0160, -0.0124]
- processbench guarded_vs_unguarded_fixed_family: -0.0044 [-0.0122, +0.0036]

## 2. Per-arm descriptive table (outer-refit, no selection; both panels)

| Row | PRMB AUROC | PB macro-F1 | PB error-side hit | PB clean-side | PB activation guard | pre-label flags |
|---|---|---|---|---|---|---|
| `prov5_cont` — R1 provenance CONT L-SML (lambda=0 anchor = fixed-family control) | 0.6506 | 0.3444 | 0.267 | 0.512 | ok (act 0.82) |  |
| `prov5_joint` — R2 provenance-merged hierarchical Joint | 0.6553 | 0.3449 | 0.259 | 0.543 | ok (act 0.80) |  |
| `internal_cont` — R3 INTERNAL CONT L-SML (= S2) | 0.5885 | 0.1343 | 0.153 | 0.319 | CATASTROPHE (act 0.74) |  |
| `internal_joint` — R4 INTERNAL hierarchical Joint (= S1) | 0.6110 | 0.1289 | 0.195 | 0.368 | CATASTROPHE (act 0.72) |  |
| `prov5_cont_gate050` — R5 Hook 2 congruence, provenance CONT, lambda=0.5 | 0.6495 | 0.3424 | 0.266 | 0.501 | ok (act 0.82) |  |
| `prov5_cont_gate100` — R6 Hook 2 congruence, provenance CONT, lambda=1 | 0.6493 | 0.3432 | 0.257 | 0.536 | ok (act 0.79) |  |
| `internal_cont_gate100` — R7 Hook 2 congruence, INTERNAL CONT, lambda=1 | 0.5823 | 0.1422 | 0.150 | 0.353 | ok (act 0.72) |  |
| `internal_joint_gate050` — R8 Hook 2 on joint fit, lambda=0.5 | 0.6109 | 0.1289 | 0.193 | 0.206 | CATASTROPHE (act 0.85) |  |
| `internal_joint_gate100` — R9 Hook 2 on joint fit, lambda=1 | 0.6009 | 0.1370 | 0.106 | 0.333 | CATASTROPHE (act 0.71) |  |
| `internal_joint_liu010` — R10 Hook 3a LIU-transplant model-inverse, lambda=0.1 | 0.6724 | 0.2641 | 0.242 | 0.486 | ok (act 0.79) |  |
| `internal_joint_liu050` — R11 Hook 3a LIU-transplant model-inverse, lambda=0.5 | 0.6683 | 0.2609 | 0.242 | 0.485 | CATASTROPHE (act 0.77) |  |
| `internal_joint_diag010` — R12 Hook 3b diagonal gate prior, lambda=0.1 | 0.6690 | 0.2754 | 0.253 | 0.469 | ok (act 0.80) |  |
| `internal_joint_diag050` — R13 Hook 3b diagonal gate prior, lambda=0.5 | 0.6619 | 0.2685 | 0.244 | 0.480 | CATASTROPHE (act 0.78) |  |
| `internal_gaff_cont` — R14 Hook 1 gated-affinity grouping, CONT | 0.6584 | 0.3153 | 0.258 | 0.501 | CATASTROPHE (act 0.80) | STRUCTURALLY_FRAGILE[processbench] |
| `internal_gaff_joint` — R15 Hook 1 gated-affinity grouping, Joint | 0.6584 | 0.3220 | 0.248 | 0.539 | CATASTROPHE (act 0.77) | STRUCTURALLY_FRAGILE[processbench] |
| `dufs_pf_lsml` — R16 historical hard DUFS-PF selector + CONT | excluded (coverage) | excluded (coverage) | | | | COVERAGE_INCOMPLETE[prmbench], COVERAGE_INCOMPLETE[processbench] |
| `internal_joint_modelinv_lam0` — R3 reference: ungated model-inverse map (lambda=0) | 0.6734 | 0.2684 | 0.245 | 0.481 | ok (act 0.80) |  |
| `fixed_family_cont_unguarded` — continuity: historical unguarded fixed-family CONT (Amendment R1) | 0.6661 | 0.3489 | 0.268 | 0.523 | ok (act 0.83) |  |
| `equal_all23` — control: equal weights over all 23 | 0.6452 | 0.3407 | 0.258 | 0.528 | ok (act 0.80) |  |
| `equal_family_active23` — control: equal-family | 0.6375 | 0.2815 | 0.210 | 0.526 | ok (act 0.72) |  |
| `permctl_gate_prov5_cont` — negative control: feature-permuted gates on R6 | 0.6473 | 0.3158 | 0.239 | 0.481 | CATASTROPHE (act 0.74) | UNSTABLE_MAP |
| `permctl_graph_internal_joint_liu010` — negative control: node-relabeled graph on R10 | 0.6732 | 0.2695 | 0.240 | 0.501 | CATASTROPHE (act 0.78) |  |
| `iu_c2_s25_l2_exoff` — IU grid | 0.6665 | 0.3394 | 0.252 | 0.549 | ok (act 0.79) |  |
| `iu_c2_s25_l2_exon` — IU grid | 0.6523 | 0.3431 | 0.256 | 0.553 | ok (act 0.78) |  |
| `iu_c2_s25_l1_exoff` — IU grid | 0.6620 | 0.3425 | 0.256 | 0.541 | ok (act 0.80) |  |
| `iu_c2_s25_l1_exon` — IU grid | 0.6549 | 0.3490 | 0.262 | 0.543 | ok (act 0.80) |  |
| `iu_c2_s10_l2_exoff` — IU grid | 0.6661 | 0.3438 | 0.254 | 0.560 | ok (act 0.79) |  |
| `iu_c2_s10_l2_exon` — IU grid | 0.6454 | 0.3429 | 0.261 | 0.526 | ok (act 0.80) |  |
| `iu_c2_s10_l1_exoff` — IU grid | 0.6531 | 0.3443 | 0.255 | 0.548 | ok (act 0.80) |  |
| `iu_c2_s10_l1_exon` — IU grid | 0.6476 | 0.3455 | 0.257 | 0.549 | ok (act 0.79) |  |
| `iu_c1_s25_l2_exoff` — IU grid | 0.6610 | 0.3439 | 0.259 | 0.534 | ok (act 0.81) |  |
| `iu_c1_s25_l2_exon` — IU grid | 0.6588 | 0.3472 | 0.260 | 0.550 | ok (act 0.79) |  |
| `iu_c1_s25_l1_exoff` — IU grid | 0.6610 | 0.3439 | 0.259 | 0.534 | ok (act 0.81) |  |
| `iu_c1_s25_l1_exon` — IU grid | 0.6588 | 0.3458 | 0.261 | 0.536 | ok (act 0.80) |  |
| `iu_c1_s10_l2_exoff` — IU grid | 0.6610 | 0.3439 | 0.259 | 0.534 | ok (act 0.81) |  |
| `iu_c1_s10_l2_exon` — IU grid | 0.6589 | 0.3450 | 0.256 | 0.556 | ok (act 0.79) |  |
| `iu_c1_s10_l1_exoff` — IU grid | 0.6610 | 0.3439 | 0.259 | 0.534 | ok (act 0.81) |  |
| `iu_c1_s10_l1_exon` — IU grid | 0.6588 | 0.3475 | 0.258 | 0.559 | ok (act 0.79) |  |

## 3. Module B — learned trajectory-axis reducer (PRMBench primary)

Primary contrast (inner-selected best of the 3x3 grid vs frozen top-1/span-max control on the same substrate): selected `iu_c2_s25_l2_exoff__sml` on 5/5 folds; winner AUROC 0.6634 vs B0 0.6669; delta -0.0034 [-0.0043, -0.0026] → **HARM**.

| Row | PRMB step AUROC | vs B0 (descriptive) |
|---|---|---|
| B0 frozen span-max control | 0.6665 | — |
| B1 label-free SML weights over 10 order statistics | 0.6671 | +0.0006 [-0.0025, +0.0036] |
| B2a max-vs-mean blend (alpha by fold {'0': 0.5, '1': 0.5, '2': 0.5, '3': 0.5, '4': 0.5}) | 0.6728 | +0.0063 [+0.0049, +0.0077] |
| B2b positional bins (label-free) | 0.6565 | -0.0100 [-0.0147, -0.0054] |
| B3 supervised LR over order statistics (competitor) | 0.6726 | +0.0060 [+0.0046, +0.0075] |

3x3 grid (substrate x trajectory fuser), descriptive AUROC: `iu_c2_s25_l2_exoff__sml` 0.6634, `iu_c2_s25_l2_exoff__iu` 0.6545, `internal_cont__sml` 0.6029, `internal_cont__iu` 0.6002, `internal_joint__sml` 0.6428, `internal_joint__iu` 0.6492
Grid cells BLOCKED on every fold (no admissible LOAO partition over 10 order-statistic units): ['internal_cont__joint', 'internal_joint__joint', 'iu_c2_s25_l2_exoff__joint'].

Pre-registered mechanism prediction (trajectory-IU beats trajectory-SML because the order statistics are near-collinear): NOT confirmed on the deployed-IU substrate (sml > iu).

## 4. Pre-label structural review (label-free, registered before evaluation)

- INTERNAL grouping blocked: PB 5/40 (cap 10), PRMB 0/5 (cap 1); K distribution PB {'3': 35}, PRMB {'3': 5}.
- Hook 1 gated-affinity grouping blocked: PB 24/40 → both Hook 1 rows STRUCTURALLY_FRAGILE on ProcessBench; PRMB 1/5.
- Gate seed std max 0.0807 (cap 0.15); no candidate arm below the 0.5 cross-fold map-cosine floor; every arm passes map agreement vs `prov5_cont` (floor 0.50).
- No row MECHANISM_INERT under the every-lane rule; Hook 3 rows at lambda=0.1 are near-inert relative to the lambda=0 reference (see Section 1).
- Amendment R2 coverage: `dufs_pf_lsml` fails closed on 197/240 PB lanes and 25/30 PRMB lanes → excluded from selection, descriptive only.

Full review: `prelabel_structure_review.md`. Evaluator outputs: `evaluation/headline.json`, `inner_selection.json`, `moduleb.json`, `report_tables.json`, `report_contrasts.json`.

## 5. Interpretation (written after evaluation; not part of the registered gates)

1. **The Step-349 objective/head mismatch was real, and fixing the head is what wins on PRMBench.** Every map that solves `(C_model + gamma I) w = v` with the fitted joint factors `u_g` (the four Hook 3 rows, their permutation control, and the ungated lambda=0 reference) beats deployed IU-PCR by 0.6-0.7pp AUROC with CIs clear of zero; the hierarchical head on the same fit and the same groups scores 0.611. The lambda=0 reference is the best of the family (0.6734).
2. **DUFS integration into the coefficients adds nothing here.** Dose-response is monotone against it on both hooks (lambda 0 > 0.1 > 0.5), the node-relabeled graph reproduces the win, and Hook 3a at lambda=0.1 is a small but significant harm relative to its own lambda=0 map (-0.0010 [-0.0015, -0.0005]). Hook 2 congruence and Hook 1 gated grouping are also at or below their ungated references. The PRMB SUPPORT is therefore MECHANISM_UNATTRIBUTED to DUFS and attributed to the regularized model-inverse head.
3. **INTERNAL grouping, not the map, is the ProcessBench failure.** Every INTERNAL-grouping row (CONT, hierarchical, model-inverse) sits at 0.13-0.28 macro-F1 while every provenance-grouping row sits at 0.34-0.35; the model-inverse rows keep normal activation (0.79) but a weaker detector ranking, so this is not the Step-349 threshold collapse. INTERNAL selected K=3 on 35/35 admissible PB lanes.
4. **K=3 is forced by the minimum-group-size-3 rule, not by instability** (label-free re-fit of the LOAO consensus on two frozen folds): K=4/5/6 partitions have median ARI 1.000 to their consensus but always contain a size-2 group and are rejected; K=8 can never be admissible with 23 features. At m=2 the within-group system is exactly determined (Amendment R1 Section 4) and the model-inverse head uses `u_g` directly, so pairs are safe; only singletons must stay forbidden.
5. **The Step-205 small-m guard costs the fixed-family row 1.5pp AUROC on PRMBench** (-0.0154 [-0.0165, -0.0144]) and is NULL on ProcessBench; the historical unguarded estimator is statistically indistinguishable from deployed IU on PRMBench (-0.0005 [-0.0018, +0.0008]).
6. **Module B:** the label-free order-statistic reducer does not beat the frozen max (HARM, -0.0034); the max-vs-mean blend (alpha=0.5 on every fold) and the supervised LR both add ~+0.006 AUROC descriptively, i.e. the informative signal is the tail plus a bulk correction, not a learned reweighting of the bulk. Trajectory-IU did not beat trajectory-SML on the deployed substrate (prediction not confirmed). The Joint fuser is BLOCKED on all folds over 10 order-statistic units (min group size 3).

**Registered next step (Step 354, to be written before any run):** (a) `target_condition` dose {30, 100, 300, 1000, 3000, 1e4} for the ungated model-inverse map on INTERNAL and on provenance groups; (b) grouping with minimum group size 2, K in {3,...,8}, same stability rule, for both the model-inverse map and the hierarchical head. Same nested folds and controls; labelled development, because both axes were chosen after seeing these results.
