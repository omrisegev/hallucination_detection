# Pre-label structural review — Joint L-SML v2

Incomplete folds: 0  |  aborts: 0

## Admission (R2 coverage completeness)
- processbench: 240 lanes; eligible LSML 15/16, IU 16/16; incomplete: {'dufs_pf_lsml': 197}; fixed control complete: True
- prmbench: 30 lanes; eligible LSML 15/16, IU 16/16; incomplete: {'dufs_pf_lsml': 25}; fixed control complete: True

## Fallback rates (Section 3.4 caps)
- processbench: INTERNAL blocked 5/40 (cap 10) fragile=False; gated-affinity blocked 24/40 fragile=True; K dist {'3': 35}
- prmbench: INTERNAL blocked 0/5 (cap 1) fragile=False; gated-affinity blocked 1/5 fragile=False; K dist {'3': 5}

## Stability
- gate seed std max 0.0807 (cap 0.15); violations: none
- cross-fold map cosine (floor 0.5): unstable arms: {'permctl_gate_prov5_cont': {'pb_omnimath_q8': 0.491, 'prmbench_qwen3_8b': 0.468}}
- inertness (Hook 1/2 rows vs lambda=0 reference):
  - prov5_cont_gate050 vs prov5_cont: min cos 0.9598, median 0.9681, lanes >=0.995: 0/45 -> inert=False
  - prov5_cont_gate100 vs prov5_cont: min cos 0.8958, median 0.9187, lanes >=0.995: 0/45 -> inert=False
  - internal_cont_gate100 vs internal_cont: min cos 0.4850, median 0.9125, lanes >=0.995: 0/45 -> inert=False
  - internal_joint_gate050 vs internal_joint: min cos 0.8045, median 0.9327, lanes >=0.995: 0/45 -> inert=False
  - internal_joint_gate100 vs internal_joint: min cos 0.1687, median 0.3092, lanes >=0.995: 0/45 -> inert=False
  - internal_gaff_cont vs internal_cont: min cos 0.2935, median 0.5454, lanes >=0.995: 4/45 -> inert=False
  - internal_gaff_joint vs internal_joint: min cos 0.3590, median 0.4905, lanes >=0.995: 4/45 -> inert=False
  - internal_joint_liu010 vs internal_joint_modelinv_lam0: min cos 0.9904, median 0.9949, lanes >=0.995: 21/45 -> inert=False (vs hierarchical head median 0.272)
  - internal_joint_liu050 vs internal_joint_modelinv_lam0: min cos 0.9342, median 0.9681, lanes >=0.995: 5/45 -> inert=False (vs hierarchical head median 0.225)
  - internal_joint_diag010 vs internal_joint_modelinv_lam0: min cos 0.9813, median 0.9912, lanes >=0.995: 11/45 -> inert=False (vs hierarchical head median 0.279)
  - internal_joint_diag050 vs internal_joint_modelinv_lam0: min cos 0.9054, median 0.9616, lanes >=0.995: 5/45 -> inert=False (vs hierarchical head median 0.229)
- dose pairs: {'internal_joint_liu010~internal_joint_liu050': 0.9888, 'internal_joint_diag010~internal_joint_diag050': 0.9895, 'prov5_cont_gate050~prov5_cont_gate100': 0.986, 'internal_joint_gate050~internal_joint_gate100': 0.6194}
- small-m census (rows with any guard/flag): {'prov5_cont': {'guarded_lanes': 45, 'flagged_lanes': 0}, 'internal_cont': {'guarded_lanes': 45, 'flagged_lanes': 5}, 'internal_joint': {'guarded_lanes': 40, 'flagged_lanes': 0}, 'prov5_cont_gate050': {'guarded_lanes': 45, 'flagged_lanes': 0}, 'prov5_cont_gate100': {'guarded_lanes': 45, 'flagged_lanes': 0}, 'internal_cont_gate100': {'guarded_lanes': 45, 'flagged_lanes': 5}, 'internal_joint_gate050': {'guarded_lanes': 40, 'flagged_lanes': 0}, 'internal_joint_gate100': {'guarded_lanes': 40, 'flagged_lanes': 0}, 'internal_gaff_cont': {'guarded_lanes': 44, 'flagged_lanes': 6}, 'permctl_gate_prov5_cont': {'guarded_lanes': 45, 'flagged_lanes': 0}, 'dufs_pf_lsml': {'guarded_lanes': 4, 'flagged_lanes': 25}, 'internal_gaff_joint': {'guarded_lanes': 19, 'flagged_lanes': 0}}

## Map agreement vs prov5_cont (floor 0.5)
- dufs_pf_lsml: median 0.991, min 0.911, PB viol 0, PRM viol 0
- equal_all23: median 0.974, min 0.915, PB viol 0, PRM viol 0
- equal_family_active23: median 0.972, min 0.931, PB viol 0, PRM viol 0
- internal_cont: median 0.836, min 0.667, PB viol 0, PRM viol 0
- internal_cont_gate100: median 0.856, min 0.665, PB viol 0, PRM viol 0
- internal_gaff_cont: median 1.000, min 0.722, PB viol 0, PRM viol 0
- internal_gaff_joint: median 0.990, min 0.906, PB viol 0, PRM viol 0
- internal_joint: median 0.914, min 0.712, PB viol 0, PRM viol 0
- internal_joint_diag010: median 0.956, min 0.841, PB viol 0, PRM viol 0
- internal_joint_diag050: median 0.962, min 0.827, PB viol 0, PRM viol 0
- internal_joint_gate050: median 0.887, min 0.743, PB viol 0, PRM viol 0
- internal_joint_gate100: median 0.789, min 0.651, PB viol 0, PRM viol 0
- internal_joint_liu010: median 0.955, min 0.842, PB viol 0, PRM viol 0
- internal_joint_liu050: median 0.963, min 0.827, PB viol 0, PRM viol 0
- iu_c1_s10_l1_exoff: median 0.992, min 0.985, PB viol 0, PRM viol 0
- iu_c1_s10_l1_exon: median 0.990, min 0.974, PB viol 0, PRM viol 0
- iu_c1_s10_l2_exoff: median 0.992, min 0.985, PB viol 0, PRM viol 0
- iu_c1_s10_l2_exon: median 0.990, min 0.974, PB viol 0, PRM viol 0
- iu_c1_s25_l1_exoff: median 0.992, min 0.985, PB viol 0, PRM viol 0
- iu_c1_s25_l1_exon: median 0.990, min 0.974, PB viol 0, PRM viol 0
- iu_c1_s25_l2_exoff: median 0.992, min 0.985, PB viol 0, PRM viol 0
- iu_c1_s25_l2_exon: median 0.991, min 0.974, PB viol 0, PRM viol 0
- iu_c2_s10_l1_exoff: median 0.985, min 0.956, PB viol 0, PRM viol 0
- iu_c2_s10_l1_exon: median 0.992, min 0.972, PB viol 0, PRM viol 0
- iu_c2_s10_l2_exoff: median 0.991, min 0.971, PB viol 0, PRM viol 0
- iu_c2_s10_l2_exon: median 0.992, min 0.970, PB viol 0, PRM viol 0
- iu_c2_s25_l1_exoff: median 0.993, min 0.986, PB viol 0, PRM viol 0
- iu_c2_s25_l1_exon: median 0.991, min 0.975, PB viol 0, PRM viol 0
- iu_c2_s25_l2_exoff: median 0.990, min 0.964, PB viol 0, PRM viol 0
- iu_c2_s25_l2_exon: median 0.991, min 0.974, PB viol 0, PRM viol 0
- permctl_gate_prov5_cont: median 0.991, min 0.886, PB viol 0, PRM viol 0
- permctl_graph_internal_joint_liu010: median 0.951, min 0.854, PB viol 0, PRM viol 0
- prov5_cont_gate050: median 0.997, min 0.989, PB viol 0, PRM viol 0
- prov5_cont_gate100: median 0.992, min 0.980, PB viol 0, PRM viol 0
- prov5_joint: median 0.993, min 0.984, PB viol 0, PRM viol 0

## Source/config fidelity
- seed rule 20260905 + 100*outer violations: none; n_arms rule 36 - fail_closed violations: none; deployed IU grid mismatches: none, roster rows missing: none

## Per-row flags
- dufs_pf_lsml: COVERAGE_INCOMPLETE[prmbench], COVERAGE_INCOMPLETE[processbench]
- internal_gaff_cont: STRUCTURALLY_FRAGILE[processbench]
- internal_gaff_joint: STRUCTURALLY_FRAGILE[processbench]
- permctl_gate_prov5_cont: UNSTABLE_MAP

## Late-freeze limitation
EXECUTION_REGISTRY.json was not written at launch; source/config hashes are bound by freeze_integrity.py AFTER the structure stage (dated 2026-09-06). Lineage rests on the committed producer revision plus the per-fold manifests, not on a launch-time freeze. This limitation is retained verbatim in the results report.
