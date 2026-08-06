# Synthetic dependency-fusion admission benchmark

Decision: **STOP_AND_REVISE**.

This benchmark is a mechanism gate, not evidence about hallucination detection. Passing permits the real-data experiment; failing blocks it until the method or its claim is revised.

## Admission gates

| gate | observed | rule | result |
|---|---:|---:|:---:|
| `required_method_completion` | 1.000000 | >= 1.000000 | **PASS** |
| `clean_fixed_su_matches_fixed_iu` | 0.000002 | <= 0.010000 | **PASS** |
| `clean_fixed_sdsf_not_harmful` | -0.000011 | >= -0.010000 | **PASS** |
| `sparse_fixed_support_recall` | 0.956250 | >= 0.600000 | **PASS** |
| `sparse_fixed_support_precision` | 1.000000 | >= 0.400000 | **PASS** |
| `sparse_fixed_su_beats_fixed_iu` | -0.000004 | >= 0.002500 | **FAIL** |
| `sparse_fixed_sdsf_beats_fixed_su` | 0.008453 | >= 0.005000 | **PASS** |
| `sparse_fixed_sdsf_ci_excludes_zero` | 0.007473 | >= 0.000000 | **PASS** |
| `oracle_detects_tail_value` | 0.009356 | >= 0.005000 | **PASS** |
| `sdsf_captures_oracle_gap` | 0.891534 | >= 0.250000 | **PASS** |

## What this decision means

The decision is intentionally conjunctive. A `STOP_AND_REVISE` result does not erase mechanisms whose own frozen gates passed; it means at least one part of the compound scientific claim failed and the real-data run remains blocked.

In v2, fixed-orientation SDSF passes its improvement, uncertainty, clean-world, support-recovery, oracle-value, and oracle-gap gates. The failed gate is the separate claim that sparse covariance cleaning improves the two-component PCR solution by itself. The defensible revision is therefore to attribute the synthetic gain to SDSF's full reliability/dependency-weighted solve, not to SU-PCR covariance cleaning alone. This interpretation narrows the claim; it does not retroactively turn the conjunctive admission decision into a pass.

## Method AUROC

| world | method | n | mean [95% CI] |
|---|---|---:|---:|
| `clean` | `iu_pcr` | 40 | 0.6660 [0.6639, 0.6682] |
| `clean` | `iu_pcr_fixed` | 40 | 0.6668 [0.6647, 0.6688] |
| `clean` | `lsml_full` | 40 | 0.6661 [0.6639, 0.6683] |
| `clean` | `mean_signrho` | 40 | 0.6404 [0.6153, 0.6607] |
| `clean` | `oracle_linear` | 40 | 0.6669 [0.6647, 0.6690] |
| `clean` | `oracle_pcr2` | 40 | 0.6668 [0.6647, 0.6690] |
| `clean` | `pcr_structured` | 40 | 0.6668 [0.6646, 0.6689] |
| `clean` | `pcr_structured_fixed` | 40 | 0.6668 [0.6646, 0.6689] |
| `clean` | `ridge_observed` | 40 | 0.6665 [0.6645, 0.6686] |
| `clean` | `sdsf` | 40 | 0.6665 [0.6645, 0.6686] |
| `clean` | `sdsf_fixed` | 40 | 0.6667 [0.6646, 0.6689] |
| `clean` | `su_pcr_fixed` | 40 | 0.6668 [0.6646, 0.6689] |
| `clean` | `su_pcr_reproduction` | 40 | 0.6667 [0.6646, 0.6690] |
| `clean` | `upcr_signrho` | 40 | 0.6649 [0.6623, 0.6674] |
| `sparse_small` | `iu_pcr` | 40 | 0.6553 [0.6496, 0.6603] |
| `sparse_small` | `iu_pcr_fixed` | 40 | 0.6639 [0.6617, 0.6660] |
| `sparse_small` | `lsml_full` | 40 | 0.6511 [0.6489, 0.6534] |
| `sparse_small` | `mean_signrho` | 40 | 0.6196 [0.5984, 0.6381] |
| `sparse_small` | `oracle_linear` | 40 | 0.6735 [0.6714, 0.6756] |
| `sparse_small` | `oracle_pcr2` | 40 | 0.6640 [0.6619, 0.6660] |
| `sparse_small` | `pcr_structured` | 40 | 0.6600 [0.6526, 0.6647] |
| `sparse_small` | `pcr_structured_fixed` | 40 | 0.6640 [0.6619, 0.6660] |
| `sparse_small` | `ridge_observed` | 40 | 0.6577 [0.6455, 0.6671] |
| `sparse_small` | `sdsf` | 40 | 0.6581 [0.6459, 0.6675] |
| `sparse_small` | `sdsf_fixed` | 40 | 0.6712 [0.6689, 0.6735] |
| `sparse_small` | `su_pcr_fixed` | 40 | 0.6640 [0.6619, 0.6660] |
| `sparse_small` | `su_pcr_reproduction` | 40 | 0.6597 [0.6517, 0.6646] |
| `sparse_small` | `upcr_signrho` | 40 | 0.6509 [0.6435, 0.6570] |
| `sparse_large` | `iu_pcr` | 40 | 0.6497 [0.6368, 0.6600] |
| `sparse_large` | `iu_pcr_fixed` | 40 | 0.6623 [0.6602, 0.6645] |
| `sparse_large` | `lsml_full` | 40 | 0.6475 [0.6454, 0.6496] |
| `sparse_large` | `mean_signrho` | 40 | 0.6136 [0.5858, 0.6380] |
| `sparse_large` | `oracle_linear` | 40 | 0.6718 [0.6695, 0.6741] |
| `sparse_large` | `oracle_pcr2` | 40 | 0.6624 [0.6603, 0.6646] |
| `sparse_large` | `pcr_structured` | 40 | 0.6582 [0.6524, 0.6629] |
| `sparse_large` | `pcr_structured_fixed` | 40 | 0.6623 [0.6602, 0.6645] |
| `sparse_large` | `ridge_observed` | 40 | 0.6501 [0.6315, 0.6654] |
| `sparse_large` | `sdsf` | 40 | 0.6502 [0.6321, 0.6651] |
| `sparse_large` | `sdsf_fixed` | 40 | 0.6708 [0.6684, 0.6732] |
| `sparse_large` | `su_pcr_fixed` | 40 | 0.6623 [0.6602, 0.6645] |
| `sparse_large` | `su_pcr_reproduction` | 40 | 0.6591 [0.6540, 0.6632] |
| `sparse_large` | `upcr_signrho` | 40 | 0.6559 [0.6512, 0.6598] |
| `dense_stress` | `iu_pcr` | 40 | 0.6598 [0.6575, 0.6621] |
| `dense_stress` | `iu_pcr_fixed` | 40 | 0.6598 [0.6574, 0.6620] |
| `dense_stress` | `lsml_full` | 40 | 0.6638 [0.6613, 0.6664] |
| `dense_stress` | `mean_signrho` | 40 | 0.6604 [0.6550, 0.6652] |
| `dense_stress` | `oracle_linear` | 40 | 0.6839 [0.6819, 0.6861] |
| `dense_stress` | `oracle_pcr2` | 40 | 0.6597 [0.6574, 0.6619] |
| `dense_stress` | `pcr_structured` | 40 | 0.6597 [0.6574, 0.6619] |
| `dense_stress` | `pcr_structured_fixed` | 40 | 0.6598 [0.6574, 0.6620] |
| `dense_stress` | `ridge_observed` | 40 | 0.6589 [0.6565, 0.6614] |
| `dense_stress` | `sdsf` | 40 | 0.6589 [0.6564, 0.6613] |
| `dense_stress` | `sdsf_fixed` | 40 | 0.6595 [0.6571, 0.6616] |
| `dense_stress` | `su_pcr_fixed` | 40 | 0.6597 [0.6574, 0.6620] |
| `dense_stress` | `su_pcr_reproduction` | 40 | 0.6598 [0.6575, 0.6620] |
| `dense_stress` | `upcr_signrho` | 40 | 0.6587 [0.6563, 0.6609] |

## Paired contrasts

Positive means the candidate is better. Deltas are AUROC fractions.

| world | contrast | n | mean [95% CI] | W/L |
|---|---|---:|---:|---:|
| `clean` | `su_minus_iu` | 40 | +0.0007 [+0.0000, +0.0018] | 25/15 |
| `clean` | `sdsf_minus_su` | 40 | -0.0002 [-0.0005, -0.0000] | 16/24 |
| `clean` | `pcr_structured_minus_su` | 40 | +0.0000 [-0.0001, +0.0000] | 17/22 |
| `clean` | `sdsf_minus_pcr_structured` | 40 | -0.0002 [-0.0005, -0.0000] | 15/25 |
| `clean` | `sdsf_minus_upcr_deployed` | 40 | +0.0016 [+0.0004, +0.0033] | 17/23 |
| `clean` | `lsml_minus_upcr_deployed` | 40 | +0.0012 [-0.0004, +0.0032] | 15/25 |
| `clean` | `fixed_su_minus_fixed_iu` | 40 | +0.0000 [-0.0000, +0.0000] | 22/18 |
| `clean` | `fixed_sdsf_minus_fixed_su` | 40 | -0.0000 [-0.0000, -0.0000] | 17/23 |
| `clean` | `fixed_pcr_structured_minus_fixed_su` | 40 | -0.0000 [-0.0000, +0.0000] | 16/23 |
| `clean` | `oracle_minus_oracle_pcr` | 40 | +0.0001 [+0.0000, +0.0001] | 29/11 |
| `clean` | `oracle_minus_su` | 40 | +0.0001 [+0.0000, +0.0002] | 24/16 |
| `clean` | `oracle_minus_fixed_su` | 40 | +0.0001 [+0.0000, +0.0002] | 24/16 |
| `sparse_small` | `su_minus_iu` | 40 | +0.0043 [-0.0016, +0.0100] | 30/10 |
| `sparse_small` | `sdsf_minus_su` | 40 | -0.0016 [-0.0095, +0.0046] | 32/8 |
| `sparse_small` | `pcr_structured_minus_su` | 40 | +0.0003 [-0.0001, +0.0009] | 16/24 |
| `sparse_small` | `sdsf_minus_pcr_structured` | 40 | -0.0019 [-0.0101, +0.0044] | 32/8 |
| `sparse_small` | `sdsf_minus_upcr_deployed` | 40 | +0.0072 [+0.0002, +0.0129] | 35/5 |
| `sparse_small` | `lsml_minus_upcr_deployed` | 40 | +0.0002 [-0.0062, +0.0083] | 12/28 |
| `sparse_small` | `fixed_su_minus_fixed_iu` | 40 | +0.0001 [+0.0000, +0.0002] | 23/17 |
| `sparse_small` | `fixed_sdsf_minus_fixed_su` | 40 | +0.0073 [+0.0061, +0.0084] | 38/2 |
| `sparse_small` | `fixed_pcr_structured_minus_fixed_su` | 40 | -0.0000 [-0.0001, +0.0001] | 18/22 |
| `sparse_small` | `oracle_minus_oracle_pcr` | 40 | +0.0095 [+0.0088, +0.0102] | 40/0 |
| `sparse_small` | `oracle_minus_su` | 40 | +0.0138 [+0.0095, +0.0216] | 40/0 |
| `sparse_small` | `oracle_minus_fixed_su` | 40 | +0.0095 [+0.0087, +0.0103] | 40/0 |
| `sparse_large` | `su_minus_iu` | 40 | +0.0094 [+0.0011, +0.0206] | 26/13 |
| `sparse_large` | `sdsf_minus_su` | 40 | -0.0089 [-0.0237, +0.0032] | 32/8 |
| `sparse_large` | `pcr_structured_minus_su` | 40 | -0.0009 [-0.0026, +0.0001] | 16/24 |
| `sparse_large` | `sdsf_minus_pcr_structured` | 40 | -0.0080 [-0.0214, +0.0034] | 32/8 |
| `sparse_large` | `sdsf_minus_upcr_deployed` | 40 | -0.0057 [-0.0213, +0.0066] | 32/8 |
| `sparse_large` | `lsml_minus_upcr_deployed` | 40 | -0.0084 [-0.0114, -0.0048] | 5/35 |
| `sparse_large` | `fixed_su_minus_fixed_iu` | 40 | -0.0000 [-0.0001, +0.0000] | 19/20 |
| `sparse_large` | `fixed_sdsf_minus_fixed_su` | 40 | +0.0085 [+0.0075, +0.0094] | 39/1 |
| `sparse_large` | `fixed_pcr_structured_minus_fixed_su` | 40 | +0.0000 [-0.0000, +0.0001] | 18/22 |
| `sparse_large` | `oracle_minus_oracle_pcr` | 40 | +0.0094 [+0.0086, +0.0101] | 40/0 |
| `sparse_large` | `oracle_minus_su` | 40 | +0.0127 [+0.0094, +0.0172] | 40/0 |
| `sparse_large` | `oracle_minus_fixed_su` | 40 | +0.0095 [+0.0087, +0.0103] | 40/0 |
| `dense_stress` | `su_minus_iu` | 40 | -0.0000 [-0.0000, +0.0000] | 18/22 |
| `dense_stress` | `sdsf_minus_su` | 40 | -0.0009 [-0.0020, -0.0001] | 11/29 |
| `dense_stress` | `pcr_structured_minus_su` | 40 | -0.0001 [-0.0002, +0.0000] | 28/12 |
| `dense_stress` | `sdsf_minus_pcr_structured` | 40 | -0.0008 [-0.0018, -0.0001] | 10/30 |
| `dense_stress` | `sdsf_minus_upcr_deployed` | 40 | +0.0002 [-0.0009, +0.0012] | 13/27 |
| `dense_stress` | `lsml_minus_upcr_deployed` | 40 | +0.0052 [+0.0043, +0.0060] | 39/1 |
| `dense_stress` | `fixed_su_minus_fixed_iu` | 40 | -0.0000 [-0.0000, -0.0000] | 12/28 |
| `dense_stress` | `fixed_sdsf_minus_fixed_su` | 40 | -0.0003 [-0.0004, -0.0002] | 7/33 |
| `dense_stress` | `fixed_pcr_structured_minus_fixed_su` | 40 | +0.0000 [+0.0000, +0.0000] | 29/11 |
| `dense_stress` | `oracle_minus_oracle_pcr` | 40 | +0.0242 [+0.0229, +0.0255] | 40/0 |
| `dense_stress` | `oracle_minus_su` | 40 | +0.0241 [+0.0229, +0.0253] | 40/0 |
| `dense_stress` | `oracle_minus_fixed_su` | 40 | +0.0242 [+0.0229, +0.0254] | 40/0 |

## Fixed-orientation result and legacy-control diagnosis

SDSF beat SU-PCR on **32/40** repetitions, with a median +0.0079, but 8 tail-amplification failures changed the mean to -0.0089. Under the deployable fixed feature contract, SDSF beat SU-PCR on 39/40 by +0.0085 [+0.0075, +0.0094]. The fixed-orientation comparison is the registered v2 result on disjoint draws; it is not post-hoc.

The label-free reliability-tail fraction correlates with the SDSF effect at -0.976; ordinary half-sample polarity stability does not expose the problem (+0.098), because the wrong orientation can be stable.

For the legacy sign(rho) control, the v1 post-hoc tail guard would have used SDSF on 34 repetitions and SU-PCR on 6, producing +0.0065 with 32 wins and 2 losses. This is a hypothesis, not a result: its 0.25 threshold was seen after v1 and must be frozen and tested on disjoint synthetic seeds before use.

## Interpretation boundary

The dense stress world intentionally violates SDSF/SU-PCR sparse-support assumptions, so it diagnoses failure behavior but cannot veto a method that passes its declared sparse world. DUFS+L-SML is secondary here because the generator defines covariance-fusion truth, not a favorable feature-selection manifold. Its decisive comparison remains the real, fixed in-scope cells.
