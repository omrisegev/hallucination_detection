# Synthetic dependency-fusion admission benchmark

Decision: **STOP_AND_REVISE**.

This benchmark is a mechanism gate, not evidence about hallucination detection. Passing permits the real-data experiment; failing blocks it until the method or its claim is revised.

## Admission gates

| gate | observed | rule | result |
|---|---:|---:|:---:|
| `required_method_completion` | 1.000000 | >= 1.000000 | **PASS** |
| `relative_polarity_accuracy` | 0.870089 | >= 0.950000 | **FAIL** |
| `clean_su_matches_iu` | 0.000707 | <= 0.010000 | **PASS** |
| `clean_sdsf_not_harmful` | -0.001664 | >= -0.010000 | **PASS** |
| `sparse_support_recall` | 0.937500 | >= 0.600000 | **PASS** |
| `sparse_support_precision` | 1.000000 | >= 0.400000 | **PASS** |
| `sparse_su_beats_iu` | 0.015699 | >= 0.002500 | **PASS** |
| `sparse_sdsf_beats_su` | -0.016156 | >= 0.005000 | **FAIL** |
| `sparse_sdsf_ci_excludes_zero` | -0.036171 | >= 0.000000 | **FAIL** |
| `oracle_detects_tail_value` | 0.008951 | >= 0.005000 | **PASS** |
| `sdsf_captures_oracle_gap` | -1.095449 | >= 0.250000 | **FAIL** |

## Method AUROC

| world | method | n | mean [95% CI] |
|---|---|---:|---:|
| `clean` | `dufs_pf_lsml` | 5 | 0.6681 [0.6592, 0.6775] |
| `clean` | `iu_pcr` | 40 | 0.6662 [0.6630, 0.6691] |
| `clean` | `iu_pcr_known_orientation` | 40 | 0.6673 [0.6650, 0.6696] |
| `clean` | `lsml_full` | 40 | 0.6666 [0.6643, 0.6689] |
| `clean` | `mean_signrho` | 40 | 0.6393 [0.6183, 0.6568] |
| `clean` | `oracle_linear` | 40 | 0.6674 [0.6651, 0.6697] |
| `clean` | `oracle_pcr2` | 40 | 0.6674 [0.6651, 0.6697] |
| `clean` | `pcr_structured` | 40 | 0.6667 [0.6638, 0.6693] |
| `clean` | `pcr_structured_known_orientation` | 40 | 0.6673 [0.6649, 0.6696] |
| `clean` | `ridge_observed` | 40 | 0.6653 [0.6607, 0.6688] |
| `clean` | `sdsf` | 40 | 0.6653 [0.6607, 0.6688] |
| `clean` | `sdsf_known_orientation` | 40 | 0.6673 [0.6649, 0.6696] |
| `clean` | `su_pcr_known_orientation` | 40 | 0.6673 [0.6650, 0.6696] |
| `clean` | `su_pcr_reproduction` | 40 | 0.6669 [0.6644, 0.6695] |
| `clean` | `upcr_signrho` | 40 | 0.6643 [0.6609, 0.6675] |
| `sparse_small` | `dufs_pf_lsml` | 5 | 0.6167 [0.6138, 0.6198] |
| `sparse_small` | `iu_pcr` | 40 | 0.6493 [0.6367, 0.6588] |
| `sparse_small` | `iu_pcr_known_orientation` | 40 | 0.6649 [0.6625, 0.6675] |
| `sparse_small` | `lsml_full` | 40 | 0.6526 [0.6497, 0.6555] |
| `sparse_small` | `mean_signrho` | 40 | 0.6047 [0.5771, 0.6283] |
| `sparse_small` | `oracle_linear` | 40 | 0.6749 [0.6727, 0.6772] |
| `sparse_small` | `oracle_pcr2` | 40 | 0.6650 [0.6626, 0.6676] |
| `sparse_small` | `pcr_structured` | 40 | 0.6556 [0.6418, 0.6645] |
| `sparse_small` | `pcr_structured_known_orientation` | 40 | 0.6650 [0.6626, 0.6675] |
| `sparse_small` | `ridge_observed` | 40 | 0.6482 [0.6317, 0.6618] |
| `sparse_small` | `sdsf` | 40 | 0.6488 [0.6326, 0.6622] |
| `sparse_small` | `sdsf_known_orientation` | 40 | 0.6725 [0.6701, 0.6748] |
| `sparse_small` | `su_pcr_known_orientation` | 40 | 0.6650 [0.6625, 0.6674] |
| `sparse_small` | `su_pcr_reproduction` | 40 | 0.6574 [0.6440, 0.6655] |
| `sparse_small` | `upcr_signrho` | 40 | 0.6490 [0.6360, 0.6579] |
| `sparse_large` | `dufs_pf_lsml` | 5 | 0.6168 [0.6096, 0.6246] |
| `sparse_large` | `iu_pcr` | 40 | 0.6443 [0.6250, 0.6597] |
| `sparse_large` | `iu_pcr_known_orientation` | 40 | 0.6657 [0.6639, 0.6675] |
| `sparse_large` | `lsml_full` | 40 | 0.6524 [0.6506, 0.6543] |
| `sparse_large` | `mean_signrho` | 40 | 0.6156 [0.5921, 0.6372] |
| `sparse_large` | `oracle_linear` | 40 | 0.6748 [0.6729, 0.6767] |
| `sparse_large` | `oracle_pcr2` | 40 | 0.6658 [0.6641, 0.6676] |
| `sparse_large` | `pcr_structured` | 40 | 0.6599 [0.6539, 0.6646] |
| `sparse_large` | `pcr_structured_known_orientation` | 40 | 0.6658 [0.6640, 0.6675] |
| `sparse_large` | `ridge_observed` | 40 | 0.6438 [0.6201, 0.6643] |
| `sparse_large` | `sdsf` | 40 | 0.6439 [0.6200, 0.6641] |
| `sparse_large` | `sdsf_known_orientation` | 40 | 0.6736 [0.6717, 0.6756] |
| `sparse_large` | `su_pcr_known_orientation` | 40 | 0.6657 [0.6639, 0.6676] |
| `sparse_large` | `su_pcr_reproduction` | 40 | 0.6600 [0.6543, 0.6645] |
| `sparse_large` | `upcr_signrho` | 40 | 0.6554 [0.6488, 0.6605] |
| `dense_stress` | `dufs_pf_lsml` | 5 | 0.6129 [0.6100, 0.6159] |
| `dense_stress` | `iu_pcr` | 40 | 0.6558 [0.6512, 0.6590] |
| `dense_stress` | `iu_pcr_known_orientation` | 40 | 0.6590 [0.6570, 0.6610] |
| `dense_stress` | `lsml_full` | 40 | 0.6623 [0.6600, 0.6644] |
| `dense_stress` | `mean_signrho` | 40 | 0.6472 [0.6320, 0.6598] |
| `dense_stress` | `oracle_linear` | 40 | 0.6828 [0.6806, 0.6850] |
| `dense_stress` | `oracle_pcr2` | 40 | 0.6589 [0.6569, 0.6609] |
| `dense_stress` | `pcr_structured` | 40 | 0.6572 [0.6551, 0.6592] |
| `dense_stress` | `pcr_structured_known_orientation` | 40 | 0.6590 [0.6569, 0.6609] |
| `dense_stress` | `ridge_observed` | 40 | 0.6499 [0.6410, 0.6569] |
| `dense_stress` | `sdsf` | 40 | 0.6498 [0.6406, 0.6568] |
| `dense_stress` | `sdsf_known_orientation` | 40 | 0.6587 [0.6566, 0.6606] |
| `dense_stress` | `su_pcr_known_orientation` | 40 | 0.6589 [0.6569, 0.6609] |
| `dense_stress` | `su_pcr_reproduction` | 40 | 0.6574 [0.6555, 0.6592] |
| `dense_stress` | `upcr_signrho` | 40 | 0.6564 [0.6544, 0.6584] |

## Paired contrasts

Positive means the candidate is better. Deltas are AUROC fractions.

| world | contrast | n | mean [95% CI] | W/L |
|---|---|---:|---:|---:|
| `clean` | `su_minus_iu` | 40 | +0.0007 [+0.0000, +0.0018] | 11/28 |
| `clean` | `sdsf_minus_su` | 40 | -0.0017 [-0.0042, -0.0002] | 15/25 |
| `clean` | `pcr_structured_minus_su` | 40 | -0.0003 [-0.0008, +0.0000] | 23/17 |
| `clean` | `sdsf_minus_pcr_structured` | 40 | -0.0014 [-0.0035, -0.0002] | 15/25 |
| `clean` | `sdsf_minus_upcr_deployed` | 40 | +0.0010 [-0.0012, +0.0027] | 23/17 |
| `clean` | `lsml_minus_upcr_deployed` | 40 | +0.0023 [+0.0006, +0.0043] | 19/21 |
| `clean` | `known_su_minus_known_iu` | 40 | -0.0000 [-0.0000, -0.0000] | 14/25 |
| `clean` | `known_sdsf_minus_known_su` | 40 | -0.0000 [-0.0000, +0.0000] | 15/25 |
| `clean` | `known_pcr_structured_minus_known_su` | 40 | +0.0000 [-0.0000, +0.0000] | 21/19 |
| `clean` | `oracle_minus_oracle_pcr` | 40 | +0.0000 [-0.0000, +0.0001] | 19/21 |
| `clean` | `oracle_minus_su` | 40 | +0.0004 [+0.0001, +0.0011] | 22/18 |
| `sparse_small` | `su_minus_iu` | 40 | +0.0082 [+0.0020, +0.0161] | 29/11 |
| `sparse_small` | `sdsf_minus_su` | 40 | -0.0087 [-0.0223, +0.0025] | 27/13 |
| `sparse_small` | `pcr_structured_minus_su` | 40 | -0.0019 [-0.0065, +0.0012] | 13/27 |
| `sparse_small` | `sdsf_minus_pcr_structured` | 40 | -0.0068 [-0.0173, +0.0021] | 28/12 |
| `sparse_small` | `sdsf_minus_upcr_deployed` | 40 | -0.0002 [-0.0118, +0.0092] | 32/8 |
| `sparse_small` | `lsml_minus_upcr_deployed` | 40 | +0.0035 [-0.0053, +0.0172] | 14/26 |
| `sparse_small` | `known_su_minus_known_iu` | 40 | +0.0001 [-0.0000, +0.0002] | 24/16 |
| `sparse_small` | `known_sdsf_minus_known_su` | 40 | +0.0075 [+0.0060, +0.0087] | 39/1 |
| `sparse_small` | `known_pcr_structured_minus_known_su` | 40 | +0.0001 [-0.0001, +0.0002] | 19/21 |
| `sparse_small` | `oracle_minus_oracle_pcr` | 40 | +0.0099 [+0.0090, +0.0107] | 40/0 |
| `sparse_small` | `oracle_minus_su` | 40 | +0.0175 [+0.0101, +0.0310] | 40/0 |
| `sparse_large` | `su_minus_iu` | 40 | +0.0157 [+0.0024, +0.0321] | 27/13 |
| `sparse_large` | `sdsf_minus_su` | 40 | -0.0162 [-0.0362, +0.0007] | 33/7 |
| `sparse_large` | `pcr_structured_minus_su` | 40 | -0.0002 [-0.0021, +0.0012] | 17/23 |
| `sparse_large` | `sdsf_minus_pcr_structured` | 40 | -0.0160 [-0.0361, +0.0004] | 33/7 |
| `sparse_large` | `sdsf_minus_upcr_deployed` | 40 | -0.0115 [-0.0322, +0.0058] | 32/8 |
| `sparse_large` | `lsml_minus_upcr_deployed` | 40 | -0.0030 [-0.0076, +0.0033] | 10/30 |
| `sparse_large` | `known_su_minus_known_iu` | 40 | +0.0000 [-0.0000, +0.0001] | 22/18 |
| `sparse_large` | `known_sdsf_minus_known_su` | 40 | +0.0079 [+0.0070, +0.0087] | 40/0 |
| `sparse_large` | `known_pcr_structured_minus_known_su` | 40 | +0.0000 [-0.0000, +0.0001] | 22/18 |
| `sparse_large` | `oracle_minus_oracle_pcr` | 40 | +0.0090 [+0.0082, +0.0097] | 40/0 |
| `sparse_large` | `oracle_minus_su` | 40 | +0.0147 [+0.0103, +0.0209] | 40/0 |
| `dense_stress` | `su_minus_iu` | 40 | +0.0015 [-0.0003, +0.0049] | 11/29 |
| `dense_stress` | `sdsf_minus_su` | 40 | -0.0076 [-0.0160, -0.0011] | 6/34 |
| `dense_stress` | `pcr_structured_minus_su` | 40 | -0.0001 [-0.0012, +0.0005] | 34/6 |
| `dense_stress` | `sdsf_minus_pcr_structured` | 40 | -0.0074 [-0.0151, -0.0013] | 6/34 |
| `dense_stress` | `sdsf_minus_upcr_deployed` | 40 | -0.0066 [-0.0148, -0.0003] | 7/33 |
| `dense_stress` | `lsml_minus_upcr_deployed` | 40 | +0.0059 [+0.0040, +0.0081] | 39/1 |
| `dense_stress` | `known_su_minus_known_iu` | 40 | -0.0000 [-0.0000, -0.0000] | 8/32 |
| `dense_stress` | `known_sdsf_minus_known_su` | 40 | -0.0003 [-0.0004, -0.0002] | 4/36 |
| `dense_stress` | `known_pcr_structured_minus_known_su` | 40 | +0.0000 [+0.0000, +0.0000] | 33/7 |
| `dense_stress` | `oracle_minus_oracle_pcr` | 40 | +0.0239 [+0.0227, +0.0250] | 40/0 |
| `dense_stress` | `oracle_minus_su` | 40 | +0.0254 [+0.0236, +0.0276] | 40/0 |

## Failure diagnosis (post-hoc, not confirmatory)

SDSF beat SU-PCR on **33/40** repetitions, with a median +0.0060, but 7 tail-amplification failures changed the mean to -0.0162. With the planted orientation supplied only as a diagnostic, SDSF beat SU-PCR on 40/40 by +0.0079 [+0.0070, +0.0087].

The label-free reliability-tail fraction correlates with the SDSF effect at -0.908; ordinary half-sample polarity stability does not expose the problem (+0.063), because the wrong orientation can be stable.

A post-hoc tail guard would have used SDSF on 33 repetitions and SU-PCR on 7, producing +0.0057 with 33 wins and 0 losses. This is a hypothesis, not a result: its 0.25 threshold was seen after v1 and must be frozen and tested on disjoint synthetic seeds before use.

## Interpretation boundary

The dense stress world intentionally violates SDSF/SU-PCR sparse-support assumptions, so it diagnoses failure behavior but cannot veto a method that passes its declared sparse world. DUFS+L-SML is secondary here because the generator defines covariance-fusion truth, not a favorable feature-selection manifold. Its decisive comparison remains the real, fixed in-scope cells.
