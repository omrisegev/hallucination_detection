# Reasoning Localization 0.3662 — Phase-3 family-expert attribution v1

Status: **complete on the opened eight-Qwen ProcessBench development panel;
no promotion**

## Question

P3C replaced equal within-family compression by ordinary two-component IU in
all three multi-view H2 families simultaneously. Its aggregate result was
inconclusive, so it cannot identify whether one family benefited while another
caused the loss. This bounded ladder changes one family expert at a time.

## Frozen roster

All arms use the compact H2 member roster already frozen by P3D:

- `entropy_level`: singleton `entropy_series`, always equal passthrough;
- `entropy_dynamics`: the frozen dynamics members plus `C7_EDIS_ONSET`;
- `partition_energy`: frozen members excluding `energy_series`;
- `topk_distribution`: six frozen top-k members.

Sampled-token energy and structural views remain absent. No feature, reducer,
threshold, component count, sign, or weight may be selected from ProcessBench
or PRMBench outcomes.

## Ladder

1. `P3E0_H2_XFIT_EQUAL_REFERENCE`: donor-fold standardize each view, equal
   mean inside each family, equal mean across four family risks.
2. `P3E1_DYNAMICS_IU_ONLY`: replace only entropy-dynamics equal compression by
   ordinary two-component IU-PCR.
3. `P3E2_PARTITION_IU_ONLY`: replace only partition-energy compression.
4. `P3E3_TOPK_IU_ONLY`: replace only top-k compression.
5. `P3E4_ALL_MULTI_IU_CONTROL`: replace all three multi-view families; this is
   a closure/control arm, not an extra combinatorial candidate.

All arms preserve the frozen H0 clean/error decision and use the top-ten step
reducer. They rerank H0 non-abstentions only.

## Fit and inference

- Five grouped folds use `sha256(row_id) mod 5`; all scorer copies of a source
  question remain together.
- Donor-fold imputation, standardization, IU fit and sign orientation are fit
  on four folds; held responses are projection-only.
- IU uses the frozen ordinary configuration: L2, two components,
  `scale_ratio=0.25`, `g2_projection_k=1`, no exclusion, no difficulty gate,
  no recomputation, no mean fallback.
- The confidence sign is oriented against the donor-only equal-view anchor.
- All five score trees are frozen and hashed before labels are imported.
- Twenty-thousand paired whole-question bootstrap draws are shared across
  arms. The four candidate-minus-E0 macro-F1 contrasts form one Bonferroni
  family.

## Interpretation and gates

Primary contrasts are E1/E2/E3 minus E0; E4 minus E0 verifies aggregate
closure. Raw best is separate from supported improvement. A CI crossing zero
is `PROMISING_UNCONFIRMED` for a positive point or `INCONCLUSIVE`, never generic
rejection.

A family expert becomes eligible for one later method-specific variant only
if it has a positive point estimate, no supported material harm, exact-error
delta at least `-0.010`, worst-cell delta at least `-0.020`, zero H0 abstention
mismatches, and stable finite donor fits. Formal ProcessBench promotion still
requires point delta at least `+0.003` and simultaneous CI lower bound above
`+0.003`. Because the population is opened development data, even a passing
arm requires independent confirmation.

The next method-specific step, if earned, changes only that surviving family:
STG/SU requires its sparse-support premise and matched random/permuted support;
DUFS-LIU requires a zero-strength IU alias and permuted-graph control; B3 or
L-SML requires its own frozen label-free objective and exact parent alias.

## Completed result

All five score trees were frozen before label import; H0 abstention mismatches
are zero. The matched cross-fit equal reference scores `0.364284`, within
`+0.000194` of the incumbent H2 score `0.364090`.

| arm | F1 | delta vs E0 | simultaneous CI | W/T/L |
|---|---:|---:|---:|---:|
| E1 dynamics IU only | 0.366876 | +0.002592 | [-0.001839,+0.007194] | 6/0/2 |
| E2 partition IU only | 0.359376 | -0.004908 | [-0.013100,+0.003050] | 3/0/5 |
| E3 top-k IU only | 0.365603 | +0.001319 | [-0.001433,+0.004355] | 4/1/3 |
| E4 all multi-view IU | 0.359577 | -0.004708 | [-0.013846,+0.004211] | 3/0/5 |

E1 and E3 are `PROMISING_UNCONFIRMED`; neither clears the `+0.003` practical
and simultaneous gates. E2 and E4 are inconclusive rather than rejected, but
their negative points show that ordinary IU is not uniformly useful and that
partition compression explains much of the all-family loss. Clean abstention
is identical by construction. E1 also raises exact error by `+0.003236
[-0.000807,+0.007364]`; its worst cell macro delta is `-0.004668`.

The next bounded method-specific study may use dynamics as its primary family
and top-k as a secondary control. It must not combine both or open STG, DUFS,
B3 and L-SML simultaneously. No PRMBench transfer opens from these unconfirmed
development results.
