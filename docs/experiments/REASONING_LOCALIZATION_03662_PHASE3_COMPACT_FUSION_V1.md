# Reasoning Localization 0.3662 — Phase 3 Compact Fusion v1

Status: **historical registration; Phase-3 development completed and frozen with no promotion (Step 346)**

Freeze note: the prospective language below is retained as the historical
registration contract.  P3B--P3S and Q1 were subsequently executed as
separately frozen development families; the unexecuted templates are now
`NOT_RUN_BY_GATE`, and no further rung is authorized on this population.

## Motivation and evidence boundary

Phase 2 did not establish a statistically supported replacement for the
historical 0.3662 system.  It did establish two useful development premises:

1. the current H0 response detector has a favorable but unresolved point
   contribution in the matched historical 2x2 audit; and
2. top-ten step pooling is a small, unresolved development improvement over
   top-five pooling.

The user explicitly authorizes carrying both premises forward.  This is an
amendment to the original survivor-only wording: they enter Phase 3 as
`PROMISING_UNCONFIRMED`, not as promoted or statistically supported winners.
H2 and H3 remain visible comparators.  H3 is not the Phase-3 parent because the
matched audit did not isolate a positive H3-localizer contribution.

## P3A/P3B question

Can ordinary label-free IU-PCR improve the *outer weighting of the four compact
H2 family scores* while the response detector and step reducer remain fixed?

The one changed factor is:

```text
four H2 family token-risk curves
  P3A: equal outer mean
  P3B: ordinary two-component IU-PCR outer weights
-> frozen top-ten step reducer
-> frozen H0 response detector and threshold evaluator
```

The four input families are frozen before results:

- entropy level;
- entropy dynamics with the frozen C7 EDIS-onset insertion;
- partition energy with `energy_series` removed; and
- top-k distribution.

Sampled-token energy remains absent, exactly as in H2.  C8 remains a separate
H3 comparator and is not silently folded into P3B.

## Fit and leakage contract

- No ProcessBench or PRMBench target is accepted by the score-fitting code.
- The existing fit-safe token preparation and deterministic token cap are used.
- IU-PCR sees only the four standardized family confidence streams.
- Orientation uses their equal confidence mean, never task labels.
- Scores for all eight Qwen ProcessBench cells are hash-frozen before the
  ProcessBench label file is imported.
- All scorer copies of one source question remain grouped in evaluation and
  bootstrap.
- This population is development-open.  A positive result is therefore at
  most `PROMISING_UNCONFIRMED` until independent confirmation.

This first compact arm intentionally does not choose among IU, SU, DUFS, LIU,
STG, tensor, or hierarchical alternatives.  Ordinary IU is the preregistered
first fusion rule.  More complex mechanisms may open only after this result is
reported and must retain an exact parent alias and their registered controls.

## Evaluation and verdicts

Primary contrasts are P3B minus P3A and P3B minus H0 on the same rows, folds,
step spans, H0 detector, and top-ten reducer.  Report macro F1, exact first
error, within one, clean abstention, W/T/L, worst cell, and paired grouped
intervals.  Three Phase-3 fusion mechanisms are reserved in the multiplicity
family, so the primary interval uses the frozen Bonferroni family size of 3.

Promotion still requires all of:

- point delta at least +0.003 versus P3A;
- multiplicity-valid lower confidence bound above +0.003;
- no material exact-error, clean-abstention, or worst-cell regression; and
- improvement over H0 as well as the exact P3A parent.

A positive point with an interval crossing zero is
`PROMISING_UNCONFIRMED`, not rejected.  An interval spanning practically
relevant benefit and harm is `INCONCLUSIVE`.  Only supported material harm,
leakage/provenance failure, or another hard gate closes the branch.

## P3C amendment after the P3B verdict

The valid detector-preserving P3B run showed supported harm from ordinary IU
*across* the four family scores.  That closes outer IU but does not test the
registered family-expert idea.  P3C therefore changes a different factor:

- entropy level remains a singleton passthrough;
- ordinary IU is fitted separately inside entropy dynamics+C7, partition
  energy without `energy_series`, and top-k distribution; and
- the four resulting family risks retain the exact equal outer mean.

This matched design isolates inner-family compression from outer-family
weighting.  It opens one fixed IU flavour for all eligible families; it does
not select a different flavour per family from task outcomes.  SU, DUFS/LIU,
L-SML, STG, and tensor variants remain later branches and cannot be inferred
from this result.

## P3D deployed U-PCR / prune-refit amendment

P3B and P3C used ordinary full-pool IU-PCR; they did not test deployed U-PCR.
The latter first estimates label-free `rho_hat` on the full fit-side pool,
removes weak experts using the frozen project thresholds, and refits on the
survivors.  This is now a separate planned P3D branch defined in
`REASONING_LOCALIZATION_03662_PHASE3_DEPLOYED_UPCR_PRUNE_REFIT_V1.md`.

The exact deployed policy is dimension-ineligible on the four outer H2 family
scores because its `<5` fallback necessarily returns a simple average.
Therefore the meaningful primary contrast uses the exact compact H2 member
views and compares deployed U-PCR with ordinary full-pool IU on the same
matrix.  A survivor-mask equal-weight control and cardinality-matched random
mask distinguish selection from reweighting and generic dimension reduction.

The same two-pass idea may later wrap IU, SU/STG, DUFS-LIU, L-SML/B3, or a
tensor/query encoder only when that method first survives its unpruned gate
and exposes a preregistered method-native label-free relevance quantity.  Each
method gets at most one full-pool/prune-refit pair; audit F1 cannot choose the
mask, threshold, flavour, or member count.
