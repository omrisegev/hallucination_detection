# H2/H3 PRMBench Diagnostic — Pre-label Parent-Alias Amendment V2

Status: `FROZEN_BEFORE_RUN`; supersedes only the invalid V1 parent-alias check.
The evaluated H0/H2/H3 arms, metrics, bootstrap, practical bounds and evidence
boundary are unchanged.

## Pre-label V1 failure

V1 attempted to alias the new H0 top-ten score directly to the Phase-1 R2
artifact. Phase-1 R2 uses top-five pooling. The check therefore changed the
reducer and could not be an exact alias; it stopped at maximum absolute error
`0.23125777605843761` before any PRMBench label artifact was loaded and before
any score-freeze manifest was written.

This is a contract-design hard failure, not a scientific result and not a
rejection of H0, H2 or H3.

## Corrected control

V2 computes two H0 reducer outputs from the exact same fitted family curve:

1. `h0_top5_control`: top-five pooling followed by the unchanged common
   response detector. This must alias the Phase-1 R2 PRMBench frozen score at
   maximum absolute error `<=1e-12`.
2. `h0_top10_candidate`: top-ten pooling followed by the same detector. This
   is the registered diagnostic reference and is not expected to alias R2.

Only the top-ten candidate enters evaluation. The top-five score is a
non-rankable implementation/provenance control. H2 and H3 remain exactly as
registered in V1. Scores still freeze before labels open.

