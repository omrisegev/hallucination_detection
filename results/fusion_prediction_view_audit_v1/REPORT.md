# Fusion prediction view — feasibility only, Step 311

IU-PCR / Joint L-SML remains the core. The new AR(1) component appends nine
window error measurements to the original 27 columns; it is not a detector
replacement, KalmanNet or Diverging Flows. No new localization quality scores.

All 110 answers support finite, varying added columns. This predictor fits
past pairs within one answer; old token B3/local-online/CIW variants used
donor/calibration answers. See the source audit protocol for exact boundaries.

| Stream | AR / last MSE median | AR / EMA32 MSE median | AR beats EMA32 |
|---|---:|---:|---:|
| entropy | 0.683 | 0.974 | 78/110 |
| spilled | 0.586 | 0.469 | 110/110 |
| energy | 0.756 | 0.948 | 76/110 |
| top1_logprob | 0.642 | 1.005 | 51/110 |
| logprob_margin | 0.748 | 0.907 | 93/110 |
| topk_entropy | 0.686 | 0.975 | 76/110 |
| topk_varentropy | 0.738 | 0.923 | 91/110 |
| topk_renyi2 | 0.659 | 0.995 | 59/110 |
| topk_tail_mass | 0.601 | 1.021 | 30/110 |

These are telemetry-prediction results, not localization quality. Tokens t>=17
are used, with at least 16 previous fitting pairs. First-token residual is
excluded from every window mean. AR is not uniformly better than EMA32.

| Original bank | Residual predictor | Median closest-column abs Spearman | >=0.90 |
|---|---|---:|---:|
| moment | ar1 | 0.871 | 358/990 |
| moment | last | 0.929 | 605/990 |
| moment | ema32 | 0.825 | 127/990 |
| context | ar1 | 0.837 | 232/990 |
| context | last | 0.925 | 576/990 |
| context | ema32 | 0.882 | 394/990 |

990 entries = 9 streams x 110 answers, not independent observations. Added
columns remain substantially redundant; lower correlation may also be noise.
40/110 have N<36, while 31 original matrices already reach centered rank N-1.
No blanket fit-validity or quality conclusion follows from those facts.

The next quality study must include unchanged IU/Joint, the same cores with
the addition, equal aggregation with the same addition, simple residual
controls and graph-zero/permutation controls. Register coverage/fallback
before labels; retain both benchmarks and matched historical anchors.

Review PASS: {'source_and_grid_checks': 110, 'batch_prediction_stream_traces': 2970, 'residual_window_and_mse_arrays': 330, 'exact_original_column_replays': 660, 'independent_correlation_rank_bundles': 660}. Maximum prediction discrepancy
1.350e-13. Three tests pass. Audit
10.34 s, review 11.74 s.
Same-session independent algebra, shared ranking primitive; no browser visual
inspection. This is an already-exposed development cohort. No winner promoted.

See [visual report](REPORT.html), [protocol](../../docs/experiments/FUSION_PREDICTION_VIEW_AUDIT_V1.md),
[summary](SUMMARY.json), [review](REVIEW.json), and the existing
[quality anchors](../fusion_graph_conditioning_v1/REPORT.html).
