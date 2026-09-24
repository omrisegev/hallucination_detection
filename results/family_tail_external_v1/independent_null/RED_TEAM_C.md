# Independent null and math review

Completed all 6,190 answer/backbone records and 53,970 official step slots, with 53,964 nonempty feature steps. No GPU or new inference. All three prediction seals were verified before evaluator labels were opened. The frozen script remained unchanged.

Independence limit: this reviewer authored the historical hist26 extraction adapter. This audit independently reimplements downstream family construction, score normalization, decisions, confusion metrics, and perturbations; it is not an independent raw feature-extraction review. Root METRICS, CONTRASTS, REPORT and other reviewers' outputs were not read.

## Results

| Cell | N answers | Tail score (%) | Family equal (%) | Bank11 (%) | Tail after entropy-family permutation (%) | Tail decisions changed |
|---|---:|---:|---:|---:|---:|---:|
| hard2verify_qwen3_8b | 200 | 41.0196 | 42.3801 | 43.6695 | 42.5856 | 78 |
| socratic_qwen3_8b | 2995 | 59.9205 | 61.1239 | 63.2213 | 59.2256 | 1080 |
| socratic_qwq32b | 2995 | 62.6897 | 62.9365 | 64.2382 | 62.1649 | 970 |

Hard2Verify uses its balanced F1 (harmonic mean of class recalls); Socratic uses mean class F1 (PRMScore). These metric columns must not be pooled across benchmarks.

All 30 arm/cell primary scores exceed the 97.5th percentile of both diagnostic null schemes. This verifies signal relative to shuffled labels, not superiority of learned fusion. The tail candidate is below Bank11 and its matched family-equal control in all three observed cells. Its advantage over continuous-family L-SML occurs only for QwQ; the root registered paired bootstrap, not these null ranges, determines comparison uncertainty.

Label null: 200 deterministic global class-count-preserving permutations and 200 within-answer class-count-preserving permutations per cell. The same label permutations are shared across all arms and both Socratic backbones. Global shuffling tests association ignoring answer structure; within-answer shuffling retains each answer's prevalence while breaking step location. The empirical plus-one upper p-value floor is 1/201; these are exploratory diagnostics, not multiplicity-adjusted superiority tests.

Feature diagnostic: a single predeclared UID-seeded permutation of the standardized level_entropy family across nonempty steps in every answer, with the other 14 families fixed and no fitting. Both frozen candidate and matched control use their original source thresholds. It preserves the family marginal and within-family channel information. It changes 2,128 candidate decisions and 1,719 family-equal decisions. Candidate performance drops on both Socratic backbones but rises on Hard2Verify. A single permutation is descriptive and does not establish that this family is universally beneficial or causal.

## Math and implementation findings

- Independent candidate and family-equal scoring replay covers 12,380 answer/arm computations. Maximum score discrepancy is 8.881784197001252e-16; every valid-step decision matches. All 6,190 label/metadata mutation checks pass. The API computes from features and names only.
- All 100 tail-mark fixtures preserve the top-ceil(20%) quota, fractional boundary ties, centering, constant-column zero, and positive-affine rank invariance.
- For K=2 and nonzero off-diagonal covariance c, the matrix [[0,c],[c,0]] has equal-magnitude leading eigenvector coordinates. Six signed numerical checks pass. The outer stage does not identify a relative reliability split. Zero covariance is degenerate.
- The final normalized weights are not equal L1 group mass: the two CUSUM families total 0.2992462715352441; the other 13 total 0.7007537284647559. Both groups have L2 norm 0.21159906784736202. Inner unit-L2 normalization explains this difference.
- The fitting marks include fractional ties and answer-centered quotas. Deployment uses continuous standardized features. A synthetic monotone transform keeps every tail mark unchanged yet changes deployed scores by up to 0.743594398197263. Binary-classifier reliability guarantees cannot simply be asserted for this deployment.

## Claim assessment

| Claim | Verdict | Evidence |
|---|---|---|
| Predictions contain step-quality signal | Supported relative to the two specified shuffle nulls | LABEL_NULL.json, all 30 arm/cell scores |
| Tail candidate improves on Bank11 or family equal in all external cells | Refuted at the observed point estimates | Independent full-population confusion calculations |
| Candidate deployment uses the retained entropy family | Supported as numerical sensitivity; no universal benefit claim | FEATURE_NULL.json, full 6,190 records |
| K=2 implies learned between-group reliability weighting | Not supported | SYNTHETIC_MATH.json and frozen deployment group norms |
| Target labels affect feature-based predictions | No effect in tested mutation interface | 6,190 exact feature-family replays with mutated label metadata |

Artifacts: `PREDECLARED.json`, unchanged `audit_null_math.py`, `SYNTHETIC_MATH.json`, `LABEL_NULL.json`, `LABEL_NULL_DRAWS.npz`, and full `FEATURE_NULL.json`. Per-answer feature rows remain in the full artifact; root will create the tracked compact summary.

Execution wall time: 108.134 seconds. FEATURE_NULL SHA256: f11a10921fd0dafc6f8d3dd47ba5bb153032c5e1a6c5505b7559146833cf65ec.
