# Automatic group-free IU — frozen Phase A4 protocol

**Boundary:** `automatic-group-free-iu-a4-v1-2026-08-13`

**Status:** pre-registered before the Llama structural evaluation

**Allowed claim:** existence (or absence) of a shared/repeatable telemetry component

**Forbidden claim:** identification, selection, orientation, or validation of a
hallucination detector

## 1. Scientific question and hard identifiability boundary

The 3,400 ProcessBench items are the same fixed responses scored by Qwen3-4B,
Qwen3-8B, and Llama-3.1-8B. Changing the scorer can change measurement
nuisance, but it cannot change whether the fixed response is correct. The
experiment can therefore ask whether a reproducible cross-scorer component
exists. It does not identify a complementary scorer-sensitive component:
low-repeatability residual variation may be scorer-specific target sensitivity,
measurement nuisance, or noise.

It cannot decide whether a shared component is hallucination, item difficulty,
response length, style, or another item property. No A4 outcome may create a
detector, choose a hallucination component, orient a risk score, set trust, or
open a correctness/step label. The target-identification verdict is frozen in
advance as `CLOSE_NO_TARGET_CONTRAST`. A separate premise verdict may pass or
fail for the usefulness of the shared/repeatable component.

The local surface audit found no legal strict-S1 target-changing pair. The
available RAGTruth full/no-context/leave-one-chunk-out conditions change
evidence while holding the answer fixed; they are self-supervised
interventions reserved for A6. They may not be imported into A4 after seeing
the structural result.

The legal data claim is “no new labels beyond the frozen mixed-v2 contract.”
The mixed-v2 signs and nonlinear transforms were inherited from earlier
label-informed development; A4 is not label-naive.

## 2. Data firewall and fixed roster

The public loader constructs new records from this whitelist only:

- `id`, `problem`, and `steps` for exact pairing, grouping, and
  text-only controls;
- `token_entropies`, `token_spilled_energies`, `token_logsumexp`, and
  `top_k_logprobs` for the 30 registered extractors.

It never copies `label`, `labels`, `first_error`, `error_label`,
`final_answer_correct`, `align_diag`, or step spans. Public fit and structural-evaluation
functions accept numeric arrays, feature names, subset IDs, item-group IDs,
and text covariates—not raw cache dictionaries. Tests must prove that a
fixture containing target keys produces the same sanitized record as one
without them and that a prohibited key is rejected at the fit boundary.

`align_diag` is not copied into the sanitized record. Before sanitization the
loader may read only its boolean-equivalent `problems` field to reject
misaligned rows; no other nested value crosses the firewall.

The primary roster is exactly:

```
epr, trace_length, spectral_entropy, low_band_power, high_band_power,
hl_ratio, dominant_freq, spectral_centroid, stft_max_high_power,
stft_spectral_entropy, rpdi, sw_var_peak, pe_mean, hurst_exponent,
cusum_max, cusum_shift_idx, epr_spilled, sw_var_peak_spilled,
cusum_max_spilled, epr_energy, min_energy, sw_var_peak_energy,
cusum_max_energy, mean_top1_logprob, logprob_margin,
mean_logprob_entropy, varentropy, renyi_entropy_2, topk_tail_mass
```

This is the 29-feature intersection emitted in the frozen no-label score
artifacts `results/leverage_balanced_processbench_transfer_v1/FIT_MANIFEST.json`
and its eight score files, whose hashes become A4 boundary inputs. It contains
every registered mixed-v2 feature except `min_spilled`, which the existing
deterministic degeneracy rule removed in both Qwen views. Llama's extra
non-degenerate copy is excluded before evaluation. The script must assert the exact roster, exact
3,400-by-3 pairing, and finite raw values; the reusable transformer is not
allowed to exercise its median-imputation branch.

## 3. Splits before transformations

Exact hashes of `(problem, steps)` define item groups. All duplicates,
including cross-subset duplicates, remain in one fold. A deterministic
five-fold `StratifiedGroupKFold` uses subset as the stratification variable and
seed 20260813. The item split is made before fitting a transform, nuisance
model, ridge value, component, or baseline.

For every outer fold:

1. Qwen3-4B and Qwen3-8B representations of the four training folds are the
   only fitting data.
2. All three views of the held item groups are evaluation-only.
3. Llama is never used to choose a transform, nuisance model, ridge,
   component, baseline, null, or interpretation.

The pooled-Qwen `FixedMixedV2Transformer` is fitted on outer-training rows and
applied unchanged to held Qwen and Llama rows. A raw confidence-oriented
z-score transform with no squared/mode operation is frozen as one diagnostic
sensitivity analysis, not as an alternative candidate.

## 4. Feature-level content/length control

The scorer views share response content and almost share token length. A
simple item shuffle is therefore vacuous. Before fitting a shared component,
each transformed feature is residualized using a text-only multivariate ridge
model with fixed ridge 1.0 and these predeclared covariates:

- log response character count and word count;
- log reasoning-step count;
- log problem character count and word count;
- log scorer-token count;
- squared versions of the six standardized continuous covariates;
- subset indicators; and
- an effect-coded Qwen view indicator (-1 for 4B, +1 for 8B, zero for the
  unseen Llama midpoint).

The design is standardized from Qwen training items only. Five group folds
produce cross-fitted Qwen training residuals separately for every feature;
the full outer-training nuisance model is then applied to held items. CorrCA
is refitted on the residual feature matrices. Removing length only from the
final scalar is prohibited.

The text-only predicted feature matrices are retained only for direct
confounding diagnostics against the actual held component score.
All headline correlations are computed within subset and then combined with
an equal-subset Fisher-z macro, so dataset identity cannot create the primary
effect.

## 5. Frozen shared-component estimator

Let `X4` and `X8` be the cross-fitted Qwen training residual matrices with `n`
rows and 29 features. Each is centered with its own training column mean. With
`A.T @ C / (n - 1)` defining the 29-by-29 sample cross-covariance of centered
matrices `A` and `C`, define

```
W = (X4.T @ X4 + X8.T @ X8) / [2 * (n - 1)]
B = (X4.T @ X8 + X8.T @ X4) / [2 * (n - 1)]
```

where `B` is explicitly symmetrized. For ridge fraction `a`, solve

```
B v = lambda [W + a * trace(W)/29 * I] v.
```

Only the leading generalized eigenvector is retained. Its algebraic sign is
fixed by making the lexicographically first coefficient among those with
maximum absolute magnitude positive; this sign has no correctness meaning.

The ridge grid is `[1e-4, 1e-3, 1e-2, 1e-1, 1]`. Nested four-fold grouped item
validation inside the outer training set selects the ridge with the highest
equal-subset Fisher-z Qwen4-vs-Qwen8 correlation; ties choose the larger ridge.
Every transform, nuisance model, and component is refitted inside the nested
fold. No training eigenvalue is treated as evidence.

The same complete fitting and ridge-selection pipeline is repeated inside
every training-pair null.

## 6. Frozen baselines

All baselines see the same outer-training Qwen pairs, the same transformations,
the same feature-level nuisance residuals, and the same held triples:

1. ridge CCA rank one, using separate Qwen loadings `a4` and `a8`. Unit-normalize
   each, flip `a8` when `a4.T @ a8 < 0`, average, and unit-normalize again. A
   zero-norm average is an explicit failed fit. Specifically, with training
   covariances `C44`, `C88`, and `C48`, eigendecompose the two symmetric
   matrices `Cmm + a*trace(Cmm)/29*I`, form their symmetric inverse square
   roots `P4` and `P8`, take the leading singular vectors `u,v` of
   `P4 @ C48 @ P8`, and set `a4=P4@u`, `a8=P8@v`. The same ridge grid, nested
   validation, and larger-ridge tie rule as CorrCA apply. Nested selection and
   every held evaluation project all views through the averaged common vector,
   never through separate-view CCA scores;
2. diagonal per-feature reliability weighting: for centered columns use
   `r_j = 2 cov(X4_j,X8_j) / [var(X4_j)+var(X8_j)]`, set
   `q_j=max(r_j,0)`, and use `q/sqrt(sum(q^2))`; all-zero `q` is a failed fit;
3. the best single feature selected by nested Qwen repeatability;
4. rank-one pooled PCA on stacked Qwen training residuals; and
5. the equal-weight 29-feature mean.

Where a ridge is required, the same grid and nested Qwen-only rule is used.
The strongest paired baseline is chosen inside each outer training fold by
nested Qwen repeatability, never by held Qwen or Llama results. Text-only
prediction is not a cross-view correlation baseline. Instead, after the
CorrCA vector is frozen, one separate ridge-1 scalar model is fitted on the
outer-training Qwen4 and Qwen8 CorrCA scores stacked together. It uses the
same training-fitted text covariates and view coding as the feature nuisance
model, but its target is the actual scalar component score—not the previously
predicted feature matrix. The unchanged scalar model predicts held Qwen4,
Qwen8, and Llama scores. R-squared and absolute Pearson correlation are
computed separately in every fold-by-subset-by-view cell. Within each outer
fold, each view receives an equal-subset arithmetic macro; the confounding
gate uses the worst (largest) macro across the three views. This is repeated
independently for every outer fold.

## 7. Evaluation statistics and nulls

For each outer-held fold and subset, report:

- Qwen repeatability: Pearson correlation between frozen Qwen4 and Qwen8
  component scores;
- Llama external structural check: Pearson correlation between frozen Llama
  score and the mean of the two frozen Qwen scores;
- correlations for every fixed baseline;
- held R-squared and absolute correlation between each actual component score
  and its training-only text-feature prediction;
- pooled values as diagnostics only.

Correlations are computed separately in every outer-fold-by-subset cell. The
primary summary is the equal-weight mean of finite Fisher-z correlations over
the 5-by-4 cells, transformed back with `tanh`; at least four items and
nonconstant scores are required in every cell. Raw out-of-fold scores from
different fitted models are never concatenated. A 5,000-draw paired bootstrap
resamples complete item groups within every fold-by-subset cell and recomputes
this exact macro. It reports individual Qwen and Llama macro intervals and the
paired CorrCA-minus-preselected-baseline interval. Llama is one external
model-family check, not general model-family validation.

Two conditional pairing nulls are mandatory. Response-word-length decile
edges are fitted on each outer-training subset. Deterministic derangements
operate within subset and fitted length decile:

1. **training-pair null:** Qwen8 item records are conditionally deranged,
   and the full transformer, residualizer, nested ridge selection, and
   component fit are rerun for 200 seeded nulls;
2. **held-pair null:** with the real fitted component frozen, held Llama item
   scores are conditionally deranged for 1,000 seeded nulls.

The script records stratum sizes, proves there are no fixed points, and aborts
if any required stratum cannot be deranged. Nulls resample item triples, not
individual feature rows. Each null seed reruns all outer folds and produces
the same equal-fold-by-subset Fisher-z macro as the observed statistic. The
95th percentile is taken over those complete per-seed macro statistics, never
over fold-level fragments.

Finally, four leave-one-subset-out fits train on the other three subsets and
evaluate the omitted subset without using Llama during selection. Component
stability is the squared cosine across outer-fold CorrCA vectors.

## 8. Non-vacuous premise gate

`PASS_SHARED_REPEATABLE_COMPONENT_PREMISE` requires every condition below:

1. primary residualized CorrCA exceeds the strongest preselected paired
   baseline on Llama-vs-Qwen macro correlation by at least 0.02, with a paired
   95% interval whose lower bound is above zero;
2. Qwen repeatability and Llama transfer are positive in every subset, where
   each subset value is its five-fold Fisher-z macro, and both equal-subset
   macro intervals exclude zero;
3. in every outer fold, the worst-view equal-subset macro from the separately
   fitted scalar text model has held R-squared at most 0.10 and held absolute
   correlation at most 0.35; this is a direct residual-confounding diagnostic,
   not a correlation between two text predictions;
4. observed Qwen repeatability exceeds the 95th percentile of the full
   training-pair null and observed Llama transfer exceeds the 95th percentile
   of the held-pair null, each by at least 0.02;
5. leave-one-subset-out Llama transfer is positive in at least three of four
   subsets and its equal-subset macro is positive; and
6. minimum outer-fold squared-cosine component stability is at least 0.70.

Failure of any condition gives `CLOSE_SHARED_REPEATABLE_COMPONENT_PREMISE`.
Passing supplies only a reusable shared-structure diagnostic for A5/A6. In
both cases the detector verdict remains `CLOSE_NO_TARGET_CONTRAST`, no detector
score is constructed, and A5 begins next.

## 9. Required artifacts

The phase writes a source/data boundary before evaluation, exact fold and
feature manifests, nested selections, held predictions, per-subset metrics,
paired bootstrap draws or their hashes, both null distributions, LO-subset
fits, raw-z sensitivity diagnostics, a machine-readable gate decision, an
artifact hash manifest, and a concise report. All transitive source and input
hashes must verify before the phase is committed.

## References

- Parra, Haufe, and Dmochowski, [Correlated Components Analysis — Extracting
  Reliable Dimensions in Multivariate Data](https://arxiv.org/abs/1801.08881),
  2018.
- Lock et al., [Joint and Individual Variation Explained](https://doi.org/10.1214/12-AOAS597),
  2013.
- Feng et al., [Angle-Based Joint and Individual Variation Explained](https://arxiv.org/abs/1704.02060),
  2018.
- Sørensen, Kanatsoulis, and Sidiropoulos, [Generalized Canonical Correlation
  Analysis: A Subspace Intersection Approach](https://doi.org/10.1109/TSP.2021.3061218),
  2021.
