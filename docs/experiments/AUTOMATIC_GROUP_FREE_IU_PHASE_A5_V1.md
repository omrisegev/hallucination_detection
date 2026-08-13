# Automatic group-free IU — Phase A5 protocol v1

**Protocol date:** 2026-08-13

**Supervision tier:** S1, with no new correctness labels beyond the frozen
mixed-v2 IU input contract

**Status:** preregistration; no A5 synthetic or real result has been opened

**Primary model:** IU-anchored equal-covariance continuous sparse latent mixture

## 1. Question and bounded claim

A5 asks whether an item-level latent likelihood can estimate a useful atomic
correction to IU-PCR while modelling conditional feature dependence, without
the manual provenance groups used by Family-NRM.

The primary model is new only in this narrow sense: it fits a continuous
two-component sample likelihood and a sparse within-component precision
jointly, then restricts the deployable correction to an affine IU-anchored
direction. It is not sparse marginal covariance, inverse-covariance anomaly
scoring, a GMM negative-log-likelihood detector, DEEM pseudo-probabilities,
higher-moment deflation, or a labelled anchored head. Those routes have already
been tested and closed.

Unlabelled likelihood does **not** identify correctness. A two-component model
is invariant to component label switching and may fit difficulty, length,
style, or another latent nuisance. A5 therefore makes one explicit, unavoidable
assumption: the frozen IU-PCR score is the target-semantic anchor. It orients
each environment-local label switch from adaptation rows, but no
majority-better-than-random rule is permitted.
Passing A5 can establish improvement conditional on that anchor; it cannot
prove that the anchor itself denotes correctness.

## 2. Observational-equivalence audit

Before any performance gate, the implementation must construct two synthetic
descriptions with identical `P(X)`:

1. the mixture bit is named target `Y` and an independent bit is nuisance `Z`;
2. the same two bits exchange semantic names while the generated `X` is
   unchanged.

With the same seed, the complete observed package `(X, w_IU)`, fitted
parameters, selected alpha, and scores must be bit-identical. Only inaccessible
diagnostic labels may change. The anchor-points-to-nuisance world is an
impossibility demonstration, not a target-recovery world.
This is a required proof-by-construction that A5 cannot solve target semantics
from `P(X)` alone. If inheriting the IU target anchor is considered
unacceptable, the registered outcome is `CLOSE_UNIDENTIFIABLE_WITHOUT_IU_ANCHOR`
and execution moves to A6.

## 3. Data and target firewall

### 3.1 Primary roster

The real premise test uses the exact 23 A0 source environments and the 17 raw
features present in all of them, in this frozen order:

1. `cusum_max`
2. `cusum_max_energy`
3. `cusum_max_spilled`
4. `epr`
5. `epr_energy`
6. `epr_spilled`
7. `logprob_margin`
8. `mean_logprob_entropy`
9. `mean_top1_logprob`
10. `min_energy`
11. `renyi_entropy_2`
12. `rpdi`
13. `sw_var_peak`
14. `sw_var_peak_energy`
15. `sw_var_peak_spilled`
16. `topk_tail_mass`
17. `varentropy`

This automatic complete-core rule is fixed by the A0 presence matrix. It
contains no `trace_length` coordinate and no A4 component. The missing-aware
30-feature model is secondary and may be run only if every primary real-data
premise gate passes; it cannot replace, tune, or rescue the primary.

### 3.2 Rebuilt target-free tensor

`results/dependency_fusion_raw/cells.npz` is forbidden as an A5 fit input. It
contains label arrays, whole-cell standardized matrices, and no item-group
identifiers. A5 instead rebuilds a sanitized tensor from the A0 raw cache
sources.

The isolated sanitizer may inspect only:

- the top-level problem key, used as the within-source `item_group_id`;
- normalized prompt/question content, inspected before sanitization solely to
  compute a global SHA-256 `content_group_id` and never copied as text;
- candidate ordinal within that problem;
- `token_entropies`, `token_spilled_energies`, `token_logsumexp`, and
  `top_k_logprobs`, solely to derive the frozen features;
- token-count information solely to write the forbidden-fit length sidecar.

One inherited A0 preprocessing exception is isolated before the public A5
sanitizer. For `seiclr_triviaqa_opt30b` only, a hash-bound crop stage may read
exactly `full_text` and `token_offsets` to reproduce the already frozen
first-answer-line span from `spectral_utils.answer_span.answer_token_slice`; it may
output only the four cropped telemetry streams above. It cannot expose text,
offsets, grader output, answers, or labels to A5 fitting. The crop implementation
and test are included in the source boundary. This exception is not described
as label-naive: it inherits the earlier grader-alignment decision but reads no
correctness value.

It must copy none of `label`, `label_lexical`, `correct`, `is_correct`, gold
answers, prompts/questions, generated/full text, or nested target-bearing
objects. The sanitizer records but does not expose unexpected keys, then fails
closed if required telemetry is absent or a nonfinite core feature is produced.
The source-specific question extractor is exactly top-level `row["question"]`
for every registered A0 raw cache; nested `gold_row`, prompt, candidate text,
and answers are never fallbacks. Prompt normalization is Unicode NFKC followed
only by whitespace collapse; case and punctuation are preserved. Each row gets
both `canonical_item_id = sha256(dataset_revision + "\\0" + split + "\\0" +
str(top_level_key))` and `content_hash = sha256(normalized_question)`. Global
`content_group_id` is the connected component under equality of either ID.

Dataset revisions/families are frozen as: `gsm8k/test -> gsm8k`,
`math500/test -> math500`, all three `trivia_qa* / validation -> triviaqa`,
and one family each for `hotpotqa`, `nq_open`, `sciq`, `squad_v2`, and
`truthfulqa`. Revision is the manifest dataset string, except all GSM8K and all
MATH500 sources deliberately share the canonical revisions above.

For every pair of sources sharing a canonical revision, audit the intersection
of top-level keys and require identical normalized-question hashes for 100% of
that intersection. The overlap matrix records expected overlap as the
intersection of canonical IDs and observed overlap as shared connected
components; any discrepancy closes as `CLOSE_INVALID_GLOBAL_ITEM_BOUNDARY`.
Cross-revision content-hash overlaps are recorded and merged but have no
minimum expected count. If a source lacks top-level `question` or fewer than
99.9% of rows receive both IDs, close as
`CLOSE_INADEQUATE_GLOBAL_GROUP_BOUNDARY`; never fall back to integer-only IDs.

Public fit/evaluation APIs accept typed arrays only and reject mappings or
arguments containing a target-like key. Unit tests use sentinel target objects
that raise on access.

The prepared boundary contains raw core values, environment and dataset-family
IDs, item-group and candidate IDs, missingness, a raw exact-token-count
sidecar, source manifests/hashes/sizes/timestamps, and deterministic group
folds. It contains no prompt text and no target. Confidence signs are inherited
from the frozen IU contract and disclosed as earlier label-informed preprocessing. Means,
standard deviations, and every local transform are fitted on the applicable
training/adaptation split only.

Length is never a fit input. Raw and log token count are available only to the
post-fit confounding gate. No A4 residualizer is reused.

Source SHA-256, byte size, and manifest SHA-256 are canonical. Local filesystem
mtime is explicitly excluded because downloading identical LFS bytes changes it;
the prepared audit writes the fixed marker
`noncanonical-filesystem-mtime-excluded` instead of a machine timestamp.

The primary structural population contains exactly one response per
`(environment_id, content_group_id)`: choose the candidate minimizing the tuple
`(sha256("A5-primary-response\\0" + environment_id + "\\0" +
content_group_id + "\\0" + item_group_id + "\\0" +
str(candidate_ordinal)), item_group_id, candidate_ordinal)`. All other
responses are excluded from structural fitting, likelihood, nulls, and gates.
They may be scored only in the final retrospective application after the local
model is fitted on the deterministic one-response subset. This makes prompt,
not response, the independent likelihood unit.

## 4. Primary model

For item `i` in environment `e`, with latent bit `z in {-1,+1}`:

```text
X_ie | z ~ Normal(m_e + z * delta_e / 2, Sigma_e)
P(z=+1) = pi_e
Omega_e = inverse(Sigma_e)
```

The two classes share `Sigma_e`, so posterior log odds are affine. Environment
means, priors, separations, and nonzero precision values are local. The
precision penalty and graph support are learned from training environments
only and then shared. Every local covariance is fitted under that frozen
support. Component priors are clipped to `[0.05, 0.95]` only during numerical
optimization; the effective-mass gate below decides whether the solution is
usable.

### 4.1 Sparse graph and mixture fit

The implementation uses the following one-way graph pipeline:

1. fit a diagonal, IU-oriented two-component mixture in each training
   environment for initialization;
2. form each environment's responsibility-weighted within-component
   covariance, convert it to correlation, and average correlations with equal
   environment mass; components are pooled within environment according to
   their effective membership because their covariance is shared;
3. fit graphical lasso on penalty grid
   `{0.01, 0.02, 0.05, 0.10, 0.20}` and define support by
   `abs(partial_correlation) > 0.01`;
4. refit each environment by EM with a shared-class covariance and a
   fixed-support Gaussian precision MLE;
5. recompute the equal-environment residual correlation after sparse EM and
   refit support once at the **same already selected penalty**, then stop. The
   penalty is not reselected and a second update is prohibited.

The fixed-support precision MLE minimizes
`trace(S Omega) - logdet(Omega)` over the registered free diagonal/edge
entries. It starts at the positive diagonal solution, rejects non-positive
definite iterates, and must satisfy relative gradient and objective tolerances
of `1e-7` and positive minimum eigenvalue above `1e-8`. The unconstrained
sparse EM uses at most 300 iterations; the scalar constrained-direction EM on
the alpha path uses at most 1,000 iterations. Both stop when relative likelihood
change is below `1e-7`. The
graphical-lasso dual-gap tolerance is `1e-4`, two orders of magnitude below
the edge threshold; any nonconverged penalty arm is excluded rather than
retained, its failure is recorded, and the fold closes if no penalty arm is
usable.
The
larger graphical-lasso penalty wins likelihood ties within `1e-8` nats. The
implementation boundary will freeze initialization seeds before sealed
synthetic execution.

### 4.2 IU anchor, orientation, and trust

Let `w_IU,e` be frozen IU-PCR fitted on the same adaptation rows with
`IU_FIT_DEFAULTS`. Let `w_mix,e = Omega_e delta_e`. Resolve only the mixture's
environment-local label switch using that adaptation-fitted IU anchor:

```text
choose sign so w_mix,e' Sigma_e w_IU,e > 0
```

Scale `w_mix,e` to have the same `Sigma_e` norm as `w_IU,e`, then form the
IU-orthogonal correction

```text
u_e = w_mix,e
      - w_IU,e * (w_IU,e' Sigma_e w_mix,e)
                 / (w_IU,e' Sigma_e w_IU,e)
w_e(alpha) = w_IU,e + alpha * u_e
```

The alpha grid is `{0, 0.125, 0.25, 0.5, 1}`. For likelihood selection, each
direction is an actual constrained equal-covariance mixture: with the sparse
`Sigma_e` frozen from adaptation, its mean separation is restricted to
`beta * Sigma_e w_e(alpha)`, while `beta >= 0`, centre, and prior are refitted
on adaptation only. Held likelihood is evaluated under that constrained
density. The final deployed ranking score is exactly
`w_e(alpha)' X`; density scale `beta` is not folded into the score.

Alpha and graph penalty are selected by the explicit nested procedure in
Section 7. Within each validation fold, first find the empirical best
penalty/alpha. For every arm compute paired per-environment likelihood losses
relative to that empirical best and their ordinary sample standard error over
validation environments. An arm is admissible when its mean loss is no greater
than one standard error (plus `1e-12` numerical tolerance). Select the smallest
alpha among admissible arms, then the larger graph penalty, then the larger
likelihood. This conservative one-standard-error rule is frozen from
development data before any sealed seed is opened and is applied identically
to the candidate, diagonal and random-support controls. If any reliability
gate fails, the returned weights and scores are copied from IU-PCR, not
recomputed, and must agree below `1e-12` relative error.

All 17 primary coordinates use the raw branch of mixed-v2. In each legal fit,
`z_j=(sign_j*x_j-mu_j)/sigma_j`, with frozen confidence sign and train-fitted
`mu,sigma`. Thus the transformed score is folded exactly into raw weight
`sign_j*w_j/sigma_j` and intercept
`-sum_j(w_j*mu_j/sigma_j)`. Maximum reconstruction error over the original raw
17-column matrix must be below `1e-10`.

## 5. Matched controls

Density-mechanism controls receive identical splits, local adaptation,
standardization, initialization budgets, and equal-environment aggregation.

1. the capacity-identical anchored sparse two-component model with the same
   support, covariance, centre/prior fit, and adaptation budget, constrained to
   `alpha=0`;
2. the anchored two-component model with diagonal covariance;
3. a one-Gaussian fixed-support model with no latent correction;
4. 32 degree- and edge-count-matched random graph supports, each refitted; the
   strongest support is selected using nested training likelihood only and is
   the registered held comparator;
5. an unanchored equal-covariance two-component mixture, density-only and
   never promotable;
6. the prior GMM-NLL anomaly scorer, demonstrating that A5 is not the closed
   direction-free anomaly route;
7. exact IU-PCR, identical in ranking to the `alpha=0` direction.

Existing SU-PCR, continuous L-SML/tetrad, Family-NRM, and prior GMM-NLL are
frozen score comparators, not density-mechanism controls; identical mixture
initialization is neither claimed nor applicable. Family-NRM is never a
selector, orienter, or tuner.

The diagonal arm and every random-support arm independently select their own
alpha by the identical inner-validation likelihood and one-standard-error rule. The
strongest random arm is chosen only after those independent selections, by its
inner-validation likelihood. The capacity-identical sparse `alpha=0` arm
reuses the candidate's frozen covariance and every other fit object, changing
only alpha. The retrospective diagonal score uses the final alpha selected by
the diagonal arm, never the sparse candidate's alpha.

Random supports use deterministic degree-preserving double-edge swaps. For arm
`d=0..31`, first order vertices lexicographically by their immutable feature
names, then encode the exact UTF-8 payload
`"A5-random-support\0" + decimal(split_seed) + "\0" +
float(penalty).hex() + "\0" + decimal(d)`. Initialize NumPy PCG64 from the
first eight SHA-256 digest bytes interpreted unsigned big-endian. Starting from
the candidate support, repeatedly choose two distinct undirected edges uniformly;
choose one of the two cross-rewirings by one RNG bit; reject self-loops,
duplicate edges, unchanged edge sets, or a support already emitted in that
split. Accept exactly `max(100,20*E)` swaps, with at most
`1000*max(E,1)` attempts. Connectivity is neither required nor repaired;
diagonal entries remain present. Each accepted arm must preserve the exact
degree sequence and edge count. A nonconverged or non-unique arm is recorded as
unusable; it is never silently replaced by a different seed. Fewer than 32
usable unique arms closes the dependency comparison as
`CLOSE_INADEQUATE_RANDOM_GRAPH_CONTROL` (and a zero-edge selected graph closes
the dependency premise directly). The quotient is learned inside the same
training fold before swaps; its mean/contrast vertices receive immutable names
derived lexicographically from the frozen original feature names. Each arm independently refits all local
mixtures and selects alpha.

The A1 factorial prior is excluded from the primary. It may appear once as a
separately named metadata-assisted negative control and cannot rescue a failed
gate or choose any hyperparameter.

## 6. Execution stages and synthetic stop rule

### A5-S0 — implementation and development seeds

Unit tests cover target firewall, likelihood monotonicity, affine posterior,
feature permutation equivariance, fixed-support recovery, group splits,
duplicate behavior, deterministic ties, and exact IU fallback. Implementation
may be debugged only on the development seed namespace `510000..519999`.

Before sealed execution, hash source, tests, simulator, exact configurations,
and the unused sealed seeds `520000..529999`. The execution is divided into
independently frozen stages: S1a is world 8 only; a PASS may open a new
preregistered S1b boundary for the already specified remaining synthetic
worlds, without changing the candidate estimator, grids, gates, or any S1a
interpretation. Only a S1b PASS may open a separately preregistered real-data
S2 implementation of Sections 7--8. Each later boundary must cite and hash all
earlier results, and an earlier result may determine only whether the next
stage opens—not alter its method or gates. No result-changing repair is
permitted inside a stage after that stage's first sealed result is opened.

Synthetic AUROC always treats planted `Y=+1` as positive and is computed within
each sealed test environment before equal-environment averaging. The oracle is
the affine Bayes score `wY_e'X`. All repetition intervals below are paired
percentile bootstraps with 20,000 draws from namespace `529000` plus a stable
hash of the gate name. Graph/alpha fitting follows the same train/validation/
test separation as the real nested algorithm, with no access to planted bits.

### A5-S1 — sealed synthetic premise

Every world has exactly 100 repetitions. World index is one-based in the list below;
repetition `r` uses seed `520000 + 200*world_index + r`, `r=0..99`.
No data-dependent redraw is allowed.

S1a opens world 8 first as the registered hard anti-repackaging stop.
Only its candidate path is needed for that gate; density controls and the other
worlds remain unopened until it passes. A world-8 failure writes the complete
100-repetition closure artifact, closes A5 immediately, and prohibits both the
remaining synthetic worlds and any real-cache transfer. A PASS permits only
the independently reviewed S1b boundary described above; the current runner's
remaining-world implementation is a prewritten diagnostic scaffold and is not
authorized to execute until that boundary is separately frozen. This execution
order is frozen before the first sealed seed is opened.

The base world has `p=17`, 12 environments, and 400 adaptation plus 400
evaluation items per environment. A seed permutation assigns eight graph-
training, two alpha/penalty-validation, and two sealed test environments. For
each item, independent balanced Rademacher bits `Y,Z` and Gaussian noise
generate

```text
X = m_e + Y*Sigma_e*wY_e*dY/2 + Z*Sigma_e*wZ_e*dZ/2 + epsilon
epsilon ~ Normal(0, Sigma_e)
```

The undirected true graph has edges `(j,j+1)` for `j=0..15` and `(j,j+3)` for
`j in {0,3,6,9,12}`. With its adjacency `A` and degree matrix `D`,
`Omega0 = I + 0.18*(D-A)`. Environment scales are
`q_ej=exp(0.10*N(0,1))`, `Sigma_e=diag(q_e)*inverse(Omega0)*diag(q_e)`, and
`m_ej=0.3*N(0,1)`. Base feature-space mean-separation vectors are zero except
`bY_(0,4,8,12)=(1,.8,-.7,.6)` and
`bZ_(2,9,15)=(1,-.9,.7)`. Set
`deltaY_e=diag(q_e)bY`, `wY_e=Omega_e*deltaY_e`, and analogously for `Z`, then
normalize each discriminant to unit `Sigma_e` norm. The frozen synthetic IU anchor is the
unit-`Sigma_e` vector proportional to `wY_e + 0.8*h_e`, where `h_e` is the
seeded unit-`Sigma_e` projection of `N(0,I)` orthogonal to `wY_e`. The favorable
sparse world uses `dY=1.4,dZ=0`.

The eleven worlds are:

1. favorable sparse Gaussian above;
2. independent/no-correction target, replacing `Omega0` by `I` and setting
   `w_IU=wY` exactly;
3. clipped-A0 small-n, using 23 environments with total item counts
   `(500,500,500,500,500,500,500,500,500,500,500,300,300,300,300,500,500,198,500,500,500,500,500)`
   split 17/3/3 by seeded environment permutation; within each environment,
   rank items by `sha256("A5-small-n" + seed + environment_index +
   item_index)` and assign the first `floor(n/2)` to adaptation and the rest to
   evaluation, so each item is its own group and the two halves differ by at
   most one;
4. Student-t noise `epsilon=Gaussian*sqrt(5/ChiSquare_5)`;
5. heteroscedastic classes, multiplying `Sigma_e` by `0.65` for `Y=-1` and
   `1.35` for `Y=+1`;
6. exact and near duplicates, two separately gated variants with 100 paired
   repetitions each, appending respectively `X_0` or
   `.999*X_0+sqrt(1-.999^2)*eta` with independent standardized `eta`;
7. anti-oriented-coordinate majority, replacing `bY` by the dense vector with
   entries `-1/sqrt(17)` on indices `0..9` and `+1/sqrt(17)` on `10..16`, then
   constructing the still target-valid IU anchor from its Bayes direction;
8. nuisance-dominant mixture with `dY=1.0,dZ=1.8` and `Z` independent of `Y`;
   half the environments have one response for each of 800 prompts, while half
   have ten responses for each of 800 prompts with one prompt-shared `Z`; the
   primary deterministic-response rule reduces both to 800 independent prompt
   units, split into 400 adaptation and 400 evaluation prompts;
9. environment-specific nuisance with `dY=1.0` and
   `dZ in {-1.8,0,+1.8}` cycling by environment;
10. no-latent one-Gaussian with `dY=dZ=0`;
11. anchor-points-to-nuisance equivalence with `dY=1.0,dZ=1.8` and the IU
    anchor constructed around `wZ`; swapping the semantic names of `Y,Z`
    changes no observed object.

The feature-deletion stress is paired with worlds 3 and 4. Graph-training
environments remain byte-identical and the full graph/quotient is fitted once.
For validation or test environment `e`, rank its original coordinates by
`sha256("A5-feature-deletion" + seed + environment_id + coordinate_index)`;
remove the first one, two, then three, and use the principal induced support of
the already fitted full graph—never relearn a lower-dimensional graph and never
impute from held rows. If the learned quotient contains a deleted coordinate,
remove that member; drop a quotient component only if it becomes empty, and
renormalize its mean over retained members. For diagnostic truth on retained
coordinates `K`, recompute the marginal Bayes direction as
`inverse(Sigma_KK) @ (Sigma @ w)_K`; do not slice `w`. Rebuild a full-rank
canonical mean-plus-Helmert basis on retained members using only the original
graph-training covariance restricted to those coordinates. Induce the learned
support through this deterministic basis map and require quotient dimension,
matrix rank, and retained raw dimension to agree; held values never define or
repair this basis.
Every gate that applies to deletion must pass separately at deletion counts
one, two, and three; no averaging across counts is allowed.

Graph-training environments use all 800 registered prompt units. The named
400/400 adaptation/evaluation halves are used only when an environment serves
as validation or sealed test; no graph-training row is discarded.

For vectors `a,b`, direction recovery is
`cos2_Sigma(a,b)=(a' Sigma b)^2/[(a' Sigma a)(b' Sigma b)]`.
Report this for final `w(alpha)` versus planted Bayes `wY` and for correction
`u` versus the planted IU-orthogonal residual of `wY`; a zero-norm planted
residual is registered as not applicable, never as one. Support F1 uses
undirected off-diagonal edges, with the same `0.01` partial-correlation
threshold as fitting, and is macro-averaged first over held environments and
then repetitions.

The favorable sparse world must satisfy all:

- final-direction median `cos2_Sigma >=0.80`, fifth percentile `>=0.50`;
- residual-correction median `cos2_Sigma >=0.60`, fifth percentile `>=0.25`;
- mean support F1 at least `0.60`;
- mean oracle-minus-IU AUROC gap at least `0.01`; otherwise close as
  `CLOSE_SYNTHETIC_NO_HEADROOM`;
- ratio of mean candidate-minus-IU to mean oracle-minus-IU AUROC at least
  `0.50`, and paired repetition-bootstrap 95% lower bound for candidate minus
  IU greater than zero.

The nuisance-dominant stress is the anti-repackaging gate: in each repetition
compare final and correction `cos2_Sigma` to their target and nuisance
directions, average over held environments, and require target greater than
nuisance for both vectors in at least 90 repetitions. Candidate-minus-IU AUROC
must also have a nonnegative paired lower bound. Failure closes A5 immediately,
without fetching the multi-gigabyte real raw caches.

In independent and no-latent worlds, alpha zero must be selected in at least
90 repetitions and exact fallback error must be below `1e-10`. Each of worlds
3--7 and 9 separately must have mean candidate-minus-IU harm no worse than
-0.005 AUROC and repetition fifth percentile no worse than -0.02; pooling
misspecifications is prohibited.

For duplicate world 6, coordinates are standardized on adaptation rows. Let
`j=0`, let `a_j` be its final standardized-coordinate coefficient before
augmentation, and `a_j,a_dup` after paired augmentation. Require median
`(|a_j|+|a_dup|)/max(|a_j_before|,1e-12) <=1.10` and median Spearman rank
correlation between paired original and augmented evaluation scores at least
`0.999999` for exact duplicates. For rho-`0.999` near duplicates, gate the
actual independently selected deployed scores and require median score
Spearman at least `0.995`; additionally require selected alphas to differ by no
more than one grid step (`0.125`). A same-fixed-alpha comparison is reported
only as a mechanism diagnostic and cannot satisfy the gate. The coefficient statistic is computed on the learned correction
only, excluding the unchanged IU anchor, so sign cancellation in IU is not
misreported as duplicate evidence.

The estimator handles this stress with an automatic, label-free
mean-plus-contrast transform fitted only on graph-training environments.
Standardize within each training environment; connect a coordinate pair only
when its Pearson correlation is at least `0.998` in every graph-training
environment; and take connected components. Order components and their members
lexicographically by immutable frozen feature name. Each component becomes its
mean plus deterministic Helmert contrast coordinates. Retain every contrast,
including an empirically exact graph-training contrast: a near-zero contrast
gets unit scale rather than being dropped, so a held departure cannot break
the alpha-zero identity. Contrasts preserve the original IU discrepancy exactly
in both constrained likelihood and deployment, but are masked out of the learned
correction; only component means may receive correction mass. A learned mean
correction is expanded equally across members. `alpha=0` therefore remains an
exact score identity, and the density direction equals the deployed direction.
The transform, contrast scale, and threshold are refit inside every
null/control.

If target recovery passes only in the favorable Gaussian world, the verdict is
`CLOSE_SYNTHETIC_MISSPECIFICATION`. Any failure above closes the route and A6
opens. Development-seed success cannot override a sealed failure.

## 7. Real label-free premise, only after all synthetic gates pass

### 7.1 Exact nested algorithm and global content purge

For every outer held environment `e`:

1. remove from all other environments every row whose `content_group_id`
   occurs in `e`; this is the outer-training population. Purged environments
   with fewer than 80 remaining primary prompt units are omitted from that
   fold; if fewer than eight training environments or fewer than three dataset
   families remain, the fold is unusable. Every one of the 23 outer folds and
   every inner validation fold must be usable; otherwise close as
   `CLOSE_INADEQUATE_ITEM_DISJOINT_ENVIRONMENTS`;
2. inside that population, leave one environment `v` out in turn, additionally
   purging from every inner-training environment every content group occurring
   in `v`;
3. for each graph penalty, rebuild the initial mixtures, equal-environment
   residual covariance, first support, sparse EM, and one fixed-penalty support
   update using inner-training environments only;
4. split `v` content groups by
   `sha256("A5-inner"+content_group_id+outer_id+inner_id) mod 2`, fit its
   standardization, IU, local mixture, covariance and `u_v` on adaptation only,
   and evaluate every alpha's constrained density on evaluation only; swap the
   halves and repeat;
5. average per-row log likelihood within swap, then equally over both swaps and
   validation environments. Jointly select one penalty and one global alpha by
   the frozen paired one-standard-error rule in Section 4.2;
6. rebuild support once on all retained purged outer-training environments at
   that penalty, without reselecting it;
7. split held `e` by
   `sha256("A5-outer"+content_group_id+outer_id) mod 2`; use adaptation only for
   local standardization, IU, mixture, covariance and `u_e`, and evaluation
   only for likelihood/score. Swap halves and repeat.

Rows never cross a content-group boundary. A split with fewer than 20 effective
members in either component is unusable, not reassigned. A secondary
non-purged/transductive analysis may be reported but cannot affect any gate.
The primary statistic averages evaluation-row log score within swap, then
swap, then environment with equal mass. Dataset-family bootstrap resamples
families, then environments within sampled families, for 20,000 fixed draws
from seed `540000`.

### 7.2 Parametric nulls and stability

Two null families each use 200 draws: seeds `541000..541199` for the sparse
one-Gaussian null and `542000..542199` for the diagonal two-component null. In
each outer fold, fit only the global null support/penalty law on the purged
outer-training population. Fit permitted held-local null centre, covariance,
prior, and separation on real held adaptation only; sample a fresh held
adaptation and evaluation population from those local parameters without
reading real held-evaluation values. Refit the local candidate from simulated
adaptation and score only simulated evaluation. Because the primary has one
deterministic response per prompt, null rows are independent prompt units and
no unmodelled K-response cluster remains.

Rerun the **complete** nested support, penalty, alpha, random-control and
local-fit pipeline. One seed generates one coupled 23-outer-fold macro
replicate, using that seed for every simulated environment with stable
fold-specific subkeys. The latent statistic is candidate minus sparse
one-Gaussian held log score; the graph statistic is candidate minus
diagonal-mixture held log score. Each observed statistic must exceed its null's
95th percentile.

A third 200-draw null reassigns whole content groups among environment labels
within dataset family while preserving every environment's prompt count;
seeds are `543000..543199`. It reruns the same complete pipeline. The observed
candidate-minus-capacity-identical-`alpha=0` held log-score statistic must
exceed its 95th percentile. Item-pair shuffle is inapplicable because A5 has no
paired-view input.

Graph stability is pairwise Jaccard of the 23 outer-fold supports. Loading
stability is not compared across environments: within each held environment,
fold both swap-fitted raw affine weights onto all its rows and use the squared
Pearson correlation of their scores; report minimum and median across
environments, and require the unsquared correlation to be positive in every
environment so a complete rank reversal cannot pass. Record support edge
count/density. Reject a local fit unless its
total adaptation count is at least `2*(p+edges)` for the shared precision and
each component's effective adaptation mass is at least
`max(20,2*(p+1))` for its mean/prior. This replaces a single impossible
per-component count of all shared and local parameters.

The primary must pass every gate:

- mean held log-likelihood gain at least `0.01` nats per observed feature and
  family-bootstrap lower bound above zero versus the capacity-identical sparse
  `alpha=0`, diagonal mixture, and sparse one-Gaussian controls;
- lower bound above zero versus the nested-training-selected strongest of 32
  degree/edge-matched random graph supports;
- observed latent and graph gains exceed the 95th percentiles of train-fitted
  one-Gaussian-sparse and diagonal-mixture parametric nulls;
- observed correction likelihood gain exceeds the 95th percentile of the
  environment-label reassignment null;
- LOEO graph-support median Jaccard at least `0.70`;
- within-environment swap score squared correlation median at least `0.80` and
  minimum at least `0.50` after IU orientation;
- converged local fits at least 95%, and both component effective masses at
  least 20 adaptation items in every scored split;
- selected alpha greater than zero in at least 16 of the 23 outer folds;
- feature permutation, deterministic rerun, missingness, affine
  reconstruction, duplicate, and zero-evidence fallback gates from the common
  contract all pass.

### 7.3 Length confounding

Within each training environment, normalize `s_A5-s_IU`, `s_A5`, and `s_IU`
using adaptation-only mean/scale. The outer-training score source is only the
group-held-out inner-validation predictions generated under the selected
outer-fold penalty/alpha; no row may be both a local-fit input and length-model
target. Fit one equal-environment ridge-1 cubic spline regression for each of
correction, A5, and IU. Every regression receives all three inputs: raw token count,
`log1p(count)`, and a saturation indicator for equality to the source
manifest's generation cap; knots for both numeric terms are pooled
outer-training percentiles `(20,40,60,80)`. Apply unchanged to normalized held
evaluation scores.

Compute R-squared and absolute Spearman inside each held environment/swap and
macro-average swaps, then environments. Use the same family bootstrap for
intervals. For correction predictability, upper 95% bounds must be below `0.10`
R-squared and `0.35` absolute Spearman. Separately require
`R2(s_A5)-R2(s_IU) <=0.02` and
`absSpearman(s_A5)-absSpearman(s_IU) <=0.05` at their upper paired bounds.
No score is pooled across environments before these calculations.

Where at least 20 exact-count strata contain at least two evaluation content
groups, report nonparametric exact-length `R2 = 1 - Var(score -
mean(score|exact_count))/Var(score)` for correction, candidate, and IU.
Right-censored cap rows form their own registered stratum and are never treated
as their unknown true length. Length failure closes the route; it is not
repaired by adding length to the fit.

## 8. One-way retrospective label veto

Only after exactly one primary score bundle, configuration, source/tensor/code
hashes, and all structural verdicts are frozen may the already opened 24-cell
development labels be read. Labels may PASS or VETO this sole primary; they may
not choose an alpha, graph, arm, seed, transform, feature, or control.

Before labels load, freeze the final penalty and alpha as the modes of the 23
outer-fold selections; ties choose alpha zero then the larger penalty.
Refit the global support at that frozen penalty on all 23 structural
environments.
Apply these objects to each of the 24 registered development cells, including
the additional `spilled_triviaqa_llama8b`. Fit candidate-local standardization,
IU anchor, sparse mixture/covariance, and diagonal control only on that cell's
deterministic one-response-per-content-group subset, then apply all frozen
local affine scores to every complete-core row. This transductive subset fit is
the registered batch-calibration deployment setting; no selection is repeated.
Freeze candidate, IU, diagonal, and existing Family-NRM scores keyed by
`(cell,item_group_id,candidate_ordinal)`, plus score-source-code hashes.

The IU used inside candidate orientation/correction is named
`IU_subset_anchor`. The promotion comparator is the separately evaluated
deployed IU-PCR, fitted on all unlabelled complete-core rows exactly as in the
frozen 24-cell benchmark and named `IU_deployed_all_rows`. Sparse-versus-
diagonal isolates the A5 dependency contribution because both use the same
deterministic fitting subset; candidate-versus-IU uses
`IU_deployed_all_rows` because that is the method A5 must beat.

A physically separate label loader then joins labels by that exact key and
fails unless the ordered key hash agrees. Every pairwise comparison uses the
intersection of finite keyed rows for the two methods; it reports the mask hash
and reruns IU on that same mask. Bootstrap units are content groups within
cell, with equal cell mass and the registered dataset-family macro. Score and
label artifacts have separate hashes.

Promotion requires all:

- equal-cell macro AUROC improvement over deployed IU-PCR at least 0.005 with
  grouped-bootstrap lower bound above zero;
- candidate minus frozen Family-NRM lower bound at least -0.002;
- no registered dataset-family macro more than 0.010 below IU-PCR;
- sparse candidate minus diagonal latent control grouped-bootstrap lower bound
  above zero. This dependency gate is mandatory for A5.

Failure produces a closure artifact and proceeds to A6. Passing freezes one S1
finalist, but does not open PopQA until the A6/S2 comparison policy is fixed.

## 9. Interpretation and immutable decisions

- A5 is S1 conditional on the inherited mixed-v2/IU semantic anchor, not fully
  label-naive unsupervised learning.
- Likelihood and stability identify density structure, not hallucination.
- No A4 component, trace length, manual family, feature DAG, or A1 taxonomy may
  select or orient the primary.
- A nonlinear, heteroscedastic, NLL, or posterior-probability deployment score
  is outside scope even if its density fit is better.
- A failed premise is recorded, not swept with a broader grid.
- The untouched confirmation boundary remains
  `popqa-gemma3-4b-it-confirmation-v1`.

## 10. Literature relation

The sparse-dependency motivation follows the weak-supervision structure work
of Varma et al., *Learning Dependency Structures for Weak Supervision* (ICML
2019), and the sparse-plus-latent precision literature of Chandrasekaran,
Parrilo, and Willsky. Those results motivate conditional precision modelling;
they do not supply semantic target identification for these continuous
telemetry features. The affine equal-covariance restriction and IU-orthogonal
trust path are specific to the present deployment contract.
