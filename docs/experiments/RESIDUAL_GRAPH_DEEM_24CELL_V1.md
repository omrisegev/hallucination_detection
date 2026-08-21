# Residual-Graph DEEM v1: frozen 24-cell development protocol

## Document status

This document specifies a new experiment; it does not report its outcome.
Implementation, execution, and interpretation must follow this protocol unless a
dated amendment is committed before the affected outcomes are inspected.

### Frozen amendment 2026-08-21: per-cell historical inventory

This amendment was made before any Phase-1 outcome was opened and controls every
later section.  The original exact-30 contract was infeasible: the historical
pipeline deliberately removed missing, constant, or saturated coordinates on a
cell-by-cell basis.  The canonical population has 24 cells, 48,607 rows, seven
schemas with `p=19..30`, and 38 absent feature-by-cell entries out of 720.  Only
6/24 cells contain all 30 registered features.  No earlier result is invalidated;
those results used their stated variable-inventory contract.

The experiment therefore preserves each cell's exact historical inventory and
population.  It never fills a missing coordinate, changes an answer-span, or
substitutes a different population.  The draft's fixed-width identifiers are
retired in favor of `inventory`, meaning all features present in that cell and
equal weight over present families.  The machine-readable source of truth is
`configs/residual_graph_deem_24cell_v1_registry.json`.

Missingness is structural, not imputation debt.  In particular,
`seiclr_triviaqa_opt30b` has a canonical short answer-span and only 19 faithful
features; eleven dynamics quantities are undefined and remain absent.  In the
three cells without `trace_length`, the `structural` family is absent from the
model while raw trace length remains an external nuisance.  The sensitivity is
named `stable_inventory_minus4` and removes the four quarantined coordinates only
when present.  It cannot rescue a failed primary result.

The experiment starts on the original 24 completed-answer Global cells. It does
not touch Localization or Early Detection during Phase 1. Those applications are
separate later phases with separate targets and data contracts.

## One-paragraph summary

Replace IU-PCR with a binary DEEM-style latent-variable fusion model that consumes
the registered per-cell gray-box telemetry inventory and emits

\[
q_i=P(H_i=1\mid x_i),\qquad H_i=1\;\text{means hallucination},
\]

so a larger public score always means greater hallucination risk. Then test two
competing uses of the residual feature graph: either hallucination risk itself is
smooth on that graph, or the graph mainly describes a nuisance subspace that
should be represented separately from hallucination. DUFS may weight the graph
metric, but it may not remove inputs from DEEM or use correctness labels. The
first decision is made only on the canonical 24 Global cells; only a stable and
non-inferior DEEM core may advance to application-specific Localization or Early
Detection protocols, and only a graph-specific win may carry a manifold claim
into those protocols.

## Research questions

The experiment answers five ordered questions.

1. Can a continuous-input binary DEEM adaptation stably replace IU-PCR on the
   original 24 cells?
2. Does an explicitly interpretable, family-additive DEEM model retain the useful
   signal of every input present in the historical cell inventory?
3. After removing the shared DEEM risk direction and length, is the remaining
   feature geometry useful for estimating hallucination risk?
4. Is the residual graph better used to smooth the hallucination logit, or to
   model and remove nuisance structure?
5. Does DUFS improve the residual metric beyond a uniform residual graph, without
   merely rediscovering length or another nuisance?

Question 1 is logically prior to Questions 3--5. A failed or unstable DEEM core
cannot be rescued by a favorable graph variant.

## Claim boundary

The 24 cells have already been used repeatedly during method development.
Consequently, Phase 1 is retrospective development evidence, not external
validation. Even a full pass permits only this conclusion:

> `INTERNAL_RESIDUAL_GRAPH_DEEM_CANDIDATE_AWAITING_NEW_GLOBAL_VALIDATION`

It does not justify the words *universal hallucination manifold*, *confirmed*, or
*externally validated*. A new dataset/model cell that did not participate in any
feature, orientation, solver, residual, graph, or threshold decision is required
for confirmation.

The DEEM paper and package model categorical or soft learner outputs. Feeding the
continuous telemetry measurements directly into a new visible energy model is
our adaptation, not a theorem supplied by DEEM. Paper-aligned hard and soft-rank
arms remain as controls.

- Paper: <https://openreview.net/attachment?id=YF1ObZwFnk&name=pdf>
- Reference implementation: <https://github.com/shaham-lab/deem>
- Existing adapter: `spectral_utils/deem_adapter.py`

## Fixed scientific constraints

### Allowed information during fitting

The following are allowed:

- the gray-box telemetry features in the frozen inventory of the current cell;
- their frozen semantic provenance families;
- unlabeled feature marginals within the current cell;
- raw response length as an explicitly declared nuisance, whether or not the
  historical cell inventory retained `trace_length` as a model coordinate;
- dataset and model identifiers only for grouping, blocked evaluation, and
  nuisance diagnostics;
- deterministic random seeds fixed in the run definition.

The following are forbidden during fitting, graph construction, DUFS gating,
orientation, hyperparameter selection, checkpoint acceptance, or candidate
selection:

- correctness or hallucination labels;
- per-feature or per-arm AUROC/AUPRC;
- external judges, extra generations, extra model calls, retrieval evidence, or
  hidden states that are not already in the registered gray-box contract;
- post-hoc sign flips based on labels;
- choosing a graph topology, `k`, Laplacian coefficient, support, or seed after
  observing target metrics;
- using dataset/model identity as a classifier input.

Labels enter only the separately isolated evaluation and conditional-null stages.

### Public score orientation

The repository contract in `spectral_utils/feature_contract.py` defines

\[
z^{\mathrm{conf}}_{ij}=s_j\,z^{\mathrm{raw}}_{ij},
\]

where larger `z_conf` means *more likely correct*. This experiment converts once
and only once to risk orientation:

\[
z^{\mathrm{risk}}_{ij}=-z^{\mathrm{conf}}_{ij}.
\]

The target used only by evaluation is

\[
y_i^{H}=1-y_i^{\mathrm{correct}}.
\]

Every emitted detector score, including controls, is oriented so that larger
means more likely hallucination. No reporting path may negate the score again.

DEEM's latent classes are permutation-identifiable. Their semantic alignment is
fixed without natural labels: class 1 is whichever latent class has the larger
mean value of the frozen, symmetric risk-consensus anchor. This is an orientation
prior; it does not prove that the latent variable is hallucination.

## What the existing evidence does and does not establish

The experiment is motivated by four earlier observations, none of which is
allowed to decide its outcome.

1. The current 24-cell deep-hard DEEM result is approximately tied with full
   IU-PCR (about 0.7544 versus 0.7542 macro AUROC). It is not an established
   replacement win. Hard DEEM also showed material seed instability, including a
   worst cell seed range of roughly 38.4 percentage points.
2. The original soft-rank DEEM configuration often collapsed. A label-free pilot
   repair with learning rate `1e-4` and 100 epochs completed all 15 registered
   fits on three pilot cells and five seeds, but it has not yet earned a full
   24-cell performance claim.
3. Family-NRM produced a bounded positive result on PRMBench: +0.460 pp response
   AUROC, with a paired interval of approximately [+0.068, +0.841] pp. The gain
   depended on the manually registered six-family provenance prior. Atomic NRM
   did not reproduce the gain.
4. The direct conditional graph audit found recurrent length-conditional
   geometry, but its valid decision was invalidated by a ProcessBench length-only
   control failure. Union-kNN, adaptive-k, and diffusion produced similar
   conditional geometry; adaptive/diffusion did not improve graph health and
   radius was weak. Topology search therefore is not the primary lever here.

The resulting working hypothesis is deliberately narrower: useful information
may exist in residual, family-structured dependencies, but geometry alone cannot
identify whether a direction is hallucination or nuisance.

## Phase 1 population: the original 24 Global cells

### Canonical source

- Registry: `configs/residual_graph_deem_24cell_v1_registry.json`
- Fit input: one rebuilt, label-free per-cell bundle from the registered raw
  telemetry source; the historical `cells.npz` is reference-only.
- Roster: `scripts/inscope_cells.py::INSCOPE`
- Existing benchmark contract:
  `docs/experiments/FROZEN_24_CELL_BENCHMARK.md`
- Unit of analysis: one completed response in one dataset/model cell.
- Target: completed-answer hallucination, `1 - correctness`.

The runner must fail unless the exact roster contains 9 QA and 15 math cells and
unless `inside_coqa_llama7b` remains excluded for its documented generation
defect.

### Frozen roster

QA:

1. `epr_triviaqa_mistral24b`
2. `losnet_hotpotqa_mistral7b`
3. `sciq_llama8b`
4. `se_nq_open_llama8b`
5. `se_squad_v2_llama8b`
6. `seiclr_triviaqa_opt30b`
7. `semenergy_triviaqa_qwen3_8b`
8. `spilled_triviaqa_llama8b`
9. `truthfulqa_llama8b`

Math:

1. `ars_gsm8k_r1distill8b`
2. `internalstates_gsm8k_qwen25_7b`
3. `lapeigvals_gsm8k_llama3b`
4. `lapeigvals_gsm8k_llama8b`
5. `lapeigvals_gsm8k_mistral24b`
6. `lapeigvals_gsm8k_nemo`
7. `lapeigvals_gsm8k_phi35`
8. `noise_gsm8k_mistral7b`
9. `noise_gsm8k_phi3mini`
10. `math500_dsmath7b`
11. `math500_qwenmath7b`
12. `math500_r1distill8b`
13. `math500_r1distill8b_mn4096`
14. `trace_gsm8k_llama8b_k10`
15. `trace_math500_qwenmath15b_k10`

The eight dataset-family blocks are `triviaqa`, `hotpotqa`, `sciq`, `nq_open`,
`squad_v2`, `truthfulqa`, `gsm8k`, and `math500`. Tie seeds and optimizer seeds
are repeated measurements, not independent samples.

## The per-cell inventory input contract

The primary DEEM input contains every feature in the exact historical inventory
and order registered for the current cell.  The possible provenance groups are
frozen by `spectral_utils/specrage_views.py`; only groups present in a cell enter
its model:

| family | members | role |
|---|---:|---|
| `entropy_level` | 1 | mean entropy level (`epr`) |
| `entropy_dynamics` | 14 | spectral, temporal, change-point, and complexity summaries of the entropy trajectory |
| `sampled_token_energy` | 4 | realized sampled-token surprisal / spilled-energy trajectory |
| `partition_energy` | 4 | full-vocabulary log-partition trajectory |
| `topk_distribution` | 6 | top-1, margin, entropy, varentropy, Renyi, and tail-mass summaries |
| `structural` | 1 | `trace_length` |

The four features currently quarantined by `fixed_stable_v1` are not silently
removed from the primary inventory test. A separately named
`stable_inventory_minus4` sensitivity removes each of them only if present.  It
is predeclared, cannot be selected after outcomes, and cannot rescue a failed
primary result.

Before any fit, an inventory stage verifies the exact per-cell names, order,
confidence signs, row count, raw-source byte hash, manifest hash, admission hash,
and finite values against the registry.  Missing-feature completion, zeros,
imputation, fingerprint joins, and positional fallback are forbidden.  A mismatch
stops the run before labels.  Constants that are genuinely present remain present;
donor standardization records `sigma=1` for such a coordinate.

### Frozen observed inventory audit

The population contains 48,607 rows and seven distinct schemas (`p=19..30`).
There are 38 absent feature-by-cell entries: 23 in `entropy_dynamics`, 12 in
`sampled_token_energy`, and three in `structural`.  `min_spilled` is absent from
12 cells; each STFT coordinate from six; `trace_length` and `dominant_freq` from
three; eight additional dynamics coordinates from one cell each.  The complete
cell lists and every ordered inventory are in the machine registry.

| observed schema summary | cells |
|---|---|
| complete 30 | `ars_gsm8k_r1distill8b`, `lapeigvals_gsm8k_llama3b`, `lapeigvals_gsm8k_mistral24b`, `math500_dsmath7b`, `sciq_llama8b`, `trace_math500_qwenmath15b_k10` |
| only `min_spilled` absent | `internalstates_gsm8k_qwen25_7b`, `lapeigvals_gsm8k_llama8b`, `lapeigvals_gsm8k_nemo`, `lapeigvals_gsm8k_phi35`, `losnet_hotpotqa_mistral7b`, `noise_gsm8k_mistral7b`, `noise_gsm8k_phi3mini`, `trace_gsm8k_llama8b_k10`, `truthfulqa_llama8b` |
| both STFT coordinates absent | `epr_triviaqa_mistral24b`, `spilled_triviaqa_llama8b` |
| STFT and `min_spilled` absent | `se_nq_open_llama8b`, `se_squad_v2_llama8b`, `semenergy_triviaqa_qwen3_8b` |
| `dominant_freq`, `trace_length` absent | `math500_qwenmath7b`, `math500_r1distill8b` |
| `trace_length` absent | `math500_r1distill8b_mn4096` |
| 19-feature short-span inventory | `seiclr_triviaqa_opt30b` |

### Raw reconstruction and physical label firewall

`results/dependency_fusion_raw/cells.npz` is a historical inventory reference,
not a fit input.  Bundles are rebuilt from the hash-frozen raw telemetry in
canonical source order.  The 23 A5 source specifications are copied into this
experiment registry, and `spilled_triviaqa_llama8b` is newly registered with raw
SHA-256 `cf01350f5bc141908e3f0563c1bc3037148fbad3a30c4eb05c63cd3c13a51e65`,
size 7,808,360 bytes, manifest SHA-256
`ca767e773b8ee5accb54b9f8c1a8ecf441bb8144e956323a1a4fe6ed0091f36f`,
`complete_h16` telemetry-only admission, and 256 admitted rows.

Canonical identities are
`row_id=<cell>::<raw_problem_key>::candidate<ordinal>` and
`group_id=<cell>::<raw_problem_key>`, where the ordinal is assigned before
admission.  Each per-cell `allow_pickle=False` fit bundle contains only `X_raw`,
feature names, confidence signs, row/group IDs, raw trace length, dataset family,
task type, and source/admission hashes.  It contains no target-like field.  Only
after Stage A scores are complete and immutable may a separate evaluator process
create or open the keyed `row_id -> y_H` sidecar.  The evaluator requires an
order-independent one-to-one join.

For every donor fit, standardize with donor parameters only and then orient once:

\[
z=(X_{raw}-\mu_{donor})/\sigma_{donor},\qquad
x_{risk}=-s_{confidence}\odot z.
\]

For an exactly constant donor coordinate, set and record `sigma_donor=1`; this
does not add or fabricate an observation.

## Model 0: paper-aligned DEEM controls

Two existing conversions remain controls rather than the proposed method.
They use the packaged `deem==0.2.0` code from upstream commit
`7740f606b8fb5506065a8c710da5a00c1425f9b7` in five isolated processes with
explicit Python, NumPy, and Torch seeding/determinism.  The common hybrid config
is one Sparsemax preprocessing layer with identity initialization, hidden
dimension one, five sampler steps, batch 1024, momentum 0.9, weighted majority
vote, `mv_rand`, and 100 epochs.  These are named “0.2.0 adapter controls”, never
“paper-exact”: the package DLP transition is not the MH transition printed in the
paper.  Package majority-vote alignment is diagnostic only; final class alignment
uses the external risk consensus defined below.

### Hard control

Each risk-oriented continuous feature is converted to a binary learner vote by
a deterministic empirical median split. This most closely resembles binary
learner predictions accepted by standard DEEM.  Its learning rate is `1e-3`.

### Soft-rank control

Each feature is converted to a two-class pseudo-probability by its empirical
rank:

\[
p_{ij}=\frac{\operatorname{rank}(z_{ij})-0.5}{n},\qquad
\pi_{ij}=[1-p_{ij},p_{ij}].
\]

These values preserve order but are not claimed to be calibrated probabilities.
Ties use deterministic average ranks, probabilities are clipped at `1e-3`, and
the repaired configuration is frozen at learning rate `1e-4`, 100 epochs, and
seeds `(0,1,2,3,4)`.

## Model 1: continuous-visible, family-additive DEEM

### Binary latent target

The proposed model explicitly defines the two latent states as

\[
H\in\{0,1\},\qquad H=1\text{ is hallucination}.
\]

For Boltzmann energy `E`, the exact hallucination logit is

\[
\ell_\theta(x)
=\log\frac{P_\theta(H=1\mid x)}{P_\theta(H=0\mid x)}
=E_\theta(x,0)-E_\theta(x,1),
\]

and the public score is `q=sigmoid(ell)` after the frozen unsupervised class
alignment.

Calling state 1 “hallucination” is possible, but the name alone does not identify
the latent class. The risk anchor supplies the allowed semantic orientation;
natural correctness labels do not.

### Exact contribution decomposition

For every family present in the cell, use a width-eight family network with an
atomic output coordinate per present feature:

\[
u_g=\tanh(W_gx_g+d_g),\qquad
c_g=w_g\odot x_g+\frac{2}{|g|}\tanh(V_gu_g+e_g),
\]

and define the complete continuous-visible energy by

\[
\ell=b+\sum_{g\in G_i}\mathbf 1^\top c_g,\qquad
E(x,h)=\tfrac12\lVert x-a\rVert^2-h\ell,\qquad q=\sigma(\ell).
\]

Thus every atomic contribution is explicit, family sums and the logit reconstruct
exactly, and the quadratic base measure makes the continuous density normalizable.
No interaction extension is part of v1.

Initialization is `a=b=0`, `w_gj=2/(|G_i||g|)`, with every other weight sampled
from `N(0,0.005^2)`.  The target-free risk anchor is equal-family/equal-feature:

\[
r_i=|G_i|^{-1}\sum_{g\in G_i}|g|^{-1}\sum_j x_{ij}.
\]

Compare its `q`-weighted and `(1-q)`-weighted means.  If the former is smaller,
swap the latent classes and transform `q`, `ell`, `b`, and every atomic
contribution together.  An absolute anchor difference at most `1e-6` is a fit
failure.

The full-batch training loss is the free-energy contrast

\[
\mathcal L_{DEEM}=\operatorname{mean}F(x_{pos})-
\operatorname{mean}F(x_{neg}),\qquad
F(x)=\tfrac12\lVert x-a\rVert^2-\operatorname{softplus}(\ell).
\]

Persistent negatives use a correctly MH-adjusted MALA kernel: proposal scale
`delta=.10`, five steps, the exact forward/reverse Gaussian proposal ratio, a
buffer initialized from all training rows, and 5% data refresh per epoch.  The
frozen optimizer is float64 CPU full-batch SGD, learning rate `1e-3`, momentum
zero, 100 epochs, no scheduler, regularizer, weight decay, or minibatching.

### Why the family constraint is useful

The constraint connects the three most informative prior results:

- DEEM supplies a binary latent fusion objective instead of IU-PCR;
- Family-NRM suggests that provenance-level residual structure can help;
- the failed Atomic-NRM experiment warns that unconstrained atomic covariance
  modes do not identify their semantic role.

It is a disclosed inductive bias, not a discovered universal partition.

## Cross-fitted residual representation

The residual graph must not be built from the same fitted posterior that it then
iteratively modifies. The graph is therefore derived from a frozen, cross-fitted
baseline DEEM.

For each cell and seed:

1. Split whole `group_id` groups, never rows, into five folds. Sort groups by
   group-median raw length, form deterministic deciles with stable-ID tie breaks,
   and assign round-robin. Natural labels are never read and sibling candidates
   never cross folds.
2. Fit Model 1 on four folds and emit held-fold `ell_0` and every atomic
   contribution coordinate on the fifth fold.
3. Repeat until every row has out-of-fold baseline values.
4. On donor folds only, standardize both predictors and fit for each coordinate
   `PolynomialFeatures(degree=3, include_bias=False)` including cross terms,
   followed by `Ridge(alpha=1, fit_intercept=True)`:

   \[
   \widehat m_g(\ell_0,\log(1+\mathrm{length}))
   =\widehat E[c_g\mid\ell_0,\log(1+\mathrm{length})].
   \]

5. Emit held-fold residual contributions

   \[
   r_{ig}=c_{ig}-\widehat m_g(\ell_{0,i},\log(1+\mathrm{length}_i)).
   \]

6. Standardize residual coordinates using donor-fold location and scale only;
   exact donor constants use recorded scale one.
7. Concatenate held-fold residuals in original row order and freeze them before
   any graph-regularized DEEM fit begins.

Dataset and model effects are constant inside a cell. They may enter pooled
diagnostic residual models or blocked nulls, but never the deployed detector
input. The primary residualizer is cubic ridge on standardized `ell_0` and
`log1p(length)` with its degree and penalty frozen in the run definition.

The runner records row IDs, donor/held indices, transform parameters, and hashes.
Any overlap between a row and its residual-model training set invalidates that
fit.

Because five independently fit coordinate systems are concatenated, graph
eligibility additionally requires two fold-artifact diagnostics: held-fold
predictability and same-fold edge enrichment, each against 999 group-level fold
permutations. `p<.05` in either diagnostic invalidates that cell's graph.

## Graph construction

### Frozen topology

The primary graph is the corrected, self-safe, local-scale union-kNN graph from
the reviewed graph audit. It must remove self by explicit row index, handle exact
duplicate feature rows, remain symmetric and nonnegative, and retain positive
affinities for mixed duplicate/unique neighborhoods.

It is constructed and stored only as SciPy CSR/COO sparse data with stable
`row_id` tie keys and a symmetric union.  Dense `N x N` materialization is
forbidden.  Graph losses sum each unique undirected COO edge once.

The legacy comparator is `k=7`. Health-oriented fixed sensitivities are
`k in {5,10,15}`. Adaptive-k, radius, and diffusion are excluded from candidate
selection because the previous topology audit did not show a clear benefit.
They require a new amendment if revisited.

### Uniform residual metric

The first residual graph uses equal family mass and equal feature mass inside a
family:

\[
d_R(i,j)^2=\sum_{g\in G_i}\frac{1}{|G_i|}\frac{1}{|g|}
\lVert r_{ig}-r_{jg}\rVert_2^2.
\]

Family balancing prevents the 14 entropy-dynamics coordinates from dominating
the one-dimensional entropy-level and structural groups merely by count.  The
same equal-present-family metric is used for raw graphs and all distance
comparisons.

### DUFS-weighted residual metric

DUFS is used only to learn nonnegative metric gates:

\[
d_{R,G}(i,j)^2=\lVert G\odot(r_i-r_j)\rVert_2^2.
\]

It does not delete features from the DEEM energy model. Every coordinate in the
cell inventory continues to enter Model 1.

To reduce self-confirming geometry, gates are learned cross-view: the gate for
family `g` is estimated on donors against a `k=7` graph built only from the other
present families, never from `g` itself.  Optimize the parameter-free external
two-step-affinity form of Eq. 7 with stochastic gates `sigma=.5`, `mu_0=.5`,
Adam `lr=.02`, 120 epochs, all donor rows, and seeds `(0,1,2,3,4)`.  Average
across folds and seeds, RMS-normalize inside each family, then multiply each
coordinate by `sqrt(1/(|G_i||g|))`.  A gate is not permitted to depend on
correctness, hallucination, or per-feature AUROC.

Report effective feature count, exact family mass, and cross-seed cosine
stability.  All gates closed or median cosine below 0.80 is a fit failure.

### Laplacian

For adjacency `W`, use the symmetric normalized Laplacian

\[
L=I-D^{-1/2}WD^{-1/2}.
\]

Isolated vertices use the reviewed fail-closed convention. The run definition
stores adjacency, degree, component, isolated-node, affinity, and Laplacian
spectral diagnostics for every cell and arm.

## Two competing graph hypotheses

The experiment does not assume in advance that a smooth graph direction is the
hallucination direction. It compares two distinct mechanisms.

### Hypothesis A: target-smooth residual graph

The residual graph represents local hallucination similarity. Add a normalized,
centered Rayleigh penalty to the DEEM objective:

\[
R_H(\ell,L)=
\frac{\widetilde\ell^\top L\widetilde\ell}
{\widetilde\ell^\top\widetilde\ell+\epsilon},
\qquad
\widetilde\ell=\ell-\overline\ell.
\]

The denominator and a posterior-variance gate prevent the optimizer from winning
by producing a constant score. Lower `R_H` means that neighboring residual
profiles receive similar hallucination logits.

### Hypothesis B: nuisance-absorbing residual graph

The graph represents shared nuisance rather than hallucination. Introduce a
small nuisance representation `U_phi(x)` and optimize

\[
\mathcal L
=\mathcal L_{\mathrm{DEEM}}
+\lambda_U\frac{\operatorname{tr}(U^\top L U)}
{\operatorname{tr}(U^\top U)+\epsilon}
+\gamma\lVert U^\top\widetilde\ell\rVert_F^2,
\]

subject to centered, whitened `U` with fixed dimension `d_U=3`. The first graph
term makes `U` capture smooth residual structure; the second discourages the
hallucination logit from reusing it. A deterministic spectral nuisance basis
using the first three nonconstant Laplacian eigenvectors is included as a
diagnostic comparator.

The G4 encoder is separate from the family contribution networks, uses width
eight and output dimension three, and is fully centered and whitened with ridge
`1e-6`.  Both smoothness and logit orthogonality are scale-normalized, with
`gamma=1`.

This branch is the direct residual-research hypothesis: the Laplacian may be most
valuable as a model of what should *not* be called hallucination. If it wins, the
claim is nuisance separation, not discovery of a hallucination manifold.

### Present-family Laplacian

A five/six-node present-family graph is estimated from absolute Spearman
dependencies among cross-fitted family residual sums
and used to regularize corresponding first-layer embeddings `B`:

\[
R_F(B)=\operatorname{tr}(B^\top L_F B).
\]

G5 regularizes the mean rows of each present family's `V_g` output matrix and is
a secondary arm. An atomic inventory-feature residual-covariance Laplacian is not
primary because the residual-identifiability study found only 1 of 8 families
stable enough for the covariance-decomposition route.

### No circular updates

The following loop is forbidden:

`posterior -> graph -> smoothed posterior -> rebuilt graph -> ...`

The allowed order is:

`cross-fitted baseline -> frozen contributions -> frozen residual transform -> frozen graph -> graph-DEEM`.

The graph is never rebuilt after graph-DEEM training starts.

## Frozen Phase 1 arms

| ID | arm | purpose | decision role |
|---|---|---|---|
| B0 | `iu_pcr_inventory` | direct incumbent under matched input contract | comparator |
| B1 | `deem_inventory_hard_adapter020` | packaged hard conversion | diagnostic baseline |
| B2 | `deem_inventory_soft_rank_adapter020_repaired` | repaired soft-rank adapter | baseline candidate |
| B3 | `deem_inventory_continuous_additive` | proposed continuous-visible DEEM, no graph | core primary |
| G0 | `deem_inventory_raw_graph_uniform_target` | raw-feature graph, target-smooth | circularity/nuisance control |
| G1 | `deem_inventory_raw_graph_dufs_target` | raw DUFS graph, target-smooth | residualization ablation |
| G2 | `deem_inventory_residual_graph_uniform_target` | residual graph, equal metric | graph primary A |
| G3 | `deem_inventory_residual_graph_dufs_target` | residual graph, DUFS metric | graph primary A |
| G4 | `deem_inventory_residual_graph_dufs_nuisance` | residual graph, explicit nuisance latent | graph primary B |
| G5 | `deem_inventory_present_family_laplacian` | present-family feature Laplacian | secondary |

Every graph arm has a `lambda=0` route that directly aliases B3 and does not
construct a graph or nuisance encoder.  Identity therefore includes objective,
posterior, contributions, and serialized score rather than relying on numerical
coincidence.  `stable_inventory_minus4` is a named sensitivity, not an extra
candidate-search dimension.

B0 is rerun on exactly the same donor-standardized risk inventory as B3.  It may
not read the historical `F` matrix or its label/data-derived `rho_polarity`.

### External method comparators

The report includes the already frozen score contracts for:

- deployed U-PCR;
- ordinary/full IU-PCR;
- Family-NRM;
- current DUFS-LIU;
- equal-weight risk consensus;
- the strongest matched one-dimensional linear direction as an evaluation-only
  diagnostic;
- a one-dimensional graph built on that linear score.

The linear controls determine whether a result needs nonlinear local geometry at
all. If the proposed graph does not beat the matched linear-score graph, the
permitted interpretation is a shared direction, not a manifold.

The competing one-dimensional graph is built target-free from the freshly rerun
B0 score with the same `k=7`, self-safe topology, and health rules.  It is included
in the whole-search maximum statistic.

## Negative and nuisance controls

Every graph primary is compared with the following controls under identical
training and evaluation code:

1. `length_only_graph`: neighbors are built only from `log1p(trace_length)`.
2. `node_permuted_graph`: graph nodes are permuted while preserving graph degree
   and spectrum.
3. `random_gate_graph`: within-family permutation of learned gate magnitudes,
   preserving nonnegativity, effective feature count, and family mass.
4. `uniform_gate_graph`: equal family-balanced residual metric.
5. `raw_graph`: no contribution residualization.
6. `posterior_permuted_on_graph`: posterior rows are permuted before measuring
   smoothness.
7. `family_permuted_residuals`: residual coordinates are permuted within family
   under a target-blind seed.
8. exact `lambda=0`: no Laplacian effect.

The control runner additionally emits the explicit `uniform`, `raw`,
`posterior_permuted`, and `family_permuted` arms through the same serialization
path, rather than computing them only as post-hoc diagnostics.

A control failure invalidates the corresponding graph mechanism. A favorable
CRT result may not rescue a failed exact-length control, and vice versa.

## Hyperparameters and selection discipline

### Solver seeds

Use seeds `(0,1,2,3,4)` for all stochastic DEEM arms. The public cell score is
the arithmetic mean of the five aligned posterior probabilities only when all
five fits pass. Per-seed results remain visible. Seeds are never treated as five
statistical observations.

### Laplacian coefficients

Synthetic studies choose the coefficients before Phase 1 labels are evaluated.
The real-cell path is frozen to

`lambda in {0, 0.01, 0.03, 0.1, 0.3, 1.0}`.

Exactly one nonzero coefficient per graph hypothesis must be nominated by the
synthetic mechanism gate. The rest are sensitivity curves and cannot replace a
failed headline coefficient.

### DUFS settings

DUFS optimizer, sparsity weight, number of epochs, family normalization, and tie
breaking are fixed on synthetic data. Phase 1 outcomes cannot select them. The
report must show the effective number of features, family mass, and gate
stability for every seed.

### No outcome-based arm fishing

The primary comparisons are B3 versus B0, G3 versus B3, and G4 versus B3. G2
separates graph benefit from DUFS benefit. G0/G1 are controls. G5 is secondary.
Hypotheses A and B are distinct co-primary mechanisms and use multiplicity
correction; one cannot be relabeled as the intended primary after results.

## Phase 0: mandatory pre-outcome verification

No natural labels may be opened until all checks pass.

### Synthetic mechanism worlds

Generate the following ten deterministic fixtures with `n=1024`, 256 groups of
four rows, PCG64 base seed `20260821`, noise SD one, signal/nuisance loading two,
5% imbalance in the imbalanced world, and 20% duplicates in the duplicate world.
Loading matrices and the expected-winner matrix are serialized and hashed in the
Phase-0 freeze before any real-cell fit:

1. binary hallucination signal shared by independent noisy features;
2. pure length manifold with target depending steeply on length;
3. smooth nuisance manifold independent of target;
4. target signal plus an orthogonal smooth nuisance;
5. linearly separable target with no nonlinear manifold advantage;
6. nonlinear local target geometry not captured by a linear score;
7. exact duplicate and mixed duplicate/unique feature rows;
8. class imbalance and near-constant features;
9. latent-class permutation and risk-anchor reversal;
10. pure noise.

The target-smooth model should win only in worlds 1 and 6. The nuisance-absorbing
model should help in worlds 3 and 4 without inventing target signal in worlds 2,
9, or 10. The full selection procedure is also run under planted nulls.

Run every graph mechanism at `lambda in {0,.01,.03,.1,.3,1}`.  For each
mechanism nominate the smallest nonzero lambda within one standard error of its
best synthetic outcome, but only if it promotes no negative world or control. If
no lambda survives, stop before natural labels.  All other values remain fixed
sensitivity curves.  The same Phase-0 suite also runs small fixtures of all seven
real inventory schemas.

### Mechanical gates

All of the following are mandatory:

- exact registered inventory/name/order/sign/source/admission/row-count contract
  on all 24 cells, including the verified spilled source;
- `H=1` and larger public score mean hallucination in every synthetic fixture;
- class-permutation alignment does not change the emitted semantic score;
- family contributions reconstruct the logit with maximum absolute error
  `<=1e-8`;
- `lambda=0` graph arms match B3 score, posterior, and objective within `1e-10`
  or a stricter deterministic hash where possible;
- every held row is absent from its baseline and residualizer donor set;
- transformations applied to held rows use donor-only parameters;
- graph symmetry error `<=1e-10`, no unintended self edges, finite nonnegative
  weights, and positive affinities for selected structural edges;
- posterior score standard deviation `>=1e-3` in at least 90% of fits;
- finite objective history and at least 90% fit completion;
- median pairwise absolute Spearman across five solver seeds `>=0.90` per cell,
  with a separately reported minimum;
- median DUFS gate cosine across seeds `>=0.80` and family-mass variation
  reported;
- no target effect in pure-noise, length-only, node-permuted, or random-gate
  planted controls after the whole-procedure correction.

### Graph-health gates

For a graph arm to be eligible in a cell:

- largest connected component fraction `>=0.90`;
- isolated-node fraction `<=0.05`;
- at least two target classes exist, but this fact is checked only by evaluation;
- exact duplicate handling passes the self-safe regression tests.

At least 22 of the 24 cells must be graph-eligible for a graph-level claim.
Ineligible cells remain in tables and count as failures; they are not silently
dropped.

## Evaluation barrier

### Stage A: fit and freeze

The fit runner must not import or access label arrays. It writes one file per
cell/arm/seed containing score, posterior, contribution, residual, graph, gate,
health, and runtime diagnostics. Each file is hashed. Resume accepts a checkpoint
only when its source hash, code hash, schema, arm registry, and diagnostics match.
Writes are atomic.  Every artifact also records objective history, graph/gate
hashes, config and environment hashes, donor/fold manifests, and determinism
settings.  A failed fit retains its partial history and health evidence.  The
runner has no sidecar path argument and a static import scan forbids label modules.

### Stage B: evaluate

A separate evaluator verifies the complete score manifest before it reads labels.
It refuses debug, incomplete, schema-mismatched, or hash-mismatched runs. It never
overwrites the immutable score-freeze manifest, and it refuses a missing seed or
a non-bijective keyed join.

### Stage C: deterministic rebuild

A verifier rebuilds the scientific output twice:

1. resume from the frozen checkpoints;
2. rebuild into a fresh output directory.

Both summaries and decisions must match the original. Any mismatch changes the
decision to `REBUILD_VERIFICATION_FAILURE`.

The resume rebuild uses the original checkpoints.  The fresh rebuild writes to a
new output directory.  Compact summaries, decisions, and declared hashes must all
match; a large artifact need not traverse the local machine to establish this.

## Metrics and blocked statistics

### Primary detector metrics

- per-cell AUROC for hallucination (`y_H=1`);
- per-cell AUPRC for hallucination;
- equal-cell macro within each dataset family;
- equal-family macro across the eight dataset families;
- separate QA and math macros;
- worst-cell and worst-family paired change;
- wins, ties, and losses relative to B0 and B3.

There is no pooled row-level AUROC across cells.

### Uncertainty

Use paired, family-blocked bootstrap intervals. Resample dataset families, then
cells within each sampled family, keeping every method and seed ensemble paired.
Report 95% intervals, the bootstrap distribution, and leave-one-family-out
changes. The eight families, not 24 cells or 120 seed fits, define the strongest
independent grouping available internally.

### Geometry diagnostics

For each graph report:

- target and logit normalized-Laplacian smoothness;
- exact-length and cross-fitted CRT conditional effects where eligible;
- smoothness of length, dataset/model proxies, baseline confidence, and residual
  family contributions;
- overlap with the one-dimensional linear-score graph;
- graph component, degree, affinity, local-scale, and duplicate diagnostics;
- DUFS gate weights by feature and family;
- raw-versus-residual neighbor overlap;
- target-smooth versus nuisance-latent actuation.

Conditional tests are mechanism diagnostics, not training objectives.

### Whole-search conditional null

If any Phase 1 outcome influences candidate nomination, ordinary global label
permutation is forbidden. The entire selection procedure is rerun under each
conditional-null world:

- exact-length swaps where sufficient identical lengths provide mobility;
- a cross-fitted flexible propensity CRT for hallucination given length;
- family-blocked target nulls for dataset/model structure.

Every draw recomputes all outcome-dependent summaries and reselects the winner
under the maximum statistic over graph hypotheses, supports, the B0 one-dimensional
graph, and eligible headline arms. Stage-A models, scores, gates, and graphs remain
frozen and are not refit under target nulls. Use `B=199` for development; run
`B=999` only if a candidate passes every other promotion gate. Exact-length and
cross-fitted propensity CRT promotion are co-required, together with the
family/group-blocked target null. Tie seeds remain robustness dimensions and are
not multiplied into the sample count.

## Frozen Phase 1 decision gates

### Gate A: stable DEEM core

B3 passes only if:

1. at least 90% of the 120 cell-seed fits are healthy and every cell has all five
   healthy seeds for its ensemble score;
2. median within-cell seed Spearman is at least 0.90;
3. B3 is non-inferior to matched IU-PCR B0: equal-family AUROC difference has a
   95% lower bound above `-0.0025` (-0.25 pp);
4. neither QA nor math macro is worse than B0 by more than 0.5 pp;
5. no orientation, collapse, leakage, or rebuild control fails.

This gate tests whether DEEM is viable as an IU-PCR replacement. It does not test
the graph.

### Gate B: graph utility

A graph primary passes detector utility only if:

1. equal-family AUROC improves over B3 by at least `+0.005` (+0.5 pp);
2. the paired family-bootstrap 95% lower bound exceeds zero;
3. QA and math are each no worse than B3 by 0.5 pp;
4. at least 14 of 24 cells improve or tie within `0.0005`;
5. the worst cell loss is no worse than -2 pp;
6. at least 90% of cells pass graph health.

### Gate C: residual and DUFS specificity

For G3 to support the proposed mechanism:

1. G3 must beat its raw counterpart G1 by at least +0.25 pp equal-family AUROC;
2. G3 must beat uniform residual G2 by at least +0.25 pp, or DUFS receives no
   incremental-credit claim;
3. G3 must beat length-only, node-permuted, and random-gate controls;
4. its conditional geometry must pass both exact-length and CRT whole-procedure
   tests after multiplicity correction;
5. it must show at least +0.02 conditional-effect advantage over the matched
   one-dimensional linear-score graph, with a paired family lower bound above
   zero.

If the last condition fails, the result is classified as a transferable/shared
direction rather than nonlinear manifold evidence.

### Gate D: nuisance-separation specificity

G4 passes only if:

1. it passes Gate B versus B3;
2. it beats the target-smooth G3 by at least +0.25 pp or shows a preregistered
   reduction in length/model/dataset dependence with no detector loss larger
   than 0.25 pp;
3. nuisance representation variance and whitening gates pass;
4. hallucination logit--nuisance dependence is lower than in B3 and G3;
5. pure-length and pure-nuisance planted controls do not create false target
   promotion.

G3 and G4 are corrected as two co-primary mechanisms. Neither may rescue the
other after inspection.

## Decisions and what they permit

The evaluator emits exactly one primary decision and two advancement flags.

### Primary decisions

- `MECHANICAL_OR_CONTROL_FAILURE_INVALIDATES_DEEM_GRAPH_AUDIT`
- `DEEM_BASELINE_NOT_STABLE_STOP`
- `NO_DEEM_ADVANTAGE_ON_ORIGINAL_24`
- `STABLE_DEEM_REPLACEMENT_WITHOUT_GRAPH_GAIN`
- `RESIDUAL_GRAPH_MECHANISM_NOT_SUPPORTED`
- `RESIDUAL_GRAPH_IS_LINEAR_DIRECTION_NOT_MANIFOLD`
- `INTERNAL_RESIDUAL_GRAPH_DEEM_CANDIDATE_AWAITING_NEW_GLOBAL_VALIDATION`

### Advancement flags

- `ADVANCE_CORE=true` only when Gate A passes. This allows a separately frozen
  DEEM-replacement test in Localization and Early Detection, without a manifold
  claim.
- `ADVANCE_GRAPH=true` only when Gate A and one graph-specific mechanism gate
  pass. This allows the corresponding frozen graph mechanism to be included in
  later application protocols.

No Phase 1 outcome automatically validates either later task.

## Phase 2: Localization, only after Phase 1 closes

Localization asks *where the first error occurs*, not merely whether a completed
answer is wrong. It therefore requires a new, committed addendum before labels
are accessed.

### Required changes

- Unit: a reasoning step or transition, with source-question grouping.
- Target: first-error boundary or step-level error, as specified by each
  localization benchmark.
- Features: the already defined DSP/step feature contract for Localization; no
  completed-answer feature may leak information from later steps.
- DEEM: same frozen core architecture where compatible, but a separate
  localization head and posterior semantics.
- Graph: built between causal step representations at comparable progress; no
  future tokens or gold boundary information.
- Splits: entire source questions remain in one fold; dataset/model family
  holdouts are preferred over random steps.
- Metrics: step AUROC/AUPRC, first-error localization error, tolerance-window
  accuracy, and source-group bootstrap intervals.

The direct ProcessBench control failure from the earlier graph audit must be
treated as an explicit adversary. Length-only and step-index-only graphs must
fail closed per localization lane, not merely in a pooled rate across tasks.

### Localization decisions

- If only `ADVANCE_CORE` is true, test continuous DEEM against the frozen IU-PCR
  localization baseline, but do not include a manifold claim.
- If `ADVANCE_GRAPH` is also true, include exactly the winning frozen residual
  graph mechanism; do not reopen topology or DUFS selection.
- A Global success cannot compensate for a failed Localization control.

## Phase 3: Early Detection, only after a separate freeze

Early Detection asks whether hallucination can be warned about from a prefix.
It is an online causal problem and receives the strictest information boundary.

### Prefix-only contract

At budget `t`, every feature, residual, graph edge, DEEM posterior, and nuisance
representation may depend only on tokens `1:t`. The following are forbidden:

- final response length;
- suffix tokens or final-answer correctness proxies;
- features normalized by realized final length;
- graph neighbors computed from completed trajectories;
- forced-closure compute counted as if it were observed online.

`trace_length` is replaced by elapsed prefix length/budget, which is known at
decision time. Any DSP feature used here must have an audited causal prefix
implementation.

### Early metrics

- AUROC/AUPRC at fixed token and relative-progress budgets;
- time-to-warning for hallucinated responses;
- false-warning rate on correct responses;
- recall at fixed false-positive rates;
- area under the prefix-performance curve;
- realized compute and latency under the actual stopping rule;
- source-group and dataset/model-family blocked intervals.

The Phase 3 addendum must freeze budgets, stopping rules, censoring treatment,
and how incomplete responses are scored before labels are opened.

## Planned implementation layout

The implementation is split into physically separate stages:

- `spectral_utils/residual_graph_deem.py`
- `scripts/build_residual_graph_deem_data_v1.py` — source registry, target-free
  bundles, and separately invoked label-sidecar construction;
- `scripts/run_residual_graph_deem_24cell_v1.py`
- `scripts/evaluate_residual_graph_deem_24cell_v1.py`
- `scripts/report_residual_graph_deem_24cell_v1.py`
- `scripts/verify_residual_graph_deem_24cell_v1.py`
- `scripts/plot_residual_graph_deem_24cell_v1.py`
- `scripts/test_residual_graph_deem.py`
- `scripts/test_residual_graph_deem_protocol.py`
- `results/residual_graph_deem_24cell_v1/`

## Required outputs

The scientific result directory must contain:

1. `RUN_DEFINITION.json` with protocol, input, environment, source-code, graph,
   hyperparameter, feature-order, and sign hashes;
2. `FIT_COMPLETE.json` and immutable `SCORE_FREEZE_MANIFEST.json`;
3. `PER_FIT.csv`, `PER_CELL.csv`, and `FAMILY_SUMMARY.csv`;
4. `CONTRIBUTION_RECONSTRUCTION.csv`;
5. `RESIDUAL_DIAGNOSTICS.csv`;
6. `GRAPH_HEALTH.csv` and `GATE_STABILITY.csv`;
7. `CONDITIONAL_GEOMETRY.csv`;
8. `CONTROLS.json` and `WHOLE_SEARCH_NULL.json`;
9. `PAIRWISE_COMPARISONS.csv`;
10. `DECISION.json` with the primary decision and both advancement flags;
11. `REPORT.md` and static figures;
12. `REBUILD_VERIFICATION.json` proving resume and fresh-rebuild agreement;
13. `REVIEWER_GUIDE.md` with exact recomputation commands and claim boundaries.

The run definition must hash every output-generating script, including reporting,
plotting, and verification code.

## Required visualizations

The gate is not only a scalar table. The report must generate visual explanations
that make failure modes visible.

1. **Architecture diagram:** risk-oriented per-cell inventory -> present
   contribution families -> baseline DEEM -> cross-fitted residuals -> DUFS metric -> frozen
   graph -> target-smooth or nuisance-absorbing DEEM.
2. **Per-cell score map:** B0, B3, G2, G3, and G4 AUROC for all 24 cells, grouped
   by dataset family and QA/math.
3. **Paired-change forest plot:** family-level changes with blocked intervals.
4. **Residual graph atlas:** for representative cells, the same 2-D layout colored
   separately by hallucination label, length, baseline logit, model/dataset
   metadata, and graph arm. Coordinates are visualization-only and never used by
   the method.
5. **Neighbor-composition panels:** local hallucination purity and length
   difference versus neighborhood radius, including exact-length matched panels.
6. **Raw-versus-residual graph comparison:** edge overlap, health, and which
   nuisance associations disappeared.
7. **DUFS gate heatmap:** feature and family mass across cells, folds, and seeds.
8. **Target-versus-nuisance actuation:** how G3 changes `ell` and how G4 allocates
   variation to `U`, with posterior variance shown to rule out collapse.
9. **Linear-versus-graph panel:** residual metric graph against the matched 1-D
   linear-score graph.
10. **Control dashboard:** length-only, permuted, random-gate, lambda-zero, and
    duplicate-row checks, with every failed lane visible.
11. **Lambda paths:** fixed synthetic-nominated headline marked before real-cell
    performance is shown.
12. **Seed stability:** per-cell rank correlation and AUROC range for all five
    seeds.

Plots must display all cells and failures; no “representative” panel may replace
the complete tables.

## Interpretation table

| observation | permitted interpretation | forbidden interpretation |
|---|---|---|
| B3 stable and non-inferior, graphs fail | DEEM is a viable IU-PCR replacement candidate | manifold improves detection |
| G3 beats B3 but not linear graph | transferable/shared risk direction | nonlinear hallucination manifold |
| G3 passes residual and conditional controls | residual local geometry is an internal candidate | universal hallucination manifold |
| G4 wins and reduces nuisance dependence | graph helps isolate nuisance from risk | graph itself is hallucination |
| only DUFS arm wins but gates are unstable | exploratory feature-weighting clue | reproducible DUFS discovery |
| length-only or permuted control passes | graph audit invalid | weak but acceptable manifold evidence |
| Phase 1 passes | later task protocol may open | Localization or Early Detection validated |

## Stop rules

Stop Phase 1 without opening later tasks if any of the following occurs:

- any per-cell inventory, order, signs, row count, source hash, or admission hash
  differs from the registry;
- continuous DEEM repeatedly collapses or fails the seed-stability gate;
- class orientation cannot be guaranteed without natural labels;
- lambda-zero identity or contribution reconstruction fails;
- graph health fails in more than 10% of cells;
- length-only, permuted, or whole-search null controls promote;
- the proposed graph improves an internal scalar only after choosing a new
  topology, coefficient, support, or seed from Phase 1 labels.

If Gate A passes but graph gates fail, close the manifold branch and retain the
DEEM replacement branch. If Gate A fails, neither branch advances.

## Minimal execution order

1. Commit this protocol and an immutable machine-readable arm registry.
2. Implement the continuous family-additive DEEM and exact contribution tests.
3. Implement cross-fitted residualization and self-safe graph tests.
4. Implement DUFS cross-view gates and both Laplacian hypotheses.
5. Pass all Phase 0 synthetic and mechanical gates.
6. Freeze code, environment, feature inventory, hashes, seeds, and synthetic
   hyperparameter nomination.
7. Fit and freeze scores on the 24 cells without importing labels.
8. Run the isolated evaluator, controls, whole-search null, and visual report.
9. Perform resume and fresh-rebuild verification.
10. Record one Phase 1 decision and the two advancement flags.
11. Only then write and freeze the Localization addendum.
12. Only after that lane closes, write and freeze the causal Early Detection
    addendum.

## Execution and storage boundary

The approximately 7.82 GiB raw-source rebuild runs one cell at a time on AIRCC
CPU.  Full fitting runs checkpointed on AIRCC B200; the expected budget is 40--70
accelerator-hours.  Large bundles, checkpoints, graphs, and scores move directly
between AIRCC and
`gdrive:hallucination_detection/cluster_results/residual_graph_deem_24cell_v1/`.
The runner refuses an existing Drive run prefix unless an exact matching resume
manifest is present; it never overwrites a foreign or mismatched run.  Large
artifacts are not routed through the local workstation.  Git contains code,
protocol, registries, manifests, compact reports, and rebuild evidence only.

Phase 2/3, Localization, and Early Detection are expressly out of scope for this
execution even if an advancement flag is positive.

## Final success criterion

The useful result is not that a Laplacian can make scores smoother. The useful
result is that the same preregistered DEEM representation and residual mechanism
are stable across the original dataset families, beat the matched IU-PCR/DEEM
baseline, survive length and linear-direction controls, and later transfer
without modification to unseen dataset/model cells. If the selected features,
gates, or mechanism change from family to family, the correct conclusion is that
we found local structures, not a typical hallucination manifold.
