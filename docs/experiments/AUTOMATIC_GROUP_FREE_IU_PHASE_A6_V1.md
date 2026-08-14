# Automatic group-free IU — Phase A6 protocol v1

**Protocol date:** 2026-08-14

**Supervision tier:** S2 self-supervision; no human or benchmark correctness
label may fit, orient, select, or tune the method

**Status:** preregistered and independently adversarially audited (`NO
BLOCKERS`); Step 265 adds the pre-telemetry transformed-coordinate affine
clarification and unsealed development primitives; no A6 response telemetry,
simulator result, natural response, or target has been opened

**Primary candidate:** reciprocal Paired Target/Nuisance Intervention IU
(`PTNI-IU`), one frozen group-free atomic correction

## 1. Question, target ontology, and bounded claim

A6 asks whether mechanically verified interventions can supply the target
information that the closed covariance, repeatability, and latent-likelihood
routes could not identify from `P(X)` alone. The target is **ordinary
conditional task-answer correctness**: whether the one designated answer
assertion parsed from a short-answer response is correct for the task stated in
its prompt. It is not a claim that every incidental token or explanation is
factually true, and it is not contextual groundedness, evidence support, or RAG
faithfulness. This target matches the sealed response-level PopQA confirmation
boundary.

Evidence removal from a fixed response changes its contextual support but need
not change the proposition expressed by the response. Therefore the existing
RAGTruth full/no-context/leave-one-chunk-out caches are ambiguous premise
diagnostics, not A6 target pairs. ProcessBench cross-scorer triples are clean
nuisance-only pairs. PRMBench modified traces contain human/benchmark target
fields and are retrospective evaluation material only. None may fit or orient
A6.

The promotable claim is deliberately narrow:

> Reciprocal, mechanically verified prompt-response interventions can identify
> a nuisance-invariant atomic correction to IU-PCR that transfers from forced
> counterfactual pairs to untouched on-policy errors and then to natural
> hallucination benchmarks, while deployment remains one affine pass over the
> original mixed-v2 telemetry.

A quartet result alone does not establish that claim. Forced counterfactual
responses may be much easier to detect than natural fluent errors. The frozen
candidate must therefore pass an untouched on-policy mechanical-error gate and
a one-way natural-response veto before confirmation.

## 2. Prior-work boundary and novelty

This route does not claim evidence perturbation or contrastive hallucination
generation as new. Xu et al. perturb sources to construct contrastive
hallucinations and test transfer to natural hallucinations
([TACL 2023](https://aclanthology.org/2023.tacl-1.32/)). GASP uses
full/no-context/leave-one-out evidence contrasts for grounding sensitivity and
also illustrates that such transfer can fail on short-answer RAG
([arXiv:2607.04223](https://arxiv.org/abs/2607.04223)). The project's own GASP,
EC-IU, and original-30 evidence heads already tested evidence ablation; adding
another graph, evidence gate, or multi-pass head would repeat a closed route.

The only proposed methodological contribution is the combination of:

1. a reciprocal prompt-response crossover that balances every prompt and
   response marginal across correct and incorrect conditions;
2. factorial target, nuisance, and target-by-render contrasts;
3. a group-free atomic nuisance-whitened direction;
4. exact IU-orthogonal target-local anchoring and fallback; and
5. distillation to one affine deployment score over the target's frozen
   present-roster subset of the nominal 30 named **mixed-v2 transformed
   coordinates** (equivalently, a nominal 30-vector with absent coefficients
   fixed to zero).

## 3. Reciprocal calibration construction

### 3.1 Unit of construction

One reciprocal source group contains two equally difficult, well-formed task
worlds `P_A` and `P_B`, with unique semantic answers `a != b`, and two
deterministically rendered response ASTs `R_A` and `R_B`. For every prompt
rendering `r`, the complete crossed matrix is scored:

```text
(P_A^r, R_A)  valid          (P_A^r, R_B)  invalid
(P_B^r, R_A)  invalid        (P_B^r, R_B)  valid
```

Within each rendering, every prompt hash and response hash occurs exactly once
with each target polarity inside its group; across the four renderings it
therefore occurs four times per polarity. `R_A` is byte-identical in all of its cells,
and so is `R_B`. Both prompts are valid counterfactual worlds; neither is
called a corrupted or malformed prompt. Prompt-only and response-only target
prediction are structurally unidentifiable, not merely hoped to be weak.
The labels `A` and `B` are assigned only after both worlds and responses exist,
using one pre-hashed fair bit per group; the assignment is shared across
renderings and scorers and cannot encode target polarity or semantic family.

Each group has one canonical prompt rendering and all three nuisance prompt
renderings:

1. lexical/template paraphrase;
2. punctuation, whitespace, and layout rendering; and
3. semantics-preserving declaration/order/notation rendering.

Thus each group produces `4 prompt-response cells x 4 renderings = 16` traces
per scorer. Nuisance family is within group and is never assigned between
groups.

### 3.2 Exact populations and disjointness

The Qwen calibration population contains exactly 900 accepted reciprocal
groups: 100 groups in each of the nine `semantic domain x target mutation`
strata. Within every stratum, 50 groups use the short response grammar and 50
use the certificate response grammar defined below. A separate Llama
quartet-audit population contains another 900 groups under the same allocation.
Each Qwen scorer and the held Llama scorer also receives a disjoint natural
calibration cohort of exactly 2,000 unmodified on-policy responses. Mechanical
labels are not constructed for the Qwen cohorts. Labels for the Llama cohort
are generated by the independent interpreter and stored in a separate sealed
sidecar, never in the feature/calibration artifact. The two quartet
populations, three scorer-specific natural cohorts, and PopQA share no AST,
semantic source-record ID, donor ID, complete prompt ID, or template ID. This
is an ID-level pre-generation invariant; common answer strings, generic refusal
strings, or coincidentally byte-identical short responses are not identities and
do not trigger adaptive post-generation filtering.

The three semantic domains are:

1. exact integer/rational arithmetic programs;
2. finite relational-table aggregation and lookup; and
3. finite logical, set, and counting tasks.

The three target mutations are defined at the typed-AST level and must be
applicable in every domain:

1. value/entity leaf substitution;
2. relation/operator substitution; and
3. constraint/condition substitution.

The nine domain-by-mutation strata, rather than a mutation nested inside a
domain, are the generalization cells. At least one predeclared derived-answer
subdomain in every split forbids either answer from occurring verbatim in
either prompt. Its gates are reported separately.

Groups and natural-cohort prompts are drawn from pre-hashed attempt schedules.
Rejected attempts may be
replaced only by the next attempt in that schedule and only before telemetry
exists. The Qwen and Llama schedules and every split assignment are frozen in
the construction boundary. Source groups are the indivisible split and
bootstrap unit. Donor and template banks are partitioned before construction;
no donor or template ID crosses an outer fold or scorer population.

### 3.3 Deterministic response construction

Each reciprocal group is assigned one response grammar by its pre-hashed
schedule. The short grammar contains exactly one semantic assertion:

```text
The final answer is <canonical-answer>.
```

The certificate grammar is a deterministic rendering of a typed response AST
derived from the task solution. It contains a sequence of exact premises,
operations, intermediate values, and one final answer. Every semantic span is
represented in that AST; filler, unsupported prose, quotations, and free-form
LLM generation are forbidden. The generator targets 40--80 response tokens
under both frozen scorer-family tokenizers; attempts outside that band are
rejected before telemetry.

`R_A` is generated only from the verified solution AST of `P_A` and `R_B` only
from `P_B`. The independent verifier parses the entire short or certificate
response, verifies every assertion and operation against the candidate prompt,
and checks the final answer. Thus diagonal certificate cells are wholly true
and every off-diagonal cell has the wrong designated final answer because
`a != b`; full-certificate verification is a construction-integrity check, not
an expansion of the answer-correctness target. The response grammar,
canonical value formatting, renderer, parser, and their source hashes are
frozen in S0a. No model, random seed, or post-telemetry admission decision
constructs `R_A` or `R_B`.

### 3.4 Matching invariants

For every accepted group and every rendering:

- `P_A` and `P_B` have identical typed-AST shape, node count, solution depth,
  changed-node count, and changed-node type;
- both prompts are grammatical, answerable, and drawn from the same rendering
  template distribution;
- their prompt token counts are exactly equal under the frozen Qwen and Llama
  tokenizers;
- neither prompt contains a corruption marker, sentinel, impossible value, or
  condition-specific boilerplate;
- each response's byte SHA-256 and response token IDs are identical across all
  prompts and renderings where that response appears; and
- short responses contain exactly the one registered final-answer assertion;
  certificate responses satisfy the frozen 40--80-token band under both
  tokenizer families and end in one mechanically parseable final-answer field.

The construction records character length, word length, both tokenizer
lengths, AST size/depth, edit distance, changed-node type, prompt perplexity,
prompt-response lexical overlap, answer-in-prompt, numeric/entity rarity,
template ID, donor ID, and response length as forbidden-fit shortcut sidecars.

### 3.5 Independent semantic verification

The generator and verifier may not call the same evaluator. The boundary
contains a canonical serialized typed AST and its SHA-256. The generator uses
one exact evaluator; an independently implemented verifier parses every
rendered prompt and response and evaluates with a separate algorithm. Integer
and rational operations are exact; floating point is forbidden.

The implementation boundary must specify the two evaluators per domain. The
minimum acceptable independence is recursive evaluation versus separately
implemented stack/bytecode evaluation for arithmetic, direct relational
operators versus an isolated in-memory relational engine for tables, and
symbolic rules versus exhaustive finite assignment for logic/counting.
Property tests mutate every AST node type and round-trip every renderer.

Every accepted group must satisfy all of the following twice, once per
evaluator:

```text
Eval(P_A) = a
Eval(P_B) = b
a != b
Parse(R_A) = a
Parse(R_B) = b
truth matrix = [[valid, invalid], [invalid, valid]]
Parse(Render_r(P)) = full original task AST for all r
VerifyEveryAssertion(P_A,R_A) and VerifyEveryAssertion(P_B,R_B)
not VerifyEveryAssertion(P_A,R_B) and not VerifyEveryAssertion(P_B,R_A)
```

The full task AST includes output format, units, quantifiers, aliases, and
constraints, not only its factual payload. Ambiguity, refusal, parser failure,
no-op mutation, multiple accepted answers, a response accepted by both worlds,
or evaluator disagreement rejects the group before telemetry.

### 3.6 Closed natural-answer grammar

The on-policy cohorts are generated ordinarily, without constrained decoding,
but their primary mechanical label uses a closed parser. After Unicode NFKC
and outer-whitespace normalization, exactly one of these complete-string forms
is accepted:

```text
<answer-atom>
Answer: <answer-atom>
The answer is <answer-atom>.
The final answer is <answer-atom>.
```

`<answer-atom>` is parsed by the domain's independent exact evaluator and may
contain only the registered canonical integer, rational, finite-set, entity, or
relation syntax, including a registered unit when the task requires it. Extra
sentences, multiple answer atoms, explanations, refusals, unregistered aliases,
or unmatched units are parse failures. Every parseable output therefore
contains exactly one semantic assertion. The parser source, normalized grammar,
domain-answer canonicalizers, and property tests over every accepted/rejected
production are hashed before collection.

For PopQA, the same four complete-string wrappers extract one answer atom; the
already frozen normalized-alias token-boundary rule then judges that atom. The
canonical PopQA full-population label remains unchanged and is still reported;
the A6 confirmation additionally requires the closed-parser subset and the
same parse-rate gate used in S2c to pass, so A6 cannot benefit from a different
output convention at confirmation.

## 4. Scorers, telemetry, and target firewall

### 4.1 Scorer boundary

The source direction and all hyperparameters are fitted only from
teacher-forced telemetry from `Qwen/Qwen3-4B` and `Qwen/Qwen3-8B`. Nested folds
may use both Qwen views with equal view mass. Each Qwen scorer's target-local
transform and IU are fitted only on that scorer's disjoint 2,000-response
natural cohort, with no mechanical labels constructed or accessible. This same
target-local procedure is used in Qwen validation and deployment.

After S2a, one joint append-only Llama boundary collects and hashes both the
900-group quartet telemetry and the disjoint 2,000-response on-policy natural
feature matrix. Its mechanical correctness sidecar is separately hashed and
remains sealed. Before S2b, a no-label verifier checks exact 1:1 row keys and
counts between the S0a prompt manifest, natural feature manifest, and opaque
sidecar key/hash envelope without reading any label payload; quartet trace keys
are likewise checked against all scheduled cells. Any missing, extra, or
duplicate key closes the boundary. Fit the target-local Llama transform and IU on the natural
features, then apply them unchanged to the quartet audit. Only an S2b PASS
authorizes joining the already frozen natural labels for S2c. This ordering
creates no circular S2b dependency.

The held scorer-family audit uses
`meta-llama/Llama-3.1-8B-Instruct` on the disjoint 900-group quartet population.
It is PASS/CLOSE only: no parameter, direction, baseline identity, threshold,
or scale may be refitted from Llama intervention results.

All three natural calibration cohorts use one greedy response per prompt,
`max_new_tokens=150`, and the same short-answer instruction later used by the
on-policy gate and PopQA. Exact resolved model/tokenizer revisions, chat
templates, stop rules, and per-item seed hashes are frozen before collection.
No mechanical correctness object is constructed for the two Qwen cohorts. The
Llama interpreter sidecar is constructed by an isolated process and remains
unread until S2b passes.

### 4.2 Frozen atomic inputs

The candidate receives only the 30 frozen mixed-v2 feature identities, in this
order:

```text
epr, trace_length, spectral_entropy, low_band_power, high_band_power,
hl_ratio, dominant_freq, spectral_centroid, stft_max_high_power,
stft_spectral_entropy, rpdi, sw_var_peak, pe_mean, hurst_exponent,
cusum_max, cusum_shift_idx, epr_spilled, sw_var_peak_spilled,
cusum_max_spilled, min_spilled, epr_energy, min_energy,
sw_var_peak_energy, cusum_max_energy, mean_top1_logprob,
logprob_margin, mean_logprob_entropy, varentropy, renyi_entropy_2,
topk_tail_mass
```

Feature signs and the four frozen mixed-v2 transforms are inherited
label-informed preprocessing and are disclosed as such. No A4 component,
feature family, `FEATURE_TO_VIEW`, learned graph, or manual group enters A6.
Mean generated-token NLL is saved as a baseline-only sidecar and is never an
input coordinate to PTNI-IU.

The public construction boundary contains typed ASTs, condition identities,
mechanical truth, and shortcut sidecars but no natural benchmark target. The
public numerical fit API accepts numeric feature arrays, reciprocal group IDs,
domain/mutation/render IDs, scorer IDs, and the mechanically implied crossover
polarity only. It rejects target-like mappings and every original,
ProcessBench, SemGrad, PRMBench, HLE, RAGTruth, and PopQA label/answer field.
Sentinel tests fail on any forbidden access.

### 4.3 Feature admission

All four telemetry streams must describe the same nonempty response-token
trace, contain finite numeric arrays, and have `K >= 2` top-logprob entries.
The construction report gives feature admission by scorer, domain, mutation,
render, response grammar, and target polarity. Source intervention moments never
use a partial quartet or source-row imputation. If any of the 16 traces in
either Qwen scorer lacks a legal all-30 feature vector, drop that reciprocal
group from both Qwen views and from every rendering before folds. For Llama,
drop the complete 16-trace group from its audit. Both the common retained Qwen
population and the separate retained Llama population must each contain at
least 95 groups in every domain-by-mutation stratum and at least 47 of each
50-group response grammar within every stratum. Within retained
groups, target-polarity and rendering admission are therefore exactly balanced.
Weights are renormalized over retained groups inside each cell, and group folds,
null strata, and matching feasibility are reverified; failure closes before
candidate fitting. No post-telemetry bin merge or group replacement is allowed.

For each natural target-local matrix, a feature identity is present only if at
least 99% of its rows are finite. Its label-free median from that same natural
matrix imputes the remaining at most 1% before the transformer/IU fit and is
then reused unchanged on evaluation rows. Features below 99% are absent for the
entire target and invoke the roster restriction in Section 5.4; at least 17
present nondegenerate identities are required. Thus real source quartets use no
imputation, while target-local natural calibration uses one explicit unlabeled
matrix and cannot impute by condition.

The Qwen source coordinate roster is the immutable-name intersection of feature
identities present under that rule in both Qwen natural cohorts. Only this
intersection is transformed and may enter source `tau`, `nu`, `iota`, `q`, or
the learned direction; fewer than 17 identities closes A6. A nominal 30-vector
is never formed by silently synthesizing an identity absent from either source
cohort.

## 5. Factorial contrasts and PTNI-IU

### 5.1 Train-only coordinate transform

For each scorer and target cell, fit the frozen mixed-v2 transformer and
ordinary IU label-free on that scorer's original natural response matrix before
any mechanical or benchmark label is accessible. Apply that transform unchanged
to intervention rows. Intervention rows never fit normalization or IU. Frozen
feature identities/signs and the learned source direction are transported in
standardized coordinate space; only this registered target-local transform,
IU fit, covariance projection, and scale conversion are permitted.

In Qwen nested evaluation, each natural calibration cohort is wholly disjoint
from quartet source groups and fixed before folds. Its transform/IU is fit once
per scorer without labels and reused in every inner/outer split; only the PTNI
source direction is refitted by quartet fold. In Llama S2b/S2c, the same
transform/IU is fit once on the 2,000 natural features before the correctness
sidecar is opened and is applied unchanged to both quartet and natural rows.
PopQA analogously fits on its own natural unlabeled response matrix. This is
transductive batch calibration, matching ordinary IU-PCR, and is reported as
such.

### 5.2 Reciprocal target and nuisance contrasts

Let `z(P,R,r)` be the transformed retained source-roster `p`-vector,
`17 <= p <= 30`. For rendering `r`, define the
prompt-balanced invalid-minus-valid target effect

```text
tau_r = 0.5 * [z(P_B,R_A,r) - z(P_A,R_A,r)
             + z(P_A,R_B,r) - z(P_B,R_B,r)].
```

For each of the four prompt-response cells `c`, define nuisance main effects
and target-by-render interactions

```text
nu_c,r   = z(c,r) - z(c,canonical)
iota_r   = tau_r - tau_canonical.
```

Let one target sample exist for every
`group x scorer x domain x mutation x rendering`. Within each
`scorer x domain x mutation x response grammar x rendering` cell, its retained
groups receive equal weight after the complete-block rule in Section 4.3; the
144 cells also receive equal weight. For nuisance and interaction
terms, canonical zero contrasts are excluded: `nu` contains the four
prompt-response cells under each of the three noncanonical renderings, while
`iota` contains the three noncanonical renderings. Prompt-response cells get
equal mass. Every moment below is a population moment whose explicitly
normalized weights sum to one; `N-1` denominators and trace-count weighting are
forbidden.

Let `mu_T` be the target-sample weighted mean. Define

```text
S_T = sum w_T (tau_r - mu_T)(tau_r - mu_T)'
S_N = sum w_N nu_c,r nu_c,r'
S_I = sum w_I iota_r iota_r'
S   = S_T + S_N + S_I.
```

Before these moments, apply the automatic redundancy quotient in Section 7.
Let `q` contain quotient coordinates whose target-local natural variance is
greater than `1e-8` in both Qwen scorers and whose source intervention second
moment `E[tau^2 + nu^2 + iota^2]` is greater than `1e-10`. Let
`c = trace(S[q,q]) / |q|`. If `|q| < 17`, `c <= 1e-12`, or any value is
nonfinite, structural evidence is zero and the method returns exact IU.
Otherwise set `S_scaled = S[q,q] / c`. For

```text
lambda in {0.01, 0.03, 0.10, 0.30, 1.0, 3.0, 10.0}
```

the frozen source risk direction is

```text
r_0[q](lambda) = (S_scaled + lambda I)^(-1) mu_T[q]
r_0[not q] = 0.
```

Positive `r_0'z` means more intervention-identified error. The sign is fixed
only by the mechanical invalid-minus-valid definition.

### 5.3 Frozen score statistics

For any confidence score `f` and reciprocal group `g`, scorer `s`, and
rendering `r`, define one and only one valid-minus-invalid quartet contrast:

```text
Delta_f(g,s,r) = 0.5 * [f(P_A,R_A,r) - f(P_B,R_A,r)
                       + f(P_B,R_B,r) - f(P_A,R_B,r)].
H(x) = 1 if x > 0, 0.5 if x = 0, and 0 if x < 0.
```

The ordering objective `J_f` is the equal-weight macro mean of
`H(Delta_f)`: groups are equal inside each `scorer x domain x mutation x
response grammar x render` cell and those cells are equal in the aggregate. No alternative four
cross-class comparisons or pooled-row AUROC is permitted for quartet
selection. Per-cell gates use the same statistic restricted to the named
cell.

For the unit correction-only confidence score
`f_unit(z) = -r_perp' z`, independent of alpha, define the signed structural
target margin `m_T` as the same equal-cell mean of `Delta_f_unit`. For a
noncanonical nuisance family `k`,
define

```text
d_N(g,s,c,k) = f_unit(z(g,s,c,k)) - f_unit(z(g,s,c,canonical))
d_I(g,s,k)   = Delta_f_unit(g,s,k) - Delta_f_unit(g,s,canonical)
R_N(k)       = sqrt(equal-cell mean d_N^2) / max(abs(m_T(k)), 1e-12)
R_I(k)       = sqrt(equal-cell mean d_I^2) / max(abs(m_T(k)), 1e-12)
m_T(k)       = 0.5 * equal-cell mean [Delta_f_unit(k)
                                      + Delta_f_unit(canonical)].
```

`c` ranges over all four prompt-response cells. The macro `R_N` and `R_I` use
the corresponding pooled equal-family numerator and the all-render `m_T`;
family gates use the formulas above. Scorers, domains, mutations, renderings,
response grammars, and groups receive equal mass at every level. If a required
`m_T` is at most `1e-12` in absolute value, the learned direction and every
`alpha>0` arm are infeasible rather than receiving an epsilon-assisted pass;
`alpha=0` remains eligible exact IU. Every stated target-margin, nuisance-ratio,
and interaction-ratio gate below refers only to these quantities.

### 5.4 IU anchoring and affine deployment

For a target-local unlabeled standardized quotient matrix, fit ordinary
two-component IU-PCR with `IU_FIT_DEFAULTS`, obtaining confidence weight `u`.
Let `C = Z'Z/n` after exact train-fitted centering. Zero both `u` and `r_0` on
locally degenerate coordinates. Define `||v||_C = sqrt(max(v'Cv,0))` and form

```text
r_perp = r_0 - u * (u' C r_0) / (u' C u)
```

If `u'Cu <= 1e-10`, `r_0'Cr_0 <= 1e-10`, or
`||r_perp||_C / ||r_0||_C < 0.25`, structural evidence is zero. Otherwise
replace `r_perp` by `r_perp * ||u||_C / ||r_perp||_C`. The deployed confidence
score is

```text
s_alpha(z) = u' z - alpha * r_perp' z,
alpha in {0, 0.0625, 0.125, 0.25, 0.50, 1.0}.
```

Thus larger scores predict correctness, the correction is IU-orthogonal in
the target covariance geometry, and `alpha=0` is the exact IU score. Here and
throughout A6, “original-feature affine” means the frozen mixed-v2 transformer
object followed by one affine weight and intercept over the target's frozen
present-roster subset of its nominal 30 named output coordinates (equivalently,
a nominal 30-vector whose absent coefficients are zero). It does **not** mean
affine in the raw telemetry values:
mixed-v2 deliberately contains nonlinear squared and empirical-rank/mode
transforms. The code must reconstruct the score in this transformed coordinate
system with maximum absolute error below `1e-10`; no additional detector,
feature pass, or nonlinear fitted head is permitted.

On every zero-evidence condition above, the returned weights, intercept, and
scores are copied bit-for-bit from the quotient IU fit; a rescaled approximation
is forbidden. The automatic quotient is identity on the registered 30-feature
input unless the exact-duplicate rule in Section 7 fires.

For a natural target with absent feature identities, restrict both the frozen
source direction and ordinary target IU to the target's present roster before
projection. Only the target-local natural-matrix medians in Section 4.3 may
fill sporadic real missing values; a feature absent from the target roster is
never synthesized from a source environment. Fewer
than 17 present nondegenerate identities returns exact IU on that present
roster.

### 5.5 Nested selection

Qwen selection uses five fixed outer source-group folds. Every outer training
set contains a complete five-fold inner split. `mu_T`, all three `S` terms,
direction, exact-duplicate quotient, and every intervention-trained control are
refitted inside each inner training split. Target-local transforms and IU use
only the separately frozen natural matrices as specified in Section 5.1.

For each lambda/alpha arm, the objective is exactly `J_f` from Section 5.3.
Define the feasible set

```text
F = {all alpha=0 arms for which ordinary IU fits}
    union
    {alpha>0 arms whose f_unit margin is positive in every
     domain-by-mutation cell and whose nuisance and interaction ratios
     are <=0.50 in every nuisance family on every inner-validation fold}.
```

If `F` is empty, the fold closes as invalid IU. Let `a_best` maximize mean inner
validation `J` over `F`; before any SE is computed, exact ties choose smallest
alpha and then largest lambda. Let `SE_best` be the reciprocal-source-group standard error
of that arm's complete equal-cell macro: the sample standard deviation of
20,000 fixed-seed stratified source-group bootstrap replicates, resampling
within domain-by-mutation-by-grammar strata and recomputing the full equal-cell
macro. The seed namespace is frozen in the implementation boundary. The one-SE set is exactly
`{a in F: J_a >= J_best - SE_best}`; choose the smallest alpha and then the largest
lambda inside it. No other arm's standard error is used. No AUROC or natural
label participates.

Outer-fold scores are never concatenated across separately fitted transforms.
Metrics are computed within `outer fold x scorer x domain x mutation x response
grammar x render` and macro-averaged. Artifacts report exact retained counts in
every such cell and fold. Bootstrap draws repeat that complete macro statistic.
The outer nested run gates the complete selection procedure.

The final arm is unique. After that procedure passes, evaluate every fixed
lambda/alpha arm by five-fold cross-fitting over all retained Qwen groups, using the same
outer fold manifest. Compute its exact equal-cell macro and reciprocal-group
standard error; an arm is eligible only if its structural feasibility
constraints pass in every held fold under the same definition `F`, alpha=0
exception, and pre-SE tie order. Retain arms within one standard error of
the best eligible macro using exactly
`{a in F_final: J_a >= J_best - SE_best}`, then choose
smallest alpha, largest lambda, in that order. Refit `mu_T`, `S`, quotient, and
`r_0` once on all retained Qwen groups at
that unique arm. Hash the fitted artifact before Llama is opened. If no arm is
eligible, ordinary IU itself did not fit; emit `CLOSE_INVALID_IU` with no
deployable A6 score. If only alpha=0 arms are eligible, select exact IU under
the frozen rule and let the registered S2a improvement gate close the route.

Leave-family-out transfer is a separate refitted procedure, never an
evaluation of the all-family direction. For held target mutation `t`, remove
all groups with mutation `t` from moments, inner feasibility, arm selection,
and every learned control; run the same fixed nested group folds on the two
remaining mutations, refit once on those mutations, and evaluate only the
removed groups. For held nuisance rendering `k`, remove rendering `k` from
every fitted `tau`, `nu`, and `iota` sample and from feasibility and arm/control
selection; canonical plus the other two nuisance renderings remain, and only
the removed rendering is evaluated. The fixed group-fold manifest and the
same `SE_best`, alpha/lambda tie rules apply. Held-family rows cannot fit a
source moment, arm, baseline identity, or threshold. The disjoint unlabeled
natural cohort, including prompts from the same semantic domain/mutation label,
remains permitted for the target-local transform/IU in every LO run because it
contains no intervention contrast or correctness label. LO-family therefore
tests direction transfer, not leave-domain-out normalization; this transductive
representation calibration is reported explicitly.

The definitions of `F`, the pre-SE tie order, `SE_best`, and the feasible
one-SE set apply identically to the PTNI full-retained-Qwen final selection,
every PTNI leave-family-out fit, and every PTNI null/placebo refit. Controls use
the separate frozen contracts below. No downstream procedure may redefine
feasibility after seeing its held statistic.

## 6. Matched controls and shortcut audits

Every fit/evaluation split reports:

1. exact IU-PCR (`alpha=0`);
2. frozen Family-NRM, comparison only and never a selector;
3. strongest single mixed-v2 atomic feature selected in inner Qwen folds;
4. mean generated-token NLL/perplexity;
5. reciprocal target-only mean direction with no nuisance/interaction matrix;
6. diagonal-only `S` whitening;
7. unrestricted retained-`p`-feature ridge logistic pair discriminator with
   `ridge in {0.01,0.1,1,10}`, a self-supervised capacity ceiling;
8. naive unreciprocated target-delta PTNI, diagnostic only;
9. reciprocal PTNI without the interaction term, diagnostic only; and
10. the supervised atomic head, retrospective ceiling only.

Controls 5--9 use the same retained source roster, target-local `z`, quotient,
population weights, trace scale `c`, outer/inner folds, and target-local IU
geometry as PTNI unless an exception is stated here:

```text
# 5: target-only mean direction; no ridge/lambda
r_target = mu_T

# 6: diagonal whitening
D = Diag(diag(S[q,q] / c))
r_diag[q](lambda) = (D + lambda I)^(-1) mu_T[q]
r_diag[not q] = 0
lambda in {0.01,0.03,0.10,0.30,1.0,3.0,10.0}

# 8: unreciprocated response-block target samples
B = {d_A(g,s,r), d_B(g,s,r)} with equal A/B weight
mu_U = E_B[B]
S_T_U = E_B[(B-mu_U)(B-mu_U)']
iota_A,r = d_A,r - d_A,canonical
iota_B,r = d_B,r - d_B,canonical
S_I_U = E[iota_A iota_A' + iota_B iota_B'] / 2
S_U = S_T_U + S_N + S_I_U
c_U = trace(S_U[q,q]) / |q|
r_U(lambda) = (S_U[q,q]/c_U + lambda I)^(-1) mu_U[q]

# 9: no-interaction PTNI
S_noI = S_T + S_N
c_noI = trace(S_noI[q,q]) / |q|
r_noI(lambda) = (S_noI[q,q]/c_noI + lambda I)^(-1) mu_T[q]
```

Controls 6, 8, and 9 use the common lambda grid shown above. Every nonpositive
or nonfinite trace scale and every `|q|<17` condition returns its exact IU arm.
Controls 5, 6, 8, and 9 each pass their source risk vector through the exact
Section 5.4 IU-orthogonal projection/normalization and deploy
`u'z-alpha*r_control_perp'z` on the common alpha grid. Their feasibility is
measured on the real reciprocal `Delta`, `d_N`, and `d_I`; omitting a fitted
moment never omits its held diagnostic.

Control 7 fits a source-only weighted ridge logistic model on the four
individual quartet rows. The response is `y=1` for the two off-diagonal/error
rows and `y=0` for the two diagonal/correct rows. Rows have equal weight within
group; groups are equal inside each of the 144 source cells; cells are equal.
For risk coefficient `w`, unpenalized intercept `b`, and ridge `rho`, minimize

```text
E_weighted[-y log sigmoid(w'z+b)
           -(1-y) log(1-sigmoid(w'z+b))] + (rho/2)||w||_2^2,
rho in {0.01,0.1,1,10}.
```

The optimizer, convergence/KKT tolerance, and failure closure are frozen in its
source boundary. Logistic receives no IU projection or alpha; its transported
confidence score is exactly `f_logistic(z)=-(w'z+b)` with the Qwen-fitted
coefficient/intercept unchanged on Llama. It is a self-supervised capacity
ceiling, not a promotable fusion rule.

Control selection is class-specific and never invents a PTNI parameter:

- IU, Family-NRM, and mean NLL are parameter-free. NLL confidence is `-NLL`.
- Single-atomic evaluates every present feature and both polarities. Maximum
  nested `J` wins; exact ties use the immutable feature order in Section 4.2
  and then positive polarity before negative. It has no one-SE or nuisance
  feasibility rule; those diagnostics are reported only.
- Target-only uses only the common alpha grid and chooses smallest alpha in its
  feasible one-SE set. Diagonal-whitening, unreciprocated, and no-interaction
  use the lambda and alpha grids above and choose smallest alpha then largest
  lambda. All four use candidate-style `F` and `SE_best`; nuisance and
  interaction feasibility is computed on their deployed unit correction even
  when the corresponding term was omitted from their fit.
- Ridge logistic is an unrestricted diagnostic ceiling with no IU arm and no
  alpha. It chooses among `{0.01,0.1,1,10}` by nested `J`; the one-SE set uses
  the pre-SE best ridge with exact ties choosing the largest ridge, and final
  selection again chooses the largest eligible ridge. Nuisance/interaction
  ratios are diagnostics, not eligibility constraints.
- The supervised atomic head is an already frozen retrospective ceiling. A6
  does not refit, select, orient, or tune it.

Every learned control receives its own nested and full-retained-Qwen
cross-fitted selection under only its contract above, with no PTNI parameter
sharing. For the final single control, apply its feature/polarity rule to the
full cross-fitted macro. The stronger of final single versus NLL is the larger
Qwen macro, ties selecting NLL. The resulting target rule is frozen before
Llama: if NLL won, always use NLL; if a single atom won, use that exact
feature/polarity only when the target-local roster contains the identity with
natural variance greater than `1e-8`, otherwise use NLL. This availability
fallback is triggered only by the label-free roster/variance rule and never by
held performance. It is applied identically on Llama, retrospective targets,
and PopQA and is named the `frozen availability-aware single/NLL composite`.
A PTNI fusion contribution is established only if PTNI beats IU and that
unique frozen composite.

The unrestricted logistic ceiling is scorable on a target only when every
coefficient identity in its frozen Qwen roster is present and nondegenerate in
the target-local roster. Otherwise its artifact is marked
`UNSCORABLE_TARGET_ROSTER` and no coefficient is dropped, imputed, or replaced;
because logistic is diagnostic only, this does not change the candidate or its
mandatory composite comparison. No target-side control identity is reselected.

Inside each outer fold, the single-versus-NLL composite is chosen using only
that outer training set and its inner validation scores; the chosen identity is
then applied unchanged to the outer held groups. Consequently the S2a paired
PTNI-minus-composite interval is fully outer-held. The full-retained-Qwen cross-fitted
identity described above is selected only after the outer procedure has passed
and is used solely for the final frozen artifact and Llama audit; it is never
back-projected into the S2a outer-fold comparison.

S0 is split so no model-derived sidecar can adapt construction. S0a freezes and
verifies the complete population using only AST, string, tokenizer, and hash
invariants. It freezes all accepted/rejected attempt IDs; no later replacement
is permitted. S0b then computes prompt perplexity with the separately pinned
external audit model `EleutherAI/pythia-410m-deduped` at an exact revision
frozen by the implementation boundary. The input is the rendered task text
without a chat template, and mean next-token NLL over the whole prompt is the
only perplexity statistic. This model never scores responses or enters PTNI.

A fixed ridge-logistic shortcut audit uses only the sidecars in Section 3.4.
Five grouped folds and ridge grid
`{0.01,0.1,1,10}` are fixed. The 95% source-group bootstrap upper endpoint of
target AUC must be at most 0.60 overall, in every domain, in every one of the
nine domain-by-mutation strata, in each response grammar, and in each rendering
family. These are simultaneous mandatory gates with no multiplicity-based
waiver. Prompt-only and
response-only hashes must be exactly 50/50 by construction; classifiers using
only either marginal must score exactly 0.5 up to `1e-12` after duplicate-row
weighting. Failure closes the data premise before candidate fitting.

After telemetry applies the frozen complete-block rule, rerun this identical
shortcut model, folds, sidecar columns, thresholds, and all simultaneous gates
on the retained Qwen population before S2a and on the retained Llama quartet
population before S2b. No sidecar feature, ridge, fold, or threshold may change.
Feature-driven admission that makes either retained population fail the already
frozen shortcut gate closes A6; it cannot define an easier admitted experiment.

## 7. Nulls and common structural gates

Controls 1--3 use 200 predeclared seeds and refit the entire source direction,
IU projection, lambda, and alpha selection pipeline. Control 4 is one
deterministic balanced refit. No control invents or claims
semantic truth for an unscored cross-group prompt-response pair. Instead, the
following exact group-block contrasts are constructed from already scored rows.

For notation, let

```text
d_A(g,s,r) = z(P_B,R_A,r) - z(P_A,R_A,r)
d_B(g,s,r) = z(P_A,R_B,r) - z(P_B,R_B,r)
q_A(g,s,r) = f(P_A,R_A,r) - f(P_B,R_A,r)
q_B(g,s,r) = f(P_B,R_B,r) - f(P_A,R_B,r).
```

Thus real `tau=.5*(d_A+d_B)` and real `Delta=.5*(q_A+q_B)`.
The nulls are:

1. **whole-quartet polarity:** one Rademacher sign `b_g` per reciprocal group,
   shared across scorers and renderings, gives `tau~=b_g*tau` and
   `Delta~=b_g*Delta`; this complements the complete 2-by-2 truth matrix and
   preserves both prompt and response marginals;
2. **response-block derangement:** a fixed-point-free group bijection `pi`
   moves the complete `R_B` response-conditioned block as a unit and reverses
   its pseudo role, giving `tau~_g=.5*(d_A,g-d_B,pi(g))` and
   `Delta~_g=.5*(q_A,g-q_B,pi(g))`; no row is duplicated and each donor block
   is used exactly once;
3. **matched random-group difference:** let `a_g,s,r` be the mean feature vector
   over all four cells and `h_g,s,r` the corresponding mean confidence score.
   A minimum-cost fixed-point-free donor bijection gives
   `tau~_g=a_pi(g)-a_g` and `Delta~_g=h_g-h_pi(g)`, so every group is used once
   as a recipient and once as a donor; and
4. **nuisance-as-target:** for each noncanonical nuisance family `k`, define
   the cell-balanced rendering shift and its confidence ordering

   ```text
   eta(g,s,k) = 0.25 * sum_c [z(g,s,c,k) - z(g,s,c,canonical)]
   D_eta(g,s,k) = 0.25 * sum_c [f(g,s,c,canonical) - f(g,s,c,k)].
   ```

   The three `k` families and four cells have exact equal weight. Fit the same
   nuisance-whitened anchored estimator with `eta` as its pseudo-target:
   `mu_eta=E[eta]`, `S_T_eta=Cov_pop(eta)`, `S_N_eta=Cov_pop(tau)`,
   `bar_eta(g,s)=sum_k eta(g,s,k)/3`, and
   `S_I_eta=E[(eta-bar_eta)(eta-bar_eta)']`, using the ordinary
   lambda/alpha grids. For its unit correction define

   ```text
   m_eta = equal-family mean D_eta_unit
   bar_D_eta(g,s) = sum_k D_eta_unit(g,s,k)/3
   R_etaI(k) = sqrt(E_g,s[(D_eta_unit(g,s,k)-bar_D_eta(g,s))^2])
               / max(|m_eta|,1e-12)
   R_sem(cell) = sqrt(E_cell[Delta_unit_real^2]) / max(|m_eta|,1e-12).
   ```

   Its pseudo feasible set contains exact IU plus alpha>0 arms only when
   `m_eta>1e-12`, `R_etaI<=0.50` in every nuisance family, and `R_sem<=0.50`
   in every domain-by-mutation cell on every inner-validation fold. Nested
   selection uses `J_eta=E[H(D_eta)]` with the ordinary `F/SE_best/tie` rule.
   Before semantic interpretation, the deterministic control must activate on
   outer-held pseudo-nuisance data: its final deployed arm has `alpha>0`;
   nested outer-held `J_eta>=0.60` macro and `>=0.55` in every
   `nuisance family x scorer x domain x mutation x response grammar` cell; its
   deployed pseudo correction margin is positive in every such cell; and the
   20,000-draw paired source-group bootstrap lower endpoint for
   `J_eta(control4)-J_eta(IU)` is greater than zero. The activation bootstrap
   draws one multiplicity per indivisible reciprocal source group within
   `domain x mutation x grammar`, reuses that multiplicity across all three
   nuisance families and both scorers, then computes each named nuisance-family
   and scorer cell before equal-macro aggregation. Failure of any activation condition yields
   `CLOSE_UNINFORMATIVE_NUISANCE_CONTROL`; it is never counted as safe nuisance
   rejection.

   Only after activation, evaluate the nuisance-trained score on **real
   outer-held semantic** `Delta_f`. The finite nonrecursive confounding set is:
   (a) real `J>=0.60` macro and `>=0.55` in every domain-by-mutation, scorer,
   grammar, and render cell; (b) real unit-correction margin positive in every
   such cell; (c) the registered non-copy `J>=0.60` with lower endpoint above
   `0.50`; and (d) both paired lower endpoints versus IU and the outer-train
   single/NLL composite greater than zero. Nulls, placebos, LO fits, direction
   cosines, duplicate/missingness stresses, and control 4 itself are explicitly
   excluded from this set. The lower endpoints in (c,d) use 20,000 fixed-seed
   source-group bootstrap draws within `domain x mutation x grammar`, with
   scorers/renderings equal in every recomputed real-semantic macro; (c) simply
   restricts those draws to the preregistered non-copy groups. Seed namespaces
   are frozen before telemetry. `PASS_ALL(a--d)` yields
   `CLOSE_NUISANCE_CONFOUNDING`; failure of at least one means the negative
   control behaved as required. This construction is deterministic and has no
   seed or p-value.

For controls 1--3, replace `tau` by `tau~`, derive `iota~` from the same
rendered and canonical pseudo-target samples, retain the ordinary nuisance
second moment, and use `Delta~` in selection/evaluation. Each seed's scalar is
the complete equal-cell `J_control = mean H(Delta~)` from Section 5.3. For every
seed, `J_seed_outer` means one unbiased nested statistic: aggregate `J_control`
over the five outer-held folds using the registered 144-cell macro. After that
nested run, the registered full-retained-Qwen cross-fitted rule produces one separately
named final deployed-arm indicator `A_seed_final`. `J_observed_outer` and
`A_observed_final` are defined by the identical real-data procedures. Inner-fit
or individual outer-fold arm counts are never used as denominators.

Null 1 is a conditional paired sign-permutation test. Its sharp working null is
that, conditional on the frozen group/shortcut strata, each complete group
contrast is sign-exchangeable: `(tau_g, Delta_g)` and
`(-tau_g, -Delta_g)` have the same joint distribution when there is no target
association. This is an explicit modeling assumption, not design-based
randomization by the A/B assignment. Its one `b_g` draw is made
globally for every source group and is reused unchanged whenever that group
appears in an inner train, inner validation, outer train, outer held, or final
fit. `J_observed_outer` must exceed the empirical 99th percentile of
`J_seed_outer` (ascending
order statistic 199 in one-based indexing), its exact Monte Carlo p-value
`(1 + count(J_seed_outer >= J_observed_outer))/201` must be at most `0.01`, and
at least 180 of the 200 `A_seed_final` indicators must equal `alpha=0`.

Controls 2 and 3 are explicitly **split-local placebo stress tests**, not
randomization nulls and not CV on one globally relabelled dataset; no p-value
or exchangeability claim is made for them. A mapping is drawn independently
inside each inner-train, inner-validation, outer-train, and outer-held
partition so a group never crosses a split. For each placebo family, all 200
complete schedule hashes **and** all 200 outer-held evaluation-assignment
hashes must be distinct. The 95th percentile of `J_seed_outer` must be at most
`0.55`; it is the ascending one-based order statistic 191, equivalently
NumPy-style `quantile(q=.95, method="higher")` for 200 values.
`J_observed_outer` minus that percentile must be at least `0.05`, and
at least 180 of the 200 `A_seed_final` indicators must equal `alpha=0`. Within a
response-grammar stratum, each partition mapping is shared across both Qwen
scorers and all four renderings. Control 4 follows its one deterministic
semantic-held evaluation above.

The frozen S0a
null stratum is `domain x mutation x response grammar x joint-Qwen/Llama prompt
token-length bin x AST-complexity bin`. Bin edges are derived only from S0a
sidecars, then adjacent bins are deterministically merged in immutable
lexicographic order until every stratum in every predeclared inner/outer
partition has at least four groups. Failure to achieve that minimum closes S0a.

Control 2 uses seeded Fisher-Yates with at most 10,000 attempts to obtain a
derangement. For control 3, eligible directed edges must be in the same stratum,
must not share source/donor/template identity, and must lie below the S0b-frozen
75th percentile of standardized shortcut-vector distance. Each seed assigns
every eligible `(recipient,donor)` edge the independent key
`SHA256(seed || partition || recipient || donor) / 2^256`; Hungarian assignment
minimizes those random keys, with ineligible and self edges infinite. Distance
defines the caliper only, so the seed changes generic-cost assignments rather
than merely breaking ties. Every row and column is used once. All 200 schedules
and perfect-matching existence are preflighted on the full S0b population
before response telemetry. The S0b boundary enumerates the shortcut vector and
its encoding, fits its label-free standardization, freezes the 75th-percentile
caliper, and includes prompt perplexity only after that sidecar has been
computed; S0a does not claim to freeze this caliper. After the frozen complete-block admission rule, the same
seeds and algorithm are rerun on the retained population before candidate
fitting; no caliper, stratum, or retry rule may change. If any finite
derangement is absent or either set of 200 required hashes is not distinct, A6 closes.
Artifacts store partition and stratum sizes, zero fixed-point/no-reuse
assertions, the exact pseudo-contrast schedule, permutation hashes, and the
number of unique realized `J_seed_outer` values for each placebo.

Before any moment, derive a source redundancy quotient from permitted Qwen
training rows only. Join coordinates only when their standardized columns are
bit-identical in both Qwen natural matrices and every applicable Qwen
intervention-training row. Connected components and Helmert member order use
immutable feature names. A component becomes its arithmetic mean plus `k-1`
orthonormal contrasts; zero contrasts are removed and their PTNI correction is
fixed to zero. The quotient mapping is frozen with the source direction and no
new group may be discovered on Llama or PopQA. A target must satisfy the same
within-component equality within `1e-10`; otherwise it returns ordinary
unquotiented target IU exactly.

IU on a verified quotient uses the sum-preserving coefficient map, and
expansion divides a mean correction equally and adds retained contrasts. The
registered 30 coordinates have unique identities, so absent an automatically
verified duplicate this map is the identity and `alpha=0` is ordinary IU-PCR.
Exact duplication must preserve the selected arm, expanded deployed score, IU
score, and fallback within `1e-10`, and duplicate-component combined correction
L1 mass within `1e-10` relative.

For rho=0.999 near duplicates no contrast is removed. Across 100 registered
stress seeds, full selected-output rank correlation with the unaugmented fit
must be at least 0.995 in every run, median absolute alpha difference at most
0.125, and median deployed-correction L1-mass ratio in `[0.90,1.10]`. For a
seed whose unaugmented deployed correction mass is at most `1e-12`, the ratio
is not divided; the augmented run must also select `alpha=0` and have correction
mass at most `1e-12`, or the seed fails. Deleting one, two,
or three seeded coordinates from held rows only must leave source-training
artifacts bit-identical, retain at least 17 coordinates, and cause
candidate-minus-IU ordering harm no worse than `-0.005` in every stress and
median no worse than zero.

Registered missingness uses 5%, 15%, and 30% MCAR plus domain-linked blocks;
for these injected source-intervention stresses only, medians are computed from
the current source inner-training intervention rows and applied unchanged to
its validation/held rows. This is distinct from the real target-local natural
matrix median rule in Section 4.3. The missingness indicator is forbidden as a
feature, and each ordering gate must remain within 0.01 of complete data. An
all-missing coordinate causes exact IU fallback. Under every feature-name
permutation, inverse-permuted weights, scores, arm identity, and control
identities agree within `1e-10`. Deterministic reruns produce byte-identical
canonical JSON and score arrays. Affine reconstruction and exact zero-evidence
IU fallback stay within `1e-10`. Full candidate and control selection, not only
a fixed direction, is rerun under all stresses.

## 8. Staged execution and hard stops

No later stage may repair a failed earlier stage. Every stage has an
append-only source/runtime/data boundary, predeclared seeds, independent
no-edit review, exact schedule verification, checkpoint/resume semantics, and
a completion artifact bound to its boundary and record hashes. A PASS may only
authorize implementation of the already specified next stage under a new
reviewed boundary; it may not change the estimator, grids, gates, target,
feature contract, or interpretation.

### A6-S0 — construction and semantic boundary

S0a runs before either audit-model or response telemetry:

- both 900-group populations have exactly 100 groups per domain-by-mutation
  stratum and all four renderings;
- the three exact 2,000-prompt natural-cohort manifests, scorer assignment,
  item hashes, attempt schedules, domain/mutation counts, and disjointness from
  both quartet populations and PopQA are frozen; the two Qwen manifests contain
  no correctness sidecar path or object, and poison-sentinel tests assert that
  none can be constructed or loaded by any Qwen fit API;
- both independent evaluators pass 100% of diagonal/off-diagonal truth and
  rendering round-trips;
- there are zero ambiguous, no-op, duplicate, donor-crossing,
  template-crossing, or marginal-imbalance cases;
- every response SHA is condition-invariant and each prompt/response marginal
  is exactly 50/50 valid/invalid;
- A/B AST complexity and both tokenizer lengths match exactly;
- the closed natural-answer parser, canonicalizers, source hashes, and
  accept/reject property tests are frozen; S0a freezes only the Llama prompt
  IDs, expected row-key schema, and isolated-sidecar protocol and asserts that
  no Llama response, feature row, or correctness sidecar exists yet; and
- response grammars, every accepted/rejected attempt, all group folds, and the
  merged null strata are frozen, with at least four groups in every stratum of
  every predeclared train/validation partition.

S0b runs the fixed external-LM prompt perplexity and sidecar shortcut audit from
Section 6, then freezes the exact control-3 shortcut-vector columns and
continuous/categorical encoding, its label-free standardization, the
75th-percentile distance caliper, eligible matching graphs, all 200 seed
schedules, perfect-matching proofs, and schedule hashes from Section 7. No
response is scored, and S0a groups cannot be replaced. The shortcut AUC gate,
matching existence, and distinct-schedule gates must all pass inside the
append-only S0b boundary before response telemetry.

Any failure yields `CLOSE_INVALID_INTERVENTION_BOUNDARY` or
`CLOSE_MECHANICAL_SHORTCUT_PREMISE`; telemetry scoring is forbidden.

### A6-S1 — sealed estimator simulator

The implementation boundary freezes exact equations and 100 unused seeds for
each of these worlds before opening any result:

1. target only;
2. nuisance only;
3. equal target and nuisance;
4. nuisance with twice the target magnitude;
5. target direction varying by intervention family;
6. null/single-Gaussian features;
7. forced-mismatch NLL-dominant error; and
8. coherent fluent error with no mean-NLL separation.

Worlds use 30 atoms, three domains, three target families, three nuisance
families, two source scorer views, five group folds, the real alpha/lambda
selection code, small-n/missingness matched to the registered A6 construction,
and a frozen valid IU anchor. Development seeds and sealed seeds are disjoint.
Exact numerical parameters and RNG byte serialization are part of the S1
source boundary, not chosen after results.

For every held simulator environment `e`, let `C_e` be its population feature
covariance and `t_e,n_e` its planted target-risk and nuisance-risk directions.
Define signed covariance cosine

```text
cos_C(a,b) = (a' C_e b) / sqrt((a' C_e a)(b' C_e b)).
Pref(a) = mean_e cos_C(a,t_e)^2 - mean_e cos_C(a,n_e)^2.
```

A direction is target-preferred iff its norm and every denominator exceed
`1e-12`, `mean_e cos_C(a,t_e)>0`, and `Pref(a)>=0.05`; equality below the
margin, a zero direction, or a tie fails. The final risk direction is the
negative of the deployed confidence coefficient; the correction risk direction
is the actually deployed `alpha*r_perp`, so selected `alpha=0` fails this
direction gate in target-plus-nuisance worlds. Both directions must be
target-preferred in at least 90/100 repetitions of every target-plus-nuisance
world. Nuisance-only and
null worlds must select exact `alpha=0` in at least 95/100. Candidate-minus-IU
AUROC must have a paired 20,000-draw 95% lower endpoint at least zero in every
target world. The twice-stronger nuisance and coherent/no-NLL worlds are
mandatory. Any unusable repetition, common-gate failure, or gate failure closes
A6 before real telemetry.

### A6-S2a — nested Qwen quartet premise

All of the following must pass:

- ordering is at least 0.60 macro and at least 0.55 in every one of the nine
  domain-by-mutation strata, each Qwen scorer, each response grammar, and each
  held nuisance render;
- paired source-group bootstrap lower endpoints for PTNI-minus-IU and
  PTNI-minus-outer-train-selected-single/NLL are both greater than zero;
- signed correction target margin is positive in every stratum, scorer, and
  render;
- correction nuisance RMS divided by absolute target margin is at most 0.25
  macro and 0.50 in every nuisance family;
- target-by-render interaction RMS has the same 0.25/0.50 limits;
- ordering in the non-copy/derived-answer subdomains is at least 0.60 with a
  lower endpoint above 0.50;
- leave-one-target-family-out and leave-one-nuisance-family-out satisfy macro
  ordering at least 0.60 and every held family at least 0.55;
- minimum signed cosine among outer and leave-family-out source directions is
  at least 0.70; and
- all null and common structural gates pass.

All paired intervals use 20,000 stratified reciprocal-group bootstrap draws
with equal scorer and domain-by-mutation mass. Failure yields
`CLOSE_INTERVENTION_PREMISE`; Llama telemetry remains sealed.

Because certificate off-diagonal cells may change intermediate assertions as
well as the designated answer, a predeclared short-grammar-only sensitivity
refits the complete selection procedure on all retained short-response groups and
reports the same cell metrics and direction cosine. It is diagnostic only: it
cannot replace the mixed-grammar primary, change an arm, rescue a failed gate,
or authorize Llama. The primary already requires its ordering gate separately
in each response grammar.

### A6-S2b — frozen Llama quartet audit

Freeze the Qwen source artifact, candidate, and all control identities first.
Collect the joint quartet/natural-feature Llama boundary described in Section
4.1, but do not join the natural correctness sidecar. On the disjoint quartet
groups, using only the target-local transform/IU learned from the unlabelled
on-policy feature matrix, require the same 0.60 macro, 0.55 per-cell and
response-grammar, positive-margin, nuisance-ratio, interaction-ratio, non-copy,
and paired improvement gates as S2a. No intervention-result or source-direction
refit, reselection, sign change, or rescue variant is permitted: the registered
target-local natural transform, IU, covariance projection, and scale are the
only deliberate Llama-side fit. Failure closes A6 with the natural labels
unopened.

### A6-S2c — untouched on-policy mechanical-error transfer

The 2,000 natural features were already collected and frozen before S2b; after
S2b passes, join their mechanical sidecar exactly once. Each row comes from one
ordinary, unmodified greedy response by the held Llama policy to a disjoint
pre-hashed prompt using the short-answer instruction and
`max_new_tokens=150`, matching the PopQA boundary. The population is equally
allocated across the three domains and target families up to integer
remainder. Exact model/tokenizer revision, greedy decoding, chat template, stop
rule, and seeds are frozen before collection. No prompt or response is edited,
counterfactually paired, or used in candidate/alpha selection.

The independently hashed closed parser in Section 3.6 extracts exactly one
answer atom, and the second domain evaluator marks it correct iff it equals the
unique answer of the full task AST under exact canonicalization. All other
outputs are parse failures. Parsing failures are reported but excluded from the
primary answer-error gate; treating them as incorrect is secondary. The
parse-failure rate must be at most 10% overall and 15% in every domain. Among
parseable responses require at least 100 correct and
100 incorrect overall and at least 30 of each per domain; otherwise return
`CLOSE_UNDERPOWERED_ON_POLICY_TRANSFER`.

On parseable outputs, the frozen candidate must achieve correctness AUROC at
least 0.55,
candidate-minus-IU grouped-bootstrap lower endpoint greater than zero, and no
domain AUROC more than 0.010 below IU. This gate tests quartet-to-natural-error
transport; it is not a substitute for the natural benchmark veto.

### A6-S3 — one-way retrospective natural-response veto

Only after S2c passes, materialize and hash exactly one target-free score bundle
for already opened natural responses. The primary correctness veto contains
only the original response cells, SemGrad SciQ/TruthfulQA, and HLE rows with a
frozen response-level answer-correctness label. Before producing scores, the S3
boundary writes and hashes an immutable row manifest keyed by
`environment, item_group_id, candidate_ordinal`: `Y=1` always means the frozen
grader judged the designated answer correct, and resampling groups the complete
`item_group_id`. Original rows use their frozen A0 binary correctness mapping;
SemGrad uses the already frozen BEM-at-0.8 binary mapping; HLE uses only rows in
the validated frozen judge sidecar and reports its non-paper-faithful judge
status. Rows without one of these exact mappings are excluded before scores
exist. The manifest's counts, source hashes, label polarity, and group IDs are
reviewed and frozen at the S3 boundary; no row can be added or removed after a
candidate score is joined.

ProcessBench first-error/no-error labels and RAGTruth support/hallucination
labels are adjacent-task diagnostics only: neither is pooled into the primary
answer-correctness macro or supplies target-transfer evidence. PRMBench
benchmark-modified traces, every A6 generated/crossed example, and supervised
heads are also separately named diagnostics. Feature/score production remains
isolated before any permitted label sidecar is joined.

Labels may only PASS or VETO this sole bundle. They may not choose a feature,
arm, sign, threshold, trust, transform, or dataset. With equal dataset-family
mass and family-grouped 20,000-draw bootstrap, require:

- candidate-minus-IU lower endpoint greater than zero;
- candidate-minus-frozen-Family-NRM lower endpoint at least `-0.002`; and
- no registered domain-family macro more than 0.010 below IU.

Failure yields `CLOSE_NATURAL_TRANSFER_VETO`. PopQA remains sealed.

### A6-S4 — untouched PopQA confirmation

Only an S3 PASS may invoke the already sealed
`popqa-gemma3-4b-it-confirmation-v1` boundary. The primary checkpoint access
smoke and registered fallback rule run before any PopQA response or label is
collected. Fit only the allowed target-local transform, IU, covariance
projection, and scale on the natural unlabeled PopQA response matrix. The Qwen
source direction, lambda, alpha, sign, roster, and score definition remain
frozen.

Apply the canonical S2 success gates exactly: paired question-group bootstrap
candidate-minus-IU lower endpoint greater than zero;
candidate-minus-Family-NRM lower endpoint at least `-0.002`; and no registered
domain-family loss worse than 0.010. In addition, the Section 3.6 wrapper parser
must accept at least 90% of responses overall and 85% in every registered PopQA
stratum, and the same three performance gates must pass on that parseable
subset under the already frozen alias rule. The canonical full-population PopQA
result remains primary and is not replaced by the subset. One frozen S2
finalist gets one opening.

## 9. Interpretation of outcomes

- S0 failure means the intervention surface does not isolate correctness from
  construction mechanics.
- S1 failure means the estimator or trust rule follows nuisance even when
  target interventions are known.
- S2a failure means reciprocal synthetic target structure is not recoverable
  from the atomic telemetry.
- S2b failure means the structure does not transfer across scorer family.
- S2c failure means forced-pair incompatibility does not transfer to untouched
  model errors.
- S3 failure means mechanical-task transfer does not improve natural
  hallucination detection.
- Only S4 success supports an S2 successor to IU-PCR on the sealed target.

No lower-stage success may be promoted after a higher-stage failure. A failed
route returns exact IU at deployment and advances the research program to A7.
