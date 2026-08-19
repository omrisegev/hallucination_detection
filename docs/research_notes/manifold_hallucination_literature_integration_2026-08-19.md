# Geometry/manifold literature integration for hallucination detection

**Date:** 2026-08-19  
**Scope:** literature integration and research planning only  
**Execution boundary:** no detector implementation, no new labels, no LLM/embedding
inference, no GPU/cluster use, and no Drive mutation

## Executive decision

**Recommendation: `DIAGNOSTIC ONLY`.** None of the newly added manifold papers
provides a deployable, label-free source of correctness semantics compatible with the
current gray-box, single-pass IU-PCR contract. We should not implement a density ridge,
local tangent router, PCNET analogue, or atomic c-STG from these papers now.

The literature does sharpen the roadmap:

1. **APORIA adds genuinely new observables**—semantic relationships among repeated
   generations—but not label-free target information. Its useful direction is fitted
   with prompt-local correctness labels, requires up to 150 generations, and scarcely
   transfers between prompts.
2. **PCNET and Density Ridge add genuinely new hidden-state observables**, but not
   label-free target information. Both use correctness labels; PCNET is explicitly
   contrastive, and Density Ridge is correct-only plus a stated both-class projection.
   They change the access tier.
3. **LTSREx/LEGO and density-ridge machinery alone add no information beyond \(P(X)\).**
   Applying them to our DSP or mixed-v2 matrix would estimate the same nuisance-prone
   geometry more carefully, not identify correctness.
4. **The strongest compatible candidate source remains verified target-changing versus
   nuisance-only interventions.** That is already the frozen A6/PTNI program, but it is
   not yet a demonstrated source: S0a established mechanics and S0b is still pending.
   The new Tiberi–Sompolinsky paper offers only a post-hoc target-direction transport
   audit after A6 passes; it does not alter, rescue, route, or improve A6 by itself.

The practical consequence is deliberately conservative: finish literature documentation,
retain LTSREx/Tiberi as conditional diagnostics, and do not start another manifold-named
fusion experiment. If the user later authorizes a method experiment, the next core-method
gate remains frozen A6-S0b/S1—not a new density or routing branch.

## Papers read and documented

| Paper | Status established from paper | Digest |
|---|---|---|
| Tiberi & Sompolinsky, *Manifold geometry underlies a unified code for category and category-independent features* | bioRxiv v2, not peer reviewed | `papers/digests/2026-03-23-713692v2-full.md` |
| Vamshi, Bhatnagar & Yang, *Geometry-Aware Hallucination Detection in Large Language Models* | arXiv v3, no venue found | `papers/digests/2601-06196v3.md` |
| Ricco et al., *A Geometric Analysis of Small-sized Language Model Hallucinations* (APORIA) | ICML 2026, PMLR 306 | `papers/digests/2602-14778v3.md` |
| Nielsen et al., *Hallucination as an Anomaly: Dynamic Intervention via Probabilistic Circuits* (PCNET/PC-LDCD) | arXiv v1 preprint | `papers/digests/2605-05953v1.md` |
| Shamsi, *Density Ridge Selective Prediction for LLM and VLM Hallucination Detection under Calibration-Label Scarcity* | five-page arXiv extended abstract | `papers/digests/2606-10198v2.md` |
| He, Wang & Mishne, *Local Manifold Explanations with Tangent Space Regression* | PMLR 334, TAG-DS 2026 | `papers/digests/local-manifold-explanations.md` |

Related already-cached work used in the comparison includes Mind the Gap, GeoFaith,
HARP, HaloScope, Semantic Entropy Probes, repeated-sampling uncertainty, and the
LTSREx-author line c-STG/DiSC/IC-PML/LEGO. These were not counted as newly digested
PDFs in this batch.

## What “new information beyond P(X)” means here

Let \(X\in\mathbb R^{n\times m}\) be the current mixed-v2, family, atomic, or causal
DSP feature matrix. Any method that constructs neighborhoods, tangents, kernels,
Laplacians, densities, ridges, or clusters using only \(X\) is a function of the same
marginal distribution \(P(X)\). In the observationally equivalent worlds already used
in this project, target and nuisance can exchange semantics while preserving \(P(X)\).
No such method can determine which latent factor is correctness in both worlds.

The new papers introduce only three possible ways out:

| Source | Outside current P(X)? | Where target semantics comes from | Current compatibility |
|---|:---:|---|---|
| Alternative response texts across generations | yes | APORIA: prompt-local correctness labels; a label-free cohesion heuristic has no guarantee | black-box but multi-pass and retrospective |
| Residual-stream/hidden-state trajectories | yes, as observables | PCNET: both-class contrastive labels; Ridge: correct-only/both-class calibration | white-box and usually supervised |
| Verified target/nuisance interventions | potentially, if A6 passes | intervention contract, not geometry | candidate compatible with affine one-pass deployment through A6/PTNI |

This is why a high AUROC from a “factual manifold” is not automatically evidence that
the manifold itself identifies factuality. If the correct subset or Fisher direction was
selected using \(Y\), the label supplied the semantics.

## Cross-paper decision table

| Paper | Geometric object | New information source | Labels required | Sampling / inference cost | Global applicability | Localization applicability | Early applicability | Compatible IU-PCR variant | Replace / augment | Primary identifiability risk | Implementation status here | Recommendation |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Tiberi–Sompolinsky | category manifolds and target-local linear regressors | none before target labels; useful environment structure after | continuous target + category | huge offline activation population | target-direction transport audit after A6 | label-only retrospective audit | budget-wise direction-drift audit only | PTNI-IU / NRM after a verified direction | augment diagnostics | local directions are defined by labels; no per-query router | not implemented | diagnostic only |
| GA-ICL | mean-pooled final hidden states, local PCA patches, class prototypes | hidden states + labelled ICL examples + judge pass | yes, both classes | 3–4 h training per model/task + extra ICL inference | supervised critic tier only | none | none | none without changing access tier | external comparator | metric learning/proxy labels—not manifold—may explain gain; HaluEval firewall unclear | not implemented | reject |
| APORIA | embeddings of 150 sibling responses per prompt | repeated-generation semantic content | operational method needs prompt-local both-class labels | 150 generations + encoder + judge; 30–50 labelled siblings | possible multi-pass premise signal | none | none | sampling residual beside IU, not one-pass IU | augment Global only | coherent hallucinations; no prompt transfer; label-conditioned selection | sampling fusion exists, APORIA geometry untested | future pilot only if multi-pass tier opens |
| PCNET | final-layer residual states; learned PC density score | hidden states and counterfactual top-k next states | yes, 250+250 and F1 threshold labels | white-box training; top-8 lookahead when gated | supervised white-box comparator | future-work only | causal architecture idea, no valid prefix evaluation | none under current contract | separate detector / score residual | labelled contrastive density is presented as factual manifold; token-label mapping unclear | not implemented | reject current tier |
| Density Ridge | 6-D hidden-trajectory kinematics and 1-D SCMS ridge | hidden trajectories; ridge itself adds no source | correct-only plus stated both-class front-end | 5 generations, 200 calibration queries, white-box | supervised sampling comparator only | none | not causal | supervised nonlinear residual only | separate score | correct ridge is target-labelled; variant selection and population unclear | not implemented | reject implementation |
| LTSREx / LEGO | local tangent spaces and explanatory coefficient fields | none; arbitrary dictionary explains chosen geometry | no target labels, but target relevance is absent | offline kNN/LPCA or LEGO + sparse regressions + graph smoothing | nuisance audit | onset-blurring diagnostic risk | transductive/non-causal as published | post-hoc audit of any frozen IU geometry | augment diagnostics | can explain a nuisance manifold perfectly | LTSREx digested; no algorithm implementation | diagnostic only |
| c-STG | context-dependent supervised stochastic gates | labels; context in our run was derived from X | yes | supervised training | tested and failed robust gate | tested and failed | tested and failed | oriented family contributions | attempted router | flexible nonlinear classifier can masquerade as reliability routing | family Global/Local/Early tested | stop on current contexts |
| DiSC-style differential geometry | feature graphs under paired conditions | condition/intervention contrast | no natural labels if condition semantics are verified | paired calibration conditions; one-pass deployment possible after distillation | high relevance through A6/PTNI | needs first-error-changing interventions not available | needs prefix-valid interventions not available | PTNI-IU then optional group audit | augment intervention direction | condition may alter nuisance rather than target unless contract is verified | not implemented; A6 supplies the stronger contract | high-priority conditional audit, not a new branch |

## Mapping to the existing methods and failures

| Existing method/result | What it already established | Consequence for the new papers |
|---|---|---|
| IU-PCR | one global affine direction from unlabeled covariance is robust but approximate | any new score must add information, not re-estimate the same covariance |
| DUFS-LIU mixed-v2 | smooth graph structure can add small gains, but broad/online transfer is weak | a density/tangent method cannot claim correctness from smoothness |
| Family-NRM / NRM-CS-IU | a provenance quotient supplies a useful label-free correction; atomic removal loses | local geometry needs a target anchor before replacing provenance |
| Atomic Operator | stable atomic geometry was negatively associated with utility; top proxy lost | “more stable tangent/ridge” is not a solution |
| Atomic NRM + supervised atomic ceiling | atomic features contain target headroom, but label-free geometry misses its direction | atomic contextual routing needs an external context, not more capacity |
| Family relevance | IU-rank strata expose +2.833pp oracle headroom; the proposed gate fails | specialization exists, but no deployable reliability key was identified |
| DSP-contextual IU | failed even positive and coherent-nuisance simulations | do not open LPCA/LTSREx/LEGO on DSP alone |
| Family c-STG | Global rank-router loses to global LR; Local/Early DSP routers fail | flexible supervised gating does not rescue current context |
| Unified-28 | Local transfer improves; Global/Early regress | one geometry is not justified across the three targets |
| Dedicated Global/Local/Early heads | two heads beat one; Early transfer remains weak | conclusions must remain task-specific |

The repeated-measurement reliability result is also directly relevant: bootstrap stability
separated reproducible structure, but it did not improve target ranking. APORIA differs
only because it obtains *new response content*; a bootstrap of the same trace does not.

## Candidate 1 — intervention-defined target geometry plus a transport audit

**Status:** scientifically strongest candidate, but still unproven and already represented
by frozen A6/PTNI. S0a passed mechanics only; the new literature does not authorize
changing that protocol.

For a balanced target/nuisance crossover, define factorial feature effects

\[
\Delta_T=\frac14\sum_{t,n\in\{-1,+1\}}t\,\mu_{tn},\qquad
\Delta_N=\frac14\sum_{t,n}n\,\mu_{tn},
\]

where \(\mu_{tn}\) is the mean atomic feature vector under target state \(t\) and
nuisance rendering \(n\). A nuisance-whitened target direction is

\[
u=\Sigma_N^{-1/2}\Delta_T,
\qquad
q=P_{\perp w_{IU}}u,
\qquad
s_\alpha(x)=s_{IU}(x)+\alpha q^\top x,
\]

with an exact \(\alpha=0\) IU fallback. A later Tiberi-style diagnostic can fit local
\(q_e\) by intervention environment and decompose the loss of one global \(q\) into
environment-centroid, scale, and direction-alignment terms.

- **Calibration input:** mechanically verified target-changing and nuisance-only pairs.
- **Deployment input:** the ordinary one-pass mixed-v2 vector only; no intervention.
- **Frozen IU content:** feature transforms, directions, IU score, and exact fallback.
- **Labels learned:** no benchmark correctness label; semantics come from verified
  intervention contracts. Natural labels remain veto/confirmation only.
- **Complexity:** calibration multiplies conditions; deployment remains affine \(O(m)\).
- **Failure world:** intervention changes response style/length rather than target;
  teacher-forced effect does not transport to natural errors; target directions flip by
  environment.
- **Falsification control:** nuisance-as-target, conditional sign permutation, placebos,
  held target/nuisance families, and observational-equivalence worlds.
- **Fair audit baselines:** pooled/global PTNI, partial-pooling environment interactions,
  norm-matched shrinkage, and permuted environment identity. Tiberi's decomposition is
  explanatory and is not itself expected to beat these scores.

**Relationship to papers:** DiSC can summarize which feature graph changes between
conditions, and Tiberi can audit whether known-environment local target directions
support one global readout. Tiberi supplies neither a per-query environment assignment
nor a score improvement. Neither paper should precede or modify A6. LTSREx may explain
a direction only after it is verified.

## Candidate 2 — repeated-generation semantic reliability beside IU

For \(K\) sibling completions and response embeddings \(e_{ik}\), define a label-free
dispersion and primary-response conformity:

\[
c_i=\operatorname{median}_{k<\ell}d(e_{ik},e_{i\ell}),
\qquad
a_i=d(e_{i1},\operatorname{medoid}(e_{i1:K})).
\]

The simplest residual candidate is

\[
s_i=z(s_{IU,i})+\beta_1z(c_i)+\beta_2z(a_i),
\]

or, for a router premise only, test whether \((c_i,a_i)\) predicts held-question
family/atomic utility beyond IU rank, length, family disagreement, answer agreement,
and Semantic Entropy.

In a label-blind premise test, the signs and magnitudes of \(\beta_1,\beta_2\) must be
fixed before correctness is opened. Learning them from correctness converts the object
into supervised fusion. Also, \(c_i\) may encode prompt difficulty shared by every
sibling rather than the reliability of the designated primary response.

- **Calibration/deployment input:** \(K\) completed generations and an external sentence
  encoder; no causal prefix.
- **Frozen IU content:** the primary-generation IU score and all feature orientations.
- **Labels learned:** APORIA's actual FDA uses labels; the proposed premise begins
  label-blind and freezes before any correctness evaluation.
- **Complexity:** \(K\) generation passes and \(O(K^2)\) distances per prompt.
- **Failure world:** a coherent misconception produces many mutually similar wrong
  answers; diverse correct paraphrases look unreliable.
- **Control:** simple answer agreement, Self-Consistency, Semantic Entropy, shuffled
  siblings, within-prompt label permutation, and coherent-hallucination strata.
- **Incremental bar:** geometry must beat the simple sampling baselines; otherwise its
  “manifold” language adds no value.

This is a legitimate separate multi-pass Global research tier, but not the recommended
next core-method experiment. The two local K=10 raw files are Git-LFS pointers rather
than available response caches, so even a faithful retrospective APORIA premise test
would require artifact retrieval and new embedding inference. That is outside this task.

## External supervised comparator — factual-manifold anomaly residual

The generic score is

\[
a(x)=\operatorname{dist}\bigl(\phi(x),\widehat{\mathcal M}^{+}\bigr)
\quad\text{or}\quad
a(x)=-\log \widehat p\bigl(\phi(x)\mid Y=1\bigr).
\]

PCNET learns \(\phi\) contrastively from both classes; Density Ridge fits
\(\widehat{\mathcal M}^{+}\) on labelled-correct states. A possible IU residual would be
\(s=z(s_{IU})+\beta z(a)\).

- **New information:** yes only if \(\phi\) consumes residual-stream trajectories.
- **Labels:** unavoidable in the published variants.
- **Calibration/deployment:** labelled, model-specific hidden states for calibration;
  white-box full-response or prefix states at deployment. PCNET correction additionally
  evaluates counterfactual next-token states. This is outside the current access tier.
- **Frozen IU content:** an IU residual could leave the IU score and orientations frozen,
  but that construction is not evaluated in either paper.
- **Complexity:** white-box state collection and supervised fitting; Density Ridge also
  uses five generations, while PC-LDCD uses top-eight lookahead when activated.
- **Fair baselines:** same-access and same-label-budget LR/MLP, plus one-class
  Mahalanobis, kNN, and one-class density on the same hidden states. Family IU or
  ordinary IU alone is not a fair peer.
- **Failure world:** correct low-density subpopulation, domain shift, or a coherent dense
  wrong mode.
- **Falsification control:** prompt/source-grouped splits, equal label budgets, fixed
  density variant, label permutation, and prefix grouping for any Early claim.
- **Decision:** retain as literature comparators only and reject implementation in the
  current access tier. On current DSP features either method is merely a supervised
  nonlinear control over the same matrix.

## Candidate 3 — hierarchical contextual atomic routing

The requested safe parameterization is

\[
g_{ij}=\operatorname{clip}
\left(g_{\mathrm{family}(j)}(z_i)+\delta_j(z_i),0,1\right),
\qquad \|\delta_j\|\text{ strongly penalized}.
\]

The baseline must be an atomic global head. A family baseline would confound routing
with increased feature resolution. However, if \(z_i\) is only IU rank, disagreement,
length, or DSP derived from the same atoms, this model adds capacity but no target
information. The independent review of the earlier proposal therefore recommends a
cheaper falsifier before any atomic c-STG:

\[
s_i=b_i+r_i^\top\delta
+z_i\sum_j r_{ij}\bigl(\gamma_{g(j)}+\eta_j\bigr),
\]

with family interactions \(\gamma\), heavily shrunk atomic residuals \(\eta\), and
training-fold-only residualization. Required controls are atomic-global, additive
atomic+context, family-tied interaction, unrestricted atomic interaction, context
permutation, family-assignment permutation, nested fixed-budget tuning, seed stability,
and no post-label architecture selection.

- **Calibration/deployment:** a frozen external context \(z\) plus atomic IU
  contributions at both stages. Current IU-rank/DSP contexts do not qualify as external.
- **Frozen IU content:** atomic directions and base contributions remain fixed; gates
  may only redistribute leverage.
- **Labels:** supervised mechanism test only; no claim of label-free target discovery.
- **Complexity:** \(O(m)\) inference, but high statistical capacity from sample-specific
  interactions even with hierarchical shrinkage.
- **Failure world:** the gate memorizes dataset, length, or prompt-family shortcuts and
  appears useful without predicting held-family reliability.

**Decision:** do not run on the current contexts. It repeats the c-STG question with more
parameters and does not meet the “new information beyond \(P(X)\)” gate. Reconsider only
if candidate 1 or 2 supplies an independently frozen context \(z\).

## Candidate 4 — LTSREx/LEGO as an explanation and veto layer

Given a **frozen target-relevant** geometry, LTSREx can regress local tangent bases on a
dictionary containing length, budget, model identity, missingness, and causal DSP states.
It is useful if the question is “what locally parameterizes this selected geometry?”

For tangent basis \(B_i\) and dictionary Jacobian \(J_i\), its local explanatory object
is schematically

\[
A_i=\arg\min_A\|B_i-J_iA\|_F^2+\lambda\|A\|_1,
\]

optionally followed by graph smoothing of the coefficient field. The coefficients
explain the selected tangent; they are not hallucination scores.

- **Calibration input:** a frozen target-relevant score/geometry plus a predeclared
  dictionary of possible nuisance/context variables.
- **Deployment input:** none in the recommended use; this is an offline audit, not a
  detector. IU features, directions, and scores remain frozen.
- **Labels:** none required for explanation, but target relevance must have been
  established upstream; LTSREx does not establish it.
- **Complexity:** kNN/LPCA per point, sparse local regressions, and possibly a graph
  solve; LEGO adds global spectral and local pseudoinverse work.
- **Failure world:** it stably explains a length, budget, model, or missingness manifold
  and is misread as factuality evidence.
- **Control:** dictionary permutation, nuisance-dominance veto, tangent-angle bootstrap,
  and a simple LPCA baseline before LEGO.
- Do not use connection-Laplacian smoothing for onset detection without a separate
  causal formulation; it is transductive and may blur abrupt errors.
- LEGO opens only if tangent instability is empirically the bottleneck.

## Previously tested and closed — evidence-drop/onset geometry

For a causal risk stream \(u_t\), the representative coordinate

\[
d_t=\operatorname{EWMA}(u)_t-\operatorname{EWMA}(u)_{t-h},\qquad
o=\min\{t:d_t>\tau\text{ for }p\text{ steps}\}
\]

uses only current/past telemetry at deployment and preserves the base IU orientations.
Its calibration choices are \(h,\tau,p\); any label-based selection must remain inside
grouped training folds. The key controls are raw level, level-plus-onset, future-suffix
replacement, and chunk replay. This direction is not reopened: onset-only and
level-plus-onset lost to raw level for Localization, richer DSP context did not predict
held reliability, and Mind the Gap is already a direct evidence-drop baseline. The
failure world is a smooth nuisance transition or delayed score response that looks like
an error onset. A new hidden-state/intervention observable could define a new study, but
a ridge over the existing causal trajectories cannot.

## Separate rankings by task

### Global completed-response hallucination detection

1. **Existing next gate, not a positive result:** A6/PTNI intervention-defined direction;
   add a separate Tiberi-style transport audit only after A6 passes.
2. **Future multi-pass pilot:** label-blind APORIA-style cohesion, but only in a separate
   cost tier and only against Self-Consistency/Semantic Entropy.
3. **Diagnostic:** LTSREx/LEGO on a verified target geometry.
4. **Reject current tier:** PCNET, Density Ridge, GA-ICL, and atomic c-STG on current
   contexts.

### First-error Localization

1. **Retain current dedicated raw-level/family-six evidence and direct baselines.**
2. **Diagnostic only:** LTSREx after a verified first-error-changing intervention exists.
3. **Reject as evidence:** APORIA and Density Ridge are completed-response methods;
   PCNET does not evaluate first error, and Mind-the-Gap-style onset is already tested.

No new paper justifies changing the current Localization head.

### Causal Early Detection

1. **Retain IU28 / the frozen two-head architecture as the existing direct bar.**
2. **Research hypothesis only:** PCNET's causal gate, under a new supervised white-box
   tier, requires a new prefix-labelled protocol before it can be compared.
3. **Reject:** APORIA and Density Ridge require completed sibling trajectories; LTSREx
   as published is transductive; current DSP context already failed.

No new paper supplies a causal, label-free early signal.

## Staged closure plan, not a new method branch

The current \(X\)-geometry question is already closed by the project's nuisance,
atomic, DSP-router, and c-STG failures. No further ridge/tangent/router run on the same
matrix is recommended. The only cheap future falsifier concerns APORIA's genuinely new
repeated-response observable, and it opens only if its public response artifact is made
locally available under a separately approved protocol.

### S0 — current \(X\)-geometry closure

- Record the existing observational-equivalence, coherent-nuisance, atomic-proxy,
  DSP-router, and c-STG results as the closure evidence.
- Gate: no new current \(X\)-geometry method proceeds without an independent observable.

### S1 — artifact and firewall feasibility

- Pre-hash one primary response and fixed \(K\) siblings for each model-prompt from
  SOCRATES; do not select prompts by class balance and do not use Fisher or labels.
- Freeze raw medoid distance and pairwise semantic cohesion before opening correctness.
- If prompt identities, sibling membership, or prompt-disjoint splits cannot be
  reconstructed exactly, stop.

### S2 — CPU-only repeated-observable premise

- Evaluate on held prompts and held model families against answer agreement,
  self-consistency, Semantic Entropy, and response length.
- Primary metric: within-fold AUROC averaged across grouped folds, never AUROC on
  concatenated OOF predictions; uncertainty by prompt bootstrap.
- Gate: a frozen cohesion statistic must improve the strongest simple sampling baseline
  with a grouped lower confidence bound above zero and no coherent-hallucination collapse.
  Failure closes APORIA. A pass establishes only premise evidence, not a method.

### S3 — decision boundary

- A passed S2 may justify a separate request for repeated-generation inference on our
  datasets. It does not modify IU-PCR, A6, Localization, or Early.
- Tiberi/LTSREx audits remain dormant until A6 independently passes. If any trained
  component is later introduced, bootstrap must repeat the fit rather than treating
  fixed OOF predictions as variance-free.

Required artifacts are protocol and source hashes, prompt-level tables, fold metrics,
grouped intervals, negative controls, `AUDIT.json`, `DECISION.json`,
`RUN_MANIFEST.json`, and a report. All outputs remain retrospective premise evidence.

## Questions the literature cannot answer for us

1. Does any label-free repeated-generation cohesion statistic add value beyond ordinary
   answer agreement or Semantic Entropy on our model/dataset families?
2. Do A6 intervention-derived directions transfer from controlled reciprocal tasks to
   natural hallucinations?
3. If A6 works, is one global affine direction adequate, or do target directions rotate
   across environments enough to require a declared environment—not per-sample—adapter?
4. Can first-error-changing and prefix-valid nuisance interventions be constructed without
   smuggling benchmark labels into Localization/Early?
5. Are hidden-state “factual manifolds” still competitive against capacity-matched
   supervised LR/MLP controls under grouped, prompt-disjoint evaluation?
6. Does any density-ridge benefit remain under one frozen variant, realistic unfiltered
   query populations, and a test set substantially larger than 60?
7. Can a future external context provide atomic routing information that is incremental
   over the atomic global head, rather than merely extra nonlinear classifier capacity?

## Independent adversarial review

An independent agent re-read the draft against the papers and project failures. Its
verdict agreed with `DIAGNOSTIC ONLY` but identified three overclaims, all accepted:

1. APORIA/PCNET/Ridge add observables, not label-free target information.
2. A6 is a candidate source whose S0a mechanics passed; S0b has not established a
   target direction.
3. Tiberi is a post-hoc known-environment decomposition, not a router or score improver.

The review also recommended removing evidence-drop/onset from the candidate roster
because that is an already-tested project branch, and replacing the promotable Tiberi
program with the closure plan above. It tightened the fair controls for hidden-state
density methods and Tiberi, and noted that an atomic supervised LR is a ceiling rather
than a same-contract gate peer. The cheapest valid unresolved falsifier is a label-blind
SOCRATES cohesion test on held prompts/model families; it requires a separately approved
artifact workflow and does not justify `PREPARE NEW INFERENCE` now.
