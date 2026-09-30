# Independent direction review: localization L-SML / Joint L-SML

Date: 2026-09-17. Review only; no new model, scorer, benchmark or quality sweep run.

LATEST USER DECISION: Omri rejects digit-based disagreement as inconsistent
with the original method. Earlier digit-inclusive recommendations below are
historical and superseded. Remove digit dependencies from future inputs, gates
and orientation anchors. The optional bank below now has 50 digit-free streams:
47 retained streams plus three general distribution features.
No digit-free replacement has been fitted or evaluated in this review.

## Recommendation

Continue dependent hierarchical fusion, but isolate group discovery from factor
estimation and score readout. Claude's proposed within-group SML is the strongest
immediate follow-up. Direct use of the residual group loading u is not an
equivalent alternative and needs additional identification/orientation rules.
Neither the claim that readout is the only remaining weakness nor a forecast of
43% PB across all rosters is established.

The most informative missing comparison is Continuous L-SML's within-group
readout using Joint's automatically discovered groups, with the existing outer
readout, folds, normalizations, orientation and gate frozen. This is the reverse
of the existing joint_lsmlgroups_hier arm.

## Evidence inspected

- Local Claude conversation 3ccf0014-bd40-400c-ba91-2d092cbb0266, including the
  completed Step398 discussion and the user's rejection of plain averaging as
  the proposed method.
- Prior Codex thread 01a0a1ce-5546-77c0-8531-13ed87fe606e, including the literature
  list, CCA/shrinkage discussions, and later alternative-view experiments.
- Refreshed origin branch references. Latest atlas head inspected:
  bb71bb3d21841f8011aec3e68121c1165e226d15.
- .worktrees/fusion-independence-atlas-v1/docs/experiments/JOINT_REDUNDANCY_ROBUSTNESS_V1.md
- That worktree's results/joint_redundancy_robustness_v1/{RUN,RUN_L24,RUN_COMBINED}.json
- That worktree's results/lsml_gate_locator_research_v1/SUMMARY_HE.md and
  results/fusion_independence_atlas_v1/report/REPORT.md.
- Source: joint_lsml.py, joint_pair_extension.py, joint_pair_jacobian.py,
  fusion_utils.py, lsml_gate_locator_research.py and the redundancy driver.
- Temporal-research PROGRESS and archived TCN/queue state files.

Reported benchmark numbers were read from the artifacts, not independently
recomputed from predictions in this review. The numerical replay claims below
belong to the original experiments.

## Current development evidence

| Method | L08 PB / PRMB within | L24 PB / PRMB within |
|---|---|---|
| Continuous L-SML | 43.7402 / .7781 | 38.9074 / .7508 |
| Joint, own groups, hierarchical | 43.5862 / .7769 | 43.1289 / .7754 |
| Joint on Continuous groups, hierarchical | 43.8135 / .7783 | 38.5442 / .7482 |
| IU-PCR | 38.5701 / .7486 | 26.0678 / .5415 |

PB is macro F1 in percent, not SLA. The incumbent is 43.2546 / .776036.
L08 Continuous versus incumbent: +.4856pp PB, 95% CI [-.5679,+1.5266];
within +.002107, CI [-.000555,+.004851]. No confirmed successor.
L24 hierarchical Joint versus L24 Continuous: +4.22pp [+2.91,+5.54].
These are development comparisons conditional on the existing research choices,
not untouched confirmation. A CI containing zero does not establish equivalence.

## Corrections and qualifications to the interpretation

1. **The outer stage sometimes explicitly averages.**
   The redundancy driver passes small_m_guard=True to hierarchical_joint_weights.
   In fusion_utils.sml_fuse_signed, three inputs trigger w_i=1/(3*std_i),
   bypassing the spectral eigen-solve. The saved cross_small_m_guarded flag is
   true in all five L08 and L24 folds. Continuous uses the same guard where
   applicable. Thus the high-performing procedure discovers groups and fits
   within-group quantities, but does not learn unequal outer reliability
   weights in these K=3 cases. It is not a plain average of all raw features;
   it also cannot be presented as fully learned weighting at both levels.
   Removing the guard after seeing results would be a new candidate, not a fix
   to the historical run. Zero residual degrees of freedom is distinct from
   proving mathematical nonidentifiability of every three-view model.

2. **K discovery is constrained and not literally smallest-stable-K selection.**
   The driver supplies k_range=(3,4). The implementation ranks admissible choices
   by median LOAO-to-consensus ARI, then mean ARI, then minimum ARI, then smaller
   K. Smaller K is a final tiebreak. L11 fold 3 actually selects K=4; the other
   inspected Joint rosters/folds select K=3. Continuous has a different K search
   and chooses K=7 on L24. This is a legitimate algorithm comparison, but it
   confounds attribution to factor fitting versus grouping/complexity control.

3. **u is not an identified correctness direction.**
   The model's within-group residual factor can encode common error or nuisance
   as well as useful minority evidence. For a pair, only u_i*u_j is identified
   from its off-diagonal residual. The pair extension explicitly records
   latent_pair_loadings_identified=False and constructs a feasible representative
   using equal fractions of residual-variance budgets. Using that representative
   directly as a score changes the statistical contract. Within-group SML on the
   observed block avoids claiming recovery of that unidentified pair direction,
   though it still has its own assumptions and can favor nuisance variation.

4. **The remaining problem is broader than one readout.**
   Group membership changes across rosters; K selection and rare-view direction
   remain relevant. A clean digit group correlates with the successful L24 result,
   but the current comparisons do not prove exclusive causal mediation. The
   atlas' failure to establish error-independence is not proof that all fusion
   must fail, nor a direct test of every conditional latent-factor assumption.

5. **Access scope must remain explicit.**
   The Step397/398 locator fits use other training answers in source-excluded
   folds. These are not strict answer-only fits. Labels select some development
   roster choices even though the fusion fits themselves do not use labels.

## One bounded next stage

Keep all 13,769 answers, the five existing source folds, current gate, stream
definitions, sign/orientation, normalization and readout mapping fixed. Compare
the existing Joint-groups/v-readout arm with Joint-groups/within-group-SML. Keep
Continuous, the incumbent, and simple aggregation as controls; do not promote
the control as the requested algorithm.

Use L24 as the declared broad-bank question and L08/L11/L14x as fixed robustness
panels. State one primary contrast, paired source-group uncertainty on PB and
PRMB within, and explicit coverage/failures. Do not select whichever roster
improves after inspecting results. Where possible, hold discovered partitions
fixed while comparing readouts; separately match K restrictions before claiming
that the factor model itself explains the gain.

If the replacement wins, attribute the result to automatic grouping plus
hierarchical spectral readout unless an additional matched factor ablation
demonstrates a Joint-specific benefit. If it fails, do not automatically escalate
to more factors or another roster search.

## Independent research direction

Make robustness to redundant descriptions an explicit design criterion for
hierarchical fusion. Begin with exact-duplicate and feature-permutation tests,
then a fixed near-duplicate perturbation. Learn grouping without labels and
measure whether copying an existing stream changes predictions or dilutes a
useful minority family. Exact copies can be collapsed by a deterministic rule;
near-duplicate invariance is a research question, not an existing guarantee.
The new streams in L11/L14x are not literal duplicate columns, so their current
result alone does not establish this invariance property.

This criterion is target-free and falsifiable, but robustness is not itself
correctness. The final method still needs a frozen, untouched localization test.

## Prior papers and what to reuse now

- Jaffe, Fetaya, Nadler, Jiang & Kluger, *Unsupervised Ensemble Learning with
  Dependent Classifiers*: https://arxiv.org/abs/1510.05830. First priority: the
  existing within-group/virtual-classifier mechanism, adapted transparently to
  continuous features. Its original guarantees do not automatically transfer.
- *GroupFS*: https://arxiv.org/abs/2511.09166. Relevant principle: discover groups
  and handle redundancy jointly. Full GroupFS is a later alternative if the
  current partition mechanism is shown to be the bottleneck, not a new broad
  selector sweep. Its clustering objective is not a localization guarantee.
- Dror et al., *Unsupervised Ensemble Regression*:
  https://arxiv.org/abs/1703.02965. Keep IU as a matched historical control;
  dependence assumptions and target-moment identification remain substantive.
- *Multi-Target Shrinkage*: https://arxiv.org/abs/1412.2041. Useful for covariance
  estimation when instability is demonstrated; not a solution to choosing the
  wrong task direction. Existing local shrinkage results do not justify a sweep.
- Ridge/TCN: preserve the existing useful innovation channels. The archived
  aligned TCN run is COMPLETE_REVIEWED; its advantage over ridge was uncertain.
  Contextual native IU subsequently lost, so earlier CCA-based proposals should
  not be repeated as though those results were unknown.
- *Diverging Flows*: https://arxiv.org/abs/2602.13061. Still a distinct source of
  possible evidence, not a redundancy/readout repair. Archived queue state is
  PAUSED_BETWEEN_JOBS (4/90), and atlas lists FM/DiFlo/DOT as PENDING_EXPANSION;
  this review did not verify live processes or conclude the methods failed.
- CRBM, tensor/position fusion, network regularization, LOCA and KalmanNet stay
  lower priority unless a specific new signal or identifiable task mechanism
  motivates their additional complexity. Earlier scalar-feature failures are
  not blanket impossibility results for all future localization adaptations.

## SLA limitation

After fetching origin and searching commit subjects and recent diffs, the
specific new SLA commit mentioned by the user was not identified. A branch/SHA
or repository URL is still required to audit it. No conclusion here relies on
that unreviewed update. Retain PB F1 and PRMB within, and add the exact frozen
SLA contract as a separate metric once located, with explicit handling of
clean answers, gate abstentions and the denominator. Changing the metric does
not turn development data into confirmation.

## User clarifications and iterative-selection proposal, 2026-09-17

Omri clarified that averaging is acceptable inside a group when the algorithm
cannot be applied further. This is not blanket rejection of every averaging
fallback, and does not automatically specify the desired between-group rule.
He accepts exploring the u_g idea with the stated qualifications. Redundancy
robustness is also his research direction; the earlier wording must not imply
that it originated exclusively in this review.

New proposal from Omri: repeatedly refit fusion and remove a feature that does
not support the structural hypothesis. Suggested diagnostics are (a) the group
with greatest weight entropy, removing its highest-residual feature, and (b)
unstable group membership across fits or K values. This is a discussion proposal,
not a frozen experiment or an authorization to launch a sweep.

Assessment: iterative structural selection is worth a bounded test, but weight
entropy is not an error criterion. Uniform weights can describe a healthy group;
small-group guards can impose that entropy mechanically. If used diagnostically,
normalize entropy by log(group size), use magnitude contributions on comparable
feature scales, and inspect signs separately. A covariance-model residual must
also be distinguished from the temporal prediction innovations already used as
features. A high residual may be useful new evidence or noise.

Prefer resampling stability at a fixed K before judging variation across K.
Compare pairwise co-assignment, not numeric cluster labels. A stable split as K
increases is not evidence of a bad feature. Before deletion, distinguish an
incorrect group assignment from a dispensable feature; test whether the feature
fits another existing group under the same structural contract.

A conservative candidate would refit after one proposed deletion, comparing
held-out structural error on the SAME remaining covariance entries before and
after refitting, plus partition stability and identifiability. Freeze the
acceptance/stopping rule, retain an immutable full-bank comparator, and report
the selected path. Do not automatically prefer smaller raw residuals on smaller
matrices or continue until a trivially exactly fitted small model remains.
These structural checks do not guarantee preservation of useful sparse evidence.

Historical connections verified in source: selectors/a1_residual.py already
contains relative-residual and greedy forward selectors and explicitly warns
that residual quality need not predict AUROC; its historical loading-scale
defaults must not be copied silently. selectors/a7_iter_consensus.py performs
iterative pseudo-target consensus refinement. Neither establishes that Omri's
proposed backward, group-stability-aware selector has already been tested on
the current localization bank. GroupFS is related conceptually but optimizes a
different graph/group-selection objective.

## Broad-bank research objective clarified by Omri

Omri wants to be able to add historical features and let the algorithm select
or accommodate them, rather than depend on a carefully chosen roster. His
concrete example is supplying log probabilities at every rank 1..15. This is a
research-design preference; no broad-bank experiment was launched here.

Source inspection confirms the current pipeline extracts all rank-1..15
probabilities, but the final L08 uses ranks 1 and 9 and L24 uses only ranks
1,3,9,10. These are direct probability risk features (1-p1 and p2..p15), not
raw log probabilities. A full log-probability bank is therefore not identical
to merely restoring omitted columns from the current risk bank.

Other development-specific asymmetries in L08/L24:
- Rank1 Top10 versus rank9 Top8; digit Top2 versus digit innovation Top1.
- VE0.75 prefix innovation, but Renyi a0.25 as a level feature.
- L24 retains an irregular subset of Renyi orders and readout combinations,
  rather than all raw families. 'All24' means all eligible step representatives
  in the consolidated atlas, not every feature ever extracted.
- Digit token-clock innovation is retained; opportunity-clock innovation is not.
- Numeric specialization uses ASCII single-digit token IDs, with a role distinct
  from general disagreement at all provided-token positions.
- Selected/provided-token surprisal, gap and rank were available but do not
  appear as a complete family in L08.
- Renormalized top15 shape statistics and raw tail-mass features mix information
  scopes; top15 versus top50 is another declared representation choice.
- The locator's overall sign is oriented to a selected digit anchor, and
  per-answer standardization makes feature variability relative to that answer.

Recommended input contract for a future study: define complete, finite families
by measurement rather than performance (e.g. all rank1..15 log probabilities,
provided-token confidence, retained/tail mass, registered distribution summaries,
and compatible temporal features). Hold a common step readout initially so the
test does not conflate feature selection with choosing a different pooling rule
for every feature. Do not expand all ranks x all transforms x all windows at once.

Keep the dependence among deterministic transforms explicit; adding many
transforms does not create independent evidence. Historical answer-level FFT,
EPR and CUSUM summaries cannot be copied to every step and called local features;
they need a declared token/step/window contract or a separate answer-gate role.
Nested roster selection is useful development evaluation, but it does not
establish that the fitting algorithm independently selected a broad feature bank.

## Optional feature-bank proposal (discussion, not frozen or run)

Omri accepts all 15 existing probability-rank features and a uniform step
readout. He clarified that different Renyi/escort orders were motivated by
complementary behavior along the generation trajectory; this is a legitimate
hypothesis, not arbitrary parameter selection. Different orders directly
emphasize different parts of the token distribution, and temporal specialization
still requires evidence.

Revised finite bank after the user rejected digit channels and requested three
additional features: 35 base streams plus 15 temporal streams = 50.
- 15 existing direct-rank risk probabilities (1-p1, p2..p15); do not silently
  substitute log probabilities or include both representations in the first run.
- 4 provided-token views: surprisal, top1 log-probability gap, censored top50
  rank, and saved-top50 mass above the provided token. The latter is a lower
  bound when that token falls outside top50; rank is censored, not exact there.
- 2 tail masses: outside top15 and outside top50.
- 7 distribution-shape views: Renyi orders .25,.5,1,2,4,infinity, plus the
  existing H0lim view (not ordinary order-zero entropy, which is constant at
  fixed support).
- 6 escort-varentropy views: orders 0,.5,.75,1,2,4.
- 13 strictly prior prefix-mean innovations for all 13 shape/varentropy views.
- 1 top-two ambiguity ratio p2/p1, equivalently exp(logp2-logp1).
- 1 adjacent top15 identity-turnover feature (definition below).
- 1 adjacent normalized-top50 Jensen-Shannon divergence (definition below).

Suggested uniform readout: Top10 on each stream's declared valid positions,
mean of all valid positions if fewer than ten; no valid positions means missing
evidence with an explicit mask, not verified correctness. Establish and freeze
non-digit orientation conventions and a non-digit gate before evaluation.
Historical digit-inclusive scores cannot be carried over to this new bank.

Historical explanation only, not an eligible candidate: the old digit event
requires BOTH provided and argmax tokens to be verified
single ASCII digits and their IDs to differ. A provided digit whose argmax is
a non-digit is not counted. For 100 past tokens, 4 digit opportunities and 2
disagreements, the token-clock background is .02 and opportunity-clock .5.
Both digit clocks are now excluded by the user's decision.

Optional later families, separately declared: log-rank representation replacing
the 15 probability columns and local versions of historical
CUSUM/variance/spectral features. These are proposals, not verified newly
available features or a launch of a combinatorial expansion.

## Three general replacements for the retired digit channels

Omri requested three additional features and explained that the research had
become too focused on preserving digit because it supplied complementary
evidence. No feature should have protected status. The atlas did not establish
independent errors, so past digit complementarity must not be relabeled as a
proved independence result.

These definitions extend the proposal only; no extraction or experiment ran.

1. **Top-two ambiguity**: p2/p1 = exp(logp2-logp1), with values near one meaning
   the two leading candidates compete closely. This is a nonlinear transform of
   existing probability columns, not a claim of new independent information.
2. **Top15 identity turnover**: 1 - |S_t intersect S_(t-1)|/15, where S_t is the
   set of retained top15 token IDs. This measures changes in candidate identity,
   which rank-probability columns alone discard. It does not compare with the
   provided token and does not select digit token IDs.
3. **Adjacent normalized-top50 JS divergence**: normalize each saved top50
   probability vector over its own retained mass, align on the union of token
   IDs at t and t-1, and use zero outside each retained support. With these
   explicitly truncated distributions q and r, calculate
   [KL(q||m)+KL(r||m)]/(2 log(2)), m=(q+r)/2, using 0 log 0 = 0.
   This is a [0,1] divergence of the retained distributions, not the unknown
   full-vocabulary divergence. Support turnover can dominate it; that is a
   diagnostic consideration, not an independence claim.

The two temporal features compare consecutive positions inside the same answer,
including across step boundaries; the first token is inactive. Use the common
Top10 readout over valid positions. The frozen engineering direction is
higher ambiguity/turnover/divergence -> higher candidate risk, with no claim
that ordinary syntactic distribution changes imply reasoning errors. Exact
definitions, finite-value checks and missing-data behavior must be frozen and
tested before any future benchmark execution.
