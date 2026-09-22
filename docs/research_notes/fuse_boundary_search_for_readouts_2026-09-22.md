# FUSE-style boundary optimization, translated to readouts (proposal and execution, 2026-09-22)

Omri's prompt: FUSE (Lee, Ma, Zhao, Nair, Spector, Cohen, Candès, arXiv:2604.18547) chooses
per-verifier binarization thresholds by minimizing a triplet-consistency violation statistic.
Could the same idea choose our per-channel readouts? This note records the translation and the
alternatives, ranked by what the project history says about each. No computation had been run when
Sections 1--4 were written. Sections 5--9 now record the bounded execution on the frozen profiles,
the negative result, and the handoff.

Source: `papers/extracted/fuse-ensembling-verifiers-with-zero-labeled-data.md` (arXiv blocked
from this container; the cached v1 extraction was used). Prior project handling of FUSE:
`HANDOFF_FEATURE_SELECTION_AND_FUSE.md` §4 (Steps 223–225), `scripts/upcr_study/
probe_triplet_consistency.py` (m=3 admissibility probe, Spearman +0.04 against label-picked
views), `docs/research_notes/feature_subset_selection_landscape.md` §2.5 and D4 (Ŝ listed as
the primary label-free objective candidate; never built).

## 1. What FUSE does at the boundary

- Verifier j is transformed by g_{j,τ_j}(v) = sign(v − τ_j).
- Under triplet conditional independence (TCI) the second- and third-order covariance tensors
  satisfy Σ_{j1 j2} ∝ v_{j1} v_{j2} and T_{j1 j2 j3} ∝ v_{j1} v_{j2} v_{j3}, so the ratio
  T_{j1 j2 j3} / Σ_{j1 j2} depends on j3 only. Ŝ = Σ_{j3} Var over pairs (j1, j2) of that ratio;
  it is zero under TCI and needs no labels (their Proposition 2.4, Eq. 4; denominators clipped).
- τ* = argmin_τ Ŝ(g_τ(V)) by coordinate descent; then the Jaffe et al. spectral estimator on the
  binarized matrix; then a triplet-averaged posterior as pseudo-label; then any ensemble rule is
  trained on it. Verifiers with estimated balanced accuracy below ½ are dropped (their App. D).
- Their footnote: binarization is not simply "losing information"; a well-placed threshold can
  make a verifier more decisive. That reconciles with Step 134 (continuous > median-binarized)
  because the threshold there was never chosen.

## 2. The translation

In the cumulative-vote design a "verifier" is (channel, readout, encoding), and the instance
matrix is (answer, n) rows with entries "is the first error at a step ≤ n". Every boundary
parameter FUSE optimizes already exists in our pipeline, unchosen or label-chosen:

| FUSE object | our analogue | how it is set today |
|---|---|---|
| threshold τ_j on the score axis | onset quantile (onset80's 0.80), crossing level of the standardized profile, softmax temperature τ per channel | fixed constants, or a single global τ |
| transform family g_j | readout family {top5, top10, max, mean, log_top5, cusum_top5, onset80} | fixed top5, or label-selected on training folds (Codex v2 `train_selected`) |
| binarization rule | argmax → cumulative vote, earliest-wins ties | fixed; the tie rule produces constant step-0 voters (review of v2, §2) |
| drop rule (balanced accuracy < ½) | none in the vote fusion | not applied |
| — | per-channel step offset δ_j (systematically early/late localizers) | never a parameter |

The FUSE recipe therefore reads: choose (readout family, boundary parameter, tie rule, offset)
per channel by minimizing Ŝ over the (answer, n) matrix of the training answers, with no label,
then fit the weights as now. It converts Codex's label-selected roster into a label-free one at
the same access level as the fusion fit itself.

## 3. Alternatives, ranked by prior evidence

A. **Ŝ as a diagnostic on frozen rosters (no new fitting).** Compute Ŝ for the existing v2
   encodings: top5 binary, selected binary, shuffle, soft cumulative, soft jumps, CT7 profiles.
   The step-0 voters have near-zero covariance with everyone, so top5-binary should have the
   worst Ŝ; the shuffle control should be near it. If Ŝ does not order rosters the way SLA does,
   stop here. Motivation: Step 223 ("the search is not the bottleneck, the objective is") and
   handoff §3.3 (label-good subsets fit the rank-one model *worse* than random ones), so the
   sign of a violation objective must be checked before anything is optimized on it. Step 153's
   finding (ρ-violating subsets score higher under continuous L-SML) is the same warning for the
   soft path; for ±1 votes we are in FUSE's own regime.

B. **Per-channel readout + boundary search by Ŝ (the direct analogue).** Coordinate descent
   over r_j ∈ 7 readouts × θ_j (onset quantile in {0.6, …, 0.95}, or a crossing level), plus
   FUSE's drop rule. Label-free, evaluated out-of-fold against fixed top5 and against the
   label-selected roster as a ceiling. Expected size: Step 231's per-feature transform search
   (label-guided) gave +0.24 pp and was fragile across folds; Codex's label selection moved
   five channels and gave about +3.5 pp over top5 soft, much of it tie repair. So the readout
   family alone is a small lever; the value of B is the contract (label-free), not the points.

C. **Optimize the boundary on the step axis: per-channel offsets δ_j.** FUSE's τ lives on the
   score axis; the localization analogue is a shift of ŝ_j by δ_j ∈ {−2, …, +2} steps chosen by
   Ŝ. Rationale: Step 423 found systematically early (Unified-28, η ≈ 0.29) and late (Mind the
   Gap, ψ ≈ 0.26) localizers, and the long-chain deficit is a *shared* late bias; a consistent
   offset is exactly a TCI violation that Ŝ can see. Caveat: offsets are identifiable only up to
   a common shift, so fix one anchor (Σ δ_j = 0 or δ = 0 for the median channel). This is the
   variant most tied to the original question (long chains).

D. **Optimize the encoding boundary: tie rule and soft temperature.** Tie-split (1/k per tied
   step) or abstain (0 instead of ±1), and a per-channel τ_j for the soft path, all chosen by Ŝ,
   with Ŝ for the soft path computed on the jumps (pmf), not on the cumulative curves, which are
   ramp-dominated (review of v2, §3). The 11 pp binary-versus-soft gap in v2 was tie handling,
   so this is where a boundary choice has already been shown to matter most.

E. **Keep fusion before the readout; make the readouts the voters.** Fuse the token-level
   channels first (Step 422 / Stage B C1, the only configuration where L-SML beats equal,
   +3.32 pp), then apply the seven readouts to the fused series and treat those seven as the
   voters in the cumulative-vote fusion; their boundary parameters chosen by Ŝ. This is the only
   variant that preserves the pre-readout advantage and still tunes readouts. Risk: readouts of
   one series are strongly dependent by construction (top5/top10/max share a peak); L-SML
   grouping is required and Ŝ then serves as the check that any triplet is admissible at all.

F. **FUSE steps 4–5: pseudo-label, then train the readout parameters on pseudo-accuracy.**
   Only after A–D, and only with the Step 199 guard (a pseudo-label seeded from a fusion
   reproduced that fusion on 25/25 cells): measure the correlation between the triplet posterior
   and the current fused score before building on it (handoff §4.4).

## 4. Recommended bounded stage

One stage on Codex's frozen v2 profiles (no new inference): A first; if the sign is right, B+D
in one coordinate-descent loop per training fold (label-free), reported out-of-fold with the
fixed-top5, label-selected, shuffle and CT7 anchors, both binary and soft, with late fraction
next to SLA on the long cells. C is a second stage because it changes what a vote means. E is
an architecture change and gets its own discussion. Report Ŝ next to every arm so that the
relation between violation and localization is visible whatever the outcome.

## 5. Executed scope and evidence boundary

The work used the frozen eleven-channel profiles, source-group folds and OOF predictions from
`cumulative-vote-fusion-v2` commit `bcf5a4bd`; no new model inference was performed. All choices
were made separately on each outer-training fold. Labels entered only assumption audits and final
OOF metrics, never clustering, readout selection, the FUSE statistic or L-SML fitting. These are
development results on the population that motivated the method, not untouched confirmation.

The executed sequence was:

1. compute clipped Eq. 4 S-hat for the frozen top5, label-selected, shuffled and CT7 rosters in
   hard, soft-PMF and soft-cumulative representations;
2. audit FUSE's majority-better-than-random and drop assumptions rather than interpreting the
   shuffle control without checking them;
3. expand the common Top-k readout to `k in {1,2,3,5,8,10,15,20,30,40}` and evaluate fold-wise
   label-free S-hat selection, including hard-selection/soft-decoding;
4. cluster the eleven channels from binary top5 residual structure into exactly three groups of
   size at least three, optimize per-channel Top-k by cross-cluster triplets, then fit continuous
   L-SML on the soft PMFs using the frozen groups; compare freezing the initial groups with one
   post-readout reclustering pass.

Per-channel step offsets (alternative C), fusion-before-readout (alternative E), pseudo-label
training and CT7 feature additions were not run. The final project decision is also **not to add
any CT7 channel to the eleven-channel bank in this stage**.

## 6. What the experiments found

### 6.1 Frozen-roster S-hat gate

Hard S-hat ordered the three comparable eleven-channel rosters exactly as hard SLA:

- selected: S-hat `0.05994`, SLA `35.04%`;
- shuffle: S-hat `0.25264`, SLA `21.99%`;
- top5: S-hat `25434.23`, SLA `21.25%` at denominator clip `1e-6`.

That apparently clean result did not survive the representation relevant to continuous
localization. For soft PMFs, shuffle had the lowest S-hat (`0.00486`) but worse SLA (`29.71%`)
than selected (`S-hat=0.02693`, SLA `36.03%`). Soft cumulative curves reversed the association
completely; their shared ramp dominates the covariance. Under the preregistered rule, Stage A
therefore stopped: `STOP_AFTER_STAGE_A`. CT7 was reported but excluded from the ordering gate
because raw Eq. 4 sums over third verifiers and is not directly comparable between seven and
eleven channels.

Interpretation: S-hat measures compatibility with the TCI moment model, not localization quality.
Independent or weakly coupled noise can have an excellent S-hat. Conversely, nearly zero
covariances can make the ratio statistic enormous after clipping. The top5 value is a useful alarm
for the known tie pathology, but its magnitude is not an accuracy scale.

### 6.2 Assumption audit and FUSE drop rule

The label-free MoM drop rule retained every channel in every fold for top5, selected, shuffle and
CT7. The label audit likewise found a majority above balanced accuracy `1/2` in every fold. Thus
the drop rule did not repair the selector or distinguish the shuffled roster. The shuffle concern
remains conceptually important: it changes the relation between a voter and the boundary, so it is
a stress control, not a model-valid negative whose S-hat must track SLA. In this realization it did
not trigger the formal majority-good failure, but it still demonstrated that TCI fit and useful
localization are different properties.

### 6.3 Top-k ladder and fold-wise S-hat selection

Within the restricted common-Top-k family, hard and soft-PMF S-hat were informative but not exact
selectors. Across the ten values of k:

- hard: Spearman `0.891`; both minimum S-hat and best hard SLA occurred at `k=40`;
- soft PMF: Spearman `0.842`; minimum S-hat occurred at `k=20`, while best SLA occurred at `k=30`;
- soft cumulative: Spearman `-0.939`; this representation is unsuitable for selection.

The OOF comparison sharpened the conclusion:

Here SLA is exact first-error-step accuracy on erroneous ProcessBench answers; `late` is the
fraction whose predicted step falls after the true first-error step. "Long" below means the four
OlympiadBench and OmniMath q4/q8 cells.

- fixed `k=30`, soft-PMF decoding: SLA `35.47%`, late `36.46%`;
- soft-PMF S-hat selection followed by soft-PMF decoding: SLA `35.17%`, late `36.65%`;
- hard S-hat selection followed by soft-PMF decoding: SLA `34.06%`, late `36.81%`;
- hard S-hat selection followed by hard decoding: SLA `31.69%`, late `27.62%`;
- frozen label-selected soft ceiling: SLA `36.03%`;
- frozen CT7 soft comparator: SLA `39.83%`.

Hard selection failed mainly at `pb_q8/fold4`: it chose `k=1`, with clipped-denominator fraction
`0.473`; the same fold's soft SLA was `21.76%`, versus `33.80%` when soft-PMF S-hat chose `k=20`.
The other nine hard selections were `k=40`. This supports using hard statistics only as a
predeclared numerical screen, not as the final selector. No clipping threshold was tuned after SLA
was opened; any future eligibility rule must be frozen from numerical reliability alone.

### 6.4 Binary clustering, cross-cluster triplets and continuous L-SML

The last diagnostic implemented the proposed separation between structure discovery and final
continuous fusion:

- binary top5 residual clustering searched all `10,395` unlabeled three-way partitions of eleven
  channels subject to minimum group size three;
- readout coordinate descent used only triplets crossing the other two groups, so a size-three
  cluster did not make the ratio variance degenerate;
- unclipped candidates lexicographically dominated clipped candidates;
- after readout selection, continuous L-SML was fit on soft PMF rows with the supplied binary
  groups; clustering was never rerun on continuous data;
- decoding used the ordinary soft cumulative/PAVA locator.

Freezing the initial groups produced SLA `34.76%`, late `37.72%`; on OlympiadBench+OmniMath the
corresponding values were SLA `30.46%`, late `41.20%`. Reclustering once after readout selection
produced SLA `34.83%`, late `38.15%`; on the long datasets, SLA `30.85%`, late `41.99%`.
The groups changed in all ten task-folds, yet the SLA difference between the two policies was only
`+0.07` percentage point overall and `+0.39` on the long cells, while late errors increased.

This directly answers whether selecting readouts can improve clustering: it changes the partition,
but the changed partition does not translate into a material localization gain. The cross-cluster
triplet objective converged numerically and avoided the size-three degeneracy; its failure is
empirical rather than a missing implementation detail.

## 7. What we learned across the sequence

1. **Representation is part of the statistical objective.** Binary S-hat is useful for detecting
   gross tie/covariance pathologies and for forming dependency groups. It is not a reliable
   end-to-end selector for a soft decoder. Soft-PMF jumps are the relevant continuous
   representation; cumulative curves mostly measure their common monotone ramp.
2. **TCI compatibility and verifier quality are separate axes.** Minimizing Eq. 4 can favor weak,
   noisy or shuffled voters. The majority-good assumption and drop rule do not turn S-hat into an
   accuracy objective.
3. **The readout contains real signal, but S-hat does not recover its optimum precisely.** Moving
   from Top5 toward Top20--40 repairs much of the tie problem, and the broad ladder correlates with
   SLA. Nevertheless, fold-wise S-hat selection loses to fixed Top30 and remains below the
   supervised readout ceiling.
4. **Denominator clipping is a first-class diagnostic.** The `q8/fold4` collapse shows that an
   argmin can be driven by a numerically inadmissible corner. A clipping guard may be useful only
   if frozen before looking at labels; the present results cannot choose its threshold.
5. **Clustering and readout optimization did not solve one another.** Alternating them once changed
   every fold's partition but barely changed SLA. The residual and triplet objectives describe
   dependence structure; neither supplies the missing direction toward exact first-error location.
6. **Continuous fusion should remain continuous.** After binary-only structure discovery,
   continuous L-SML on PMFs preserved more information than hard decoding, consistent with the
   earlier finding that binarization costs localization quality. It still did not beat fixed Top30,
   the supervised ceiling or CT7.
7. **The long-chain problem remains.** The cluster-triplet arms retain late fractions above `41%`
   on OlympiadBench+OmniMath. The experiments rearranged dependence and readouts but did not create
   an independent early-error signal.

## 8. Decision and handoff

The FUSE/triplet readout direction is closed as a final label-free selector on this development
population. Nothing is promoted and no claim of confirmation is made. Preserve these narrower uses:

- binary residual clustering as a dependency diagnostic;
- hard S-hat and clipped fraction as numerical/pathology screens;
- soft-PMF S-hat as a secondary diagnostic inside a predeclared admissible region;
- fixed Top20/30/40 and the supervised selected roster as transparent comparison anchors.

Do not use soft-cumulative S-hat for selection, do not use hard argmin S-hat followed by soft
decoding as the final rule, and do not retrospectively freeze `k=30` as a confirmed choice. Do not
add CT7 features to the eleven-channel bank in the next continuation; CT7 remains an external frozen
comparator. Further development with Claude should start from the existing eleven features and treat
this series as negative evidence about the selector, not as evidence that readouts or clustering are
irrelevant in general.

## 9. Artifacts

- Stage A and plots: `results/fuse_boundary_search_v1/stage_a/`
- Assumption/drop audit: `results/fuse_boundary_search_v1/stage_a_prime_assumption_audit/`
- Top-k ladder: `results/fuse_boundary_search_v1/stage_a_prime_topk_ladder/`
- Fold-wise and cross-representation selection:
  `results/fuse_boundary_search_v1/stage_a_prime_topk_selection/`
- Binary clusters, cross-cluster triplets and continuous L-SML:
  `results/fuse_boundary_search_v1/binary_cluster_triplet_continuous_lsml_v1/`
