# FUSE-style boundary optimization, translated to readouts (proposal, 2026-09-22)

Omri's prompt: FUSE (Lee, Ma, Zhao, Nair, Spector, Cohen, Candès, arXiv:2604.18547) chooses
per-verifier binarization thresholds by minimizing a triplet-consistency violation statistic.
Could the same idea choose our per-channel readouts? This note records the translation and the
alternatives, ranked by what the project history says about each. No computation was run.

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
