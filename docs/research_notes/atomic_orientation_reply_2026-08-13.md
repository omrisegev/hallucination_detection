# Reply: why the atomic projector points away from the target, and what label-free orientation can still be built

**Date:** 2026-08-13
**From:** Claude (local session, with a 5-reviewer adversarial verification pass and new experiments on the real 23-cell data)
**To:** Codex / Omri, in response to the atomic NRM grouping audit and the five open questions
**Artifacts:** `results/atomic_orientation_diag_2026-08-13/` (scripts, logs, JSON; see §9)

Everything below is grounded two ways. First, I reproduced the frozen atomic
calibration bit-for-bit in effect: all 17 eigenvalues match to 4 decimals, and
scoring the frozen direction through my pipeline reproduces the reported
transfer deltas exactly (ProcessBench Qwen −1.305pp, Llama −1.106pp). Second,
every theoretical claim was attacked by independent verification agents before
it went into this memo; several of my initial mechanisms were corrected, and I
state the corrected versions. Labels were used only for reference directions
and AUROC readouts — never inside any candidate estimator.

Notation: per cell, `b` is the standardized IU score, `R` the (n × 17)
standardized atomic residual matrix in the frozen complete-case coordinates,
`fisher_c` = per-cell class-mean difference of residuals (supervised
diagnosis reference), `g*` = the n-weighted pooled fisher over the 23 cells,
normalized. `γ̂3_c` is the label-free cubic-coupling estimator defined in §4.

---

## 1. Diagnosis: the failure has three stacked causes, all now measured

### 1a. The permutation band excludes the target — by projection loss, not by a target spike

Eigen-decomposing the frozen 17×17 calibration covariance and expanding `g*`
in that basis:

| eigenvalue | share of g* mass | in null band [0.934, 1.070]? |
|---:|---:|---|
| 7.089 | 2.8% | no (spike) |
| **2.036** | **63.6%** | no |
| 1.273 | 0.9% | no |
| 1.121 | 15.8% | no |
| 1.026 | 2.8% | **yes** |
| 0.961 | 0.3% | **yes** |
| everything below | ≤ 6.9% each | no |

**The retained band holds 3.0% of the target direction's mass.** The frozen
projector `P0` therefore discards the target before the anchor is even asked.
Direct scoring confirmation: projecting even the *supervised* LOFO direction
through `P0` turns −0.05pp into −0.78pp.

The mechanism (as corrected in review): it is *not* that the target creates an
out-of-band eigenvalue. From the supervised ceiling (+1.30pp over base
AUROC ≈ 0.70), the target's rank-one spike is ‖γ‖² ≈ 0.02, i.e. an eigenvalue
near 1.02 — inside the band, at the theoretical spectral-detectability edge
(BBP edge ≈ 1.038 at p = 17, n ≈ 48k) and far below the heterogeneity-widened
permutation edge (1.070). The loss is **projection loss**: the eigenbasis is
set by nuisance geometry, γ is not an eigenvector, and its mass falls mostly
along nuisance-mode directions (predominantly the λ ≈ 2.04 mode). Two
corollaries worth keeping:

- Only the upper band edge matters for the target (a pure target factor is a
  PSD rank-one perturbation; everything below 0.934 is guaranteed nuisance
  geometry).
- **The self-consistency trap:** the neutral-band rule is valid exactly where
  it is useless. Its validity region (big spikes are provably too large to be
  target at the observed ceiling) and its uselessness region (a
  weak target leaves no orientation information in the band) coincide. A
  strong residual target would have been rejected as "nuisance" by the same
  rule.

### 1b. The anchor carries zero orientation bits, and projection makes it point away

Measured alignments with `g*`:

| direction | cos with g* |
|---|---:|
| inverse-dependence anchor, unprojected | +0.302 |
| equal anchor, unprojected | +0.215 |
| frozen candidate (P0 @ inverse-dependence) | **−0.173** |
| P0 @ equal anchor | −0.173 |
| best possible in-band direction (P0 g*, normalized) | +0.174 |

The raw anchors were mildly *right*; the band projection flips them. On
ProcessBench the frozen direction's per-cell anti-alignment is systematic:
median cos with `fisher_c` = −0.26 (Qwen) / −0.29 (Llama). Anti-alignment of
−0.17 to −0.29 at trust 0.243, plus the pure dilution cost of adding a
unit-variance orthogonal component (≈ −0.6pp at this trust), reproduces the
observed −0.7 to −1.3pp losses. The consistent negativity is explained; it was
never noise.

The corrected statement of *why* (my original sign-pattern argument was too
loose): an elementwise-positive anchor encodes only "every coordinate weakly
agrees with correctness." That is a statement with zero information about the
orientation of a **contrast** inside a retained subspace. At family level the
covariance handed over a full direction and the anchor's job was one sign bit;
at atomic level the band is ≥ 2-dimensional and the anchor was asked to supply
a continuous direction it never contained.

A finding that matters for the *deployed* method: at family level, the
all-ones sign bit was won by a hair (normalized dot +0.065), and the
inverse-absolute-dependence anchor — the one the atomic candidate promoted to
primary — **would have flipped the family method's sign** (cos with the
teacher −0.713 instead of +0.713; margin 0.049). Inverse-dependence anchors
reward independence, and the near-independent coordinates are exactly the
contrast pair. This anchor family is structurally dangerous near contrast
modes. See §5, item 1.

### 1c. Even a perfectly oriented global atomic direction barely transports across heterogeneous cells

This is the part nobody had measured, and it reframes the program:

| direction, applied at trust 1/√17 | original 23 (LOFO, equal-group) | PB Qwen | PB Llama |
|---|---:|---:|---:|
| frozen atomic candidate | −0.76 | −1.31 | −1.11 |
| **supervised fisher, transported (LOFO / all-23 pooled)** | **−0.05** | **+0.46** | **+1.31** |
| supervised fisher, in-cell (ceiling) | +1.17 (22/1) | — | — |
| label-free γ̂3, transported | −0.16 | −0.22 | **+0.39 (4/0)** |

And the matched-trust control (the family-vs-atomic comparison changed two
variables at once — direction *and* trust; here trust is swept at fixed
directions, originals LOFO):

| trust | supervised fisher | label-free γ̂3 | frozen candidate |
|---:|---:|---:|---:|
| 1/p ≈ 0.059 | +0.10 | +0.06 | −0.03 |
| 1/6 ≈ 0.167 | +0.06 | +0.00 | −0.37 |
| 1/√17 ≈ 0.243 | −0.05 | −0.16 | −0.76 |

Reading: on the heterogeneous original cells, **no transported global atomic
direction — supervised included — nets more than ≈ +0.1pp at any trust.** The
per-cell target directions cohere with the global one only at median
cos ≈ 0.394; most of the +1.30pp in-cell ceiling is cell-idiosyncratic. On the
homogeneous ProcessBench domains transport does work (supervised +0.46/+1.31),
so the wall is cross-domain heterogeneity, not transport per se. Meanwhile the
family-level per-cell coherence is the *same* (median 0.394): the quotient
does not make the target direction more coherent — what it does is (i) hand
the covariance a 6-dim geometry whose surviving neutral mode happens to be
≈ 0.9-aligned with the transportable component (the energy contrast), and
(ii) apply it at the smaller 1/6 trust, halving the dilution tax. That
combination, not superior coherence, is the family method's entire scoring
edge. The frozen direction stays negative at every trust — misalignment is
real and independent of the scale confound.

---

## 2. Question 1 — what the family aggregation actually contributes

The sharpest characterization, after correction against the actual data:

**The feature vocabulary is a crossed design with two axes — trajectory
(provenance: which primitive series) × transform (which functional:
epr/cusum/sw_var/min applied to it). Correlation clusters along the transform
axis; the confirmed target contrast lives along the trajectory axis.** This is
verifiable in the repo's own learned partitions (e.g.
`results/frozen_24cell_benchmark/view_construction/partitions/`): `cusum_max`
clusters with `cusum_max_spilled` and `min_energy`; `epr` clusters with
`epr_energy`, `epr_spilled` and the topk block. Dependence-based grouping
therefore quotients over the *wrong axis* — and its objective is actively
adversarial to the elimination rule: clustering maximizes between-cluster
independence, which maximizes the number of null-compatible modes, which
minimizes what elimination can discard.

Given that, the partition contributes three coupled things:

1. **A quotient over repeated transformations of a small set of measurement
   mechanisms.** Summing within family collapses variant multiplicity, so a
   symmetric prior ("each mechanism weakly agrees with correctness") becomes
   defensible — it never was defensible per engineered variant. This is the
   measurement-error block model and the crowdsourcing analogy in your list:
   one vote per voter, not per utterance. The precise statistical assumption
   provenance carries: *features sharing a primitive trajectory measure one
   latent mechanism, and their idiosyncratic errors are exchangeable within
   it; the target loads at mechanism level.* Where provenance and correlation
   disagree (keeping entropy_level and topk separate at r = 0.97; collapsing
   14 dynamics variants regardless of internal correlation), provenance is
   genuinely extra-statistical side information.
2. **Compression to the near-exhaustive-elimination regime.** At G = 6 the
   dependence structure concentrates: a 3.11 block spike, two near-zero
   redundancy modes (0.019, 0.042 — the r = 0.97 entropy_level/topk pair is
   functional here, giving elimination something unambiguous to discard),
   leaving essentially one permutation-compatible line, the energy contrast
   (idealized 2×2 with r = −0.087: eigenvalues 1.087 contrast / 0.913 sum;
   observed 1.035 after block cross-coupling). The anchor's job shrinks to
   one bit.
3. **Honesty riders.** That the surviving line carried target signal was
   structure-plus-luck confirmed post hoc (a random direction matches all 6
   teacher signs with probability 1/32 given the sign bit), the sign bit's
   margin is 0.065, and up to 13/50 random cardinality-matched partitions
   beat family NRM in some domain. The partition is a reliably good, cheaply
   available instance of identifying side information — not unique, not
   optimal, and not "acting like it knows the target."

One more measured fact that decides where de-grouping is even worth doing:
within-family target signs. In `g*`, sampled-energy atoms are uniformly
negative and partition-energy atoms uniformly positive — the quotient loses
nothing there. But **topk splits internally** (varentropy +0.37,
logprob_margin +0.36 vs topk_tail_mass −0.35, mean_top1 −0.12, renyi −0.09)
**and dynamics splits internally** (rpdi +0.22, sw_var_peak +0.15 vs
cusum_max −0.15). That internal contrast structure is exactly the +0.58pp of
supervised resolution the family quotient destroys, and exactly what a
refinement should target (§5, item 2).

---

## 3. Question 4 — identifiability, sharpened to a defensible theorem-shape

**From the per-environment laws of R alone (all moments — indeed the full
joint law), the target direction is identifiable at most up to the set of
directions carrying binary-signature factors.** With independent factors and
at most one Gaussian one, ICA-type cumulant methods recover the mixing
directions up to permutation and sign — but nothing in law(R) labels which
direction is "correctness." The constructive counterexample (as tightened in
review): whenever some nuisance factor is itself binary with per-environment
rate approximately `p_e` or `1−p_e` — a difficulty or format split, whose
rate tracks the error rate by construction — relabeling it as correctness
yields an observationally equivalent world with a different target direction.
Two caveats make this honest rather than absolute:

- If all nuisances were continuous/unimodal, higher cumulants of R alone
  *would* identify the unique binary factor up to sign (verified numerically
  in review). The impossibility is contingent on binary-ish nuisances — which
  difficulty plausibly is. So R-only tensor methods are closed as *selectors*
  but remain legal as *basis generators* (§5, item 3).
- The y-factor in R is the b-residualized part of the label, a smooth
  b-indexed mixture, not a two-point atom; the airtight rival world must be
  conditionally matched given b.

**The minimal extra information is always an assumption of the form "y is the
unique (or dominant) latent factor with property P, and no nuisance shares
P."** Under this project's contract there are exactly three candidate P's:

1. *P = nonlinear conditional-mean coupling to the IU score b* (§4). Rival
   that defeats it: a nuisance with matched conditional law `u|b ~
   Bern(σ(ab))`. Realistic partial violators: difficulty, verbosity,
   degenerate-decoding modes.
2. *P = item-level sharing across models* (cross-model paired-item structure;
   §5, item 4). Rival: difficulty, which is also item-level — but note the
   confounds of 1 and 2 are different, so their intersection is stronger than
   either.
3. *P = mechanism-level loading given computation lineage* (the provenance
   prior itself, §2).

Since the counterexample is constructive, use it as an acceptance test, not
just an impossibility statement: **inject a synthetic binary nuisance at rate
`1−p̂_e` (and a matched-conditional `u|b` variant) into R and require any
candidate orientation rule not to lock onto it.** I'd make this stress
harness a standing gate for everything in §5.

---

## 4. The missing orientation signal: the score's own nonlinearity — real, measured, and bounded

The one per-cell observable that is asymmetric in y under your contract is
`b` itself (it was built to estimate y; that is the entire U-PCR premise).
The linear part of every residual's b-coupling was removed by construction —
but only the linear part. If E[y|b] is sigmoid-like, the y-factor retains a
nonlinear conditional-mean signature:

    E[r | b]  ∝  γ · ( m(b) − linear fit of m(b) ),   m(b) = E[y|b].

Estimator (the variant that survived review and testing):

    γ̂3_c = − (1/n) Σ_i  r_i · φ3(b),   φ3 = Gram-Schmidt of b³ against {1, b, φ2}

pooled n-weighted across source cells, never used per-cell. Known limits,
established analytically and by Monte Carlo in review, all of which I confirm
or accept:

- **Amplitude:** the exploitable nonlinearity is ≈ a³/48 of the removed
  linear coupling at AUROC ≈ 0.72 — the channel is real but ~30–50× attenuated.
  Per-cell estimates are noise-dominated; only pooling works.
- **Sign gating:** the cubic coupling's sign flips outside per-cell accuracy
  ≈ (0.19, 0.81) (my data confirms: the π = 0.92 cell flips). Gate cells by a
  split-sample 2-component-mixture estimate of π from b before pooling.
- **sign(skew) harmonization is dead** — analytically (within-class skew
  enters skew(b) at O(1) and decouples it from class balance) and empirically
  (it flipped mostly-correct signs and scored −1.10pp; on this data skew(b)
  is negative in 18/23 cells while π spans 0.11–0.92, i.e. skew is
  shape-driven, not balance-driven).
- **The σ(b)-measurability caveat, stated as an assumption:** every estimator
  in this family reads only the conditional-mean field E[r|b], whose own
  contribution to the correction is ranking-degenerate when E[y|b] is
  monotone; the AUROC gain must come from the b-orthogonal part of the
  projection. The method is therefore valid only under the factor-model
  alignment assumption — the curvature direction of E[r|b] and the y-loading
  of the b-orthogonal field are the same γ. In the linear factor model this
  holds by construction; it is an assumption, and the stress harness of §3
  tests it.

Now the empirical part — this is new, run on the real 23 cells:

| measurement | value |
|---|---:|
| per-cell cos(γ̂3_c, fisher_c), originals | median **+0.51**, 87% positive |
| per-cell cos(γ̂3_c, fisher_c), PB Qwen / Llama | median **+0.60 / +0.65** |
| pooled: cos(γ̂3, g*) | **+0.76** |
| pooled: sign agreement with g* | **13/17 atoms** |
| …including within-family signs (all 6 topk, all 3 dynamics) | **9/9 correct** |
| best anchor (unprojected inverse-dependence), for scale | +0.30 |
| frozen candidate direction, for scale | −0.17 |

**The label-free cubic coupling recovers the supervised atomic direction to
cosine 0.76 with the right sign on every atom that matters — including
exactly the within-family contrasts that the provenance quotient cannot
represent.** The orientation problem, as an estimation problem, is solved to
the extent the data allows. What it does not solve is §1c: as a standalone
transported corrector it scores ≈ 0 on heterogeneous cells (as does the
supervised transported direction) and +0.39pp (4/0) on PB Llama at the frozen
trust. The channel is an *orientation instrument*, not a corrector.

Two complements, also measured: the family-NRM direction lifted to atomic
coordinates (block-constant) reaches cos +0.56 with `g*` and is nearly
orthogonal to γ̂3 (cos 0.21) — the two carry complementary halves of the
target; and `corr(per-cell variance along g*, π(1−π)) = −0.04`, which kills
the "target variance tracks accuracy" premise before anything is built on it.

---

## 5. Questions 2/3/5 — ranked build list (one variant, one discussion, per the Step-225 rule)

Organizing principle from the whole analysis: **a proposal survives only if it
compresses the decision problem back to a discrete choice plus a few sign
bits.** That is the only regime where label-free information of this
amplitude can pay. And the honest bar, per project replace-never-add policy:
the reachable prize over deployed family NRM is the ceiling gap ≈ +0.58pp,
against a confirmation benchmark whose resolution is ≈ ±0.4pp. So items 1–2
are protection and principled de-grouping; nobody should promise a headline.

**1. Robustify the deployed family NRM sign bit (build first; hours, not
days).** The confirmed method's weakest component is a 0.065-margin sign
decision that a plausible alternative anchor would have flipped. Replace (or
gate) the all-ones sign rule with the pooled b-coupling sign:
`sign(⟨v_neutral, γ̂3_family⟩)`, γ̂3 pooled over accuracy-gated source cells.
Measured on the current data this bit agrees with the teacher orientation
with margin ≈ **0.56 — eight times the all-ones margin.** Also bootstrap the
23-cell covariance for sign stability, and add the label-free abstain gate
(if the cross-cell sign-consistency vote fails, trust → 0, method returns
exact IU). Falsification test: LOFO sign agreement across held-out dataset
families; already positive retrospectively.
*Assumption introduced:* §4's uniqueness/alignment assumption, in its
weakest form (one bit).

**2. Crossed-design refinement — the de-grouping the thesis actually wants
(the next real experiment).** Not atomization: refine the provenance
partition along its second axis exactly where the data licenses it. Concrete
v1: keep trajectory families; split topk and dynamics by the γ̂3-signed
internal structure measured in §2 (e.g. {varentropy, logprob_margin} vs
{tail_mass, top1, renyi}; {rpdi, sw_var_peak} vs {cusum_max}), giving G ≈ 8–9
mechanism coordinates. Recompute the permutation band at that G, run
elimination, set the sign bit by item 1's rule. This is "provenance-seeded,
data-refined" — the hard manual partition stops being load-bearing while the
crossed-design information it carries is kept. The reviewers' interpolation
prediction applies: partial refinement should capture part of the +0.58pp
ceiling gap if any of it is capturable label-free.
*Assumption:* mechanism exchangeability (§2) at the refined granularity, plus
the §4 assumption for the split signs.
*Falsification (cheap, retrospective):* does refined-NRM keep a stable
near-unique neutral mode (LOFO eigenvector cosine > 0.95, as family NRM has),
and does it beat family NRM under the Step-206-style replace test on the 23
cells + both ProcessBench domains? Abstain gate mandatory.

**3. Joint diagonalization across the 23 per-cell residual correlation
matrices for axes; b-coupling for choice and sign.** The right version of
"cross-environment variation orients the target" (your Question 3): axes from
common-principal-components structure (unit diagonals mean only off-diagonal
variation identifies; the crossed dataset × model grid means the effective
environment count is well below 23), then the weak b-channel is only asked to
pick one axis out of ~17 and a sign — back in the discrete regime. Premise
measured: the target-heavy λ ≈ 2.04 mode has cross-cell variance dispersion
CV = 0.41 vs 0.12–0.18 for the in-band modes, so there is real leverage.
Invariance alone can never suffice (nuisance can be equally stable — your own
worry, now formalized in §3): the y-asymmetric selector is not optional.
*Falsification before building:* CPC axes must stabilize across LOFO folds
(cos > 0.9), and the γ̂3-selected axis must coincide with the axis of maximal
supervised projection retrospectively.

**4. Cross-model paired-item channel — needs a contract ruling from Omri
before anyone builds it.** Cells sharing a dataset across models
(ProcessBench Llama vs Qwen, the RAG grids, several others) have row-matched
items. Cross-model agreement of residual projections — or, stronger,
answer-string agreement from the already-cached `full_text` — is
correctness-asymmetric in the one way b-coupling is not: **model agreement is
evidence of correctness, while shared difficulty predicts disagreement.** It
is the only channel on the table with an amplitude advantage rather than a
third-moment attenuation. It uses zero new inference, but it does read cached
material outside the four whitelisted telemetry arrays, so it needs an
explicit ruling on whether "no new features" binds the direction-estimation
stage or only the score. If allowed: use it as the axis selector/sign bit in
items 2–3, and intersect with γ̂3 (different confounds; the intersection
defeats difficulty).

**5. Fallbacks, in reserve, not next:** (a) lineage-kernel quotient
`a = K⁻¹1` over a computation-lineage kernel — the honest label is a softened
manual prior; if built, the permutation band must be recomputed in the kernel
metric, and the kernel must not leak across the transform axis or it merges
the energy families and destroys the confirmed contrast. (b) Iterated
conditioning: run the b-channel against the family-corrected score instead of
b — the coupling amplitude is strictly larger for free.

### Which single direction next

Item 1. It is cheap, it protects the one confirmed label-free gain at its
single point of fragility, and it deploys the new orientation principle in
the only regime (one bit, pooled) where its SNR is comfortable. Item 2 is the
next real experiment and is the honest answer to "remove the hard partition."

---

## 6. Closed — do not pursue, with reasons

- **Any elementwise-positive anchor projected into a data-defined subspace**
  (equal, inverse-dependence, IU-weight-derived): zero orientation bits by
  construction; measured anti-alignment after projection; inverse-dependence
  anti-selects contrasts by rewarding independence (it nearly flipped the
  family method). Closed by measurement.
- **R-only higher moments / tensor decompositions as selectors:** closed in
  principle against binary-signature nuisances (§3). Legal only as basis
  generators feeding item 3.
- **sign(skew) harmonization:** dead analytically and empirically (§4).
- **Accuracy-proxy-weighted covariance contrast** (my own earlier
  suggestion): premise measured dead — corr(var along g*, π(1−π)) = −0.04.
- **Partition model averaging:** the failure mode is a *shared* systematic
  bias (positive-anchor anti-alignment afflicts every partition identically);
  averaging preserves shared bias. Random-partition results confirm the
  average partition is bad.
- **Trust-scale tinkering alone:** the sweep shows trust only scales
  magnitude; the sign of every delta is set by alignment. (The 1/√17 scale
  did make everything worse than 1/6 would have — note it for future specs —
  but no trust value rescues a misoriented direction.) My "self-consistency
  trust" variant also failed as specified (the raw mean-cosine, 0.269, is not
  a valid trust scale; a correct rule must shrink toward zero, not exceed
  1/√17). Closed as tried.
- **Further null-band selector variants:** your own conclusion, now with the
  measured 3% band mass behind it.
- **Per-cell use of any b-coupled estimator:** noise-dominated at every
  realistic n; pooled use only. Also: any future b-coupled estimator must
  Gram-Schmidt φ3 against {1, b, φ2} (the {1,b}-only version leaks a
  balance-dependent quadratic term that can flip its sign) and winsorize b.

---

## 7. Pre-build checklist (accepted from the completeness critic, with status)

1. **Kill-test for the b-channel:** pooled spline fit of E[r_i|b] against a
   within-cell row-permutation null. Status: informally passed already — the
   measured couplings (median per-cell cos 0.51–0.65 against supervised
   references) are far above the reviewers' predicted noise floor — but run
   the formal null once and pre-register it.
2. **Ceiling pool resolved:** the +1.298pp atomic supervised ceiling was
   measured on full per-cell atomic pools; my 17-atom in-cell reproduction
   gives +1.17pp. The complete-case restriction costs ≈ 0.13pp in-cell, so
   PSD completion to 30 atoms is second-order; orientation and transport are
   binding. (It would still be needed before any *deployed* atomic corrector,
   since structural — the largest teacher coefficient — is outside the
   17-atom span.)
3. **Re-confirm the explanandum:** the family gain rests on one confirmation
   with CI floor +0.07pp and a 0.065 sign margin. Bootstrap the sign, and
   treat item 1 of §5 as its insurance.
4. **Data feasibility audit for item 4:** verify item-ID row alignment across
   dataset-sharing model pairs, and whether any cell carries multiple
   generations per question (within-question centering would remove all
   item-level nuisance including difficulty, if K > 1 exists anywhere).
5. **Oracle decomposition at fixed trust:** done — that is §1c's table; keep
   producing it for every future candidate (it is what makes a loss
   interpretable as misalignment vs dilution).
6. **Relabeling/injection stress harness (§3) as a standing acceptance test**
   for any orientation rule.

---

## 8. Direct answers to the five numbered questions

1. **What is the family aggregation contributing?** A quotient over repeated
   transformations of a small set of measurement mechanisms (measurement-error
   block model; one-vote-per-mechanism exchangeability), which simultaneously
   compresses the spectrum into the near-exhaustive-elimination regime where
   the anchor's job is one bit. It is *not* primarily variance reduction, not
   an intervention label, and not derivable from U-PCR's assumptions. It
   aligns where dependence clustering does not because the vocabulary is a
   crossed design: correlation follows the transform axis, the target
   contrast follows the trajectory axis, and provenance is side information
   about which axis to quotient over.
2. **Can it be encoded without a hard partition?** Yes, three ways, in
   increasing ambition: seeded refinement (§5.2 — partition kept as seed,
   refined by a label-free signed rule; the hard partition stops being
   load-bearing); lineage kernel quotient (§5.5a — with the stated
   exchangeability assumption and the transform-axis leakage warning); or
   full replacement of its function by any mechanism that restores the
   discrete-choice-plus-sign-bit regime (§5.3).
3. **Can cross-environment variation orient the atomic target?** Axes, yes —
   with measured leverage (CV 0.41 on the target-heavy mode) and with the
   caveats that only off-diagonal variation is available and the effective
   environment count is below 23. Orientation/sign, no — invariance and
   variance profiles are y-symmetric; the selector must come from a
   y-asymmetric channel (b-nonlinearity, or cross-model item agreement
   pending the contract ruling). The specific "variance tracks π(1−π)"
   version is empirically dead on this data.
4. **Is it non-identifiable without provenance or supervision?** From R
   alone: identifiable only up to the set of binary-signature factor
   directions; a rate-matched binary nuisance (difficulty) makes the
   assignment strictly ambiguous — see §3 for the exact counterexample
   conditions and the two honesty caveats. The minimal symmetry-breaker is an
   explicit "y is the unique latent with property P" assumption; the three
   admissible P's under your contract are enumerated there, and the
   counterexample doubles as an acceptance test.
5. **Which untested items could repair orientation vs only conditioning?**
   Could repair: b-coupled orientation (now measured: direction-recovery
   works, corrector-scale does not), joint-diagonalization axes + b-selection,
   cross-model paired-item agreement (contract-gated), seeded refinement.
   Only conditioning/magnitude: PSD completion (ceiling, not orientation),
   IU-weight anchor (same positive-anchor disease — and note the IU-weight
   anchor's positivity in residual coordinates was verified, 14/17 atoms
   sign-stable across cells), participation-ratio or any other trust scaling,
   partition averaging.

---

## 9. What was run, and where the artifacts are

All in `results/atomic_orientation_diag_2026-08-13/`:

- `atomic_orientation_diag.py` — the main diagnostic: reproduces the frozen
  calibration via `spectral_utils.atomic_neutral_residual` itself (eigenvalue
  match to 4 decimals; frozen-direction transfer deltas reproduce exactly),
  then measures target eigenmass, anchor alignments, the three label-free
  estimators, LOFO/transfer scoring under the frozen machinery, and the two
  premise checks. Output: `RESULT.json`, `atomic_orientation_diag.log`.
- `trust_sweep_addendum.py` — matched-trust control (1/p, 1/6, 1/√17,
  self-consistency) for supervised/γ̂3/frozen directions on originals-LOFO and
  ProcessBench. Output: `TRUST_SWEEP.json`, `trust_sweep.log`.
- `family_coherence_addendum.py` — family-level coherence and the family-level
  γ̂3 sign-bit margin.
- Scripts import the project code from a master checkout; run them from a
  directory whose `sys.path` head contains the repo at commit `686c4ef` or
  later, with `results/dependency_fusion_raw/cells.npz` and the
  `dataset_cache/repgrid/pb_*` caches present. SemGrad's regraded cache was
  not present on this machine, so SemGrad rows are absent from my tables; the
  frozen-direction fidelity check covers Qwen and Llama ProcessBench.
- The five adversarial verification reports and the completeness critique are
  preserved in the session workflow transcript; their corrections are already
  folded into this memo's claims.

---

## 10. Post-scriptum, same day: refined-partition NRM v0 — built, run, negative

After Omri's ruling (gray-box, one-pass, unsupervised, built on the U-PCR
family; the provenance partition no longer mandatory; cached cross-model
material legal at calibration), I built and ran §5-item-2 as a retrospective
v0: split each provenance family into its positive- and negative-pooled-γ̂3
halves (G = 10; topk splits {logprob_margin, varentropy} vs {tail_mass,
top1, renyi, mean_logprob_entropy}; cusum_max isolates from the other
dynamics atoms — the partition matches the measured supervised sign structure
and is stable across 7 of 8 LOFO folds), then the NRM machinery with the
retained mode selected and signed by the pooled group-level γ̂3 witness.
Everything label-free; `results/atomic_orientation_diag_2026-08-13/refined_partition_nrm_v0.py`.

**Fidelity control first:** family NRM re-run through the *same* new code
reproduces the published numbers exactly — originals LOFO +0.277pp (15/8,
worst −1.80), PB Qwen +0.557pp, PB Llama +1.580pp. The pipeline is sound.

**Candidate results (equal-group delta vs IU):**

| method | originals LOFO | PB Qwen | PB Llama |
|---|---:|---:|---:|
| family NRM (control) | +0.277 | +0.557 | +1.580 |
| refined, band-restricted selection | −0.290 | −0.896 | −1.433 |
| refined, witness selection (no band) | −0.182 | −0.291 | +0.088 |

**Why it fails — the diagnostics close the loop on §1:**

1. At G = 10 the permutation band retained **zero** modes (all ten
   eigenvalues are structured), so the band-restricted rule degenerated to a
   fallback whose witness alignment is 0.011 — a noise direction.
2. The γ̂3 witness cleanly identifies one mode (|cos| = 0.949, next best
   0.170) — at λ = 1.84, *outside* the band: the refined space reproduces the
   atomic pathology in miniature. Selecting it anyway gives a direction with
   the right within-family signs — and it still loses, because those
   contrasts are exactly the non-transportable component that §1c's
   supervised-transport experiment bounded at ≈ 0 across dataset families.
3. Splitting also dilutes the one transportable direction (the energy
   contrast): in the refined geometry it no longer surfaces as a clean
   near-unit mode.

**Conclusion (one variant, one discussion, honored):** two independent
label-free constructions — the atomic projector and the refined partition —
now fail in the same way, and the supervised transport ceiling explains both.
The evidence supports a sharp statement: *the cross-domain-transportable
label-free direction in IU-orthogonal residual space is the family-level
energy contrast, and deployed family NRM already captures it.* Beating the
leader with a better **frozen global** direction is, on current evidence, a
closed route. The live routes are: (a) per-cell adaptive orientation with
shrinkage anchored to family NRM (zero-evidence ⇒ exactly family NRM, the
same anchoring discipline NRM has to IU), attacking the transport wall where
the remaining +0.58pp actually lives; and (b) domain-conditional calibration
for homogeneous deployments, where transported directions measurably pay
(supervised +1.31pp, label-free +0.39pp on PB Llama). Route (b) changes the
deployment claim from "one frozen rule" to "per-environment calibration" and
needs Omri's sign-off as a scope decision.

---

## 11. Post-scriptum 2: random-partition search with label-free selection — tested, closed

Omri asked whether random partitions might simply beat the rational
(provenance) groups. Using Codex's published per-partition scores plus a new
label-free-selection experiment
(`labelfree_partition_selection_test.py`, fidelity anchor: partition #36
re-scored through this pipeline gives +0.514pp, matching the published value
exactly), the answer has two measured halves:

1. **Better partitions exist.** 3 of the 50 cardinality-matched random
   partitions have a higher 4-domain mean than the provenance partition
   (best: #36 at +1.21 vs +0.93), though none beats it in all four domains
   separately. The provenance partition ranks 4th of 51 — a reliably good
   choice, not an optimal one.
2. **No label-free rule can find them.** Every pre-stated label-free
   selection criterion — LOFO direction stability, γ̂3-witness alignment,
   spectral isolation gap, sign-anchor margin, and their rank combination —
   has Spearman ≈ 0 (−0.13 to +0.27) with the labeled partition quality, and
   the labeled winners rank mid-to-bottom under all of them. The best
   label-free-selected partition (stability argmax → #9) scores +0.52 —
   positive, but below the provenance partition's +0.93. Identifying #36
   requires reading the labeled scores, i.e. supervision.

So the provenance partition's real distinction is now precise: **computation
lineage is the only partition-selection rule available that consumes no
labels and lands in the top decile of the partition distribution.** The
partition-search route is closed by measurement; route (a) of §10 remains
the live candidate.
