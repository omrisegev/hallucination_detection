# Research proposal — the depth channel on the localization population (v1)

Claude, 2026-09-18. Branch `claude/whitebox-layer-views-v1`,
worktree `.worktrees/whitebox-layer-views-v1` (sparse).
Data handoff: `docs/HANDOFF_WHITE_BOX_LAYER_VIEWS.md`.
Line this is measured against: `docs/HANDOFF_TOKEN_PROBABILITIES.md`.

**Status: proposal for approval. Nothing measured, nothing launched, no cluster job
submitted. This document exists to be read and approved or rejected before any code.**

Incorporates Omri's corrections of 2026-09-18 — broad bank primary rather than CT7, CT7's
SLA computed once on the token line rather than twice, and geometry routed to the gate
rather than discarded. Those three changed the proposal materially; they are marked
inline where they bite.

---

## 0. The claim under test, in one paragraph

Step 414 found that the output distribution of one greedy pass supports about 2.46
conditionally independent signals, and that more cannot be manufactured by adding further
transforms of it. Step 421 extracted, for the whole localization population, a field that
is not a transform of that distribution: how the distribution is *built* across 36 layers.
This proposal tests two separable claims. **(A)** that the depth field adds measurement
dimensions the output-distribution bank does not already carry, and **(B)** that the depth
field improves localization. These are different claims, they can come apart in both
directions, and the project already has a documented instance of them coming apart — so
the proposal deliberately refuses to let one stand in for the other.

---

## 1. What is actually new

### 1.1 The inventory, by axis

`cluster/layer_lens.py` fixes the axis order: `MODULES = ("attn", "mlp", "resid")`, layers
`0..35`, tokens in generation order. `L = 36`, `V = 151,936`, both models.

**Token-resolved — 468 channels per token. Eligible for the locator.**

| array | shape | channels | what it is |
|---|---|---|---|
| `lens_H` | `[3, 36, T]` | 108 | entropy of the layer-`l` lens distribution, full vocabulary |
| `lens_logp_tgt` | `[3, 36, T]` | 108 | lens log-prob of the token actually in the chain — depth-resolved spilled energy |
| `lens_logp_top1` | `[3, 36, T]` | 108 | max lens log-prob — commitment |
| `lens_kl_final` | `[3, 36, T]` | 108 | `KL(lens_l ‖ lens_final)` — the DoLa contrast direction |
| `resid_norm` | `[36, T]` | 36 | `‖x_l,t‖` |

**Answer-resolved — no token axis. NOT eligible for the locator. Eligible for the gate.**

| array | shape | channels | what it is |
|---|---|---|---|
| `cov_eigs` | `[36, 32]` | 1,152 | top-32 eigenvalues of the centred token covariance of `x_l` |
| `hid_proj` | `[36, 256]` | 9,216 | token-mean of `x_l` under a fixed seeded Gaussian projection |

This split is the single most important structural fact in the document. `cov_eigs` and
`hid_proj` have no token index, so no arrangement of them can produce a per-step risk.
Anyone counting "432 + 1,152 + 9,216 new features" for localization is counting wrong. The
locator-eligible field is **468** channels.

**Not new at all.** `final_lens_H` is the top-15 renormalised final-layer lens entropy —
i.e. the existing gate input, re-derived. It is in the file as the join/verification
quantity, not as a new view.

### 1.2 What each new channel is, and is not, a transform of

| channel | relation to the existing output-distribution bank |
|---|---|
| `lens_*` at `l = 35`, resid tap | a near-duplicate of existing views, up to the full-vocab vs top-15 difference |
| `lens_*` at `l < 35` | **not** a transform of the output distribution — a different distribution |
| `lens_kl_final` | a *contrast between* depths; has no output-distribution analogue at all |
| `resid_norm` | representation magnitude; no analogue |
| `cov_eigs`, `hid_proj` | representation geometry; no analogue |

The honest expectation is that the top layers are near-duplicates of what we have and the
interest is concentrated in the middle-depth contrasts. Stage 1 measures where, rather than
assuming.

### 1.3 Two fidelity notes that must not be lost

**`lens_H` is full-vocabulary; the cached `token_entropies` is top-15 renormalised.** These
are different statistics, not a rescaling. This is the defect that would have aborted the
extraction run.

**`cov_eigs` is not EigenScore.** INSIDE (Chen et al., ICLR 2024) computes covariance
eigenvalues of sentence embeddings over **K = 10 sampled generations**. Ours is the token
covariance within **one teacher-forced pass**, K = 1 by construction. The mechanism is
borrowed; the statistic is different and the multi-pass diversity that carries INSIDE's
signal is absent. Per the 2026-08-05 rule, this is *inspired by* INSIDE and will be
described that way — never labelled with the method's name, and never reported as a
reproduction of it.

---

## 2. The two roles, and why they must stay separate

The system has two components with different decisions, different axes and different
failure profiles. The proposal assigns the new features to roles explicitly rather than
letting them drift.

### 2.1 Locator — *where* is the first error

A ranking **within** one answer; the readout is argmax over steps. Only within-answer
variation can change the output: any answer-constant transform of a channel changes
nothing. Locator-eligible input is therefore **token-resolved only**, i.e. the 468 channels.

### 2.2 Gate — *does this answer contain an error at all*

A decision **per answer**, requiring a scalar comparable **across** answers. Between-answer
variance is the signal here, not a nuisance. Answer-resolved geometry (`cov_eigs`,
`hid_proj`) is native to this role — it is exactly the axis on which INSIDE-family methods
were designed to operate.

*This is Omri's correction, and it changes the proposal.* My first draft treated the
answer-level arrays as out of scope because they cannot feed a step readout. They cannot —
but that argument only removes them from the locator. It says nothing about the gate, and
the gate is the component that most needs a new kind of input.

### 2.3 Why the gate is the right target — verified, not asserted

From `results/claude_feature_bank_token_lsml_v1/GATE_ISOLATION.json`, recomputed for this
document:

| cell | gate-free SLA | gated SLA | cost of the gate | clean accuracy |
|---|---:|---:|---:|---:|
| pb_gsm8k_q4 | 47.83 | 42.51 | **−5.31** | 34.72 |
| pb_gsm8k_q8 | 49.76 | 44.44 | **−5.31** | 27.46 |
| pb_math_q4 | 34.85 | 30.13 | −4.71 | 41.63 |
| pb_math_q8 | 31.99 | 28.79 | −3.20 | 35.71 |
| pb_olympiadbench_q4 | 30.56 | 25.57 | −4.99 | 42.48 |
| pb_olympiadbench_q8 | 31.01 | 26.17 | −4.84 | 35.69 |
| pb_omnimath_q4 | 30.96 | 26.61 | −4.35 | 55.60 |
| pb_omnimath_q8 | 30.43 | 26.75 | −3.69 | 52.70 |

The gate costs between 3.20 and 5.31 points **in every one of the eight cells**, and its
clean accuracy ranges from 27.46% to 55.60% — near or below a coin flip in half of them.

There is a structural reason, and it is the argument for this direction. **The gate and the
locator are currently fed by the same signal.** CT7's gate is a frozen tail-15 statistic of
its own entropy bank; the token-level arm's gate is a threshold at 0.33 on the fused
locator composite itself. A gate that is a thresholded function of the locator cannot
discriminate independently of it — when the locator is confused, so is the gate, in the
same direction. An answer-level channel of a *different kind* is the structurally correct
repair, and the depth field is the only such channel this project has ever extracted.

---

## 3. Omri's four questions, answered directly

**Which new features are added?** Section 1: 468 token-resolved channels (locator-eligible)
and 10,368 answer-resolved channels (gate-eligible). Not one bank of 11,000 — two banks,
for two roles, on two axes.

**Gate or locator?** Both, in separate stages with separate evidence, never pooled into one
headline. Token channels → locator (Stage 2). Answer geometry → gate (Stage 3). Stage 3 is
**not** conditional on Stage 2, and is arguably the higher-expected-value half.

**Do we work on the new features alone?** No. Every measurement runs three arms — gray bank
alone, depth field alone, and the union — because the question is never "is depth good", it
is "does depth add". Depth-alone exists only as a diagnostic and can never be promoted on
its own; the mandate is explicit that a standalone auxiliary score is a control, not a
candidate.

**Do we combine probabilities and internals?** Yes, and the union is the *primary* arm, not
an afterthought. Fusion stays the core: depth enters as a supporting component improving the
inputs to the existing IU-PCR / L-SML fusion. Every Stage-2/3 claim carries the same fusion
core with and without the component on matched IDs, plus the ablation of learned fusion
against equal averaging under that component — so that a gain cannot be attributed to
fusion when the auxiliary channel alone explains it.

**Between answers, or only within an answer?** The sharpest question, and it has different
answers per role. This is section 4.

---

## 4. The axis contract — between-answer vs within-answer

The token-probability line's documented defect (its §4) was fitting on the pooled
between-answer axis and reading out within the answer: the fit saw mostly between-answer
variance — long and hard answers run hotter — and was then applied to a within-answer
ordering. The visible symptom was `chosen_surprisal` taking a **negative** weight in all
five folds despite being oriented so that higher means more risk. Between answers surprisal
tracks difficulty; within an answer it tracks error. That is what mixing the axes produces.

This proposal fixes the axis per role, in advance:

| | locator | gate |
|---|---|---|
| decision | rank steps inside one answer | one scalar per answer, thresholded across answers |
| informative axis | **within-answer only** | **between-answer, necessarily** |
| representation | every channel standardized **within the answer** before fusion | answer-level statistic; between-answer comparability is the point |
| principal confound | answer length and step length | answer length, difficulty, subset |
| control required | — | explicit length control; the step-length prior is already documented as inflating every step readout |

**The previous line already demonstrated this split empirically, which is the strongest
support Stage 3 has.** In `EVIDENCE_DOMAIN_INDEPENDENCE_V1.md`, answer length in steps is
**orthogonal to all thirteen evidence families** (|corr| ≤ .05) — the most independent
channel ever measured in this project — and yet it is *constant within an answer*, so after
answer-standardization it is exactly zero and carries no within-answer information at all.
Its .667 pooled AUC is entirely the between-answer length prior. That document's own verdict:
"**It can only inform the gate, never the step locator.**" Removing it is precisely what took
2.83 down to 2.46.

So the project has already found one genuinely independent channel and found that it is
gate-only. `cov_eigs` and `hid_proj` are answer-constant in exactly the same way. That is an
argument for routing them to the gate — and a warning that their pooled AUC, whatever it
turns out to be, will be partly the same length prior unless it is explicitly controlled.

**Two distinct things must not be conflated: the axis of the representation, and the scope
of the weight fit.** The representation contract above is fixed. The fitting scope is a
separate question with two declared arms:

- **answer-only fit** — normalization, groups, weights and any reducer estimated from the
  current answer alone. This is the mandate's stated primary objective.
- **pooled donor-fold fit** — weights estimated across answers, applied to answer-local
  representations. This is what CT7 and the token arm actually do.

`PROGRESS.md` records that the matched answer-only versus pooled comparison on one fixed
bank "was specified three times and has never been run", and that no decision to abandon
answer-only fitting was ever recorded. This proposal does not resolve that, and does not
quietly pick pooled by default: whichever arm is run is **declared with its access level**,
and if both are run they appear as separate rows, never averaged.

---

## 5. Stage plan

### Stage 0 — access and provenance. No metric, no claim.

- **S0.1** Verify the Drive backup: nine directories, 13,769 `rows/*.npz`, manifests
  present, byte counts matching the cluster. The Drive copy is the durable one; cycle-2 is
  very likely being retired while our account is now cycle-3.
- **S0.2** Verify the join. *Already done:* `JOINED.json` resolves 13,769 records over
  exactly the nine cells (prmbench 6969; pb_{math,olympiadbench,omnimath}_{q4,q8} 1000 each;
  pb_gsm8k_{q4,q8} 400 each), matching the extraction manifests.
- **S0.3** ~~Pin the provenance of the participation-ratio anchors.~~ **Done — found in
  `.worktrees/a6-s0b/docs/experiments/EVIDENCE_DOMAIN_INDEPENDENCE_V1.md`**, and it changes
  Stage 1's estimator (see below). The anchors are:

  | number | what it is |
  |---|---|
  | 2.75 | marginal PR over the 13 family virtuals |
  | **2.83** | **conditional** PR, centred within label class — the headline |
  | **2.46** | the same after removing `n_steps`, i.e. the **12 within-answer families** |
  | 1.80 | CT7's seven views — **not** from this document, still to be pinned |

  Estimator: `(Σλ)² / Σλ²` over the correlation matrix of **family virtuals**, where each
  virtual is the equal mean of its z-scored, oriented members. Population: the **94,203
  labelled PRMBench steps** (14.0% error) — *not* the eight ProcessBench cells the SLA work
  runs on. Stage 1 runs on that same PRMBench population for comparability, and says so.

  Remaining under S0.3: pin CT7's 1.80 to its own artefact.
- **S0.4** Plan a move off `cycle2_*`. Not urgent, not blocking, not silent.

Reduction runs **on the cluster** (CPU): 5.5 GB of npz, and the within-answer shuffle null
has to be computed at token level before the step reduction, so it cannot be done from a
reduced artefact.

### Stage 1 — the redundancy geometry (locator side)

**Decision question.** Conditioned on the output-distribution bank already in use, how many
*additional* effective dimensions does the depth field carry at step level, and is that
distinguishable from what the identical pipeline returns on shuffled data?

**Reference bank — primary is the broad bank, secondary is CT7.** *Omri's correction, and I
accept the reasoning.* CT7 is the most redundant bank we own — conditional PR 1.80 out of 7
views, six of which are one entropy signal — so measuring "how much depth adds" against the
densest available bank would inflate the addition almost by construction. More decisively,
the ceiling being tested was measured on the broad bank's 13 evidence families, not on CT7;
if the claim is "does depth break the ceiling of the output-distribution channel", the
comparison must be against the thing the ceiling was measured on. My objection that CT7 is
the frozen candidate confuses two purposes: freezing matters for *candidates*, coverage
matters for *measurements*. Both are reported, answering two different questions — the broad
bank answers the scientific question about the ceiling, CT7 answers the engineering question
about whether to extend the candidate.

**Representation.** Step level via the adopted top-10 token-mean within step; answer-local
standardization; **within-label-class centring**, because that is the axis on which the
three-source threshold was defined. A marginal ratio is not comparable and will not be
presented as if it were.

**Arms.** (a) gray bank alone — reproduces the anchor and validates the estimator; (b) depth
field alone; (c) union. The quantity of interest is **PR(c) − PR(a)**.

**The estimator is virtual-level, not column-level.** This is the correction S0.3 forces,
and it is a better answer to Omri's scale objection than the one I first wrote. The anchor's
PR is computed over **13 family virtuals** — each the equal mean of its z-scored, oriented
members — not over the 50 raw streams. So "PR over 468 depth columns" would not have been
comparable to 2.46 in the first place, regardless of any width control. Stage 1 therefore
reduces the depth field to **depth family virtuals** on the same recipe (the natural
partition is tap × quantity × depth band) and computes PR over `12 + k` virtuals.

**Controls.**

1. **Within-answer token shuffle**, independently per channel, R replicates, recomputing the
   whole step readout, the virtuals and the ratio. Every arm is read against *its own* null.
2. **Redundant-family calibration, which the anchor document hands us for free.** Adding
   output-distribution families does **not** move the count: every subset tried — 3, 4, 5, 6,
   7 families — lands at PR 2.7–2.9. That is a measured baseline for "what adding more of the
   same looks like", and it is a sharper comparator than a synthetic null. Depth families
   have to beat *that*, not beat zero.
3. **PR as a function of the number of depth virtuals added**, rather than one number, with
   both controls tracked alongside. The shape of that curve is the finding; a single scalar
   is not.

No number in this line is read against the bare integer 3.

**Free structural sub-questions once the matrix exists.** Are `attn`, `mlp` and `resid`
independently informative (TriLens's claim) or is the tap axis near-degenerate — PR({resid})
versus PR({resid, attn, mlp})? And the adjacent-layer correlation profile along depth: where,
if anywhere, does depth stop being a smooth near-duplicate of itself? The project's prior is
unflattering — prefix innovations sat at .87–.96 with the family they were differenced from
and moved the effective count by at most 0.15 — and adjacent layers are plausibly the same
story. Finding that *is* a result.

**Uncertainty.** Per-cell and pooled, paired **source-group** bootstrap over the 1,979-group
scheme, not per-answer.

**What Stage 1 can and cannot close.** *Omri's correction, and it is the most important
constraint in the document.* PR measures linear redundancy in a correlation matrix. It knows
nothing about the label. The project already has an explicit counter-example in the opposite
direction: at Step 415 the chosen-token channel raised the effective signal count 2.46 → 2.78
— the largest addition ever measured here — and **lowered quality by 0.73 points**. PR and
quality have already demonstrably decoupled in this codebase. Therefore:

- A **zero** result in Stage 1 closes the *narrow* justification — that depth is what unlocks
  L-SML, and that below three sources the correct output remains averaging. It does **not**
  close the question of whether the depth field carries localization signal. Those are two
  questions and the first draft of this plan wrongly fused them.
- A **positive** result in Stage 1 is likewise not evidence of localization gain, for the
  same reason in reverse.

Stage 1's output is therefore *geometry that shapes Stage 2*, plus a scientific finding about
the ceiling — not a go/no-go on the line.

### Stage 2 — locator variant. One variant, derived from Stage 1's geometry.

Deliberately unspecified in advance beyond its constraints, per "one variant, one discussion,
then build". Which layers, which taps, which quantity gets read off Stage 1 rather than
guessed. Constraints fixed now so they cannot drift:

- a handful of channels, **not 432** — explicitly prohibited by the data handoff and already
  closed by the pool-composition and per-feature-ranking results;
- the fusion core with and without the component on matched IDs, plus learned-fusion versus
  equal-averaging under the component;
- **gate held fixed, both directions**, with gate-free per-subset SLA reported beside macro-F1
  under the Mind-the-Gap protocol, because ProcessBench macro-F1 is strongly gate-dependent;
- effective count reported beside every fusion claim (Step 205 guard);
- no digit channels, and no digit dependency inherited through a frozen gate or a
  baseline-plus-correction recipe;
- CT7 is frozen — anything that changes it is a new candidate with a new name.

### Stage 3 — gate variant. Answer-level geometry. **Independent of Stages 1 and 2.**

The component with the largest measured deficit (section 2.3) and the one place the
answer-level arrays are native. Scheduling note: this does not wait on Stage 1, because
Stage 1 is a locator-side measurement and has no bearing on the gate.

- **Input.** `cov_eigs` and `hid_proj`, reduced to a small answer-level summary. The
  INSIDE-inspired direction is the eigenvalue spectrum of the mid-depth token covariance; the
  K=1 caveat in §1.3 applies and will be stated wherever the number appears.
- **Contract.** The no-error decision is defined explicitly, its calibration source is
  declared (answer-only, pooled-unlabeled, or externally calibrated — these are different
  access levels and a pooled unlabeled gate is a hybrid scope, not answer-only fitting).
- **Controls.** Answer length first and always — the length prior is documented as inflating
  readouts in this project and the gate is the component most exposed to it. Plus the current
  gate as the incumbent row, and a same-signal control that shows how much of any gain is
  simply "a gate not derived from the locator".
- **Endpoint.** Clean accuracy and error-detection jointly, per cell, with the locator held
  **fixed** so that the gate's contribution is not confounded with the locator's — the exact
  mirror of the Stage-2 rule.

---

## 6. Comparators carried in every quality table

Mandatory rows per the continuity rule; a missing asset gets a visible reason, never silent
omission.

| row | PB macro-F1 | gate-free mean SLA | note |
|---|---:|---:|---|
| CT7 (frozen candidate) | 41.19 | *pending, token line* | see §9 |
| token-level L-SML | 34.08 | 35.92 | LOCO-5 @ 0.33, development-only |
| token-level equal mean | 32.34 | 32.59 | the averaging control |
| plain token entropy, Top-10 (Step 334) | 35.44 | not measured | simple reference |
| Chen et al. Shannon Drop | — | 39.27 | published; a *derivative* readout, ours is a level |
| Chen et al. Shannon Avg | — | 25.34 | published baseline |
| chance | — | 16.58 | |

**Separate panel, context only — different task, do not pool.** The prior white-box lineage
on final-answer detection, `results/whitebox_vs_graybox_matched_v1/`: white ≈ gray on 31,440
candidates across 13 cells (AUROC −0.0013, [−0.0169, +0.0123]), mean per-cell Spearman 0.8677.

*Omri's precision, which I had blurred.* That 0.8677 is between **fused final scores**, on a
different task and a different representation. It is a prior about redundancy **of outputs**,
not about dimensional independence **of channels** — and that distinction is precisely what
Stage 1 measures. The correlation of the layer views with entropy has never been measured at
all. So the prior is worth showing and is not evidence about this question in either
direction.

---

## 7. What would count as success, and what would count as failure

Declared before measurement, so neither can be renegotiated afterwards.

| stage | success | failure | failure is still publishable as |
|---|---|---|---|
| 1 | `PR(union) − PR(gray)` separated from shuffle and dimension-matched controls, with a legible depth profile | not separated | a measured ceiling result: depth is linearly redundant with the output channel at step level |
| 2 | locator gain on **both** benchmarks, gate fixed, paired interval excluding zero, gate-free SLA reported beside macro-F1 | no gain, or gain attributable to the auxiliary score alone | a negative with a mechanism, closing the depth-locator hypothesis |
| 3 | clean accuracy up at equal or better error detection, locator fixed, length-controlled | no gain | evidence the gate's deficit is not representational |

No promotion on development data. A frozen selection rule and primary endpoints precede any
untouched confirmation, and the full cached population remains **development** evidence.

---

## 8. Operational rules

- `hidden_states[L]` is **never** read: HuggingFace applies the final norm before appending
  it, so reading it double-norms and corrupts the KL reference every other layer is measured
  against. Reconstruct `x_l = x_{l-1} + a_l + m_l`.
- `HID_PROJ_SEED = 20260811` **never** changes — a changed seed silently invalidates every
  cross-cell comparison.
- `cov_eigs` stays **float32**; residual Gram eigenvalues exceed the float16 maximum on a
  trained model (47,008 non-finite entries on record from exactly that overflow).
- Cluster gate order stays **smoke → N=30 pilot → full**. A non-login `ssh aircc` shell leaves
  `SLURM_CONF_SERVER` unset and every `squeue`/`sbatch` dies with `fatal: Could not establish
  a configuration source` — it reads exactly like an outage and is not one.
- Attention was never captured for any population in this project. An attention channel is a
  new extraction, not an analysis.
- **`GIT_LFS_SKIP_SMUDGE=1` on every worktree creation**, both lines, otherwise each new
  worktree drags ~28 GB of pickles nobody needs. Standing rule from Omri, 2026-09-18.
- Disk: 43 GB free as of this writing. The two ENOSPC failures during this worktree's
  creation were caused by a 29 GB worktree that has since been deleted, not by this repo's
  size. The sparse cone can be widened if a stage needs it; this worktree is currently 139 MB.
- **Do not run `git sparse-checkout add` or `reapply` here** — both wipe local-only ignored
  files, which already cost us Codex's local npz results in the atlas worktree on 2026-09-17.
  Stage a path outside the cone with `git add --sparse`.

---

## 9. Division of work with the token-probability line

**CT7's gate-free SLA is computed there, not here.** *Omri's ruling.* The number belongs to
the project; the computation belongs to the token line, where it is already task A, the
roster and OOF scores are already loaded, and
`scripts/diagnostics/gate_isolation_token_lsml_v1.py` already computes gate-free SLA — it
needs only CT7's step scores fed through it. Two sessions computing it separately is exactly
how a project ends up with two disagreeing values of "CT7 SLA". One implementation, one
number, shared.

Correction to my own handoff, which that session needs: **do not re-derive CT7.**
`CT7_DEV_SCORES.npz` is backed up on Drive. The "about 270 s" I wrote understated the real
work — it covered only the digitfree bank, and not the BOCPD channel or the chosen-token
statistics.

Note for this worktree: `spectral_utils/frozen_locator_ct7.py` and
`results/chosen_token_calibration_v1/` are not present in the main checkout — they live on
the token line's branch. Another reason the computation belongs there.

---

## 10. What I am asking to be approved

1. **Stage 0** — verification and provenance only. No metric, no claim.
2. **Stage 1** as specified in §5: broad bank primary, CT7 secondary, three controls, no
   go/no-go authority over the line.
3. **Stage 3 scheduled independently of Stage 1**, on the argument in §2.3 that the gate is
   the weakest component and the only one where the answer-level geometry is native.
4. **Stage 2 deferred** until Stage 1's geometry exists, then proposed as one variant for
   discussion before it is built.
5. The axis contract in §4, including the declaration that the answer-only versus pooled
   fitting comparison remains open and will not be silently resolved by default.
