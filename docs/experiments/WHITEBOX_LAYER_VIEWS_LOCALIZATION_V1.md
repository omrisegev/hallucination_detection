# Research proposal — the depth channel on the localization population (v3)

Claude, 2026-09-18. Branch `claude/whitebox-layer-views-v1`,
worktree `.worktrees/whitebox-layer-views-v1` (sparse).
Data handoff: `docs/HANDOFF_WHITE_BOX_LAYER_VIEWS.md`.
Line this is measured against: `docs/HANDOFF_TOKEN_PROBABILITIES.md`.

**Status: proposal, revised after review. Nothing measured, nothing launched.**

v2 incorporated the review of 2026-09-18; its two breaking points are marked **[BREAKS v1]**
and the stage order changed so the gate runs first. **v3 adds what a history-mining pass over
all worktrees turned up**: the negative per-layer-fusion prior (§0.1), the correction that 9.69
is not a participation ratio at all (§1), the gate endpoint that actually decides Stage 1
(§6), the mandatory length / random-step / final-layer comparator rows (§7), and the binding
lessons register (§13).

---

## 0. The claim under test

Step 414 found that the output distribution of one greedy pass supports about 2.46
conditionally independent signals, and that more cannot be manufactured by adding further
transforms of it. Step 421 extracted, for the whole localization population, a field that is
not a transform of that distribution: how the distribution is *built* across 36 layers.

Two separable claims, deliberately not allowed to stand in for each other:

- **(A)** the depth field adds measurement dimensions the output-distribution bank lacks;
- **(B)** the depth field improves localization or the no-error decision.

The project has a documented case of these coming apart — Step 415 raised the effective count
2.46 → 2.78, the largest addition ever measured here, while quality *fell* 0.73 points.

### 0.1 The prior this line must state up front, and v1/v2 did not

**Per-layer lens fusion has already been tried in this project, and the registered primary was
negative.** Steps 243–245b, record `docs/experiments/WHITEBOX_LAYER_FUSION_RESEARCH_RECORD.md`:

| arm | macro AUROC |
|---|---|
| `lens-96` (4 metrics × 3 taps × 8 layers), DUFS-LIU fusion | 0.7253 |
| **final-layer target NLL — a single number** | **0.7298** |
| all-layer residual fusion | "much weaker"; depth NRM ~11.2 AUROC points below final-layer NLL |

A 245-summary label-free screen then reached 0.784612 against a strengthened per-cell atomic
oracle at 0.784186 — **+0.000426, 95% paired interval [−0.006351, +0.006833]**. Depth gains
also reversed under whichever transfer axis was not held out: NRM was +0.438pp leave-model-out
and −0.244pp leave-dataset-out; organic layer grouping was +0.136pp LODO and negative on *every*
other control.

So the honest prior is stronger and more specific than the 0.8677 correlation v2 cited: **the
incumbent that every depth arm must beat is the final-layer statistic**, on a different task
(final-answer detection) but with the same field and the same fusion machinery. This does not
close the localization question — different task, different readout, and the gate was never the
target there — but it is the result this line is trying to overturn, and it goes in front of the
reader, not in an appendix.

**Consequence for reporting:** `final_lens_H` / the cached entropy is a **mandatory row in every
depth table**. A depth arm that does not clear the final layer has not cleared the incumbent.

---

## 1. The measurement-tag rule

**Every participation-ratio number must carry three tags, and two numbers may never be
compared unless all three match:**

| tag | values |
|---|---|
| **observation unit** | token · step |
| **centring** | marginal · conditional (within label class) |
| **column construction** | raw streams · family virtuals |

`2.46` is **(step, conditional, virtuals)**. Anything compared to it must be all three.

This is a rule rather than a one-off correction because **two independent sessions have now
failed at exactly this point**: the token line reported an "effective rank" of 9.69 that was
(token, marginal, raw) against anchors that were (step, conditional, virtuals) — three
mismatches at once — and v1 of this proposal planned a participation ratio over 468 raw depth
columns against an anchor computed over 13 family virtuals. Same failure, different tag.

Known values, tagged:

| number | tags | what it is |
|---|---|---|
| 2.75 | step, marginal, virtuals | 13 family virtuals |
| **2.83** | **step, conditional, virtuals** | 13 family virtuals, includes `n_steps` |
| **2.46** | **step, conditional, virtuals** | the **12 within-answer** families, `n_steps` removed |
| 1.80 | step, conditional, views | CT7's seven views |
| 9.69 | **not a participation ratio at all** | see below |

**Correction to v2's tag table.** v2 labelled 9.69 as "(token, marginal, raw)". That is wrong
and it understates the problem. 9.69 is an **inverse participation ratio of the weight vector**,
`1 / Σ(|w_i|/Σ_j|w_j|)²` — it equals 11 when the eleven weights are uniform and 1 when one
channel carries everything. So 9.69/11 says *"L-SML's weights came back close to uniform"* and
says nothing whatever about how many independent directions the views span. It is also
**sign-blind**, being a function of `|w|`: `chosen_surprisal`'s −0.234 pushed the count *toward*
"uniform" exactly as an equal positive weight would, so the statistic was structurally unable to
see the single most interesting thing that fit did.

**Naming rule, adopted now to prevent the third occurrence.** Three different quantities in this
codebase are called some variant of "effective rank":

| quantity | formula | role here | name to use |
|---|---|---|---|
| weight concentration | `1/Σ(|w_i|/Σ|w_j|)²` | diagnostic of a fit | **`weight_ipr`** |
| view redundancy | `(Σλ)²/Σλ²` of a correlation matrix | the **instrument** in Stage 2 | **`view_pr`** |
| spectrum shape of `cov_eigs` | `exp(spectral entropy)` | a **feature** in Stage 1 | **`spectral_effective_rank`** |

Stage 1 feeds the third as an input while Stage 2 measures the second as an instrument. They get
disjoint names in code and in every table, or they will be conflated exactly as above.

CT7's **1.80** is *not* open (v1 said it was): it is PR 1.80 over 7 views. And 1.80 < 2.46 is
itself the argument for choosing the measurement reference bank by **coverage** rather than by
candidate status — the frozen candidate is the *most* redundant bank we own.

---

## 2. What is actually new — and the split that governs everything

### 2.1 Two structurally different sub-channels **[BREAKS v1]**

| sub-channel | arrays | varies within an answer? | eligible for |
|---|---|---|---|
| **depth telemetry** | `lens_H`, `lens_logp_tgt`, `lens_logp_top1`, `lens_kl_final` `[3,36,T]`; `resid_norm` `[36,T]` | **yes**, per token (468 channels) | locator **and** gate |
| **answer geometry** | `cov_eigs` `[36,32]`, `hid_proj` `[36,256]` | **no**, constant per answer (10,368 channels) | **gate only — arithmetically** |

For the second row this is not a hypothesis. Constant within answer ⇒ zero variance after
answer-standardization ⇒ exactly zero contribution to a step locator. Identical to `n_steps`.

**The consequence v1 missed.** v1 identified this mechanism, used it to route the geometry to
the gate, and then failed to apply it to its own estimator. Including `cov_eigs`/`hid_proj` in
the participation-ratio computation would **inflate the count mechanically, in exactly the way
`n_steps` inflated 2.46 to 2.83.** Therefore: Stage 2 reports the count **with and without**
the answer-constant sub-channel, and **the number compared against 2.46 is computed without
it.**

### 2.2 What each channel is, and is not, a transform of

| channel | relation to the existing output-distribution bank |
|---|---|
| `lens_*` at layer 35, resid tap | near-duplicate of existing views, up to full-vocab vs top-15 |
| `lens_*` at layer < 35 | **not** a transform of the output distribution — a different distribution |
| `lens_kl_final` | a *contrast between depths*; no output-distribution analogue |
| `resid_norm` | representation magnitude; no analogue |
| `cov_eigs`, `hid_proj` | representation geometry; no analogue; answer-constant |

### 2.3 Two fidelity notes

`lens_H` is **full-vocabulary**; the cached `token_entropies` is **top-15 renormalised**.
Different statistics, not a rescaling. `final_lens_H` is the comparable one.

**`cov_eigs` is not EigenScore.** INSIDE (Chen et al., ICLR 2024) uses covariance eigenvalues
of sentence embeddings over **K = 10 sampled generations**; ours is token covariance within
**one teacher-forced pass**, K = 1 by construction — the multi-pass diversity that carries
their signal is absent. Inspired by, never labelled as, that method.

---

## 3. The two roles

**Locator** — rank steps *within* one answer; readout is argmax over steps. Only within-answer
variation can change the output. Input: token-resolved channels only.

**Gate** — decide *whether* an answer contains an error; needs a scalar comparable *across*
answers. Between-answer variance is the signal here, not a nuisance.

### 3.1 Why the gate goes first — the narrow version of the argument **[BREAKS v1]**

v1 argued "a gate that is a thresholded function of the locator cannot discriminate
independently of it". Too broad. The precise statements are:

1. **What is established.** The gate's information today is *entirely determined by the
   locator's ranking* — CT7's gate is a frozen tail-15 statistic of its own entropy bank, and
   the token arm's gate is a threshold at 0.33 on the fused locator composite. So the gate
   contributes **zero independent evidence**.
2. **What does not follow.** That a differently-fed gate will be better. Only that it *can* be.
3. **What does follow strongly.** The evidence class that *can* add to the gate and *cannot*
   add to the locator is exactly the **answer-constant** class — which is precisely what
   `cov_eigs` and `hid_proj` are.

So the routing is a **prediction**, not a default.

### 3.2 The measured deficit, verified

From `results/claude_feature_bank_token_lsml_v1/GATE_ISOLATION.json`:

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

3.20–5.31 points in **every one of the eight cells**; clean accuracy 27.46%–55.60%.

### 3.3 The prior that orders the stages

The only genuinely independent channel this project ever measured — answer length in steps,
orthogonal to all thirteen evidence families at |corr| ≤ .05 — turned out to be **gate-only**,
because it is answer-constant. Our two answer-level arrays are in exactly that structural
class. **The highest-prior prediction of the entire white-box line is therefore: improvement
at the gate, zero at the locator.** The gate is also where 3.20–5.31 points per cell are
sitting. It goes first.

---

## 4. The axis contract

The token line's documented defect was fitting on the pooled between-answer axis and reading
out within the answer: the fit saw mostly between-answer variance — long and hard answers run
hotter — and `chosen_surprisal` took a **negative** weight in all five folds despite being
oriented so higher means more risk. Between answers surprisal tracks difficulty; within an
answer it tracks error.

| | locator | gate |
|---|---|---|
| decision | rank steps inside one answer | one scalar per answer, thresholded across answers |
| informative axis | **within-answer only** | **between-answer, necessarily** |
| representation | every channel standardized **within** the answer | answer-level statistic; cross-answer comparability is the point |
| principal confound | step length, step position | **answer length**, difficulty, subset |
| required control | — | explicit length control, always |

The length warning is concrete: `n_steps` alone reaches .667 pooled AUC, entirely from the
between-answer length prior. Any answer-level gate channel will carry that prior unless it is
explicitly controlled, so the gate's length control is not optional hygiene — it is the first
control, and an uncontrolled gate number is not reportable.

**Representation axis and weight-fitting scope are different things.** The representation
contract above is fixed. Fitting scope has two declared arms — **answer-only** (the mandate's
stated primary objective) and **pooled donor-fold** (what CT7 and the token arm actually do).
`PROGRESS.md` records that the matched comparison between them was specified three times and
never run, and that no decision to abandon answer-only was ever recorded. This proposal does
not silently default: whichever arm runs is declared with its access level, and if both run
they are separate rows, never averaged.

---

## 5. Pre-registered before any fitting

Written down now so they cannot be discovered afterwards.

### 5.1 Fusion before versus after the readout — **the decisive contrast** **[NEW]**

In the token arm, L-SML beat equal averaging with a paired interval excluding zero — the first
time in this project. The leading mechanistic hypothesis is unflattering: at token level
per-channel noise is large and independent, so L-SML has real work to do down-weighting noisy
channels — work the step-level Top-10 mean already does for free by averaging. On that
hypothesis **L-SML is winning as a noise-weighter, not as a dimension-finder.**

The per-layer lens channels are in exactly the same structural situation: noisy, and
near-duplicate between adjacent layers. So L-SML will very plausibly "work" here too, for the
same unflattering reason. Therefore, pre-registered from day one:

> **Same bank, same folds: fuse before the readout versus fuse after it.** If the L-SML
> advantage collapses when fusion is applied after the step readout, it is noise weighting and
> not a new dimension — and that result is reported as the explanation, not buried.

### 5.2 The last-layer alignment control — an **entry gate**, not an analysis **[NEW]**

`lens_H` at the final layer is, up to the top-15 versus full-vocabulary difference, the same
statistic as the cached `token_entropies` — and `final_lens_H` sits in the *same npz*. This is
a free positive control and it runs before anything else:

1. **Alignment check.** Correlation at the last layer against `final_lens_H` must come back
   ≈ 1.0. If it does not, there is a token-alignment bug, and it is found *before* the science
   rather than after. **No stage proceeds until this passes.**
2. **Depth-decay curve.** Correlation against the existing entropy family as a function of
   distance from the last layer. This is the most direct available answer to "is this just
   another transform of the same signal, and if so from which depth does it stop being one" —
   considerably more direct than any participation ratio, and it is nearly free.

### 5.3 Endpoints, frozen now

| component | endpoint | explicitly NOT used |
|---|---|---|
| locator | **gate-free per-subset Step-level Localization Accuracy** (Mind-the-Gap protocol), erroneous answers only, no no-error decision at all | ProcessBench macro-F1 |
| gate | clean-answer accuracy **and** erroneous-answer detection, reported **separately**, per cell | any single merged score |

**Macro-F1 is not admissible for a stage that changes the gate.** A changed gate makes macro-F1
incomparable to CT7's 41.19, for exactly the reason that already produced one wrong conclusion
in this project when 41.19 was compared against 34.08 across different gates. The gate-free SLA
is structurally gate-immune, which makes it the right instrument precisely here.

**The comparison anchor is CT7's gate-free SLA = 39.89** — measured since v1 was written — not
41.19.

---

## 6. Stage plan

### Stage 0 — verification and provenance. No metric, no claim.

- **S0.1** Verify the Drive backup: nine directories, 13,769 `rows/*.npz`, manifests, byte
  counts. Drive is the durable copy; cycle-2 is likely being retired.
- **S0.2** Verify the join. *Done:* `JOINED.json` resolves 13,769 records over exactly the nine
  cells, matching the extraction manifests.
- **S0.3** Anchors pinned — see §1. Source:
  `.worktrees/a6-s0b/docs/experiments/EVIDENCE_DOMAIN_INDEPENDENCE_V1.md`. Estimator:
  `(Σλ)²/Σλ²` over the correlation matrix of **family virtuals**, each the equal mean of its
  z-scored oriented members; population the **94,203 labelled PRMBench steps** (14.0% error),
  *not* the eight ProcessBench cells the SLA work runs on. Any PR reported here runs on that
  same population and says so.
- **S0.4** **The §5.2 alignment control.** Gate for everything downstream.
- **S0.5** Plan a move off `cycle2_*`.

Reduction runs **on the cluster** (CPU): 5.5 GB of npz, and the within-answer shuffle null must
be computed at token level before the step reduction, so it cannot come from a reduced artefact.

### Stage 1 — **the gate** (was Stage 3; now first)

Highest-prior prediction of the line, and the largest measured deficit.

- **Population.** The gate is **ProcessBench-only**: 6,800 answers over 8 cells, 4,442 erroneous
  / 2,358 clean. PRMBench does not score this stage (it scores the PR measurement and the
  locator). Stated because a "9-cell gate result" would be meaningless.
- **Input — rotation-invariant summaries only.** The prior lineage explicitly ruled the geometry
  family inadmissible until its pooling semantics were verified, and allowed *only*
  rotation-invariant summaries. That contract is already implemented in
  `spectral_utils/whitebox_layer_fusion.py :: extract_geometry()`: `hid_proj` enters **only**
  through cosines and normalized distances to the final and adjacent layer, `resid_norm` only
  through log-ratios, `cov_eigs` only through top-share, negated `spectral_effective_rank` and
  negated spectral entropy. **Reuse it; do not invent a second summary set, and never consume
  `hid_proj` coordinates raw.** The INSIDE-inspired direction is the mid-depth covariance
  spectrum, with the K=1 caveat stated wherever the number appears.
- **Pre-fit numerical check, before any fitting.** Report the covariance **condition number** of
  the input bank. `cov_eigs` spans orders of magnitude by construction, and the project's
  precedent is unambiguous: a collapse was traced to the median second-moment condition number
  rising 65.8 → 23,334.6, and **centering, not scaling, was the culprit** — scale-only equal
  weighting recovered pooled OOF AUROC .675 → .716. Centering and scaling decisions are taken
  and reported separately.
- **The founding constraint on what a gate may be fed.** Answer-standardized fused step risks
  separate erroneous from clean answers at **AUC 0.43–0.52 — chance** — while raw telemetry
  summaries of the same answers separate them at **0.74–0.78 in every cell**. A gate must be fed
  a cross-answer-comparable **raw** statistic. The label-free **q = 0.3 quantile constant** is
  the precedent (it matched a fitted threshold: 31.16 vs 31.31), not a fitted threshold.
- **Incumbent — corrected.** The gate to beat is **tail15-Top10 at q = 0.33**, *not* LOCO-5.
  Step 423 measured tail15 beating LOCO-5 for both locators (+4.21pp for CT7, +3.25pp for the
  token arm), and LOCO-5's own label-selected optimum at 0.41 still sits 0.48pp *below* tail15
  at its registered setting. **LOCO-5 is the worse gate, not a mistuned one**, so its 0.41
  ceiling is worth close to nothing.
- **Controls, in order.** (i) answer length — first, always; an uncontrolled number is not
  reportable; (ii) tail15 @ q=.33 as incumbent; (iii) a same-signal control quantifying how much
  of any gain is merely "a gate not derived from the locator"; (iv) a presence/count confound
  control that must land near chance if the effect is real (precedent: `digit_presence` .568
  against `digit_rate` .729).
- **Locator held fixed**, so the gate's contribution is not confounded with the locator's.
- **Endpoint — and this is the part that decides the stage.** Clean accuracy and error detection
  separately per cell **is not sufficient**. Steps 367–371 are the trap: gates that improved
  answer-level separability substantially — one moved family-macro F1 .6500 → .6936 and AUROC
  .7423 → .7926 — nonetheless **reduced** end-to-end ProcessBench localization (36.62 → 35.53),
  because 797 erroneous answers were newly closed against 328 reopened and **338 exact
  localizations were lost against 85 gained**. Therefore Stage 1 reports, at a **matched opened
  fraction**:

  | required | why |
  |---|---|
  | clean accuracy, per cell | the deficit being targeted |
  | erroneous-answer detection, per cell | the cost side |
  | **exact localizations gained vs lost** | the number that actually decides |
  | opened fraction per cell | without matching it, the comparison is not fair |

  Separability is not an operating point. A Stage-1 "win" on the first two rows and a loss on
  the third is a loss.
- **Access declared**: answer-only, pooled-unlabeled, transductive-within-cell, or externally
  calibrated are four different scopes. The registered midrank rule is **transductive** — it
  uses other answers' scores at scoring time — and that must be declared, not inherited silently.

### Stage 2 — the depth-redundancy geometry (locator side)

**Decision question.** Conditioned on the output-distribution bank, how many *additional*
effective dimensions does the **token-varying** depth sub-channel carry at step level?

- **Reference bank: broad bank primary, CT7 secondary.** Coverage, not candidate status: CT7 is
  the densest bank we own (1.80 of 7 views, six of them one entropy signal), so measuring an
  addition against it inflates the addition by construction; and the ceiling under test was
  measured on the broad bank's 13 families.
- **Estimator: virtual-level**, per §1 — the depth field is reduced to depth family virtuals on
  the same recipe (natural partition: tap × quantity × depth band), PR computed over `12 + k`.
- **Answer-constant arrays excluded** from the number compared to 2.46, and the with/without
  pair reported (§2.1).
- **Two distinct reference levels, which answer different questions:**

  | reference | what it is | what it answers |
  |---|---|---|
  | **2.7–2.9 level** | measured PR for every subset of 3/4/5/6/7 output-distribution families | "does depth behave like *more of the same*?" — a **comparator** |
  | **within-answer token shuffle** | shuffle tokens within each answer, independently per channel, R replicates, recompute readout → virtuals → PR | "is any of this above noise?" — a **null** |

  The 2.7–2.9 level is **not** a null: it was measured for transforms of the output
  distribution only, and the white field is structurally a different modality. Both are needed;
  they answer different questions.
- **PR as a curve** against the number of depth virtuals added, not one scalar.
- **Sub-questions, free once the matrix exists.** TriLens: PR({resid}) vs
  PR({resid, attn, mlp}) — are the three taps independently informative? Plus the
  adjacent-layer correlation profile along depth, read beside the §5.2 depth-decay curve.
- **Uncertainty.** Per-cell and pooled, paired **source-group** bootstrap, not per-answer.
- **Authority.** None over the line. PR measures linear redundancy in a correlation matrix and
  knows nothing about the label; Step 415 is the counter-example. A zero closes only the narrow
  claim that depth is what unlocks L-SML above three sources.

### Stage 3 — the locator variant, derived from Stage 2's geometry

One variant, discussed before it is built. Constraints fixed now: a handful of channels, not
432; the fusion core with and without the component on matched IDs; learned fusion versus equal
averaging under the component; the §5.1 before/after contrast; gate held fixed; effective count
reported beside every fusion claim (Step 205 guard: L-SML is information-free at three features
and numerically undetermined at four); no digit channels; CT7 frozen.

---

## 7. Reporting contract

Required by Omri, 2026-09-18, and applies to every artefact this line produces.

1. **Numbers and comparisons go in tables.** Not prose, not a paragraph of figures.
2. **Every feature and method is explained in words**, not only by its code identifier. A
   reader who has never opened the repo must know what `lens_kl_final` or "family virtual"
   *is*. Code names may appear beside the explanation, never instead of it.
3. **Effective views / effective independent signals are reported beside every fusion claim**,
   with all three tags from §1 attached.
4. **Every quality table carries the simple-average control and the existing methods** — not
   only the new arm against its own ablation.

Mandatory comparator rows:

| row | gate-free mean SLA | note |
|---|---:|---|
| **CT7 (frozen candidate)** | **39.89** | the anchor for this line |
| token-level L-SML | 35.92 | development-only |
| token-level equal mean | 32.59 | **the simple-average control** |
| Chen et al. Shannon Drop (Mind the Gap) | 39.27 | published; a *derivative* readout, ours is a level |
| Chen et al. Shannon Avg (Mind the Gap) | 25.34 | published baseline |
| **`final_lens_H` / cached entropy — the final layer alone** | to measure | **the depth incumbent** (§0.1) |
| **`argmax(step length)` — the length prior** | to measure | **mandatory** |
| **`random_step`** | to measure | **mandatory** |
| chance | 16.58 | |

**The last three rows are standing requirements, not optional context.**

- The **final-layer** row is the incumbent from §0.1: depth fusion has already lost to a single
  final-layer number once (0.7253 vs 0.7298, different task).
- **`length` and `random_step`** are mandatory in every ProcessBench localization table by a
  standing instruction from Step 354, and the reason is quantitative: `argmax(step length)`
  alone reaches **29.7% raw exact peaks** against entropy Top-10's 31.5% and random-step's
  15.5%; under the full gated protocol the length control scores **33.69**, only 1.75pp below
  entropy Top-10. Error steps have a median of 107 tokens against 72 for others. Any readout
  that does not clear the length row has measured the length prior.
- Subsequent work confirmed the coupling is mostly **legitimate evidence and not separable**:
  removing it costs every stream .04–.06 AUC, length-calibrating CT7 costs −8.16pp, and the
  cost is uniform across short and long chains (interaction +0.43 [−3.68, +4.60]). So the rule
  is *report length as a declared control row*, **not** *calibrate it out of the readout*.
- Report **short-chain and long-chain subsets separately**. The macro mean hides a real
  asymmetry: our level readout beats the published derivative readout on GSM8K and loses
  6.6–12.5 points on the long subsets, and `LEN` alone scores 41.06 SLA on GSM8K against 26.44
  on the long ones.

Separate panel, different task, **never pooled**: the prior white-box lineage on final-answer
detection, `results/whitebox_vs_graybox_matched_v1/` — white ≈ gray on 31,440 candidates over
13 cells (AUROC −0.0013, [−0.0169, +0.0123]), mean per-cell Spearman 0.8677. That correlation
is between **fused final scores**, on a different task and representation: a prior about
redundancy *of outputs*, not about dimensional independence *of channels*. The correlation of
the layer views with entropy has never been measured at all — §5.2 measures it for the first
time.

---

## 8. Success and failure, declared in advance

| stage | success | failure | failure still publishable as |
|---|---|---|---|
| 0 | last-layer correlation ≈ 1.0 | it is not | an alignment bug caught before the science |
| 1 (gate) | clean accuracy up at equal-or-better error detection, length-controlled, locator fixed | no gain | evidence the gate's deficit is not representational |
| 2 (PR) | added count separates from the shuffle null **and** from the 2.7–2.9 level | it does not | a measured ceiling result: depth is linearly redundant at step level |
| 3 (locator) | gain on **both** benchmarks, gate fixed, paired interval excluding zero | no gain, or gain explained by the auxiliary score alone, or collapsing in the §5.1 before/after contrast | a negative with a mechanism |

No promotion on development data. Frozen selection rule and endpoints precede any untouched
confirmation; the full cached population remains **development** evidence.

---

## 9. Operational rules

- `hidden_states[L]` is **never** read — HuggingFace applies the final norm before appending it,
  so reading it double-norms and corrupts the KL reference. Reconstruct `x_l = x_{l-1}+a_l+m_l`.
- `HID_PROJ_SEED = 20260811` never changes; a changed seed silently invalidates cross-cell
  comparisons.
- `cov_eigs` stays **float32** (47,008 non-finite entries on record from float16 overflow).
- Cluster gate order: **smoke → N=30 pilot → full**. A non-login `ssh aircc` shell leaves
  `SLURM_CONF_SERVER` unset and every `squeue`/`sbatch` dies with `fatal: Could not establish a
  configuration source` — reads like an outage, is not one.
- Attention was never captured for any population here; an attention channel is a new
  extraction, not an analysis.
- **`GIT_LFS_SKIP_SMUDGE=1` on every worktree creation**, both lines.
- **Never** `git sparse-checkout add`/`reapply` in this worktree — both wipe local-only ignored
  files. Stage outside-cone paths with `git add --sparse`.

---

## 10. Division of work with the token-probability line

CT7's gate-free SLA is computed **there**, once — the number belongs to the project, the
computation belongs to that line, where the roster and OOF scores are already loaded and
`scripts/diagnostics/gate_isolation_token_lsml_v1.py` already computes it. Two sessions
computing it separately is how a project ends up with two disagreeing values of "CT7 SLA". That
number is now in: **39.89**.

Correction to my own handoff for that session: **do not re-derive CT7** — `CT7_DEV_SCORES.npz`
is on Drive, and my "about 270 s" estimate covered only the digitfree bank, not the BOCPD
channel or the chosen-token statistics.

---

## 11. What is being approved

1. Stage 0, including the §5.2 alignment control as a hard entry gate.
2. The §4 axis contract, with answer-only versus pooled left explicitly open.
3. **Stage 1 = the gate, first.**
4. Stage 2 as specified: token-varying sub-channel only for the 2.46 comparison, with/without
   pair reported, virtual-level estimator, both reference levels, no authority over the line.
5. Stage 3 deferred until Stage 2's geometry exists, then proposed as one variant.
6. The §5.1 before/after contrast pre-registered now.
7. The §7 reporting contract.

---

## 13. Binding lessons from the project history

Mined from `HISTORY.md` across all worktrees. Each is a rule this line follows, with the step
that earned it. Items already absorbed into sections above are not repeated.

### 13.1 Holdout structure - the 8 ProcessBench cells are fully crossed for the first time

Depth gains in this project have reversed under whichever transfer axis was **not** held out:
NRM was +0.438pp leave-model-out and -0.244pp leave-dataset-out; organic layer grouping was
+0.136pp LODO and negative on every other control (LOMO -0.049, LOCO -0.091, same-model -0.075,
same-dataset -0.154). The old 13-cell roster was not fully crossed, so a simultaneous
dataset+model holdout left a single source.

**The 8 PB cells are 4 subsets x 2 models - properly crossed.** So this line reports
**leave-subset-out and leave-model-out separately**, plus both fixed-axis controls. A gain on
one axis that reverses on the other is not a gain.

### 13.2 L-SML diagnostics that must ship with every weight vector

- **Degeneracy.** Equation 15 is identically zero below 4 views and *undetermined* at 4: the
  same numbers in a non-contiguous versus contiguous array differ by 5.55e-17 and that flipped a
  partition, swinging AUROC 0.6833 to 0.7802 - a 9.68pp swing with no deliberate jitter. Any
  depth roster with 3-4 groups is in that regime; run the invariance check and report the
  `degenerate` / near-tie flag.
- **The small-m guard silently replaces the learned outer stage with equal weights.** In the
  current leading arm `cross_small_m_guarded` is **true in all five folds**, within-group SML
  eigenvectors sit at |cos| >= .93 with uniform (mostly >= .997), and a real cross-group
  eigen-solve costs about 3pp. A guarded fit **is an average** and must be described as one.
- **Two-member groups are gauge, not inert.** An unidentified pair returns equal-and-opposite
  weights, making its contribution `w*(z_a - z_b)` - a real difference channel whose split is
  fixed by the group structure rather than the data. Report the **identified rank**, not the
  column count, and ship a collapsed-pair control.
- **Negative weights are the norm and are a diagnostic, not a result.** A complementary view
  once received a negative coefficient in 99.98% of answers and collapsed the learned fuser
  (equal 41.40 vs learned 30.53); 59.19% of native fits contain a negative coefficient. Report
  the sign of every fitted weight and the fraction of fits carrying a negative one.
- **Count and publish inadmissible, unconverged and fallback fits per fold.** Never average over
  silently substituted arms.

### 13.3 Orientation and label hygiene

- **Never `max(auc, 1-auc)`.** Found three separate times here. On a supervised score it inflated
  a cell by +12.6pp; as a fold-wise rule it created a one-sided noise floor that *credits a view
  more the closer its mapping sits to chance*. Orientation comes from a declared label-free
  anchor.
- **Freeze detector scores before opening targets**, and report the post-selection interval. A
  33-candidate gate search previously returned a +0.271pp winner whose post-selection interval
  was [-1.079, +1.594].
- **A label-selected optimum is a ceiling, never a candidate.**

### 13.4 Reporting discipline

- **Three lanes, never merged**: raw SLA, gated exact-error accuracy, and end-to-end ProcessBench
  macro-F1 including clean-trace abstention.
- **Per-cell before pooled.** A macro mean is a summary, never a gate - a previous verdict was
  wrong precisely because a per-feature mean across cells hid an effect that was +12pp on 7 cells
  and flat on 17.
- **Intersect IDs before comparing**; report coverage as its own axis. Unmatched row sets have
  reversed orderings here more than once.
- **Positive class fixed once: incorrect = 1.** A prior white-box report's AUPRC was unreusable
  because it had used correctness as the positive class.
- **Ratio metrics need a declared minimum denominator**, set before looking at the p-value; a
  headline p = 0.0096 was withdrawn when cells with sub-1pp denominators were excluded.
- **Check a cell's existing caveats before making it the anchor of a comparison.** A headline gap
  moved +37.99pp to +22.53pp when an anchor cell with 6 positives out of 256 - where trace length
  alone scores 0.925 - was replaced.
- **Name which fusion insertion point a channel enters at.** The project maintains an explicit
  five-point map (token/step evidence, predictor-residual ensemble, background-before-innovation,
  answer gate, final decision readout). A gain at one is not a gain at another, and both the
  "versus simple combination" and "versus unstable fit" contrasts must be reported.

### 13.5 Verification discipline for Stage 0

- **Prove the gate is non-vacuous.** Step 421's own pre-flight had a gate that would have
  "passed" with zero rows checked, because ProcessBench rows key on `id` and PRMBench on `idx`.
  Assert a positive row count.
- **Re-verify the label contract from the raw source, not a derived NPZ.** PRMBench annotations
  are one-based; a writer that used `flags[step]` shifted every annotation and dropped
  final-step errors. Repair changed 6,035 target arrays and 15,147 step flags, and earlier
  reviews missed it because they validated against the derived NPZ.
- **Enumerate every downstream consumer of the layer sidecars and re-derive it.** Stale numbers
  have three independent carriers here - resume-safe CSV rows, enlarged pools, and lookup NPZs -
  and the standing lesson is that *a sampled self-check is not a freshness guarantee*.
- **Validation evidence travels inside the artefact's own metadata.** A resume that measured
  nothing once overwrote the real verdicts of four completed cells, and the corrupted report
  still looked populated. A run that measured nothing must never overwrite one that did.
- **Every fitted hyperparameter needs an inertness guard.** A previous sweep never ran the
  mechanism it was named for - the regularization parameter changed the answer in 0 of 350
  groups. If changing a parameter never changes an output, the mechanism is not running.

### 13.6 Structure-free controls

Ship a shuffled or structure-free version of any structured mechanism, **and** a label-using
ceiling that bounds what it could possibly fix, *before* running it. Precedents: a
dependency-free uniform control beat the real numeral-provenance mechanism (36.24 vs 35.66) and
the label-using ceiling had already bounded the fixable share at 14%; and on the 20-stream bank
equal weighting on the right 6-stream subset reproduced the frozen leader (40.27 vs 40.37) with
every fitted rule worse than equal - **selection was the gap, not weighting**.

### 13.7 What the depth field's claim to novelty actually is

An exhaustive label-free screen of **5,443 definitions**, 114 retained representatives across 13
families and **2,879 classified pairs**, found **no pair passing the combined independence
contract**. The structural conclusion was that more independent sources require internal layers,
repeated samples, or a second model. **The depth field is the first of those three ever
extracted in this project.** That is its entire claim to novelty, and it is an untested
prediction - not a property it has been shown to have.
