# Research proposal — the depth channel on the localization population (v2)

Claude, 2026-09-18. Branch `claude/whitebox-layer-views-v1`,
worktree `.worktrees/whitebox-layer-views-v1` (sparse).
Data handoff: `docs/HANDOFF_WHITE_BOX_LAYER_VIEWS.md`.
Line this is measured against: `docs/HANDOFF_TOKEN_PROBABILITIES.md`.

**Status: proposal, revised after review. Nothing measured, nothing launched.**

v2 incorporates the review of 2026-09-18. Two of its points break things v1 asserted, and
they are marked **[BREAKS v1]**. The stage order has changed: the gate now runs first.

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
| 9.69 | token, marginal, raw | not comparable to any of the above |

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

- **Input.** `cov_eigs` and `hid_proj`, reduced to a small answer-level summary; the
  INSIDE-inspired direction is the eigenvalue spectrum of the mid-depth token covariance, with
  the K=1 caveat stated wherever the number appears.
- **Controls, in order.** (i) answer length — first, always, and an uncontrolled number is not
  reportable; (ii) the current gate as incumbent; (iii) a same-signal control quantifying how
  much of any gain is merely "a gate not derived from the locator".
- **Locator held fixed**, so the gate's contribution is not confounded with the locator's.
- **Endpoint** per §5.3: clean accuracy and error detection, separately, per cell.
- **Access declared**: answer-only, pooled-unlabeled, or externally calibrated are three
  different scopes; a pooled unlabeled gate is a hybrid, not answer-only fitting.

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
| chance | 16.58 | |

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
