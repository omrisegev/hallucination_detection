# Plan — White-box per-layer views on the localization population (v1)

Claude, 2026-09-18. Branch `claude/whitebox-layer-views-v1`,
worktree `.worktrees/whitebox-layer-views-v1` (sparse).
Companion handoffs: `docs/HANDOFF_WHITE_BOX_LAYER_VIEWS.md` (the data),
`docs/HANDOFF_TOKEN_PROBABILITIES.md` (the line this is measured against).

Status: **plan only. Nothing has been measured. No experiment launched.**

---

## 0. Why this line exists

Step 414 concluded that more than roughly three conditionally independent sources cannot
be built by adding transforms of the output distribution of one greedy pass, and named
three ways out: internal layer states, multiple samples, or a second model. Step 421
delivered the first of those for the **entire** localization evaluation population —
13,769 answers, 145,597 steps, 6,968,779 tokens, nine cells, teacher-forced, zero gate
failures.

The single question this line exists to answer:

> **Does the depth axis carry measurement dimensions the output-distribution bank does
> not already carry — and if so, does that convert into localization?**

Those are two questions and they are deliberately staged in that order. The first is
cheap, label-light and falsifiable. The second is only worth asking if the first passes.

---

## 1. What the field can and cannot be used for

`cluster/layer_lens.py` fixes the axis order: `MODULES = ("attn", "mlp", "resid")`,
layers `0..35`, tokens in generation order.

**Token-resolved (usable for localization), 468 channels per token:**

| array | shape | channels |
|---|---|---|
| `lens_H` | `[3, 36, T]` | 108 |
| `lens_logp_tgt` | `[3, 36, T]` | 108 |
| `lens_logp_top1` | `[3, 36, T]` | 108 |
| `lens_kl_final` | `[3, 36, T]` | 108 |
| `resid_norm` | `[36, T]` | 36 |

**Answer-resolved (NOT usable for a step-level localization readout):**
`cov_eigs` `[36, 32]` and `hid_proj` `[36, 256]` have no token axis. They are answer-level
summaries. They can feed *detection* (the 24-cell final-answer task, where the prior
white-box lineage lives) but they cannot produce a per-step risk. This is a scoping fact,
not a defect, and it should stop anyone from quietly counting 432 + 1152 + 9216
"features".

`final_lens_H` (top-15 renormalised) is the quantity comparable to the cached
`token_entropies`. `final_lens_H_fullvocab` and `lens_H` are full-vocabulary and are
**different statistics**, not a rescaling.

---

## 2. Stage 0 — access and provenance (no science)

Local free disk is about 14.5 GB and the field is 5.5 GB, so the reduction runs **on the
cluster**, not locally. A full local pull is possible but leaves no headroom and is not
the plan.

- **S0.1** Verify the Drive backup is complete and readable: nine directories, 13,769
  `rows/*.npz`, manifests present, byte counts matching the cluster. The Drive copy is the
  durable one — cycle-2 is very likely being retired while our account is now cycle-3.
- **S0.2** Verify the join: every `row_id` in the nine `MANIFEST.json` files resolves in
  `results/localization_full_benchmark_v3/evaluation/JOINED.json` and the per-cell counts
  match (prmbench 6969; pb_{math,olympiadbench,omnimath}_{q4,q8} 1000 each;
  pb_gsm8k_{q4,q8} 400 each). *Already confirmed on the JOINED side: 13,769 records, nine
  cells, exactly those counts.*
- **S0.3** Pin the provenance of the existing participation-ratio anchors
  (**1.80 / 2.46 / 2.83**) to the Step 414–417 artefacts: which bank, which centring,
  which standardization, which population. Stage 1 is only interpretable against them if
  the estimator is reproduced **exactly**, on the same population, before the depth
  channels are added. If the estimator cannot be reproduced, Stage 1 reports its own nulls
  and does not cite those three numbers.
- **S0.4** Plan a move off `cycle2_*`. Not urgent, not blocking, not silent.

Stage 0 produces no metric and no claim.

---

## 3. Stage 1 — how many dimensions does depth actually add

This is the measurement the data handoff mandates before any fusion, and it is the whole
of the next stage.

**Decision question.** Conditioned on the output-distribution bank already in use, how
many *additional* effective dimensions does the depth field carry at step level — and is
that number distinguishable from what the same pipeline returns on shuffled data?

**Representation.** The contract is fixed by the evaluation, not chosen here:

- step-level, via the project's adopted **top-10 token-mean within step** readout;
- **answer-local standardization** of each channel before anything else. Section 4 of the
  token-probability handoff is explicit that fitting on the pooled between-answer axis and
  reading out within the answer is what produced the misleading 9.69. This line does not
  repeat that.
- **within-label-class centring**, because that is the axis on which the three-source
  threshold was defined. A marginal ratio is not comparable and will not be reported as if
  it were.

**Three nested measurements, same population, same folds, same estimator:**

| # | channel set | what it answers |
|---|---|---|
| a | output-distribution bank alone | reproduces the anchor; sanity check on the estimator |
| b | depth field alone (468 channels) | how much structure depth has *by itself* |
| c | union | the only number that matters: **PR(c) − PR(a)** = dimensions *added* |

**Noise floor, run for all three.** Shuffle tokens within each answer, independently per
channel, R replicates, then recompute the entire step readout and the ratio. Every number
above is read against **its own** shuffled null. No number in this line is read against
the bare integer 3.

**Two structural sub-questions, free once the matrix exists:**

- **TriLens.** Are `attn`, `mlp` and `resid` independently informative, or is the tap axis
  near-degenerate? Compare PR({resid}) with PR({resid, attn, mlp}).
- **Depth profile.** Adjacent-layer correlation along depth. The project's own prior is
  unflattering: prefix innovations sat at .87–.96 with the family they were differenced
  from and moved the effective count by at most 0.15. Adjacent layers are very plausibly
  the same story, and finding that is a result.

**Uncertainty.** Per-cell and pooled, paired **source-group** bootstrap (the 1,979-group
scheme already used for the Mind-the-Gap intervals), not per-answer.

**Gate out of Stage 1.**

- If `PR(c) − PR(a)` is not separated from its shuffled null: the depth field is
  characterized as **redundant with the existing channel for localization**, we report
  that plainly, and Stage 2 does not run. Given Step 414 this is a genuine finding, not a
  failure, and it is the more likely outcome on the project's own track record.
- If it is separated: Stage 2 opens, with the *geometry* of the separation deciding what
  Stage 2 builds. Which layers, which taps, which quantity — read off Stage 1, not guessed.

---

## 4. Stage 2 — one variant, only if Stage 1 passes

Not specified in advance beyond its constraints, deliberately: "one variant, one
discussion, then build" (2026-08-05), and the variant is supposed to be *derived from*
Stage 1's geometry.

Its constraints are fixed now, so they cannot drift later:

- **Fusion stays the core.** Depth enters as a supporting component — better inputs to the
  existing IU-PCR / L-SML fusion — never as a standalone detector presented as the method.
  Required: the same fusion core **with and without** the component on matched IDs, plus
  the ablation of learned fusion against equal averaging *under* that component. If only
  the auxiliary score explains the gain, that is what gets reported.
- **A handful of channels, not 432.** Explicitly prohibited by the data handoff and by the
  pool-composition and per-feature-ranking closures already in the ledger.
- **Gate held fixed, both directions.** ProcessBench macro-F1 is strongly gate-dependent.
  Report **gate-free per-subset SLA beside macro-F1**, under the Mind-the-Gap protocol.
- **Effective count reported beside every fusion claim** (Step 205 guard: L-SML is
  information-free at three features and numerically undetermined at four).
- **No digit channels**, and no digit dependency inherited through a frozen gate or a
  baseline-plus-correction recipe.
- **CT7 is frozen.** Anything that changes it is a new candidate with a new name.

---

## 5. Comparators carried in every Stage-2 table

Mandatory rows, per the benchmark-continuity rule. Missing assets get a visible reason,
never silent omission.

| row | PB macro-F1 | gate-free mean SLA | note |
|---|---|---|---|
| CT7 (frozen candidate) | 41.19 | **not measured** | the single most valuable missing number in the line |
| token-level L-SML | 34.08 | 35.92 | LOCO-5 @ 0.33, development-only |
| token-level equal mean | 32.34 | 32.59 | the averaging control |
| plain token entropy, Top-10 (Step 334) | 35.44 | not measured | simple reference |
| Chen et al. Shannon Drop | — | 39.27 | published, per-subset, derivative readout |
| Chen et al. Shannon Avg | — | 25.34 | published baseline |
| chance | — | 16.58 | |

Separate panel, context only — **different task, do not pool**: the prior white-box
lineage on final-answer detection, `results/whitebox_vs_graybox_matched_v1/`. It found
white-box ≈ gray-box on 31,440 candidates across 13 cells (AUROC −0.0013,
[−0.0169, +0.0123]) with mean per-cell Spearman **0.8677** between the two risk scores.
That is a directly relevant prior that the white-box channel is largely *redundant* with
the gray-box one — on a different task, at answer level, with a different representation.
It does not settle the localization question and it is not evidence for this line either
way, but it is the honest prior and it belongs in front of the reader.

---

## 6. Prohibitions, carried from the data handoff

- `hidden_states[L]` is **never** read. HuggingFace applies the final norm before appending
  it; reading it double-norms and corrupts the KL reference every other layer is measured
  against. Reconstruct `x_l = x_{l-1} + a_l + m_l`.
- `HID_PROJ_SEED = 20260811` **never** changes. A changed seed silently invalidates every
  cross-cell comparison.
- `cov_eigs` stays **float32**. Residual Gram eigenvalues exceed the float16 maximum on a
  trained model; `results/whitebox_layer_fusion_v2/DATA_INVENTORY.md` records 47,008
  non-finite entries from exactly that overflow.
- `lens_H` is full-vocabulary and is **not** comparable to cached `token_entropies`
  (top-15 renormalised). Use `final_lens_H` when the comparable quantity is wanted.
- Cluster: a non-login `ssh aircc` shell leaves `SLURM_CONF_SERVER` unset and every
  `squeue`/`sbatch` dies with `fatal: Could not establish a configuration source`. It reads
  exactly like an outage and is not one. Export it, or use `bash -lc`.
- Cluster gate order stays **smoke → N=30 pilot → full**.
- Attention was never captured for any population in this project. Building an attention
  channel is a new extraction, not an analysis.

---

## 7. Worktree notes

Sparse cone: `cluster configs data docs scripts spectral_utils
results/localization_full_benchmark_v3 results/whitebox_layer_views_localization_v1`.
1,155 files checked out, 10,746 skipped — `results/` at large and `papers/` are excluded
on purpose, because local free disk is about 14.5 GB and a full checkout of this repo has
already hit ENOSPC twice.

**Do not run `git sparse-checkout add` or `reapply` in this worktree.** Both wipe
local-only ignored files; that incident is already on record for the atlas worktree
(2026-09-17). The cone was set once, at creation, while the worktree held no local files.
To stage a path outside the cone, use `git add --sparse`.

`results/localization_full_benchmark_v3/evaluation/JOINED.{json,npz}` were imported
verbatim from `master` in commit `ed391c4`, because the extraction driver and the join keys
had never lived on the same branch.

---

## 8. Open decisions — for Omri, before Stage 1 is built

1. **Which bank is the "gray" reference in the added-dimension measurement?** CT7's seven
   views, the eleven token-level channels, or `digitfree_broad50`. This changes what
   "added" means and the ASK rule applies. My recommendation: **CT7's seven views**, on the
   grounds that it is the frozen candidate and the anchor everything else is quoted against.
2. **Is CT7's gate-free SLA in scope for this line?** It is the missing comparator in
   section 5, it needs no GPU, and re-extraction is about 270 s — but it belongs to the
   token-probability line, not this one. Cheap here; scope creep if unasked.
3. **Does Stage 1's gate carry the authority to close the line?** That is, if depth adds no
   dimension above its shuffled null, do we write that up and stop, or do we run Stage 2
   anyway on the grounds that a dimension count is not a localization measurement.
