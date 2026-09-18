# Token-probability line, Stage B — pre-registration

Written **before** the code is written and before any Stage B number exists.
Covers §5.1, §5.2 and §5.3 of `docs/HANDOFF_TOKEN_PROBABILITIES.md`.
Nothing here may be edited after the first Stage B run; corrections go in a
dated amendment below.

Status: **awaiting Omri's sign-off.** Not yet run.

## 0. Why a 2×2 and not two experiments

§5.2 fixes a protocol deviation (fit on a pooled non-answer-standardized token
matrix, read out by argmax *within* the answer — fitting on one axis, applying
on another). §5.1 asks whether the L-SML gain is noise-down-weighting that a
step-level readout already does for free. They are two binary axes over one
fixed bank, one fold structure, one roster. Run separately, the second is
confounded by the first. The four cells cost barely more than two arms because
the token matrices are built once and reused.

This is a diagnostic with a written prediction per cell, not a candidate search.
No cell may be promoted to a candidate on the strength of this table.

## 1. Fixed, not varied

| item | value |
|---|---|
| bank | the eleven channels of `spectral_utils/claude_feature_bank_v1.py`, manual signs, no digit channel |
| roster | 13,769 answers / 145,597 steps; v3 labels, v2 source groups |
| folds | the same 5 source folds (2782/2740/2730/2768/2749) |
| readout | Top-10 mean of token risks within each step |
| length calibration | none (per the negative Step 420 ablation) |
| gate | reporting only; gate-free SLA is primary, LOCO-5 @0.33 and tail15 shown beside it |
| bootstrap | 10,000 draws, unit = source group, same seeds as Stage A |

## 2. The two axes

**Standardization.** `pooled` = donor-fold pooled token mean/std, the current
arm. `answer_local` = each channel standardized within the answer before fusion,
aligning the fit axis with the readout axis.

**Fusion stage.** `before` = fuse token risks, then Top-10 mean per step, the
current arm. `after` = Top-10 mean per channel first, giving a steps×11 matrix,
then fuse at step level.

## 3. The energy-pair degeneracy — declared, and what we do about it

`energy_level` and `energy_innovation` form a two-member L-SML group. A pair is
unidentified: it pins down its residual product, not the individual loadings.
They come back with equal and opposite weights to 15 digits in all five folds.

**This pair is not inert.** Weights `+w` and `−w` make its net contribution
`w·(z_level − z_innovation)` — a real difference channel whose split between the
two members is arbitrary. The bank therefore spans ten identified directions,
not eleven, and the eleventh degree of freedom is pure gauge.

It is inherited identically by all four cells, so it does not confound the 2×2
*contrasts*. It does affect every absolute level. Decision:

- The 2×2 runs on the **declared eleven-channel bank**, so that cell C1
  reproduces the published 35.92 exactly and the table bridges to the handoff.
- One **additional control at C1 only**: the same bank with the pair collapsed
  into a single explicit difference channel (ten identified channels, same
  span). This measures the cost of the gauge once instead of leaving it
  unstated in four places.
- If the control moves C1 by more than the width of its own interval, the 2×2
  is re-run on the collapsed bank and this document is amended.

## 4. Predictions — recorded before running

Omri's two, on the protocol:

- **P1.** The conditional participation ratio falls from the token-level 9.69 to
  roughly **2–3** at step level.
- **P2.** The L-SML advantage over equal weighting **shrinks substantially or
  vanishes** when fusion moves to after the readout.

Per cell. C1 is known and is the replication anchor, not a prediction.

| cell | standardization | fusion | L-SML − equal | mean gate-free SLA |
|---|---|---|---|---|
| **C1** | pooled | before | +3.32 pp *(known)* | 35.92 *(known)* |
| **C2** | answer-local | before | smaller than C1 | **above** 35.92 |
| **C3** | pooled | after | much smaller than C1, may include zero | at or above 35.92 |
| **C4** | answer-local | after | **smallest of the four, closest to zero** | **highest of the four** |

Reasoning recorded with the prediction: the `chosen_surprisal` sign flip
(−0.234 in all five folds despite being oriented toward risk) is the signature
of axis mixing — between answers surprisal tracks difficulty, within an answer
it tracks error. Answer-local standardization should remove the flip. If it does
not, the flip has another cause and P2's mechanism story is wrong.

**Additional pre-registered prediction.** `chosen_surprisal` carries a
**non-negative** weight in a majority of folds in C2 and C4.

## 5. Falsification — what would make this a positive result

If the L-SML advantage in **C3 and C4 stays inside C1's interval** — i.e. does
not shrink — then noise-down-weighting is not the explanation, and the fusion is
doing something a step-level average does not do. That would be a substantially
stronger finding than anything currently in this line, and it is the outcome
this design is built to be able to see. It is not the outcome predicted.

Conversely, if C4 shows the highest level *and* the smallest gain, the honest
reading is that the correct arm is a well-standardized step-level one and the
L-SML gain was a repair of a representation defect, not a fusion gain.

## 6. The participation-ratio floor is computed, not inferred (§5.3)

Per Omri: **the noise floor is a separate computation, not an interpretation.**

- Conditional (within-label-centred) participation ratio at step level on the
  eleven channels, for each of the four cells.
- A **shuffled null, actually run**: tokens permuted within each answer
  independently per channel, then the whole pipeline recomputed and the PR
  measured again. Same number of draws as any other estimate here.
- Prediction: the real PR lands near 2–3 (P1) while the shuffled null stays
  near 9. If the null also collapses, the PR contrast says nothing about
  dimensionality and must not be reported as if it did.
- The 9.69 is **not** comparable to the project's 1.80 / 2.46 / 2.83: those are
  conditional PRs on answer-standardized step views, this is a marginal ratio on
  a pooled non-answer-standardized token matrix. Three differences at once. Any
  comparison must be to the new step-level conditional numbers, not to 9.69.

## 7. Per-cell adaptation — the free screen

My reading of the request; correct me if it was a different idea. Stage A shows
a strong subset asymmetry: the token arm beats CT7 on GSM8K-8B and loses ~6–8 pp
on OlympiadBench and Omni-MATH. That invites "fit or choose per subset".

The screen costs nothing because the 2×2 already produces per-cell SLA for every
arm. Compute, per benchmark cell, the best arm's SLA, and average — the
**per-cell oracle ceiling**. Compare against the best single pooled arm.

- If the ceiling is within ~1 pp of the best pooled arm, per-cell adaptation has
  no headroom and the idea is closed cheaply, before anything is built.
- If it is well above, that bounds what any per-cell rule could buy and the
  design question becomes label-free per-cell selection.

This is a **ceiling selected with labels**, exactly like the 0.41 gate optimum.
It is not a candidate and may not be reported as a score.

## 8. Reporting rules

- Report the effective independent-view count beside every fusion claim
  (Step 205 guard); below three conditionally independent views, equal weighting
  is the correct L-SML output and must be said so.
- Gate-free SLA is primary. Any macro-F1 is reported with its gate named.
- CT7 stays frozen and appears only as the Stage A comparator.
- Development-only throughout; the cached population has been inspected before.
