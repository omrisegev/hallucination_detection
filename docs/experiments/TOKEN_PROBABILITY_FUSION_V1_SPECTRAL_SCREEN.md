# Can the L-SML spectra tell us which readout width to use? — no, and the question is smaller than it looks

Runs `scripts/diagnostics/spectral_descriptor_screen_v1.py`; output
`SPECTRAL_DESCRIPTOR_SCREEN.json`. Label-using screen, development-only.

## The question, and a trap in it

Omri asked whether the right `K` could be read off the L-SML's own spectral quantities —
its residual, the group count its clustering picks, the eigenvalues of the covariance.

**The trap:** our L-SML is fitted on donor folds, so its residual, its group count and
its covariance spectrum are properties of a **fold**, not of an answer — they cannot vary
per answer by construction, and cannot select anything per answer. Anything usable has to
be recomputed from the answer's own tokens. That turns out to be entirely feasible: a
~500-token answer easily supports an 11×11 covariance, and an **answer-local L-SML fit
succeeds on every answer tried**, at about 0.3 s each, returning group counts of 3–6 and
residuals spanning 4.5–9.9. So the question is answerable, and this answers it.

## The screen

Rather than build a selector and then discover it does nothing, the screen asks the
narrowest decision-relevant question first:

> among the answers where `K=10` and `K=40` **disagree** about being right, can any
> label-free descriptor tell which of the two is correct?

Chance is 0.500.

## Result — the economics kill it before the AUCs do

| | |
|---|---|
| erroneous ProcessBench answers | 4,442 |
| where `K=10` and `K=40` disagree | **632 (14.2%)** |
| of those, `K=40` is the right one | 54.9% |
| **value of a PERFECT selector between the two** | **0.70 pp** |

The two widths agree on 85.8% of answers. Even an oracle that always picked the better
of the two would gain **0.70 pp** — which is inside the noise of every interval computed
in this line.

## And the descriptors carry no signal anyway

| descriptor | AUC | | descriptor | AUC |
|---|---|---|---|---|
| log total tokens | **0.567** | | `eff_rank_entropy` | 0.525 |
| mean tokens/step | 0.560 | | **`lsml_residual`** | **0.479** |
| sd tokens/step | 0.560 | | margin at K=10 | 0.519 |
| margin at K=40 | 0.541 | | **`participation_ratio`** | **0.518** |
| **`log_condition`** | **0.466** | | **`top1_share`** | **0.492** |
| peak SNR | 0.529 | | **`lsml_group_count`** | **0.495** |
| adaptive `K*` | 0.529 | | `n_steps` | 0.528 |

**Every spectral and L-SML quantity is at chance.** The answer-local L-SML's group count
is 0.495 and its residual 0.479 — the two things Omri specifically asked about are the
two least informative in the table. The covariance eigenvalue descriptors
(`participation_ratio` 0.518, `top1_share` 0.492, `eff_rank_entropy` 0.525,
`log_condition` 0.466) are no better.

The only descriptors that move at all are **structural, not spectral** — total tokens
and tokens per step, at 0.56–0.57. That is the length dependence we already knew about,
showing up again, and at a strength that would convert almost none of the available
0.70 pp.

## Why this closes the family, not just this pair

The screen is the **easiest** version of the problem: a binary choice between two known
widths. Selecting a K per answer from the full ladder is a ~34-way decision, strictly
harder. A descriptor that cannot win the two-way case will not win the many-way one.

This is consistent with, and explains, the adaptive-readout result: the per-answer
criterion captured about 4% of a large-looking oracle ceiling. That ceiling was measured
over the whole ladder ("does *any* K localize correctly"), which is a much looser bound
than anything achievable — and the binary slice measured here shows why. The realistic
room between two sensible widths is under a point, and nothing in the answer's own
spectrum points at it.

**Closed:** choosing the readout width from per-answer spectral or L-SML descriptors.

**Not closed by this:** the readout width itself is still worth a point or so in point
estimates (K=20 over K=10), it simply cannot be *selected* per answer, and its interval
against K=10 already included zero.

## Note on a bug this screen caught in itself

The first run reported a 100% disagreement rate with K=40 right 100% of the time — an
impossible result. `hit10[i]` is an `int`, so `x and not y` produced a mix of `bool` and
`int`, NumPy inferred an **integer** array, and `ids[a | b]` became fancy indexing by
position rather than a boolean mask. The masks are now built with `dtype=bool` and an
assertion forbids an answer landing in both classes. The impossible number is what made
it visible; a subtler corruption would not have been.
