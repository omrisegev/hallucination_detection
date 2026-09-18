# HANDOFF — The token-probability line

Claude, 2026-09-18. Step 422. Everything in this line is derived from the cached
top-50 log-probabilities, the provided-token id, and the full-vocabulary logsumexp.
No model pass is needed for any of it.

## 1. Where the line stands

| arm | representation | fitting | gate | PB macro-F1 | gate-free mean SLA |
|---|---|---|---|---|---|
| CT7 (frozen candidate) | step, answer-standardized | pooled, 5 source folds | frozen non-digit tail15 | **41.19** | not measured |
| token-level L-SML | token, pooled-standardized | pooled, 5 source folds | LOCO-5 @ 0.33 | 34.08 | **35.92** |
| token-level equal mean | same | same | same | 32.34 | 32.59 |
| plain token entropy, Top-10 (Step 334) | token | none | — | 35.44 | not measured |

**Do not compare the 41.19 and the 34.08.** They use different gates and ProcessBench
macro-F1 is strongly gate-dependent. I made that mistake and it cost a wrong conclusion.
The gate-free column is the comparable one and CT7's entry in it is the single most
valuable missing number in this table.

## 2. The result that changed status this session

Under the **Mind-the-Gap protocol** — per-subset Step-level Localization Accuracy on
erroneous answers only, with no no-error decision at all (Chen et al., ICML 2026,
Table 3; their ProcessBench subset sizes are exactly ours, 400/1000/1000/1000):

```
gate-free mean SLA over the 8 cells:  L-SML 35.92   equal 32.59   chance 16.58
difference +3.33 pp, 95% CI [+1.90, +4.75]
10,000 paired source-group bootstrap draws, 1,979 groups
```

The interval excludes zero, the gain holds in **all eight cells**, and it is **larger
gate-free (+3.33) than gated (+1.73)** — so it is not a gate artefact. To my knowledge
this is the first arm in the project where L-SML beats equal weighting with a paired
interval that excludes zero. Step 413's ladder had every fusion rule within 1 pp of
equal20.

**Do not over-read it.** The absolute level is still below CT7 under CT7's own gate, and
the leading mechanistic hypothesis is unflattering: at token level, per-channel noise is
large and independent, so L-SML has real work to do in down-weighting noisy channels —
work that the step-level Top-10 readout already does for free by averaging. If that is
what is happening, the gain should **vanish** when the same bank is fused after the
readout instead of before. That test is §5.1 and it is the most informative thing left
in this line.

## 3. Versus the literature, split cleanly by length

| subset | ours (L-SML) | Chen et al. Shannon Drop | Chen et al. Shannon Avg |
|---|---|---|---|
| GSM8K 4B / 8B | **47.83 / 49.76** | 43.42 / 46.11 | 27.94 / 27.66 |
| MATH 4B / 8B | **34.85** / 31.99 | 32.03 / **32.90** | 24.17 / 24.62 |
| OlympiadBench 4B / 8B | 30.56 / 31.01 | **43.06 / 41.52** | 24.95 / 26.30 |
| Omni-MATH 4B / 8B | 30.96 / 30.43 | **38.04 / 37.04** | 23.67 / 23.40 |
| mean | 35.92 | 39.27 | 25.34 |

We beat their best on short chains and lose 6.6–12.5 points on long ones, while beating
their baseline everywhere. **Their score is a derivative** (EMA smoothing, then the mean
of the worst M drops); **ours is a level**. That is exactly the asymmetry you would
predict: a level signal drowns as the chain grows, a drop signal does not.

This does **not** contradict Step 414's finding that prefix innovations add no
dimension. That was measured at **step** level on sequences with a median of 8 steps.
This is **token** level on chains of hundreds of tokens. Different measurement, so a
token-level derivative channel is a motivated new experiment, not a reopened dead end.

## 4. Protocol audit of the token-level arm

Matches the plan: the eleven channels and their manual signs; Top-10 mean of token
risks within each step; no digit channel; no length-calibrated readout (correct per the
negative Step 420 ablation); LOCO-5 kept outside the locator and declared
development-only because its subset was historically label-selected.

**One deviation that matters.** The fusion is fit on a **pooled donor-fold token matrix
with no answer-local standardization**, while the readout is **argmax within the
answer**. The evaluation contract is answer-local step standardization, and all seven
CT7 views are answer-standardized. So the fit sees mostly *between*-answer variance —
long and hard answers simply run hotter — and is then applied to a *within*-answer
ordering. Fitting on one axis, applying on another.

Two consequences follow directly, and both are observed:

- **The 9.69 "effective rank" is not comparable to our 1.80 / 2.46 / 2.83.** Those are
  conditional (within-label-centred) participation ratios on answer-standardized
  **step** views. This is a marginal ratio on a pooled, non-answer-standardized
  **token** matrix. Three differences at once.
- **`chosen_surprisal` receives a negative weight (−0.234) in all five folds**, despite
  being oriented so that higher means more risk. Between answers surprisal tracks
  difficulty; within an answer it tracks error. The sign flip is what mixing the two
  axes produces. It is also the Step 389 signature of a redundant pool: there,
  `noreset` was "chiefly subtracted, not positively voted".

Two smaller items: the protocol document says PRMBench uses the **max** token risk in a
span while the code uses Top-10 for both benchmarks — the code is right, fix the doc.
And `energy_level` / `energy_innovation` form a **two-member L-SML group**, which is
unidentified; they come back with exactly equal and opposite weights (to 15 digits) in
all five folds, so the bank is effectively **ten** channels, not eleven.

## 5. Open experiments, in the order I would run them

**5.1 Fusion before versus after the readout, same bank, same folds.** The decisive test
of the noise-weighting hypothesis. If the L-SML advantage collapses at step level, that
single result explains every previous negative in this family and is worth more than
another candidate.

**5.2 Answer-local standardization arm.** Not a new idea — a correction of the deviation
in §4. Same bank, same folds, each channel standardized within the answer before fusion.
Aligns the fit axis with the readout axis, and tells us whether 9.69 was dimensions or
between-answer variance.

**5.3 Conditional participation ratio at step level on these eleven channels**, plus a
noise floor: shuffle tokens within each answer independently per channel and recompute.
Prediction: the token-level 9.69 falls to roughly 2–3 at step level, and the shuffled
null stays near 9. If both hold, the extra dimensions are per-token noise.

**5.4 Hold the gate fixed, both directions.** This arm under CT7's frozen tail15 gate,
and CT7 under LOCO-5. Requires rebuilding CT7's step scores, which were deleted with the
27 worktrees; the Drive backup has the inputs and re-extraction takes about 270 s.

**5.5 A token-level derivative channel.** Motivated directly by the OlympiadBench and
Omni-MATH deficit in §3, not by the literature in the abstract. Note that the project's
own "tailor, never transplant" rule applies: take the mechanism (smoothed evidence flux,
worst-M aggregation), do not transplant their exact recipe and do not label it with
their name.

**5.6 Retune the gate, honestly.** LOCO-5's inherited 0.33 costs 2.77 pp against its own
optimum at 0.41 (34.08 → 36.84). That optimum is label-selected and is a **ceiling, not
a candidate**. A label-free rule — the q=0.3 quantile constant from Step 331/332 is the
precedent — is what should actually be fitted.

## 6. Standing rules this line has to respect

- Plain averaging between features is not an acceptable final method, but it **is** the
  correct output of L-SML below three conditionally independent views (Step 205 guard).
  Report the effective count beside every fusion claim.
- No digit-derived channels, and no digit dependency inherited through a frozen gate or
  a baseline-plus-correction recipe.
- CT7 is frozen. Any change is a new candidate with a new name.
- ProcessBench macro-F1 is gate-dependent: never compare arms whose gates differ without
  holding the gate fixed, and report the gate-free per-subset SLA beside it.
- One variant, discussion, then build. Omri wants to be consulted on direction before a
  new build.

## 7. Artefacts

- `spectral_utils/claude_feature_bank_v1.py`, `scripts/run_claude_feature_bank_v1.py` —
  the arm (branch `codex/claude-feature-bank-token-lsml-v1`).
- `results/claude_feature_bank_token_lsml_v1/{RESULTS.json, OOF_SCORES.npz, SMOKE.json}`
  — per-step OOF scores for both arms plus the gate composite, enough to redo every
  analysis in §5.1–5.4 and §5.6 with no GPU.
- `scripts/diagnostics/gate_isolation_token_lsml_v1.py` +
  `results/claude_feature_bank_token_lsml_v1/GATE_ISOLATION.json` — this session's
  separation, including the full threshold sweep curve.
- `spectral_utils/frozen_locator_ct7.py`,
  `results/chosen_token_calibration_v1/FROZEN_CANDIDATE_CT7.json` — the frozen candidate.
- `docs/experiments/CLAUDE_FEATURE_BANK_TOKEN_LSML_V1.md` — the arm's protocol.
- `docs/HANDOFF_WHITE_BOX_LAYER_VIEWS.md` — the companion line, and the only channel we
  have that is not a transform of this one.
