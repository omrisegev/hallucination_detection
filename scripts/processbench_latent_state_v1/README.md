# ProcessBench latent-state localization v1

## Purpose

This experiment tests one specific extension of the frozen GL-LIU pipeline:

> After ordinary IU-PCR fuses the five strong token curves into one risk
> sequence, can an unsupervised hidden-state model locate the entry into an
> error-like regime better than taking the largest token score?

It does not create another feature contract. It does not tune DUFS. It keeps
the frozen mixed-v2 DUFS-LIU answer detector unchanged and changes only the
local placement rule.

The experiment is exploratory because ProcessBench labels were already opened
by earlier project studies. A positive result needs confirmation on new data.

## Data

Expected cache layout:

```text
cache/localization/processbench/
  pb_qwen3_4b/processbench_{gsm8k,math,olympiadbench,omnimath}.pkl
  pb_qwen3_8b/processbench_{gsm8k,math,olympiadbench,omnimath}.pkl
```

The eight cells contain two teacher-forced scorer models over the same 3,400
ProcessBench examples. They are eight score cells but only four independent
dataset families.

## Fixed feature and fusion path

For trace `i` and token `t`, construct the existing core vector

\[
x_{it}=[H,\operatorname{SWVar}(H),|\operatorname{CUSUM}(H)|,
\operatorname{SWVar}(S),|\operatorname{CUSUM}(S)|]_{it},
\]

where `H` is entropy and `S` is sampled-token spilled energy. The repository's
frozen preprocessing standardizes the curves and derives their relative
orientation without labels.

Ordinary two-component IU-PCR (`lambda=0`) estimates weights `w` and produces

\[
r_{it}=w^\top x_{it}.
\]

The original localizer predicts `argmax_t r_it`. The experiment adds two HMM
placement rules over exactly the same `r_it`.

## Primary model: reversible IU-HMM

The hidden state is `Z_it in {0,1}`. State 1 is defined as the state with the
higher mean IU-PCR risk. This fixes state identity without correctness labels.

\[
r_{it}\mid Z_{it}=s\sim N(\mu_s,\sigma^2),\qquad
A=\begin{bmatrix}1-a&a\\b&1-b\end{bmatrix}.
\]

Both states share one variance. This prevents a high-variance component from
being misnamed the high-risk state. The local score is the smoothed entry
posterior

\[
e_{i1}=P(Z_{i1}=1\mid r_i),\qquad
e_{it}=P(Z_{i,t-1}=0,Z_{it}=1\mid r_i),\quad t>1.
\]

The prediction is `argmax_t e_it`. The model may leave and re-enter the risk
state, which matches earlier evidence that uncertainty bursts are not always
persistent.

## Falsification control: absorbing IU-HMM

The control replaces the transition matrix with

\[
A=\begin{bmatrix}1-h&h\\0&1\end{bmatrix}.
\]

The initial distribution is fixed to `pi=(1,0)`. Every trace starts in the
pre-change state, so this control cannot learn a first-token shortcut.

It directly tests the stronger assumption that observable telemetry changes
permanently after the first reasoning error. It is reported beside the primary
model and is never selected after labels are opened.

## Label-free fitting and guards

- The HMM receives only scalar IU-PCR risk arrays and sequence boundaries.
- It cannot receive labels, text, reasoning steps, or step spans.
- Three deterministic starts use seeds `11, 23, 37`.
- The valid start with greatest unlabeled sequence log likelihood is selected.
- At least two starts must be valid, and their mean exact argmax agreement must
  be at least `0.80`; otherwise the HMM is rejected as seed-unstable.
- A model is invalid if parameters are non-finite, either state has less than
  2% occupancy, state separation is below 0.25 shared standard deviations, the
  risk-state order reverses, or a free transition reaches a numerical boundary.
- If every start is invalid, the result falls back exactly to the ordinary
  IU-PCR argmax. The fallback is visible in the diagnostics.

Seed locator agreement, transition probabilities, occupancy, state separation,
emission variance, posterior concentration, no-credible-entry frequency,
position/length dependence and error-aligned entry curves are all reported.

## Compared systems

All our systems use the same mixed-v2 DUFS-LIU global detector.

1. ordinary core-five IU-PCR localizer;
2. frozen temporal-LIU core-five localizer;
3. leading DUFS-LIU core-five localizer;
4. reversible IU-HMM entry localizer, the primary hypothesis;
5. absorbing IU-HMM, the falsification control;
6. full Mind the Gap control with its own detector and locator.

The runner requires exact score-hash equality with the existing factorial
artifacts for the global detector, temporal localizer and DUFS localizer.

## Two-process audit boundary

Run fitting first:

```bash
.venv/bin/python scripts/processbench_latent_state_v1/run.py fit \
  --cache-root cache/localization/processbench \
  --out-dir results/processbench_latent_state_v1
```

The `fit` command accesses only the five registered telemetry fields. The
fitting function receives sanitized rows in which labels and step spans do not
exist. It writes numeric NPZ files, source/input fingerprints and an
authoritative `FREEZE_MANIFEST.json`. It refuses to overwrite a non-empty
experiment directory.

Review the label-free diagnostics. Only then run evaluation explicitly:

```bash
.venv/bin/python scripts/processbench_latent_state_v1/run.py evaluate \
  --cache-root cache/localization/processbench \
  --out-dir results/processbench_latent_state_v1
```

The second process verifies each score file, diagnostic file, source file,
input pickle and frozen-baseline hash before it reads labels and step spans. It
also reproduces the Mind-the-Gap detector and locator hashes. It reuses
`evaluate_two_stage` with exactly 100 repeated 50/50 calibration/evaluation
splits and seed 0. The report verifies hashes of all evaluation tables.

Finally build the visual report:

```bash
.venv/bin/python scripts/processbench_latent_state_v1/report.py \
  --out-dir results/processbench_latent_state_v1
```

## Metrics

- **Exact localization:** predicted token maps to the annotated first erroneous
  step, evaluated only on erroneous traces.
- **Within one step:** the prediction is at most one reasoning step away.
- **Clean accuracy:** the global detector abstains on a fully correct trace.
- **ProcessBench F1:** harmonic mean of exact localization on erroneous traces
  and clean accuracy.
- **Signed step error:** negative is early; positive is late.
- **Normalized token distance:** distance from the first token of the gold step,
  divided by trace length.

Every result is reported per cell, across all eight cells, and across the six
cells outside the earlier GL-LIU component-selection set.

## Promotion rule

The reversible model is only promising when all of the following hold:

1. no HMM fallback occurs;
2. exact localization improves across all eight cells;
3. exact localization improves on the six non-selection cells;
4. end-to-end F1 improves across all eight cells;
5. end-to-end F1 improves on the six non-selection cells;
6. no individual cell loses more than one F1 point.

The range across the repeated calibration splits is descriptive split
variability, not an independent-data confidence interval.

## Tests

```bash
.venv/bin/python scripts/test_latent_state_localizer.py
.venv/bin/python scripts/test_processbench_latent_state_v1.py
```
