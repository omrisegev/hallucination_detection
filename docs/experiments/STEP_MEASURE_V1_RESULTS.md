# Results — redesigning the step-measurement stage

Claude, 2026-09-19. Pre-registration: `STEP_MEASURE_V1_PREREGISTRATION.md`, written before
these numbers were read. Artefacts: `results/token_probability_fusion_v1/{STEP_MEASURE.json,
STEP_MEASURE_PEAKS.npz, figures/step_measure.*}`. Development-only.

Scope as fixed by Omri: the fusion is not reopened. Input is the finished fused token
series; only the stage that turns the tokens inside a step into one number is varied; what
it fits is fitted per model. Anchor `raw x Top-10` replays **35.92** exactly.

## 1. The sweep

Gate-free SLA, mean over the eight ProcessBench cells, 4,442 erroneous answers.
The raw Top-K column reproduces the established sweep value for value.

| width | raw Top-K | whitened Top-K | raw best window | whitened best window |
|---:|---:|---:|---:|---:|
| 1 | 20.42 | 20.56 | 20.42 | 20.56 |
| 2–3 | 28.47 | 26.77 | 25.80 | 24.95 |
| 4–5 | 31.93 | 30.34 | 28.67 | 26.39 |
| 8–10 | **35.92** | 33.68 | 29.81 | 27.62 |
| 16–20 | **37.01** | 35.45 | 29.60 | 30.31 |
| 32–40 | 36.74 | **35.63** | **29.94** | **30.06** |
| 64–80 | 34.18 | 34.67 | 28.56 | 28.30 |

(Top-K uses K ∈ {1,3,5,10,20,40,80}; the window uses w ∈ {1,2,4,8,16,32,64}; rows pair
them by nearest width.)

## 2. All three predictions falsified

**P1 — whitening. FALSIFIED, and in the opposite direction to the one predicted.**
The prediction was that if the K=10-to-20 optimum exists because the noise is correlated,
whitening moves the optimum **down** toward 1 and the peak **up**. The optimum moved
**up**, 20 → 40, and the peak moved **down**, 37.01 → 35.63. At matched width the cost is
significant: **−2.24 pp [−3.91, −0.57]** at K=10. Ceiling against ceiling,
**−1.38 [−3.08, +0.27]**.

The autocorrelation is therefore **not nuisance — it is carrying signal.** Removing the
linearly predictable part of the token series removes evidence, which is the opposite of
the detection-theory intuition that motivated the arm. Supporting this reading: whitening
also lowers split-half reproducibility at every width (0.391 → 0.261 at K=10, 0.669 →
0.510 at K=40), so the whitened decision is *less* stable under a token resample, not more
— exactly what happens when a transform destroys signal rather than noise.

**P2 — contiguity. FALSIFIED, decisively.** The best contiguous window peaks at 30.37
against Top-K's 37.01: **−6.65 pp [−8.56, −4.72]**, the largest effect in the experiment.
The informative tokens inside a step are **scattered, not adjacent**. Combined with the
step-level profile, the picture is specific: at *step* resolution the error is a sharp
isolated impulse (+0.65 SD, neighbours at zero), while at *token* resolution inside that
step it is not a burst at all.

**P3 — per-model scope. FALSIFIED, to two decimals.** Per-model 35.63 against pooled
35.63, contrast **+0.00 [−0.16, +0.14]**. The fitted dynamics are nearly identical between
the two backbones — lag-1 AR coefficient 0.400–0.403 for Qwen3-4B against 0.407–0.409 for
Qwen3-8B, stable across all five folds. The two models' token dynamics are the same to the
precision that matters, so per-model scope buys nothing at this stage. This repeats what it
did for the fusion stage (−0.22 short / −0.65 long, both intervals covering zero).

## 3. What the three failures say together

Both arms were attempts to **sharpen** the statistic — whitening sharpens in time,
contiguity sharpens in position — and both lose. Meanwhile every curve in the sweep rises
with width, and the incumbent's own family peaks at K=20, wider than the deployed K=10.

The direction that works is **more aggregation, not less**. The evidence for an error is
diffuse across the tokens of a step and positively correlated, and the statistic that suits
it is a blunt average over many tokens. The Top-K mean is not a badly tuned matched filter;
it is a reasonable summary of diffuse evidence, and the sharpening toolbox is the wrong
toolbox for it.

That reframes what the remaining two directions should look for. A graph or a
self-supervised step encoder should **not** be built to localise more precisely inside a
step. It should be built to *weight* the step's tokens better — a learned pooling, where
the question is which tokens deserve mass, not which single token or which contiguous run
is the error.

One live number this run reconfirms independently: **raw Top-20 scores 37.01, +1.10
[−0.03, +2.22] over the deployed Top-10.** The interval still just touches zero, exactly as
it did in the earlier calibration, so this remains a point estimate and not a demonstrated
gain — but it is now measured twice on the same population by two independent scripts, and
it is the largest unclaimed number on the locator.

## 4. Caveats

**A reproducibility discrepancy I have not diagnosed.** My split-half criterion matches the
established `READOUT_CALIBRATION_C1` run within 0.015 at K = 1, 3, 5, 10, 20 and 40, but
diverges at K=80 (mine 0.727, the established run 0.678) — probably how each caps K against
a half-step shorter than K, since at K=80 most steps are shorter than the window. The
established run stays canonical for that criterion, and **no new conclusion about
criterion 1 is drawn here.** Under my numbers the criterion would pick K=80; under the
established ones it picks K=40 at a 0.27 pp cost. That disagreement is not resolved.

**The whitener is linear and causal.** AR(8) via Yule–Walker removes the linearly
predictable component only. A nonlinear or non-causal predictor might separate signal from
correlated noise where this one does not, and P1's falsification is a falsification of
*linear causal prewhitening*, not of the idea that some of the autocorrelation is nuisance.

Nothing here changes CT7, the gate, the bank, the fusion, or any frozen release, and no
untouched confirmation set is involved.
