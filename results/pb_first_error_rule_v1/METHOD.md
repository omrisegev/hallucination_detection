# Change-point first-error rule for ProcessBench (2026-11-10)

## 1. Problem
ProcessBench asks for one output per answer: the index of the first wrong step, or -1 (no error). The C4U rule
for PRMBench flags the top k_a steps (k_a = rounded sum of the Dawid-Skene step posteriors). That rule has no
"no error" output, so the ProcessBench F1 of C4U alone is about 0. The NC rule borrows the gate and the count
from the PRM threshold. Mind the Gap (ICML 2026) uses a different idea: the error is the step where the evidence
drops. This study writes that idea as a probability model with an explicit "no error" state.

## 2. Model
Answer a has steps s = 1..n and an evidence vector x_s (d channels, oriented so that a high value means risk).
The latent variable is the first-error step tau in {1, ..., n, inf}; tau = inf means "no error".
Each step has one of three states, fixed by tau (a left-to-right chain):

    state 0 "clean"         s < tau          density f0
    state 1 "first error"   s = tau          density f1
    state 2 "after error"   s > tau          density f2

Channels are conditionally independent given the state (naive Bayes; the same assumption as Dawid-Skene):
f_k(x) = prod_j f_kj(x_j). With lam1_s = log f1(x_s)/f0(x_s) and lam2_s = log f2(x_s)/f0(x_s):

    log p(x | tau = t) - log p(x | tau = inf) = L_t = lam1_t + sum_{s > t} lam2_s

Prior: P(tau = inf) = pi, and P(tau = t | tau <= n) is a truncated geometric with hazard h:
P(tau = t) = (1 - pi) h (1-h)^(t-1) / (1 - (1-h)^n).

Posterior: P(tau = t | x) = prior(t) exp(L_t) / [ pi + sum_u prior(u) exp(L_u) ]  (L_inf = 0).

Decision: MAP, tau_hat = argmax over {1..n, inf}. Output -1 if the MAP is inf.
Why MAP: ProcessBench scores each answer by exact match of tau_hat (error answers) or of -1 (clean answers).
For a 0-1 loss per answer, the Bayes decision is the posterior mode. So, if the model is correct, MAP gives the
highest expected number of exact matches. (F1 is a harmonic mean of the two class accuracies; MAP optimizes their
weighted sum, not F1 itself. We do not tune for F1.)
"No error" has a direct meaning: P(tau = inf | x) is the posterior probability that the answer is clean.

Two special cases connect the model to earlier rules:
- f2 = f0 (the after-error steps look clean): L_t = lam1_t. This is the Mind the Gap "drop" model: only the
  first-error step is different. With one channel and a monotone lam1, MAP becomes "first step with
  lam1_t > threshold", that is a first-passage rule such as the PRM native rule p > 0.5.
- f2 = f1: L_t = sum_{s >= t} lam1_s. This is the classical level-shift change-point (the error state persists).
On ProcessBench the after-error steps are neither: with Qwen-PRM-7B only 29% of after-error steps have p(error) > .5,
against 68% of first-error steps and 3% of clean steps. So the 3-state chain is the appropriate model. (Oracle check
with gold-fitted densities, Qwen-7B PRM only: level-shift F1 59.5, drop 63.0, 3-state 70.2.)

## 3. Estimation (no gold labels)
f0, f1, f2 are histograms on per-channel global quantile bins (B = 20 for one channel, 10 for several), with a
monotone likelihood-ratio constraint (isotonic regression of the bin log ratio). Two estimators:
- EM to convergence (all parameters free, only the data likelihood).
- Anchored plug-in: one EM step. The responsibilities are set from the PRM native partition (first step with
  p > .5 = state 1, later steps = state 2, no flag = clean answer). The densities of ALL channels are estimated from
  this partition; pi = share of answers without a flag; h = geometric MLE of the flagged positions. Then the
  posterior is computed once. No gold label is used anywhere.

Finding: EM to convergence is NOT a usable estimator here. With one channel (PRM alone), free EM moves away from the
first-error structure (Qwen-7B F1 72.8 after one step -> 37.4 at convergence). With telemetry channels and no PRM,
EM drives pi to about 0 (every answer gets an error). The likelihood of the chain does not identify "first error"
from these channels: there is no anchor that fixes which state is "clean". This is the same failure as the
earlier DS gate experiments (clean answers cannot be detected from telemetry). The anchored plug-in uses the PRM
partition only to name the states; the change-point posterior then combines all channels.

## 4. Arms
| arm | channels | estimator |
|---|---|---|
| CP PRM (anchored) | PRM risk (logit) | anchored |
| CP PRM + drop | PRM risk, first difference of PRM risk | anchored |
| CP PRM + C4U rank | PRM risk, frozen C4U rank | anchored (main arm: "C4U + change-point rule") |
| CP PRM + TELE16 | PRM risk, 16 telemetry channels (global rank-normal, backbone) | anchored |
| CP TELE16 (PRM anchor only) | 16 telemetry channels | anchored on the PRM partition, PRM not a channel |
| FREE CP TELE16 backbone / standalone | 16 telemetry channels; backbone = global rank-normal G16; standalone = base-model answer-z N16 | EM from a PRM-free start; also with pi fixed to 1/2 |

The C4U rank is the frozen recipe (c4u_new.c4u, R1 inputs) exported unchanged from the PC. Parity: C4U NC from the
exported arrays gives avg F1 69.68 / macro SLA 57.40 for Qwen-7B (the stored values).
