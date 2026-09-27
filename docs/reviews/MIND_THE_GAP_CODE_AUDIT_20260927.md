# Mind the Gap: audit of the released code against the paper (2026-09-27)

Chen, Chen, Yue, Li, "Mind the Gap: Catching Hallucinations via Evidence Drop on the Reasoning
Manifold", ICML 2026. Official code: https://github.com/QJ0114/evidence-drop, commit
`ff14a7d5bd0f3969b47555d0daa9a0c8b9dbb71f`, vendored byte-identical in
`papers/code/mind_the_gap_evidence_drop_ff14a7d/`. Every code reference below is to that snapshot;
every paper reference is a line number in
`papers/extracted/mind-the-gap-catching-hallucinations-via-evidence-drop.md`.

Requested by Omri after asking what exactly is missing from the paper to reproduce its ProcessBench
numbers. Until today the digest card recorded the code as "not inspected".

## 1. Bottom line

1. **The released code contains no step-level, no ProcessBench and no SLA code at all.** It is 14 files
   for sequence-level selective prediction on GSM8K and MATH only. The repository has one branch and three
   commits, all on 2026-05-05; no step-level file ever existed in its history. **Table 3 (ProcessBench SLA)
   cannot be reproduced from the release**, and the token-to-step question stays open.
2. **The released code does not implement the paper's headline method.** The paper's method is Shannon
   Drop: the drop statistic on the negative entropy of the renormalized top-20 distribution. The code
   implements a drop statistic on **the log-probability of the top-1 token**, with **an extra running-mean
   step the paper never mentions**. Its baseline, called "LN_E, Length-Normalized Entropy", is in fact the
   paper's **LN-S** baseline.
3. **The extra running mean changes what the statistic measures.** The first difference of a running mean
   is exactly the prefix innovation divided by position, so the released statistic gives the same drop
   21 times more weight at token 20 than at token 500. The paper's statistic weighs both identically.

## 2. What the paper says the method is

- Evidence per token (Eq. 9-10, lines 415-436): renormalize the top-K probabilities,
  `P~(v) = P(v) / sum_{top-K} P`, and take `E_i = -H(P~)`, the negative entropy. K = 20.
- Drop statistic (Eq. 11-12, lines 447-469): EMA over the **raw** evidence sequence, flux
  `Delta_j = E~_j - E~_{j-1}`, keep negative fluxes, risk = minus the mean of the M most negative. M = 5,
  EMA span 5.
- Baselines (Appendix D, lines 1700-1732): Shannon = mean renormalized top-K entropy; LogTokU = negative
  mean of the summed top-K log-probabilities; **LN-S = under greedy decoding, the negative mean
  log-probability of the maximum-likelihood token**.
- SLA (lines 597-605): success if "the first significant Evidence Drop (i.e., the first step t where
  Delta_t exceeds a step-wise threshold) matches the human-annotated first erroneous step". The paper
  indexes Delta by token in Eq. 11 and by step here, and never defines the step-wise threshold.
- ProcessBench protocol (lines 537-538): teacher forcing over the provided chains.

## 3. What the code actually does

`utils/metrics.py` holds the entire method.

| Stage | Code | Paper |
|---|---|---|
| Per-token signal | `get_logprob_curve`, lines 19-32: `max(obj.logprob for obj in step_data.values())`, the log-probability of the **top-1** candidate, full softmax, no renormalization | `-H` of the renormalized top-20 |
| Running mean | `calculate_running_mean`, lines 35-47: cumulative mean `cumsum / (t+1)` | **absent** |
| Smoothing | `calculate_ema_numpy`, lines 6-16: `alpha = 2/(span+1)`, seeded with the first value, applied **to the running mean** | EMA applied to the raw evidence |
| Flux and risk | `calculate_risk_with_running_mean_drop`, lines 68-98: `np.diff`, keep `< 0`, sort ascending, mean of the first `drop_k`, negate; fewer than `drop_k` drops are averaged as they come; no drops gives 0 | Eq. 11-12, same |
| Baseline "LN_E" | `calculate_risk_baseline_mean`, lines 50-65: minus the mean of the same top-1 log-probability curve | this is the paper's **LN-S**, Appendix D |

Under greedy decoding the top-1 candidate is the generated token, so in the released generation scripts
the signal is the log-probability of the token the model emitted, the negative of what our bank calls
`chosen_surprisal`. Under **teacher forcing** the provided token is generally not the top-1 candidate. If
the same function were applied there, `max` would return the confidence of the model's own best guess,
not anything about the provided token. The ProcessBench scripts are not released, so which one the paper
used for Table 3 cannot be checked.

### 3.1 The running mean, measured with their own code

The first difference of a running mean `m_t = mean(x_0..x_t)` is `(x_t - m_{t-1}) / (t+1)`: the
**prefix innovation**, divided by position. Calling their functions on synthetic curves:

| Check | Result |
|---|---|
| `max |diff(running mean) - prefix_innovation/(t+1)|`, 600 tokens | 3.7e-17 |
| Released statistic, one dip of -3 at token 20 vs token 500 | 0.02200 vs 0.00103, **ratio 21.3** |
| Paper statistic (EMA on the raw series), same two curves | 1.00000 vs 1.00000, ratio 1.00 |

So the released statistic is a smoothed, position-discounted prefix innovation of the top-1
log-probability. It concentrates on the opening tokens of an answer. For sequence-level selective
prediction that may still work; for localizing a step late in a chain it would be heavily biased early.

## 4. What the code does settle

| Item the paper leaves open | Resolved by the code |
|---|---|
| Prompt and thinking mode | Raw completion prompt with one few-shot example, **no chat template**, so Qwen3 thinking is never engaged. MATH: `Problem: ... Solution:` (`math/ln_e_drop.py` 29-35). GSM8K: `Question: ... Answer:` with `####` (`gsm8k/ln_e_drop.py`). Consistent with the digest's inference that the reported accuracies are non-thinking |
| Decoding | `temperature=0`, `max_tokens=1024`, `logprobs=20`, stop strings (`math/ln_e_drop.py` 120-127). The paper's `top_p = 0.95` does not appear, and is irrelevant at temperature 0 |
| Where the curve ends | Truncated at the final answer, re-tokenized to count. MATH keeps the `\boxed{...}` answer; GSM8K cuts **before** `####` (inconsistent between the two datasets) |
| EMA convention | `alpha = 2/(span+1)`, seeded with the first value: the same convention as our `derivative_step_channel_v1.ema` |
| Serving | vLLM, `bfloat16`, `max_model_len` 8192, prefix caching on |

## 5. What the code does not settle

- **Token-to-step mapping, the step-wise threshold, the ProcessBench prompt, no-error rows, tolerance.**
  None of the ProcessBench pipeline is released. The 6.2 pp gap between the paper's reported Shannon Drop
  mean on our eight ProcessBench cells, 39.27, and our faithful rebuild of the paper's series, 33.08
  (Step 424), cannot be closed from the release.
- **The calibration quantile.** `eval.py` takes `--tau` as an input (line 44, accept if `score <= tau`,
  line 74). The script that computes `tau` from the incorrect-answer scores is not released, so the paper's
  alpha versus (1 - alpha) contradiction (digest trap 1) stays open.
- **The drop method cannot even be calibrated end to end.** Only `ln_e.py` writes the incorrect-answer
  score file used for calibration; `ln_e_drop.py` does not, although the README says every method writes it.
- **AURC** is not computed anywhere in the release.
- **Shannon, LogTokU and every Drop variant other than the running-mean LN-S one** are not implemented, so
  the headline MATH/Qwen3-8B number, 88.26 selective accuracy for Shannon Drop, cannot be reproduced.

## 6. Consequences for this project

1. **Our replication followed the paper, and that was the right target.** Step 424 rebuilt `-H` of the
   renormalized top-20, EMA 5, first difference: the method as published. The project rule is fidelity to
   whatever carries the authors' name, and the published method is what the paper describes. The release
   is a different method, and results from it should be labelled as the released-code variant.
2. **The reproduction gap is not a mistake on our side that the code would reveal.** The code does not
   contain the part that is missing.
3. **The released statistic is built from pieces we already own.** Its signal is the chosen-token
   log-probability in generation mode, the negative of our `chosen_surprisal`, and its running-mean
   difference is exactly a prefix innovation. We found on 2026-09-27 that the derivative of
   `chosen_surprisal` is our strongest single PRMBench channel (within-AUC .7957; .7707 with step length
   regressed out). Whether that success is related to what the authors actually ran is an open question,
   not a finding: our channel uses the PROVIDED token under teacher forcing, a different EMA and a per-step
   readout, and has no position discount.
4. **Citing the paper.** Any comparison to Mind the Gap should state which object is compared: the
   published method, the released code, or the reported numbers. They are three different things.

## 7. Open items

- Run the released statistic, as written, on our teacher-forced telemetry with both possible readings of
  its signal (top-1 log-probability, and the provided token's log-probability), under our existing
  token-to-step collapses. Label it as the released-code variant.
- Ask the authors for the ProcessBench script. The README promises further updates; the repository has not
  changed since 2026-05-05.
