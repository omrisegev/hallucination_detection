---
slug: mind-the-gap-catching-hallucinations-via-evidence-drop
title: "Mind the Gap: Catching Hallucinations via Evidence Drop on the Reasoning Manifold"
authors: "Qunjie Chen, Yufei Chen (corresponding, yufeichen@tongji.edu.cn), Linye Li — School of Computer Science and Technology, Tongji University; Xiaodong Yue — Artificial Intelligence Institute, Shanghai University"
arxiv_id: "none found in the PDF (no arXiv stamp on any page)"
venue: "ICML 2026 (Proceedings of the 43rd ICML, Seoul, South Korea; PMLR 306, 2026)"
year: 2026
source_pdf: "papers/Mind the Gap -  Catching Hallucinations via Evidence Drop on the Reasoning.pdf"
extracted_text: papers/extracted/mind-the-gap-catching-hallucinations-via-evidence-drop.md
last_digested: 2026-08-03
---

> **REWRITTEN 2026-08-03 from the actual PDF (25 pages incl. Appendices A–F).** The previous card
> was `abstract-only` and stated the PDF did not exist. It does. Everything below is grounded in
> `papers/extracted/<slug>.md`; the earlier card's `UNVERIFIED` markers on models, baselines and
> scores are now resolved. Code: https://github.com/QJ0114/evidence-drop (not inspected).

## Summary

Reasoning is modelled as a trajectory of latent evidence states on a low-dimensional **Evidence
Manifold**, evolving as a Markov process. Under two locality assumptions (the training data
contains no direct transitions between non-adjacent evidence states; adjacent transitions preserve
the true local structure), Theorem 3.1 shows an off-manifold transition drives the optimal
predictive distribution to **uniform** — so a hallucination is not high *absolute* uncertainty but
a **sudden decline** in evidence. The detector is therefore training-free and single-pass: take
negative renormalized top-K entropy as an evidence proxy, EMA-smooth it, and score a trace by the
**mean of its M most negative first-differences**. Because the score is built from *located* drops,
the same statistic localizes to a reasoning step.

## Datasets & models used

- **Sequence-level**: GSM8K (Cobbe et al., 2021a), MATH (Hendrycks et al., 2021). *The paper says
  "MATH" — the strings `MATH-500` / `MATH500` / `500` appear nowhere in it.*
- **Step-level**: ProcessBench, reported per sub-benchmark — **GSM8K, MATH, OlympiadBench,
  Omni-MATH** (Table 3).
- **Appendix E.3**: ProofWriter, OWA and CWA at depth-3.
- **Models**: **Qwen3-4B** and **Qwen3-8B** (Yang et al., 2025a) as primary backbones; one
  **Qwen3.5-27B** MATH row in Table 1 (α = 0.05 only). Hardware: 4× NVIDIA A100-40GB.

## Methods it compared itself against

All from Appendix D with the paper's own formulas (Eq. 46–48). Each appears in an **Avg** and a
**Drop** variant, so the comparison isolates the aggregation rule rather than the signal.

| Baseline | Per-step quantity | Sequence score |
|---|---|---|
| **Shannon Entropy** | `H_i = −Σ_{v∈V_K} P̃(v\|y_<i) log P̃(v\|y_<i)` over the renormalized top-K | `φ = (1/T) Σ H_i` |
| **LogTokU** (Ma et al., 2025) | evidence mass `M_i = Σ_{v∈V_K} log P(v\|y_<i)` | `φ = −(1/T) Σ M_i` |
| **LN-S** (Malinin & Gales, 2020) | `log P(v\|y_<i)` of the **actually generated** token | `φ = −(1/T) Σ log P` |

Appendix E.2 additionally compares against **Self-Consistency** (N = 5/10/15) as the
generation-based / Semantic-Entropy-family representative.

## Experiments — methodology & scores

**Decoding**: greedy, temperature τ = 0, nucleus p = 0.95, decoding top-K fixed at 20 (the same K
used in Eq. 10). **Method defaults**: M = 5, EMA span = 5, top-K = 20.

**ProcessBench protocol**: *teacher-forcing / forced decoding* — "given a pre-defined reasoning
chain, we perform a single forward pass to compute the logit distributions for each token
transition", so the step-wise drop maps onto the human-annotated segments.

**Calibration**: test data split 50/50 into `D_cal` and `D_eval`; τ̂ is a quantile of the risk
distribution of the samples the model got **wrong** on `D_cal`; decision is `Accept if φ ≤ τ̂`.

**Metrics**: Selective Accuracy (Eq. 13) at α ∈ {0.05, 0.10, 0.50}; **AURC ×1000, lower is
better**; Step-level Localization Accuracy (SLA).

### Headline numbers

| Setup | Metric | Score | Notes |
|---|---|---|---|
| MATH / Qwen3-8B, α=0.05 | Selective Acc | **88.26** vs Shannon Avg 70.24 | +18.02 — the paper's headline |
| GSM8K / Qwen3-4B, α=0.05 & 0.10 | Selective Acc | **100** | Shannon Drop |
| GSM8K / Qwen3-8B, α=0.05 | Selective Acc | LogTokU Avg **0** → Drop **96.40** | the "rescuing weak baselines" claim |
| MATH / Qwen3-8B | AURC ×1000 | Shannon **288.8 → 190.0** | Table 2 |
| MATH / Qwen3-4B | AURC ×1000 | LogTokU **481.3 → 299.1** | Table 2 |
| GSM8K / Qwen3-8B | AURC ×1000 | LN-S 45.4 · LogTokU 155.8→73.8 · Shannon 41.9→**48.3** | **Shannon Drop is worse here** |
| ProcessBench GSM8K / Qwen3-8B | SLA | Shannon **27.66 → 46.11** | Table 3 |
| ProcessBench OlympiadBench / Qwen3-8B | SLA | Shannon **26.30 → 41.52** | Table 3 |
| ProcessBench Omni-MATH / Qwen3-8B | SLA | Shannon **23.40 → 37.04** | Table 3 |
| ProcessBench MATH / Qwen3-8B | SLA | Shannon **24.62 → 32.90** | Table 3 |
| ProofWriter OWA-d3 / Qwen3-8B | AURC | Shannon **717.17 → 589.02** | Table 7; LN-S Drop *hurts* (711.82→801.13) |

**Pretrained accuracies — the operating point everything above is measured at**: GSM8K Qwen3-4B
**87.63 ± 0.31**, Qwen3-8B **91.07 ± 0.26**. For MATH, Table 1 says 59.24 ± 0.21 — **use Appendix
E.2 Table 6 instead** (caveat 5 below): Qwen3-8B **66.12 / 66.40 / 65.96**, Qwen3-4B
**57.92 / 58.56 / 57.88**.

## Connection to our pipeline

- **Same premise as our own pivot.** "Sequence-level aggregation inevitably obscures the
  fine-grained dynamics of uncertainty evolution" is the argument that moved us off EPR (the DC
  component of H(n)) toward the AC spectrum. Two independent groups on the same premise is
  positioning support.
- **Their score is a special case of ours.** `φ` is the mean of the M worst first-differences of an
  EMA-smoothed entropy trace — one statistic of one derived series. Our pool already carries
  `cusum_max` / `cusum_shift_idx` (a change-point *location*), `sw_var_peak` (windowed variance
  peak) and the full FFT/STFT views over the same trace.
- **This is the reference implementation for Extension F** (step-level localization,
  `Research_Directions.md`), and its ProcessBench teacher-forcing protocol is directly reusable via
  `cluster/backfill_views.py`'s existing forced-decoding path (`forward_batch`,
  `build_prompt_ids(kind="stored_text")`, `candidate_quantities`).
- **Genuinely training-free and single-pass**, like ours — a fair cost-class comparison, unlike
  INSIDE (K=10) or Semantic Entropy (multi-sample).
- **Selective accuracy / AURC are new metric machinery for us** — nothing in this repo implements
  risk-coverage, AURC, or selective accuracy as of Step 218.

## Notes / open questions

**Five traps for anyone reproducing this. Each verified against the extract.**

1. **The quantile direction is self-contradictory.** §4 and App. C.2 both say τ̂ is the
   **(1−α)-quantile** of the calibration risk distribution. But Eq. 43 states the guarantee as
   `P(φ ≤ τ | H0) ≤ α` with `Accept if φ ≤ τ̂`, which requires the **α**-quantile. Table 1 settles
   it empirically: selective accuracy *decreases* monotonically as α grows (Shannon Drop GSM8K-4B:
   100 → 100 → 90.51), which only happens under the α-quantile. **Implement the α-quantile.**
2. **Two incompatible definitions of "evidence".** §3.3 Eq. 10 defines `Ê_i := −H(P̃)`, negative
   renormalized top-K **entropy**. Appendix B Eq. 36/39 defines `E = log Σ_{v∈Top-K} q(v)`, log
   top-K probability **mass**. Theorem 3.1 is proved for the second; the method implements the
   first. **The theory does not cover the statistic that was actually run.**
3. **Table 5 panels (a) and (b) are byte-identical, row for row** — `84.58/73.40/87.26`,
   `85.77/81.56/87.92`, `86.94/81.55/88.26`, `85.93/83.33/90.75`, `55.43/65.66/74.81`. Only one of
   the "Max Drops M" and "EMA Span" ablations can actually have been run. Do not cite the defaults
   as ablation-confirmed. Also: **M = 10 (90.75) beats the M = 5 default (88.26)** for Shannon Drop.
4. **Table 3 contains duplicate cells**: `LN-S Drop` equals `Shannon Avg` exactly for Qwen3-4B on
   GSM8K (both 27.94) and on MATH (both 24.17). Likely a copy error — not reproduction targets.
5. **Table 1's MATH "Pretrained" is 59.24 ± 0.21 for *both* Qwen3-4B and Qwen3-8B** — identical
   across two different models, while GSM8K correctly differs (87.63 vs 91.07). Appendix E.2
   Table 6 gives the real per-model figures (66.1 vs 57.9), i.e. the 4B value was copied into the
   8B row. **Cite Table 6.**

**Genuinely underspecified — these become *our* pre-registered choices, not theirs:**

- **No prompt template, thinking mode, max generation length, or seed appears anywhere.** The three
  accuracy figures (GSM8K-8B 91.07, GSM8K-4B 87.63, MATH-8B ~66) are consistent with Qwen3
  **non-thinking** and not with thinking-on.
- **The Drop variants of LN-S and LogTokU are never defined.** App. D gives only their *Avg*
  formulas, and the §3.3 Drop pipeline is written solely for `Ê_i = −H(P̃)`. The orientation of the
  per-step series before EMA + M-worst-drops is unstated, and it flips the metric.
- **SLA's token→step aggregation is undefined.** `Δ_j` is a *token*-level first-difference (§3.3),
  but SLA is defined on "the first **step** t where ∆t exceeds a step-wise threshold" (§4). How
  token Δ collapses to step Δ is the dominant free parameter for Table 3 and is never given.
  Neither is the handling of ProcessBench rows with no error, nor whether "matches" allows tolerance.
- **The Appendix C.2 finite-sample correction (Eq. 44) needs a confidence level δ** the paper never
  supplies.

**Honest reading of the result's strength**: Drop is not a uniform win. In Table 2 Shannon Drop is
*worse* than Shannon Avg on GSM8K/Qwen3-8B (41.9 → 48.3), and in Table 7 LN-S Drop is much worse
than LN-S Avg under both ProofWriter settings. The consistent wins are on the harder distributions
(MATH, OlympiadBench, Omni-MATH) — the same "helps where the signal is hard" pattern our own work
keeps finding.
