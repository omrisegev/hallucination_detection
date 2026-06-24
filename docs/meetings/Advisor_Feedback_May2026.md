# Advisor Feedback — May 2026
## Meeting with Ofir Lindenbaum & Bracha Laufer-Goldshtein

**Date:** May 2026  
**Action items and research questions from post-meeting notes.**

---

## Point 4 — Code Investigation: Feature Normalization (CRITICAL — read first)

**Question:** Are we currently normalizing features before the ρ-filter and Nadler fusion?

**Answer: NO. We are not normalizing.**

### What the code currently does

In every spectral phase notebook (Phase 4/5/6/7), `best_nadler_on` does:

```python
# 1. Orient by AUC sign (multiply by +1 or -1)
oriented = {n_: feats_dict[n_] * sign_m[n_] for n_ in feat_names}

# 2. Spearman ρ filter — uses scipy.stats.spearmanr (rank-based, scale-INVARIANT ✓)
rho[(a, b)] = spearmanr(oriented[a], oriented[b])[0]

# 3. Nadler fusion — uses np.cov(X.T) (covariance, scale-DEPENDENT ✗)
fused, _ = nadler_fuse(*[oriented[n_] for n_ in s])
```

Inside `nadler_fuse`:
```python
X = np.column_stack(views)
C = np.cov(X.T)          # <-- raw covariance, NOT correlation
M = diag(rs) @ pinv(C) @ diag(cs)
w = abs(eigenvector of M); w /= w.sum()
return X @ w, w
```

### Why this matters

Our 12 features span very different scales:

| Feature | Typical scale | Variance ratio vs epr |
|---|---|---|
| `trace_length` | 100–500 | ~10,000× |
| `sw_var_peak` | 0.01–2.0 | ~100× |
| `epr` | 0.8–2.5 | 1× (reference) |
| `rpdi` | 0.8–1.5 | ~1× |
| `spectral_entropy` | 2–5 | ~5× |

`trace_length` has variance ~10,000× larger than `epr`. Since Nadler's `C = np.cov(X.T)` is the raw covariance matrix (not the correlation matrix), its diagonal and off-diagonal entries are dominated by high-variance features. This means:

1. **Nadler weights are biased toward high-scale features** — not because they are more discriminative, but because they dominate the covariance structure. The algorithm cannot distinguish "large because informative" from "large because of units."

2. **Simple average comparison (Point 1) would be meaningless** without normalization — averaging raw `trace_length` (~300) with raw `epr` (~1.5) gives a score dominated entirely by `trace_length`.

3. **The ρ filter is unaffected** — `spearmanr` is rank-invariant, so our correlation-blocking step is correct regardless of scale.

4. **The final AUC is partly protected** — `roc_auc_score` is also rank-invariant, so a linearly dominant `trace_length` still gives the right AUC *for that single feature*. But the **Nadler weights are suboptimal**, meaning we may be losing the complementarity signal that motivates using fusion in the first place.

### Fix required

Add z-score normalization after sign orientation and before fusion:

```python
def zscore(arr):
    std = arr.std()
    return (arr - arr.mean()) / std if std > 1e-8 else arr - arr.mean()

# In best_nadler_on, replace:
oriented = {n_: feats_dict[n_] * sign_m[n_] for n_ in feat_names}
# with:
oriented = {n_: zscore(feats_dict[n_] * sign_m[n_]) for n_ in feat_names}
```

This makes the covariance matrix `C` equivalent to the Spearman correlation matrix (approximately), so Nadler weights reflect statistical complementarity rather than scale dominance.

**Important note:** This fix may change our reported AUC numbers. We should re-run Phase 5 and Phase 7 (GSM8K) with normalized features to see the true performance of Nadler fusion.

---

## Point 1 — Nadler Fusion Weighting vs. Simple Averaging

**Status:** Blocked on Point 4. Must fix normalization first.

### Plan (after normalization fix)

Add a `simple_average_fusion` function that takes the same z-scored, sign-oriented feature subset and computes the unweighted mean:

```python
def simple_average_fusion(*views):
    X = np.column_stack(views)
    return X.mean(axis=1), np.ones(X.shape[1]) / X.shape[1]
```

Then for every candidate subset evaluated in `best_nadler_on`, compute both:
- `auc_nadler` = AUROC of Nadler-weighted fusion
- `auc_mean`   = AUROC of unweighted mean

Report:
- **Nadler lift** = `auc_nadler − auc_mean` (can be negative if weights hurt)
- For the best Nadler subset, report both AUCs

### What to expect

If Nadler's covariance weighting is actually useful:
- Correct weights should upweight low-correlation complementary features
- Expected lift: +0.5% to +3% over simple average
- If lift ≈ 0%: the simple average already captures all the information → Nadler overhead is not justified

If normalization reveals that Nadler lift was previously an artifact of scale bias:
- Some results may drop slightly after normalization
- Some results may improve (previously dominated by `trace_length` may now use all features)

This ablation is essential for the thesis: it directly answers "why Nadler, not just average?"

### Implementation target

Add to `Spectral_Analysis_GSM8K_vs_LapEigvals.ipynb` (Phase 7) as a new cell after the fusion cell. The GSM8K dataset is the cleanest comparison case since it matches LapEigvals exactly.

---

## Point 2 — Theoretical Basis for Temperature Variation

### Literature to explore

The cross-temperature fusion result from Phase 5 (T=1.0 + T=1.5 combined features beat either single temperature) needs theoretical framing. Here are the most relevant threads:

#### A. EPR paper foundation
The original EPR framework explicitly discusses temperature dependence. At temperature T, token entropy H(n) = −∑ p_T(token) log p_T(token) where p_T ∝ logit/T. Different temperatures probe different moments of the same underlying logit distribution. This is our starting point.

#### B. Mode fragility hypothesis
**Core idea:** Correct answers correspond to high-probability, low-entropy modes that are *stable* under temperature increase. Hallucinated answers correspond to multi-modal or flat distributions that look qualitatively different at T=1.5 than at T=1.0.

Statistical mechanics analogy: in equilibrium physics, applying different temperatures to a system probes its *free energy landscape*. Stable configurations (deep minima) are occupied at all temperatures; metastable configurations (shallow minima) only appear at low T and disappear at high T. Language generation is non-equilibrium, but the analogy holds qualitatively.

**What to look for in literature:**
- Does this "mode stability under temperature perturbation" idea appear in any LLM uncertainty paper?
- Check: Rabe et al. 2023, Kadavath et al. 2022, Xiong et al. 2024 (uncertainty survey)

#### C. Complementary moments argument
T=1.0 captures E[H(n)] in the natural regime — the *mean entropy* of the generation. T=1.5 captures how this changes under perturbation — the *sensitivity* to thermal noise. These are mathematically different quantities, analogous to measuring both the mean and the derivative of a function. Their combination is information-theoretically richer than either alone.

More formally: let f(n, T) = H_T(n) be the entropy of token n at temperature T. Our Phase 5 features are measurements of f(·, 1.0) and f(·, 1.5). Their cross-temperature Spearman ρ < 0.75 empirically confirms they carry independent information — which is exactly the Nadler independence requirement.

#### D. Fluctuation-dissipation connection
In statistical physics, the fluctuation-dissipation theorem says: the response of a system to external perturbation is related to its internal fluctuations. Temperature variation in LLMs is exactly this kind of perturbation. The difference in behavior between T=1.0 and T=1.5 *probes the system's susceptibility to noise* — and hallucinations are precisely the responses that are most susceptible (least grounded in training data).

This is a novel framing that could be a thesis contribution if stated carefully.

#### E. Papers to cite (search these)
1. **SIA paper** (arXiv:2604.06192) — "Stochastic Integration Approach" — theoretical grounding for entropy dynamics in LLM generation. May have temperature discussion.
2. **Temperature scaling / calibration**: Guo et al. (2017) "On Calibration of Modern Neural Networks" — foundational, but different use case.
3. **Semantic entropy** (Farquhar et al., 2024) — they mention temperature sampling but mainly for generating multiple samples, not for feature extraction.
4. **Self-consistency** (Wang et al., 2023) — uses multiple temperature samples to detect consistency. Different angle but cites the mode stability idea.
5. **SPREG** (arXiv:2604.17884) — entropy spike intervention. May discuss how temperature affects entropy spikes.

### Research question formulation
*"Is there theoretical justification for treating LLM outputs at different sampling temperatures as independent views of the same underlying uncertainty? Specifically: does temperature variation expose 'mode fragility' in hallucinated responses while leaving grounded responses more stable, and can this be grounded in the entropy production rate framework?"*

This framing connects our empirical finding (cross-T fusion improves AUC by 0.9–3pp) to a theoretical mechanism.

### Suggested next step
Before finalizing the code, do a targeted literature search (30 min) with this query:
> "hallucination detection sampling temperature uncertainty complementary views entropy"

Key question to answer: **has anyone else used multiple temperatures as independent views for ensemble hallucination detection?** If not, our contribution may be novel. If yes, we need to position against it.

---

## Point 3 — Stronger Model for GPQA Diamond

### Current situation
- Mistral-7B on GPQA Diamond: ~30% accuracy (random chance for 4-way MCQ is 25%)
- Qwen2.5-7B on GPQA Diamond: ~30% accuracy  
- Spectral AUC: 57–65% (barely above chance for Mistral, near-chance for Qwen)

At 30% accuracy, the model doesn't *know enough* to be wrong in an interesting way. Hallucination detection requires a model that can reason but sometimes fails — not a model that is fundamentally at its knowledge ceiling.

### Model options

| Model | GPQA Diamond accuracy | Colab feasibility | Access |
|---|---|---|---|
| Llama-3.3-70B-Instruct | ~60% | A100 with 4-bit quant | Gated (need HF token) |
| Qwen2.5-72B-Instruct | ~65% | A100 with 4-bit quant | Open |
| DeepSeek-V3 (API) | ~70%+ | API only, no GPU needed | API key |
| Llama-3.1-70B-Instruct | ~55% | A100 with 4-bit quant | Gated |

**Recommendation: Qwen2.5-72B-Instruct** — highest accuracy on GPQA, fully open (no HF gating), and supported by our existing pipeline without code changes. The only change needed is `model_id = 'Qwen/Qwen2.5-72B-Instruct'` and adding `load_in_4bit=True` to the `load_model` call for Colab memory constraints.

### Code change needed in `load_model`

```python
def load_model(model_id, quantize_4bit=False):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    bnb_cfg = BitsAndBytesConfig(load_in_4bit=True,
                                  bnb_4bit_compute_dtype=torch.float16) if quantize_4bit else None
    mdl = AutoModelForCausalLM.from_pretrained(
        model_id, device_map='auto',
        torch_dtype=torch.float16,
        quantization_config=bnb_cfg,
        trust_remote_code=True)
    mdl.eval()
    return mdl, tok
```

Note: 4-bit quantization may slightly affect entropy values — worth documenting.

### Expected impact on results
With Qwen2.5-72B at ~65% accuracy on GPQA Diamond:
- More correct/incorrect split (65:35 vs 30:70) — better class balance for AUC
- The model has genuine reasoning capability, so its entropy traces will reflect reasoning uncertainty rather than pure knowledge absence
- Spectral AUC should improve substantially — the entropy H(n) difference between correct and incorrect solutions should be much more pronounced when the model actually "knows the subject"

---

## Summary: Priority Order

| # | Action | Urgency | Effort |
|---|---|---|---|
| 1 | Fix z-score normalization in all phase notebooks | HIGH (affects all results) | Low (1 line per notebook) |
| 2 | Add `simple_average_fusion` ablation to Phase 7 | HIGH (advisor request) | Low (1 new cell) |
| 3 | Re-run Phase 5 and Phase 7 with normalization fix | HIGH (validates true performance) | Medium (new Colab run) |
| 4 | Literature search on temperature variation theory | MEDIUM (thesis framing) | Medium (1 hour research) |
| 5 | Upgrade GPQA to Qwen2.5-72B-Instruct | MEDIUM (new results) | Low (1-line model change) |

**Critical dependency:** Points 1 and 2 must be done in order (normalize first, then compare vs average). Everything else is independent.
