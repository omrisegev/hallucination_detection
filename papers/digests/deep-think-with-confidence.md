---
slug: deep-think-with-confidence
title: "Deep Think with Confidence (DeepConf)"
authors: "Yichao Fu (UCSD, work done during an internship at Meta FAIR), Xuewei Wang, Yuandong Tian, Jiawei Zhao (Meta AI) — Fu and Zhao equal contribution"
arxiv_id: "arXiv:2508.15260v1 [cs.LG], 21 Aug 2025"
venue: "arXiv preprint (no venue stamp in the PDF). Project page: jiaweizzhao.github.io/deepconf"
year: 2025
source_pdf: "papers/DEEP THINK WITH CONFIDENCE.pdf"
extracted_text: papers/extracted/deep-think-with-confidence.md
last_digested: 2026-08-03
---

> **First digest, 2026-08-03.** This paper was cited throughout Extension E (Step 148) as the
> streaming baseline but was never in `papers/index.md` and had no card. Digested now because the
> step-localization work (Extension F) replicates it. All numbers below are copied from the
> extract.

## Summary

Test-time scaling by self-consistency wastes compute and saturates, because majority voting treats
every reasoning trace as equally good. DeepConf scores each trace by **local** confidence rather
than a trace-wide average, then (i) *filters* to the top-η% most confident traces and (ii) weights
their votes by confidence. The same local signal doubles as an **online early-stopping** rule: kill
a trace the moment its sliding-window confidence drops below a threshold calibrated on a small
warm-up. It needs no training and no hyperparameter tuning, and it plugs into vLLM.

## Datasets & models used

- **Benchmarks**: AIME24, AIME25, BRUMO25, HMMT25 (competition math) and GPQA-Diamond
  (graduate STEM). 30 problems each for the math sets.
- **Models**: **DeepSeek-8B** (= `deepseek-ai/DeepSeek-R1-0528-Qwen3-8B`), **Qwen3-8B**,
  **Qwen3-32B**, **GPT-OSS-20B**, **GPT-OSS-120B**.

## Methods it compared itself against

- **Pass@1** — single-trace accuracy.
- **Cons@K** — unweighted self-consistency majority voting (Wang et al., 2023), the primary baseline.
- **Mean / average trace confidence** (a.k.a. self-certainty, following Kang et al., 2025) — the
  *global* aggregation DeepConf argues against.

## Experiments — methodology & scores

**Confidence definitions** (§3.1):

| Quantity | Definition |
|---|---|
| Token confidence | `C_i = −(1/k) Σ_{j=1..k} log P_i(j)` over the top-k tokens (Eq. 2) |
| Group confidence | `C_Gi = (1/\|G_i\|) Σ_{t∈G_i} C_t` over an **overlapping** sliding window (Eq. 4) |
| Bottom-10% group | mean of the lowest-10% group confidences in the trace (Eq. 5) |
| Lowest group | `min_Gj C_Gj` — the special case (Eq. 6) |
| Tail | mean `C` over the final `\|T_tail\|` tokens, e.g. 2,048 (Eq. 7) |

**Sampling frame**: a pool of **4,096 complete traces per problem**; offline runs resample a working
set of size K from that pool; **all metrics averaged over 64 independent runs with fresh
resampling**. Filtering cutoff is *recomputed within each sampled working set on every run*.

**Online** (Alg. 2): warm up with `N_init = 16` traces, set `s = Percentile_{100−η}` of warm-up
trace confidences using Lowest Group Confidence with a **2,048** window; terminate any new trace
whose current group confidence falls below `s`; keep sampling until consensus
`β = V(â)/Σ_a V(a) ≥ τ = 0.95` or budget `B ∈ {32,64,128,256,512}`. **DeepConf-low = η 10%,
DeepConf-high = η 90%.**

**Decoding** (Table 11): DeepSeek-8B `T=0.6, top-p 0.95, top-k —, 64k`; Qwen3-8B / Qwen3-32B
`T=0.6, top-p 0.95, top-k 20, 32k`; GPT-OSS-20B/120B `T=1.0, top-p 1.0, top-k 40, 130k`.
**Prompting** (App. F): Qwen3 and GPT-OSS get *"Please reason step by step, and put your final
answer within \boxed{}."* appended; GPT-OSS additionally keeps the provider system prompt at
`reasoning effort = high`; DeepSeek-8B uses its official system prompt with the problem in the user
message.

### Headline numbers

| Setup | Metric | Score | Notes |
|---|---|---|---|
| AIME25 / GPT-OSS-120B | Accuracy | **99.9** (deepconf@512) vs cons@512 97.0 vs pass@1 91.8 | the abstract's headline |
| AIME25 / DeepSeek-8B | Accuracy | **87.4** (Tail@10) vs cons@512 82.3 vs pass@1 76.9 | Table 1 |
| AIME24 / Qwen3-32B | Accuracy | **90.8** (Bottom-10@10) vs cons@512 85.3 | Table 1 |
| AIME25 / GPT-OSS-120B (online) | Tokens | **−84.7%** vs full parallel thinking | Fig. 1 |
| Offline, avg over 23 pairs | Accuracy@512 | Tail(10%)@10 **84.6** · Tail(2k)@10 **84.5** · L(2k)@10 **84.4** · Bottom-10@10 84.0 · Bottom-50@10 84.0 · Mean@10 83.9 | App. B.4 |

**Qwen3-8B × AIME24 — our reproduction targets** (this model/dataset appears only in the
appendix tables, *not* in Table 1 or Fig. 5):

| Source | Numbers |
|---|---|
| Table 6 | Maj 80.1 · Mean 80.1 · **Mean@10 86.7** · Mean@90 80.5 · Head(10%) 80.1 · Head(10%)@10 80.5 · Head(10%)@90 80.0 |
| Table 7 | Tail(10%) 80.3 · **Tail(10%)@10 87.1** · Tail(10%)@90 80.7 · Tail(2k) 80.4 · **Tail(2k)@10 86.7** · Tail(2k)@90 80.7 |
| Table 8 | Maj 80.1 · L(512) 80.1 · L(1K) 80.1 · L(2K) 80.1 · **L(512)@10 86.7 · L(1K)@10 86.7 · L(2K)@10 86.7** · L(2K)@90 80.3 |
| Table 9 | Maj 80.1 · **B(10%)@10 86.7** · B(10%)@90 80.3 · B(50%) 80.1 · **B(50%)@10 86.7** · B(50%)@90 80.4 |
| Table 5 (best over B) | Maj 81.4 · Top90 82.1 · Top75 82.2 · Top50 84.9 · **Top25 86.9** · Top10 86.7 |
| Table 10 (online) | Maj@512 **2.32e8 tok / 80.0%** · high **1.33e8 (−42.8%) / 80.4%** · low **0.90e8 (−61.1%) / 86.5%** |

**Directly measured trace length** (useful for cost planning): Qwen3-8B AIME24 majority@512 =
2.32e8 tokens over 30 problems × 512 traces → **≈15,100 output tokens per trace**. DeepSeek-8B on
AIME25 = 4.01e8 / 15,360 → **≈26,100 tokens per trace**.

## Connection to our pipeline

- **This is our Extension-E baseline, and our implementation of it is an approximation.**
  `spectral_utils/streaming_utils.py:deepconf_lowest_group_conf` uses `−H(t)` as the token
  confidence proxy and a **64-token** window; the paper uses `C_i` over top-k logprobs and a
  **2,048** window. Every Step-148 claim measured against "the best DeepConf window" used the
  approximation, so correcting it may move those numbers.
- **Same signal family as ours, different task.** Their group/bottom/tail confidences are local
  window statistics on a per-token trace — the family our `sw_var_peak`, `cusum_max` and spectral
  views live in. Their task is *best-of-N selection*, ours is *per-answer detection*; the natural
  head-to-head is to enter our fused score as one more confidence measure in their voting harness.
- **Positivity constraint on our arm.** `V(a) = Σ_t C_t · I(answer(t)=a)` needs strictly positive
  weights (Fig. 2 axes run ~10–22; the vLLM default threshold in App. G.4 is `17`). A z-scored
  L-SML / U-PCR score is signed and scale-free, so it needs a stated positivity/scale mapping
  before it can weight votes. The *filtering* variant needs only an ordering and is mapping-free.

## Notes / open questions

- **`C_i` excludes the sampled token in their code.** Eq. 2 says "the top-k tokens", but App. G.4
  Step 5 is `# logprobs[0] is the sampled token; use the remaining candidates` →
  `new_conf = -sum(logprobs[1:]) / len(logprobs[1:])`, with `top_logprobs=20` (G.3). Including
  index 0 shifts every confidence value and therefore every threshold. **Follow the code.**
- **The paper's own conclusion is weaker than "local beats global".** App. B.4: local signals "are
  **not inferior** to global average trace confidence and, on average, deliver equal or better
  accuracy". The spread over 23 pairs is 83.9 → 84.6, i.e. **0.7pp**, and the winner is the *tail*,
  not the lowest group. Do not overstate this when using it for positioning.
- **Aggressive filtering sometimes hurts.** η=10 regresses on 8 of 20 settings (−4.69% to −0.31%),
  from "confidently wrong" cases; overall average is only +1.22%. Qwen3-8B on AIME25 is a stark
  example: Maj 82.6 → **Mean@10 74.0**, L(2K)@10 74.2. η=90 is the safe setting (+0.17% avg).
- **Reproducing the pool is the expensive part**, not the method: 4,096 traces × 30 problems ×
  ~15k tokens ≈ 1.8e9 output tokens per (model, dataset). The method itself is pure offline
  arithmetic over saved logprobs.
- **vLLM must be pinned to a commit, not a tag** — App. G.1 pins
  `31f09c615f4f067dba765ce5fe7d00d880212a6d` (Python 3.12.0, CUDA 12.8), and G.2–G.6's source edits
  are written against that API.
- **Which tables contain which model** is easy to get wrong: **Table 1 covers only DeepSeek-8B,
  Qwen3-32B and GPT-OSS-120B**, and **Fig. 5 is DeepSeek-8B only**. Qwen3-8B and GPT-OSS-20B appear
  exclusively in Appendix Tables 5–10.
