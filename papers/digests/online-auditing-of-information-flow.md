---
slug: online-auditing-of-information-flow
title: "Online Auditing of Information Flow"
authors: "Mor Oren-Loberman, Vered Azar, Wasim Huleihel (Dept. of Electrical Engineering–Systems, Tel Aviv University)"
arxiv_id: "arXiv:2310.14595v1 [cs.LG]"
venue: "arXiv preprint (v1, 23 Oct 2023) — the extract is the preprint. Journal version published as IEEE Trans. Signal and Information Processing over Networks, vol. 10, pp. 487–499, 2024, and at ICASSP 2024 (per Google Scholar / IEEE Xplore; NOT stated anywhere in the extract)."
year: 2023
source_pdf: papers/Online Auditing of Information Flow.pdf
extracted_text: papers/extracted/online-auditing-of-information-flow.md
last_digested: 2026-07-29
---

## Summary

Formulates fake-news detection on a social graph as a **sequential (quickest) detection**
problem: minimize a joint risk that prices *both* decision error and the delay before deciding.
Information spreads along a directed graph as a Markov chain over discretized edge weights
`W_e ∈ {0,…,Z−1}`; because only arbitrary parts of a propagation trace are observed, the
observed process is a hidden Markov chain. The authors reduce the joint minimization over
(stopping time `T`, decision rule `δ`) to a pure optimal-stopping problem on the posterior
`Π_ℓ = P(H₁|F_ℓ)` (Thm 2), give the optimal rule as a **two-sided threshold on `Π_ℓ`** solved
by a Bellman recursion (Thm 3), and show it is representable as an **SPRT** with Wald-calibrated
boundaries (Thm 4, Prop 1). On Weibo they match QuickStop's accuracy in **roughly half the
decision time**.

## Datasets & models used

- **Weibo dataset** [ref 6 in the paper] — the only dataset. Per the paper's Table 3:
  2,746,818 users; 3,805,656 tweets; **4,664 labeled events** (2,313 rumors / 2,351 non-rumors);
  avg 816 posts per event; avg time length 2,460.7 hours per event; min 10 / max 59,318 posts.
- Graph `G` reconstructed as the union of all propagation traces. Because Weibo contains the
  *full* trace and the method only needs a partial one, they **uniformly subsample 50% of the
  observations**.
- No language models involved. The only learned component is a **linear SVM** over averaged
  followee/follower user-feature vectors, whose `[0,1]` output is bucketed into `Z = 4` equal
  intervals to assign edge types.

## Methods it compared itself against

- **QuickStop** [ref 17] — the direct predecessor and the paper's main comparison; also a Markov
  optimal-stopping formulation. This paper generalizes it by adding the **graph structure** and
  **missing/partial observations** (hence hidden Markov rather than Markov).
- SVM-TSu / SVM-TSa [7] — dynamic series-time structure SVMs (user-features-only vs. full).
- DTCu / DTCa [1] — decision-tree tweet-credibility models (user-only vs. + content).
- SVM-RBFu / SVM-RBFa [18] — RBF-kernel SVMs (user-only vs. + content).
- CSI [13] — hybrid RNN + FC + integration deep model.
- PPC-R / PPC-C / PPC-R+C [5] — propagation-path classification with GRU, CNN, and RNN+CNN.

All baselines except QuickStop are **fixed-observation-budget** classifiers — they require a
pre-determined number of observations and do not make a real-time stop/continue decision. That
is the gap the paper targets.

## Experiments — methodology & scores

80% train / 20% test (test split is 10% genuine + 10% fake). Hyperparameters fixed at
`c_I = c_II = 10`, `c = 0.05`, `ε = 0.001`, `Z = 4`; path lengths truncated to a maximum for
tractability (the authors note truncation "can only impair the performance"). Metrics:
Accuracy, FP (genuine → fake), FN (fake → genuine), and **detection time = average number of
events before declaring**.

| Setup | Metric | Score | Notes |
|---|---|---|---|
| QuickStop (Weibo) | Accuracy / FP / FN | 0.85 / 0.08 / 0.20 | Table 4 |
| **Ours** (Weibo) | Accuracy / FP / FN | **0.86 / 0.08 / 0.18** | Table 4 — accuracy is a wash |
| QuickStop (Weibo) | Decision deadline | 12.75 events | Table 4 |
| **Ours** (Weibo) | Decision deadline | **6.29 events** | Table 4 — **≈2× faster at equal accuracy** |
| Ours, fake traces only | Detection time | 5.6 events | §4 text |
| Ours, genuine traces only | Detection time | 7.2 events | §4 text — asymmetry is by design, `c` only prices delay under H₁ |

Fig. 7 plots accuracy vs. number of events against all ten baselines; the paper claims it
dominates all of them on both accuracy and detection time. **Fig. 7 is a figure only — no
numeric table for the ten non-QuickStop baselines exists in the extract**, so the only literal
numbers available are the QuickStop row above.

The risk being minimized (Eq. 6–7):

```
inf_{T,δ}   c_I·P_{H0}(δ_T = 1)  +  c_II·P_{H1}(δ_T = 0)  +  c·E[T·1_{H1}]
             \___________________  ______________________/     \__________/
                                 \/                              delay cost,
                          error cost c_e(T,δ)                 charged only under H₁
```

Thm 2 collapses this to `inf_T E[g(Π_T) + cTΠ_T]` with `g(π) = min{c_II·π, c_I(1−π)}`, and the
terminal rule is just `δ_T = 1{c_II·Π_T > c_I(1−Π_T)}`. Thm 3 gives
`T* = inf{ℓ : Π_ℓ ∉ (π_low(z₁ℓ), π_up(z₁ℓ))}` with the two thresholds obtained from the Bellman
recursion (20) — note they are **time- and observation-dependent**, recomputed each step. Thm 4
rewrites this as an SPRT on the likelihood ratio `Λ_ℓ`, and Prop 1 + Wald's approximations let
you set the boundaries from *target* error rates: `B_low = q/(1−p)`, `B_up = (1−q)/p`, giving
`P_e,1 ≤ p(1+O(q))` and `P_e,2 ≤ q(1+O(p))`.

## Connection to our pipeline

**This is the one paper in Oren-Loberman's list that touches an open thread of ours: Extension E
(streaming / earliest-prefix detection), where the Step-148 pilot passed G1 and failed G2 and
PROGRESS records "earliest-prefix edge is the surviving thread."**

What we do today in Extension E is score prefixes at *fixed absolute budgets* and compare AUROC.
This paper supplies exactly what that formulation lacks: **a stopping rule, and a risk that
prices delay**. The mapping is mechanical:

| Paper | Our streaming setting |
|---|---|
| `ℓ` = propagation event index | `n` = generated-token index |
| `Z_ℓ` = discretized edge weight, `Z=4` levels | quantized per-token view — `token_entropies` H(n) or `token_spilled_energies` ΔE(n), both already saved per token |
| `α_I(z\|z′)`, `η_I(z)` under H₀/H₁ | first-order Markov transition matrices over quantized entropy levels, estimated separately for correct vs. hallucinated traces |
| `Π_ℓ` posterior that the item is fake | posterior that the answer being generated is wrong |
| `c` = cost per time slot of letting misinformation spread | compute cost per generated token — early abstention saves decoding |

**What transfers**: Thm 2's reduction, Thm 3's two-threshold stopping rule, and Thm 4's SPRT
representation with Wald-calibrated boundaries. These are generic once you can compute
`A_I(Z_ℓ|F_{ℓ−1})`.

**What does not transfer, and it is most of the paper**: the entire graph/path apparatus —
Eq. (11)/(13), the marginalization over all directed paths `P_ℓ`, and the hidden-Markov
structure induced by partial observation. A decoded token trace is *one* path, observed in full
and in order, so `A_I(Z_ℓ|F_{ℓ−1})` degenerates to `α_I(Z_ℓ|Z_{ℓ−1})` and what remains is a
classical SPRT on a two-state HMM. The paper's hard part is the part we do not have. **Cite it
for the formulation and the calibration, not for the theorems.**

**The blocking caveat**: the offline stage (Algorithm 1) is **supervised** — labeled traces are
needed to train the SVM edge classifier *and* to estimate `α_0, α_1, η_0, η_1`. Dropping this in
as-is puts a supervised transition-matrix estimator in a pipeline whose entire positioning is
label-free. Two honest options: (a) adopt it as a clearly-labeled **supervised streaming
baseline** (cf. `SUPERVISED_ORACLE_CORRECTION.md` conventions), or (b) estimate `α_0/α_1`
without labels — which is an open research problem in its own right, not a plug-in.

**Explicit non-overlap with the U-PCR / L-SML line**: nothing here addresses view selection,
orientation, or fusion. It is orthogonal to Steps 203–206.

## Notes / open questions

- **The deployed algorithm is not the theorem.** Algorithm 2 is a *first-order approximation* of
  Thm 3: a single fixed threshold `π₁` plus an `ε`-convergence stop (`|Π₀ − Π₁| < ε`), not the
  Bellman-recomputed `(π_low, π_up)` pair. The authors say the exact algorithm would decide
  *faster* at the same accuracy. Quote the 6.29-vs-12.75 number as "achieved by the
  approximation."
- **The metric lesson is the actionable one for Extension E.** Accuracy is a wash (0.86 vs 0.85);
  the entire contribution is 2× speed at equal accuracy. If we adopt this framing, the right
  reporting pair for streaming detection is **(AUROC at budget, tokens consumed)** — not AUROC
  alone. Our G2 gate was defined on detection quality only, which may be why it read as a
  failure.
- **Cost asymmetry is a design choice worth copying or rejecting deliberately.** `c·E[T·1_{H1}]`
  charges delay *only under H₁*, which is what produces the 5.6-vs-7.2 event asymmetry. For
  answer-level hallucination detection the compute cost is paid on *both* hypotheses, so the risk
  would need `c·E[T]` instead — a small but non-cosmetic change to Thm 2's `g(π)`.
- `c_I = c_II = 10`, `c = 0.05` are stated without any tuning or sensitivity analysis in the
  extract.
- Single dataset (Weibo), single language, single platform. No ablation over `Z` (the number of
  edge types) in the extract.
- **Version gap**: this digest is grounded in arXiv v1 (Oct 2023). The TSIPN 2024 journal version
  is ~15 months later and its numbers/baselines may differ; not checked.
- Related but not read: the same group's `Testing for a Hidden Geometry in Random Graphs`
  (arXiv:2606.16715) and `Testing dependency of weighted random graphs` (arXiv:2409.14870, IEEE
  T-IT 2026). See `papers/extracted/inhomogeneous-submatrix-detection.md` for the third.
