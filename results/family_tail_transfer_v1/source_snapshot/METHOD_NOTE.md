# What tail-mark L-SML estimates, and what it assumes

This note responds to section 3 of the 2026-09-24 handoff. It states the learning object before any further experiment. Where no derivation exists, it names the assumption being tested. It is an adaptation *inspired by* L-SML (Jaffe et al., 2016). It is not the original binary model.

## Recipe

For one answer with `n` steps (or tokens) and a channel `X_i`:

1. **Mark.** `B_i = 1` on the `k = ceil(0.2 n)` highest positions of `X_i`, and `0` elsewhere. In the tie-aware version, positions tied at the boundary share the remaining mass equally. A channel that is constant within the answer then carries no mark.
2. **Centre within the answer.** `B~_i = B_i − k/n`.
3. **Fit.** Pool `B~` over the fit answers (steps at step level; the length-weighted 60,000-token sample at token level). Standardize, then run continuous L-SML (`lsml_continuous`): group discovery, then a rank-one cross-group term. The weights are the flattened products of the cross-group and within-group weights, and their sign is oriented by Spearman correlation with the entropy anchor.
4. **Apply.** Apply the weights to the **continuous** standardized channels `X`, then take the Top10 step readout (token level only), then answer-z.

## The learning object

Let `Y ∈ {0,1}` be the step label (1 = error), `π = P(Y=1)`, and `d_i = E[B_i | Y=1] − E[B_i | Y=0]`. By the law of total covariance,

`Cov(B_i, B_j) = E[Cov(B_i, B_j | Y)] + π(1−π) d_i d_j`.

- **Conditional independence given Y** (the SML assumption) makes the first term zero off the diagonal. The off-diagonal covariance is then rank one with loadings `∝ d_i`, and the leading direction recovers `d_i` up to sign.
- **L-SML** allows dependence inside latent groups: the first term is block-structured. The rank-one term is identified *across* groups only. It needs at least three groups (K≥3), and a group structure that is correct is not guaranteed.
- **Nothing in our data guarantees either assumption.** The 15 families were built to put correlated residuals *inside* families. A low aggregated residual correlation between families does not prove conditional independence given each label value.

## Where this adaptation departs from the model

- **The population is not i.i.d.** Steps (and tokens) within an answer are sequential and dependent. Each answer contributes exactly `k` marks per channel, which induces negative dependence between positions of the same answer.
- **`π` is not one constant.** It varies by answer: the error fraction of answer `a`. Centring removes the fixed mark mass `k/n`, not the answer-level variation in `π_a`. The pooled estimate mixes answers with different `π_a` and different `n`, and at token level long answers dominate the sample.
- **The marks are ranks, not classifier votes.** `d_i` is the difference in tail-exceedance rates for a within-answer quantile rule. It is not the accuracy of a fixed binary classifier.

## From binary weights to continuous application

No derivation connects the two. Take a simple equal-variance shift model, `X_i | Y ~ N(μ_i Y, σ_i²)` after standardization.

- The optimal linear weight under conditional independence is `∝ μ_i / σ_i²`.
- For a fixed upper-quantile mark, `d_i` is an increasing but **nonlinear** function of the separation `δ_i = μ_i / σ_i`.

So the tail weights `∝ d_i` preserve sign, and ordering in `δ_i`, only under that model. They are not the optimal continuous weights. **Applying them to continuous values is a hypothesis tested empirically, not a derived result.**

Two controls isolate what the learning contributes:

- **Votes:** the same weights applied to the marks themselves. This isolates the value of keeping continuous magnitudes.
- **Learned-partition equal:** the fit's own groups with equal weights. This isolates the value of the learned magnitudes beyond the discovered grouping.

## Why the tail was expected to help

An earlier diagnostic (`results/tail_label_share_v1/`, 21.8% label-driven covariance share in the top-20% co-exceedance vs 17.6% in the full covariance) suggested that the tail co-movement is somewhat more label-driven than bulk co-movement. That is motivation, not evidence of a better weighting.

Weight changes such as the larger `chosen_std_excess` weight are **descriptive**. A mechanism claim would need a controlled component swap with the other components held fixed.
