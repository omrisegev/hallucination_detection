# Extracted: Unsupervised Feature Selection Through Group Discovery (GroupFS)

**Source**: arXiv:2511.09166v1, HTML at https://arxiv.org/html/2511.09166v1 (fetched 2026-07-17).
**Extraction method**: LaTeX pulled verbatim from the page's MathML `<annotation encoding="application/x-tex">` blocks (636 math annotations), tags stripped. Every equation below traces to that fetched text — no equation is paraphrased. Where the paper prints an explicit equation number the HTML shows only Eq. (1) numbered; the remaining display equations are unnumbered in the source and are cited here by section.

**Title**: Unsupervised Feature Selection Through Group Discovery
**Authors**: Shira Lifshitz, Ofir Lindenbaum, Gal Mishne, Ron Meir, Hadas Benisty
**Venue**: AAAI 2026. **License**: CC BY 4.0.

---

## Abstract (verbatim key sentences)

"We introduce GroupFS ... a fully differentiable, end-to-end framework that simultaneously learns feature groups and selects the informative ones in a purely unsupervised manner. Our approach constructs two graphs: one over the sample space and another over the feature space, enforcing Laplacian smoothness on both. A feature-grouping and gating mechanism, guided by sparse regularization, dynamically discovers relevant feature groups." Across nine benchmarks (images, tabular, biological) GroupFS "consistently outperforms state-of-the-art unsupervised FS in clustering and selects groups of features that align with meaningful patterns."

---

## 3 Preliminaries

### 3.1 Graphs and Spectral Analysis

Data matrix `X = [x_1,...,x_N]^T ∈ R^{N×d}`, row `i` = sample `x_i = X_{i:} ∈ R^d`, column `k` = feature `x^{(k)} = X_{:k} ∈ R^N`.

**Eq. (1) — self-tuning kernel (Zelnik-Manor & Perona 2004):**

```
W_ij = exp( - ||x_i - x_j||_2^2 / (gamma_i * gamma_j) )                         (1)
```

where `gamma_i` is the distance from `x_i` to its `K`-th nearest neighbor (sample-dependent scaling that adapts to local density).

Degree matrix `D = diag(d_1,...,d_N)`, `d_i = sum_j W_ij`. Two operators:

```
normalized graph Laplacian:  L_sym = I - D^{-1/2} W D^{-1/2}
random walk matrix:          P = D^{-1} W
```

`P^t` = transition probabilities after `t` steps of the random walk.

**Laplacian Score (LS)** for feature `x^{(k)} ∈ R^N`:

```
LS(x^{(k)}) = (x^{(k)})^T L_sym x^{(k)} = sum_{i=1}^N lambda_i <v_i, x^{(k)}>^2
```

with `{(lambda_i, v_i)}` the eigenpairs of `L_sym`. Smaller LS = smoother over the sample manifold = more informative. Total: `sum_k (x^{(k)})^T L_sym x^{(k)} = tr(X^T L_sym X)`.

### 3.2 Gumbel-Softmax (Concrete distribution)

Given class probabilities `pi = [pi_1,...,pi_C]`, temperature `T > 0`, i.i.d. Gumbel noise `g_c ~ Gumbel(0,1)`:

```
m_c = exp((log pi_c + g_c)/T) / sum_{h=1}^C exp((log pi_h + g_h)/T)
```

As `T -> 0`, `m ∈ R^C` approaches a one-hot sample. Reparameterization trick makes it differentiable.

### 3.3 Stochastic Gates (STG, Yamada et al. 2020)

Each feature `x^{(k)}`, `k ∈ {1,...,d}`, multiplied by a stochastic gate:

```
z_k = max(0, min(1, mu_k + eps_k)),    eps_k ~ N(0, sigma^2)
```

`mu_k` learnable. Expected number of selected features:

```
E[||z||_0] = sum_{k=1}^d P(z_k > 0) = sum_{k=1}^d Phi(mu_k / sigma)
```

`Phi(.)` = standard Gaussian CDF, `sigma` = fixed gate noise.

---

## 4 GroupFS

**Problem setup.** `X ∈ R^{N×d}`, `N` samples, `d` raw features assumed partitionable into `C` latent groups `{G_1,...,G_C}`, unknown a priori.

**Overall Loss:**

```
L = L_s + lambda_1 * L_f + lambda_2 * L_reg
```

### 4.1 Sample-wise Smoothness Loss L_s

**Feature Association.** Given a batch `X_B ∈ R^{B×d}`, learn a feature-to-group assignment matrix `M ∈ R^{d×C}` via Gumbel-Softmax; row `M_{i,:}` = soft membership of feature `i` across the `C` groups:

```
M_ij = exp((log pi_ij + g_ij)/T) / sum_{k=1}^C exp((log pi_ik + g_ik)/T)
```

`pi_ij` learnable logits, `g_ij ~ Gumbel(0,1)`, `T` temperature annealed during training. As `T -> 0` each row -> one-hot (feature `i` assigned to a single group).

**Group Importance.** Attach one stochastic gate `z_j` per GROUP (reduces gating params from `d` to `C`). Feature-level weights aggregate the gated group assignments:

```
zhat_i = sum_{j=1}^C M_ij * z_j,    i ∈ {1,...,d}, j ∈ {1,...,C}
```

Broadcast `zhat` across the batch -> `Zhat ∈ R^{B×d}` (identical rows). Mask input:

```
Xtilde = X_B (elementwise*) Zhat
```

**Smoothness Objective.** Build a dense random-walk matrix `P_Xtilde` (Sec. 3.1, via the Eq. (1) affinity) over the batch-masked input `Xtilde` AT EACH ITERATION:

```
L_s = - (1/(B*d)) * tr( Xtilde^T * P_Xtilde^t * Xtilde )
```

`P_Xtilde^t` = t-step diffusion operator (t-th power of random-walk matrix). Maximizing the trace aligns retained features with the graph's low-frequency directions. (`L_s` carries a negative sign because it is based on the random-walk matrix `P` and must be maximized — App. C.)

### 4.2 Feature-wise Smoothness Loss L_f

Embed each feature into a `C`-dim space:

```
F = M Q ∈ R^{d×C}
```

`Q ∈ R^{C×C}` trainable linear projection (allows interactions between clusters). Construct a feature similarity graph (nodes = features), compute its normalized Laplacian `L_feat ∈ R^{d×d}` (analogous to the sample graph of Sec. 3.1). The term `tr(F^T L_feat F)` penalizes rapid changes of `F` across similar features. Add an orthogonality penalty `||F^T F - I||_F^2`:

```
L_f = (1/(d*C)) * [ tr(F^T L_feat F) + beta * ||F^T F - I||_F^2 ]
```

`beta` weights the orthogonality penalty. **After each update step, center the columns of `F` to zero mean and renormalize them to unit l2-norm** (avoids trivial constant/zero solutions).

### 4.3 Group Sparsity Loss L_reg

Penalize the expected number of active gates weighted by each group's relative size, using the STG activation probability `P(z_j > 0) = Phi(mu_j / sigma)`:

```
L_reg = (1/C) * sum_{j=1}^C P(z_j > 0) * (1/d) * sum_{i=1}^d M_ij
```

Increases with both the likelihood group `j` is active and the proportion of features assigned to it; minimizing keeps fewer and smaller groups active.

---

## 5.1 Implementation Details

**Learnable parameters:** (i) `d×C` logits of the Gumbel-Softmax assignment matrix `M`; (ii) `C` gate means `{mu_j}` for the STG group importances; (iii) `C×C` transformation matrix `Q`.

**Initialization.** Gates initialized to `mu_j = 0.5` (unbiased prior, Yamada et al. 2020). Warm-start the logits using spectral-clustering assignments based on `L_sym`: for a feature `i` assigned to cluster `j*`,

```
log(pi_ij) = { Delta   if j = j*
             { 0       otherwise ,      Delta = log(p_main / p_rest)
```

with `p_rest = (1 - p_main)/(C - 1)` and `p_main = 0.7` in all experiments. `Q` initialized as a random orthonormal matrix, each row scaled inversely to the feature-cluster sizes estimated from the spectral-clustering logits (balanced influence on `F`). "These initial group assignments are not fixed; they are gradually overwritten during training."

**Selection rule.** "We select features by sorting the groups according to their gate means and retaining those from the top-ranked groups." (Sec. 4.1 Group Importance.) In the synthetic study: "We retain the top-ranked groups by gate mean until at least 10 features are covered."

---

## Appendix B.1 — GroupFS Hyperparameters (verbatim)

- `lambda_1 ∈ {0.1, 1, 10, 100}`, chosen such that `L_s` and `L_f` have comparable magnitudes during the FIRST training epoch.
- `lambda_2` selected via a coarse grid search, from high values (all gates closed) to low values (all gates open); the lowest-loss model is retained (sparse, non-degenerate for most datasets).
- `beta = 1 / lambda_1` (so the orthogonality term has an effective coefficient of 1).
- Number of groups `C`: heuristic (App. D) or small grid search.
- `K = 7` in the self-tuning kernel (Zelnik-Manor & Perona 2004).
- diffusion time `t = 2` (Lindenbaum et al. 2021, differentiable).
- STG noise `sigma = 0.5` (Yamada et al. 2020).
- Optimizer: Adam, learning rate `lr = 1e-3`, PyTorch default settings.
- Final models selected by LOWEST total loss.
- Temperature schedule per Gumbel-Softmax grouping layer:

```
temp(e) = max( min_t , start_t - (start_t - min_t) * e / E )
```

`e` = current epoch, `E` = total epochs. `start_t = 10` (first epoch), `min_t = 1e-2`, all runs.
- Synthetic (two-moons): 500 epochs, batch size 100.
- Real datasets: batch size in `{32,64,128}` (small) / `{64,128,256,512}` (large); epochs per Table 5.

---

## Appendix D — Selecting the Number of Clusters C (verbatim algorithm)

Follows Zelnik-Manor & Perona (2004): after computing the spectral embedding, apply the closed-form Procrustes alignment (Schönemann 1966) to match the (arbitrarily rotated) embedding to a binary cluster-indicator matrix. For each candidate `C ∈ {2,3,...,C_max}` compute distortion `E(C)`, choose the `C` that minimizes it (or a local minimum):

1. Normalized graph Laplacian of the FEATURE graph: `L_feat = I - D^{-1/2} W D^{-1/2}`. Extract its `C` smallest eigenvectors -> `U_C ∈ R^{d×C}`; row-normalize `Utilde_C(i,:) = U_C(i,:) / ||U_C(i,:)||_2` (von Luxburg 2007).
2. Run k-means with `k = C` on the rows of `Utilde_C` -> labels `l ∈ {1,...,C}`.
3. Binary cluster indicator `Y ∈ {0,1}^{d×C}`, `Y_ij = 1{l_i = j}`.
4. Orthogonal Procrustes: `R* = argmin_{R^T R = I} ||Utilde_C R - Y||_F^2`, where `Utilde_C^T Y = U Sigma V^T`, `R* = U V^T`.
5. Distortion `E(C) = ||Utilde_C R* - Y||_F^2` (lower = features cleanly partition into `C` groups).

Effective group count for the synthetic study: `C = 2 + (d - 10)` (one group per informative cluster + one per nuisance feature). For `C <= 2 + (d-10)` the model nearly always achieves `RG_sim = 1`, `TPR = 1`, `FDR = 0`.

---

## Synthetic-data evaluation metrics (Sec. 5.2) — used to design our smoke test

Relevant-Group Similarity (adapted from Imrie et al. 2022 Group Similarity). Ground-truth groups `G = {G_1, G_2}`, predicted `Ghat = {Ghat_1,...,Ghat_C}`; keep predicted groups overlapping an informative group `Gtilde = {Ghat_j : Ghat_j ∩ G_1 != empty or Ghat_j ∩ G_2 != empty}`:

```
RG_sim = (1/max(|G|,|Gtilde|)) * sum_{i=1}^2 max_{Ghat_j ∈ Gtilde} J(G_i, Ghat_j),   J(A,B) = |A∩B|/|A∪B|
```

TPR = fraction of informative features {1:10} selected (want 1). FDR = fraction of selected features from noise set {11:20} (want 0).

**Synthetic construction (Sec. 5.2):** 20-dim extension of two-moons. Features 1-5 are noisy linear transforms of the moons' first coordinate, 6-10 of the second, each `x' = sqrt(rho) x + sqrt(1-rho) eps`, `eps ~ N(0,1)`. Features 11-20 are i.i.d. `N(0,1)` noise. `rho ∈ [0.6, 1]` controls intra-group coherence; default `rho = 0.95`, additive noise std 0.05, 500 epochs, BS 100.

---

## Baselines compared (Sec. 5, Table 1 / App. E.2)

ALL (no selection), LS (Laplacian Score, He et al. 2005), MCFS, CAE (Concrete Autoencoder, Abid et al. 2019), **DUFS** (Differentiable Unsupervised Feature Selection / Gated-Laplacian, Lindenbaum et al. 2021), MGAGR, CompFS (Imrie et al. 2022). GroupFS wins or ties on 7/9 datasets in clustering accuracy. DUFS is the direct predecessor: STG gates on a Laplacian-score objective WITHOUT group discovery.
