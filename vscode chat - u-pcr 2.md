
User: Before you edit the plan, read the U-PCR paper again.
read my implementation again.
1. find differences in my implementation.
2. list the theorems, assumpstion and methods they are discussing in the paper.


GitHub Copilot: Updated todo list

Good call — I'll re-read both from source before touching the plan, so the diff is grounded in the actual current text, not memory. Fetching the paper and your U-PCR implementation in parallel.  





Fetched https://ar5iv.labs.arxiv.org/abs/1703.02965

Fetched https://raw.githubusercontent.com/omrisegev/hallucination_detection/master/spectral_utils/fusion_utils.py

Updated todo list

User: Continue

GitHub Copilot: Updated todo list

I've now re-read the full paper and your complete U-PCR implementation. Let me persist the diff findings, then give you the two lists.

Updated memory file [](file:///auto/cag-psv/osegev/.vscode-server/data/User/workspaceStorage/8e5a25eae0ab6bdc3137c0a3d1ca04cf-1/GitHub.copilot-chat/memory-tool/memories/N2I5NDBiMWYtMDM0ZS00MDVjLTg3NjEtYjM5MzkxOGI2NDc1/plan.md)

Both re-read from source. Here are the two lists you asked for.

---

## Part 1 — What the paper contains (Dror, Nadler, Bilal, Kluger 2017, arXiv:1703.02965)

### Assumptions
- **A1 — Known first two moments of the response:** $\theta_1=\mathbb{E}[Y]$ and $\mathrm{Var}(Y)$ are given. The paper argues this is *necessary* (else the data can be arbitrarily shifted/scaled without changing expert correlations).
- **A2 — Linear ensemble class:** $\hat y_{\mathbf w}(x)=\theta_1+\sum_i w_i\,(f_i(x)-\mu_i)$, Eq. (2), with $\mu_i=\mathbb{E}[f_i]$; risk is MSE.
- **A3 — Independent errors w.r.t. the optimal predictor:** with $f_i=g+h_i$, $g(x)=\mathbb{E}[Y\mid X]$, the assumption is $\mathbb{E}[h_i h_j]=0$, Eq. (13). This is the regression analogue of Dawid–Skene.
- **A4 — $m\ge 3$ experts** (needed for the linear system to identify $\mathbf a$).
- **A5 — Near-optimal experts** for the perturbation analysis: $f_i=g+\epsilon h_i$, Eq. (17), studied as $\epsilon\to0$.
- Predictors are **mean-centered** $Z_{ij}=f_i(x_j)-\hat b_i-\theta_1$ — centered only, *original scale retained*.

### Theorems / Lemmas
- **Lemma 1, Eq. (6):** optimal weights satisfy $\boldsymbol\rho=C\mathbf w^\*$, i.e. $\mathbf w^\*=C^{-1}\boldsymbol\rho$, where $\rho_i=\mathrm{Cov}(f_i,Y)$. $C$ is observable; $\boldsymbol\rho$ is the unlabeled unknown.
- **Theorem 1:** under A3, A4 and *known* $g^2$, $\boldsymbol\rho$ is consistently estimable at rate $O_P(1/\sqrt n)$, because off-diagonals obey $C_{ij}=g^2+a_i+a_j$, Eq. (14) — $\binom{m}{2}$ linear equations for $m$ unknowns $a_i$, with $\rho_i=g^2+a_i$, Eq. (11).
- **Lemma 2, Eqs. (18)–(19):** eigen-perturbation of $C(\epsilon)=g^2\mathbf{1}\mathbf{1}^\top+\epsilon(\mathbf a\mathbf 1^\top+\mathbf 1\mathbf a^\top)+\epsilon^2 D$. Consequences: at $\epsilon=0$, $C$ is **rank-1** ($g^2\mathbf{1}\mathbf{1}^\top$) → nearly-identical experts make $C$ ill-conditioned, so $C^{-1}\boldsymbol\rho$ is unstable; $\boldsymbol\rho\approx\mathbf v_1$ up to a shift; $C$ is **rank-2** spanned by $\mathbf 1,\mathbf a$ (this is the theoretical basis for using 1–2 eigenvectors, à la PCR*).

### Methods / algorithm
- **Estimate $\mathbf a$ (hence $\boldsymbol\rho$)** by the additive linear solve, Eq. (15): $\hat{\mathbf a}=\arg\min\sum_{i<j}L(\hat C_{ij}-g^2-a_i-a_j)$, with **both absolute and squared loss** (Remark 1, absolute = robust to A3 violations).
- **$g^2$ family:** $\hat{\boldsymbol\rho}(q)=\hat{\boldsymbol\rho}(0)+\tfrac q2\mathbf 1$, Eq. (16).
- **Select $g^2$** by the projection residual, Eq. (20): $\hat g^2=\arg\min_{q\in[0,\mathrm{Var}(Y)]}\|\hat{\boldsymbol\rho}(q)-(\mathbf v_1^\top\hat{\boldsymbol\rho}(q))\mathbf v_1\|/\|\hat{\boldsymbol\rho}(q)\|$ (alternative $\hat g^2=\lambda_1/m$, Eq. 18).
- **U-PCR weights, Eq. (21):** $\mathbf w=\tfrac1{\lambda_1}(\mathbf v_1^\top\hat{\boldsymbol\rho})\mathbf v_1$, with a **second-eigenvector term** added when $\lambda_2>0.1\,\mathrm{Tr}(\hat C)$.
- **Practical rules (Sec. 4.3):** (i) **difficulty gate** — if $\hat g^2/\mathrm{Var}(Y)\le\epsilon_L\approx0.1$, *abstain*; (ii) **exclusion** of weak experts ($\hat\rho_i<\rho_{\max}/3$ or $\hat\rho_i<0.05\,\mathrm{Var}(Y)$), then recompute; (iii) if $\le4$ experts remain after exclusion, use the **simple average** instead of Eq. (21).
- Oracle baseline $\mathbf w_{\text{or}}=(ZZ^\top)^{-1}Z(\mathbf y-\theta_1)$, Eq. (22).

---

## Part 2 — Differences in your implementation

### Material deviations (change the method's semantics)
1. **You z-score; the paper does not (headline).** `upcr_pipeline` builds $F=\text{zscore}(f_i)\cdot\text{sign}$ → unit variance, so `upcr_fuse`'s $C=FF^\top/n$ is a **correlation** matrix. The paper mean-centers *only* and keeps original scale — deliberately, because $\mathrm{Var}(f_i)\approx\mathrm{Var}(Y)$ carries accuracy information and knowing $\mathrm{Var}(Y)$ is the whole reason scale must be preserved. Consequence: your $\hat{\boldsymbol\rho}$ and $\hat g^2$ are on the correlation scale, so `var_y` (grid bound) and the exclusion thresholds lose their calibrated meaning. It's a *defensible adaptation* (your features aren't on the label scale, and units are heterogeneous), but it's a real departure and it's why several downstream quantities aren't paper-calibrated.
2. **`var_y` hardcoded 0.25** (= $p(1-p)$ at $p{=}0.5$) instead of the actual $\mathrm{Var}(Y)$. Wrong for imbalanced cells, and it caps $\hat g^2\le0.25$.
3. **Squared loss only.** The $a$-solve uses `np.linalg.lstsq` (L2). The paper's Remark 1 also uses **absolute loss** for robustness to A3 violations — exactly the regime you're in (redundant/dependent features). The robust variant is missing.
4. **No difficulty gate.** `upcr_fuse` always returns weights; the paper *abstains* when $\hat g^2/\mathrm{Var}(Y)\le0.1$. You compute the residual/`g2_hat` but never gate on it.
5. **No "$\le4$ experts → simple average" fallback.** After exclusion you fall back to *top-3 by $\hat\rho$* but still apply the Eq. (21) eigenvector weights; the paper switches to a plain average in that regime.
6. **Partial recompute after exclusion.** You refit $\hat{\boldsymbol\rho}$ on the kept experts but **reuse `g2_hat` from the full-$C$ grid** (no re-gridding on $C_k$). The paper says "recompute the various quantities after removal."
7. **$g^2$ residual uses a $k$-dim projection, not strictly $\mathbf v_1$.** When `auto_components=2`, the Eq. (20) residual is computed against a 2-D eigen-subspace — your own `upcr_proj_residual` docstring warns this *absorbs* dependence rather than measuring it.

### Minor
- $C=FF^\top/n$ divides by $n$ (not $n-1$); no $+\theta_1$ offset on the output (irrelevant for AUROC ranking); header attributes U-PCR to "Tenzer et al., AISTATS 2022," but arXiv:1703.02965 is Dror–Nadler–Bilal–Kluger 2017.

### Faithful — *not* differences (worth knowing)
- Additive system $C_{ij}=\rho_i+\rho_j-g^2$ ✓; the shift $\hat{\boldsymbol\rho}(q)=\hat{\boldsymbol\rho}(0)+\tfrac q2\mathbf 1$ ✓ (row-sums-2 identity); Eq. (21) weight, generalized to $k$ components ✓; exclusion thresholds $\rho_{\max}/3$ and $0.05\,\mathrm{var\_y}$ ✓; auto-2-component rule $\lambda_2>0.1\,\mathrm{Tr}(C)$ ✓; 300-point grid over $[0,\mathrm{var\_y}]$ ✓ (just a discretization of Eq. 20).

### One constraint note
The **exclusion step is faithful to the paper** but *is* feature-subset selection — which conflicts with your "no subsets" rule. That's the `min_frac=0` / `exclude_frac=inf` toggle R1 already calls for; it's not a bug, just a paper/your-constraint tension.

---

The most consequential items for your ladder: **#1 (z-scoring vs. scale-preserving), #2 (var_y), #3 (absolute loss), and #4 (difficulty gate)** — the first two change what `var_y` and $\hat g^2$ mean, and #3–#4 are paper-prescribed robustness/abstention you don't yet have. Want me to fold these into the R1/R3 rungs of the plan (e.g., R1 note the z-score↔var_y coupling; add an "absolute-loss + difficulty-gate" sub-rung) and then make the edit?

Created 2 todos