# Feature-selection implementations vs. their source papers — fidelity audit

**Asked for by Omri, 2026-07-22** (WS8): verify that our GroupFS and Gated-Laplacian/DUFS
implementations are correct and loyal to the papers, setting aside the extensions we
deliberately added (the pseudo-label gates are Omri's own idea and are out of audit scope).

Sources, both in `papers/`:

| Paper | File | Implements |
|---|---|---|
| **GroupFS** — Lifshitz, Lindenbaum, Mishne, Meir, Benisty, AAAI 2026 (arXiv:2511.09166) | `Unsupervised Feature Selection Through Group Discovery.pdf` (digested; `papers/extracted/…`) | `a2.select`, `a2.select+groups`, `a2.groups@good5` |
| **DUFS** — Lindenbaum, Shaham, Svirsky, Peterfreund, Kluger, NeurIPS 2021 (arXiv:2007.04728) | `Differentiable Unsupervised Feature Selection based on a Gated Laplacian.pdf` (read directly for this audit) | `a2.dufs`, and the gate machinery `a6_pseudolabel_gates` inherits |

Code audited: `spectral_utils/selectors/a2_groupfs.py`, `spectral_utils/selectors/a6_pseudolabel_gates.py`.

**Headline: both implementations are faithful in every equation that defines the method.**
Every objective term, the gate parameterization, the graph operators, and the readout rules
match the papers. All deviations are in *protocol* (compute budget, hyper-parameter selection)
rather than in *method*, and all but two were already documented in the code. The audit found
four things worth acting on — listed at the end.

---

## 1. GroupFS — `a2_groupfs.py`

### Faithful (verified term by term)

| Item | Paper | Code | ✓ |
|---|---|---|---|
| Total objective | `L = L_s + λ₁·L_f + λ₂·L_reg` | line 246 | ✓ |
| Sample smoothness | `L_s = −(1/(B·d))·tr(X̃ᵀ P_X̃ᵗ X̃)` | line 235 (`-(Xtil*(Pt@Xtil)).sum()/(B*d)`) | ✓ |
| Feature graph + orthogonality | `L_f = (1/(dC))[tr(FᵀL_feat F) + β‖FᵀF−I‖²_F]`, `F = MQ` | lines 240-241 | ✓ |
| Group sparsity | `L_reg = (1/C)Σ_j Φ(μ_j/σ)·(1/d)Σ_i M_ij` | line 244 (`(Pz*M.sum(0)).sum()/(C*d)`) | ✓ |
| `β = 1/λ₁` | App. B.1 | line 375 | ✓ |
| Gumbel-Softmax assignment | `M_ij = softmax((log π_ij + g_ij)/T)` | line 231 | ✓ |
| Temperature anneal | `T(e)=max(min_t, start_t−(start_t−min_t)·e/E)`, `start_t=10`, `min_t=1e-2` | line 228, consts 89 | ✓ |
| STG gates, **per group** | `z_j = max(0,min(1,μ_j+ε))`, `ε~N(0,σ²)`, `σ=0.5` | line 232, `mu` shape `(C,)` line 221 | ✓ |
| Feature weight from group gates | `ẑ_i = Σ_j M_ij z_j` | line 233 (`(M @ z)`) | ✓ |
| Gate-open probability | `P(z_j>0) = Φ(μ_j/σ)` | line 243 | ✓ |
| Self-tuning kernel Eq. (1), `K=7` | `W_ij = exp(−‖x_i−x_j‖²/(γ_iγ_j))`, γ = K-th NN | lines 121-130, `K_NN=7` | ✓ |
| Random walk `P=D⁻¹W`, `t=2` | App. B.1 | lines 133-139, `DIFFUSION_T=2` | ✓ |
| `L_feat = I − D^{-1/2}WD^{-1/2}` | App. D | lines 142-145 | ✓ |
| Warm start: spectral clustering, `p_main=0.7`, `Δ=log(p_main/p_rest)`, `μ=0.5`, `Q` orthonormal scaled by inverse cluster size | Sec. 4 init | lines 186-198 | ✓ |
| Sample graph rebuilt per minibatch from gated input | "at each iteration" | line 234 | ✓ |
| `F` columns centered + unit-ℓ2 each step | Sec. 4.2 | lines 238-239 | ✓ |
| `C` by App-D Procrustes distortion | App. D | lines 166-183 | ✓ |
| Adam, `lr=1e-3` for logits + `Q` | App. B.1 | line 223 | ✓ |

### Deviations (documented in the module docstring, all protocol-level)

1. Input already z-scored + sign-oriented upstream (paper does not specify preprocessing).
2. Rows subsampled to ≤1200; minibatch ≤256 (paper: batch 32–512, `O(B²)` graph either way).
3. `λ₂` chosen by cross-seed **selection stability**, not the paper's lowest-total-loss grid
   search — we have no per-cell held-out clustering signal to score a loss against.
4. `C` capped at 8 with a 5%-knee rule (our pools are `p≤30`, the paper's are `d≫100`).
5. Gate `lr` raised to 2e-2 and 120/180 epochs instead of the paper's 500–5000, to fit a CPU budget.
6. **Deviation 8 in the docstring — the substantive one.** Selection does **not** use the
   paper's group-gate readout ("sort groups by gate mean, retain top-ranked"). The joint group
   gates saturate open at every `λ₂` in a ×512 sweep under our budget (the sample-trace term
   rewards opening all gates, and `φ(μ/σ)`'s gradient vanishes past `μ≈2σ`), so a group is
   instead selected iff the **median per-feature DUFS gate** of its members is open. GroupFS's
   group discovery is kept; its selection signal is replaced.

### New findings

- **(G1) The docstring is unfair to itself on `λ₁`.** It lists λ₁-snapping as "deviation 5",
  but App. B.1 specifies exactly that rule — λ₁ ∈ {0.1,1,10,100} chosen so `L_s` and `L_f`
  have comparable magnitude in the *first epoch*. `_snap_lambda1` (line 302) implements it.
  This is **FAITHFUL** and should be moved out of the deviation list.
- **(G2) Interpretive consequence of deviation 8 — this one matters for how we report.**
  Because GroupFS selection runs through DUFS gates, `a2.select` and `a2.dufs` share the *same*
  gate mechanism and differ only in **readout granularity** (group-median vs per-feature). So
  the bench line "GroupFS 0.7481 vs DUFS 0.7502" is *not* paper-GroupFS vs paper-DUFS; it is
  one gate mechanism read out two ways, and the group-granular readout costs 0.21pp. Any
  advisor-facing text comparing the two must say this.
- **(G3) `λ₂` is effectively vestigial in the deployed path.** `_train_groupfs` runs once at
  `λ₂=λ₀` purely to refine the grouping logits (line 380); its gates are discarded. The paper's
  `λ₂` sweep from "all gates closed" to "all gates open" is never performed for group gates.
- **(G4) Minor:** `F` normalization is applied inside the forward pass (gradients flow through
  it) rather than the paper's "after each update step". Numerically close, not identical.

---

## 2. DUFS / Gated Laplacian — `a2.dufs` + the `a6` gate machinery

### Faithful

| Item | Paper | Code | ✓ |
|---|---|---|---|
| Loss Eq. (6) | `L(μ;λ) = −tr[X̃ᵀ L_X̃ X̃]/m + λ·Σ_i P(Z_i≥0)` | `a2_groupfs.py:271-273`, `a6:200-202` | ✓ * |
| `L_rw = D⁻¹K` (the paper's own definition, §2.1) | §2.1 | `_random_walk_power` = `D⁻¹W` | ✓ |
| Negative sign on the score term (it is maximized) | Eq. (6) | `Ls = -(…)` | ✓ |
| `t = 2` Laplacian power | §4.2.1 + App. S3 ("we keep t = 2 in all of our examples") | `DIFFUSION_T=2` | ✓ |
| STG Eq. (1): `Z_i = max(0,min(1,μ_i+ε_i))`, `ε~N(0,σ²)`, σ fixed | §2.3 | `torch.clamp(mu + randn*STG_SIGMA, 0, 1)` | ✓ |
| `μ_i = 0.5` at init ("all gates approximate a fair Bernoulli") | §2.3 | `MU_INIT=0.5` | ✓ |
| Regularizer Eq. (2): `Σ_i(½ − ½erf(−μ_i/(√2σ)))` | §2.3 | `0.5*(1+erf(mu/(σ√2)))` — the same quantity by `erf(−x)=−erf(x)` | ✓ |
| Readout: "remove the stochasticity and retain features such that `Z_i>0`" ⟺ `μ_i>0` | §4.2 | `np.where(mu > 0.0)[0]` | ✓ |

\* **Normalization note (benign):** the paper divides the trace term by `m` and leaves the
regularizer as a plain **sum**; our code divides the trace by `(B·d)` and uses the **mean** of
`P(Z_i≥0)`. Both terms are therefore scaled by the same `1/d`, so `L_code = L_paper/d` at equal
λ — identical minimizer, and Adam is scale-adaptive. Not a behavioural deviation.

### Deviations — **not previously documented anywhere**

- **(D1) Kernel.** DUFS App. S1 Eq. (8)-(9) uses a **global** bandwidth:
  `σ̂_b = max_i(C·‖x_i − x_k‖)` with `k = 2` nearest neighbors (5 for ISOLET/GISETTE) and
  `C = 5` (2 for COIL20/PIX10), giving `K_ij = exp(−‖x_i−x_j‖²/σ̂_b)`. Our code uses the
  **per-point self-tuning** kernel `exp(−d²/(γ_iγ_j))` with `k = 7` — i.e. GroupFS's Eq. (1).
  The paper cites Zelnik-Manor & Perona but then explicitly takes the **max** to globalize it.
  This is a real departure from DUFS. It is defensible (it makes `a2.dufs` graph-identical to
  `a2.select`, which is what makes deviation G2's comparison meaningful at all) but it must be
  stated rather than implied.
- **(D2) Optimization budget.** Paper: SGD, lr 0.3–1, **5,000–26,000 epochs**, full batch on
  most datasets. Ours: Adam, lr 2e-2, **120–180 epochs**, batch 256. Same class of deviation as
  GroupFS's, but the DUFS arm never declared it.
- **(D3) λ selection — ours is STRICTER than the paper's.** DUFS §5 sweeps λ over a range and
  keeps the run with the best **clustering accuracy**, and "labels are utilized to evaluate
  clustering accuracy". That protocol peeks at labels. Our λ is chosen by label-free cross-seed
  selection stability. Worth stating positively in the thesis: our DUFS numbers are obtained
  under a stricter, fully label-free protocol than the paper's own.

### New finding — actionable

- **(D4) DUFS's parameter-free loss, Eq. (7), is not implemented.**
  `L_param-free(μ) = −tr[X̃ᵀL_X̃X̃] / (m·(Σ_i P(Z_i≥0) + δ))` — the paper's own answer to
  "obviate the need to tune λ", and the variant it uses for all two-moons experiments.
  We instead built a stability search that trains **20 models per cell** (4 λ-multipliers ×
  5 seeds) purely to choose λ. Implementing Eq. (7) is a ~2-line change, removes the λ
  hyper-parameter entirely, is *more* faithful to the paper, and would cut the dominant cost of
  the a2/a6 families by roughly 20×. **Recommend adding `a2.dufs_pf` (and an `a6` counterpart)
  as a bench variant.**

---

## Verdict

| Component | Verdict |
|---|---|
| GroupFS objective, gates, graphs, init, `λ₁`, `C` | **FAITHFUL** |
| GroupFS *selection readout* | **DELIBERATE DEVIATION** (documented, empirically forced — group gates saturate under our budget) |
| GroupFS `λ₂` protocol | **DELIBERATE DEVIATION** (label-free stability vs lowest-loss grid) |
| DUFS objective, gates, readout, `t=2` | **FAITHFUL** |
| DUFS kernel | **UNDOCUMENTED DEVIATION** (self-tuning local vs the paper's global max-bandwidth) → document it (D1) |
| DUFS optimizer/epochs | **UNDOCUMENTED DEVIATION** (compute budget) → document it (D2) |
| DUFS λ protocol | **DEVIATION IN OUR FAVOUR** — label-free vs the paper's label-peeking (D3) |
| DUFS Eq. (7) parameter-free variant | **NOT IMPLEMENTED** → recommended addition (D4) |

**No correctness bug was found in either implementation.** The four follow-ups are: fix the
`λ₁` mislabel (G1), state the GroupFS-vs-DUFS confound wherever those two are compared (G2),
document the DUFS kernel + budget deviations (D1, D2), and add the Eq. (7) parameter-free
variant (D4).
