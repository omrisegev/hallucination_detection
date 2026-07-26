# SPEC — the L-SML residual drops its eigenvalue scale (proposal, not yet applied)

**Status**: PROPOSAL. Nothing changed in the codebase. Raised 2026-07-26 by Omri, from the Step 203
trimming study (`results/pruning_study/`).
**Origin**: Omri's question — *"L-SML clustering is supposed to cluster the dependent features
together, isn't that the assumption? Use this to explain why we're going the opposite direction
algorithmically."* That question is what exposed this.

---

## 1. The claim

`_estimate_von_voff` (`spectral_utils/fusion_utils.py:392`) returns the **unit-norm** eigenvector,
while Paper 1's Lemma 1 defines `v^on` by

> `r_ij = v^on_i · v^on_j`  for `i ≠ j`, `c(i) = c(j)`

i.e. `v^on` must satisfy *outer product = covariance*. The unit eigenvector `v` satisfies
`λ₁ · v_i·v_j ≈ r_ij`, so the code is short by a factor of `λ₁` (equivalently, the loadings are
`a = √λ₁ · v`, not `v`).

`_residual_lsml` (`:427`) then scores the fit as `Σ (v_i·v_j − r_ij)²` using that unscaled vector.

**Quoted source** (`papers/extracted/unsupervised-ensemble-learning-with-dependent-classifiers.md`):
> *"There exists two vectors von, voff ∈ Rm such that for all i ≠ j, rij = voff_i · voff_j if
> c(i) ≠ c(j), von_i · von_j if c(i) = c(j) (10)"*

## 2. Evidence

A **perfectly clustered** group of `m` identical duplicates (`r_ij = 1`) — the ideal case the
clustering step exists to produce — is scored as an increasingly **bad** fit as the group grows:

| group size `m` | `v_on` entry | predicted `r_ij` | actual `r_ij` | misfit / pair |
|---|---|---|---|---|
| 2 | 0.7071 | 0.500 | 1.0 | 0.250 |
| 3 | 0.5774 | 0.333 | 1.0 | 0.444 |
| 5 | 0.4472 | 0.200 | 1.0 | 0.640 |
| 8 | 0.3536 | 0.125 | 1.0 | 0.766 |
| 11 | 0.3015 | 0.091 | 1.0 | 0.826 |

Because `v` is unit-norm its entries are `≈ 1/√m`, so the predicted covariance is `≈ 1/m` while the
truth is `1`. On a 5-duplicate block: unit-norm `v_i·v_j = 0.200` vs `λ₁·v_i·v_j = 0.800` vs true
`r_ij = 1.0`.

**Consequence**: misfit per pair grows with **group size × dependence strength** — it is largest
exactly where the clustering *succeeded*.

Reproduce:
```bash
python -c "
import numpy as np; from spectral_utils.fusion_utils import _estimate_von_voff, _residual_lsml
for m in (2,5,11):
    R=np.ones((m,m)); c=np.zeros(m,int); v,_=_estimate_von_voff(R,c)
    print(m, round(v[0]*v[1],4), round(_residual_lsml(R,c)/(m*(m-1)),4))"
```

## 3. Why this explains Step 203

Step 203 measured two things that looked like a strange property of the data:

| Observation | Number |
|---|---|
| Within-size Spearman(misfit, AUROC) | **+0.223**, positive in **24/25** cells |
| Repair worst-fitting group vs repair a **random** group | **−2.22pp**, W/L 7/18, p = 0.032 |

I explained these as *"redundancy and informativeness travel together, so poor fit marks where the
signal is."* **That is the symptom, not the cause.** The cause is mechanical: the misfit is inflated
by group size and coupling strength, so:

- *"worst-fitting group"* ≈ *"biggest, most tightly-dependent group"* — which is precisely what the
  clustering was built to find and what the fusion exists to exploit;
- *"remove whatever most improves the fit"* ≈ *"dismantle the largest tight cluster"*;
- so **the selection step optimises against the clustering step**. That is the algorithmic answer to
  Omri's question.

## 4. Blast radius — this is NOT confined to selection

`detect_dependent_groups(method='residual')` chooses **K by minimising this residual**
(`fusion_utils.py:519-528`). So the mis-scale sits in the **deployed detector**, not just in the
trimming experiments.

**Predicted symptom**: since misfit falls as groups get smaller (the `1/m` effect), residual
minimisation should be biased toward **large K / many small groups**. Observed on the 25 in-scope
cells (`K_range` caps at 8):

| K chosen | cells |
|---|---|
| 4 | 4 |
| 5 | 1 |
| 6 | 5 |
| 7 | 7 |
| 8 | 8 |

**15/25 cells sit at K ≥ 7, pinned against the ceiling** — consistent with the predicted upward bias,
though not proof of it on its own (a genuinely 8-group structure would look the same).

Dependents to re-check if this changes: `detect_dependent_groups` (K choice), `lsml_fuse`,
`lsml_continuous` (both report `meta['residual']`), `scripts/lsml_theorem_validation.py`,
`a1_residual` (selects by minimising residual), `adaptive_k`'s residual-based rules, and the whole
Step 203 study.

## 5. Proposed change

In `_estimate_von_voff`, scale each group's eigenvector by `√λ₁`, and likewise for `v_off`:

```python
vals, vecs = eigh(sub)
lam = max(float(vals[-1]), 0.0)
v_on[idx] = vecs[:, -1] * np.sqrt(lam)     # Lemma 1: v_i * v_j == r_ij
```

Guard the degenerate cases already handled (`len(idx) == 1`, eigendecomposition failure) and clip
negative leading eigenvalues to 0.

**Keep it behind a flag** (`scale_loadings: bool = False`) for the first pass, so the current numbers
remain reproducible while the fixed variant is measured beside them. Do not silently change the
default — `GOOD_6 = 0.7594` is the project's standing anti-regression anchor and it is computed
through this path.

## 6. Pre-registered checks (state before running)

**Unit level**
- U1 — a perfect `m`-duplicate block scores misfit/pair `≈ 0` for all `m ∈ {2..11}` (currently
  0.25 → 0.83).
- U2 — a block-diagonal synthetic with known groups recovers `r_ij` to within tolerance.

**Behavioural predictions — these are the ones that make the claim falsifiable**
- P1 — chosen K **falls**, and the K ≥ 7 pile-up (15/25) thins.
- P2 — the sign of Spearman(misfit, AUROC) **weakens or flips** from +0.223.
- P3 — the localizer's deficit vs the random-group control (−2.22pp) **shrinks or reverses**.

If P1 holds but P2 and P3 do not, the scaling is a genuine bug **and** the inversion has a second,
independent cause — that would be a more interesting result than the fix.

**Regression / anchors** (any drift here voids everything downstream, SPEC_gap_ladder §8)
- R1 — `GOOD_6` macro stays **0.7594** on the flag-off path, exactly.
- R2 — `ars_gsm8k_r1distill8b` reference: K=4, residual 88.455, group sizes [5,7,7,11] on flag-off.
- R3 — report `GOOD_6` on the flag-**on** path as a *new number*, not as a correction, until P1–P3
  are read.

**Reporting discipline** (Omri, Step 203): report effect sizes with W/L and Wilcoxon; **do not gate
on 1–2pp differences**, and do not adopt anything on an average alone.

## 7. What this would mean for Extension I

Research_Directions Extension I currently proposes **I1 — sign-flip the selectors** (maximise misfit
instead of minimising). If this scaling fix lands and P2/P3 hold, then **I1 is the wrong remedy**: the
criterion never needed inverting, it needed scaling, and the sign-flip would be curing a symptom.

Run this spec **before** I1, and treat I1 as the fallback for the case where the fix does not move
the sign.

## 8. Honest caveats

- The paper's Eq. (14) residual is quoted above from `papers/extracted/`; the **appendix** definition
  of Eq. (14) has not been re-read line by line. It is possible the implementation deliberately uses
  a normalised residual for K-comparability, in which case this is a *documented deviation* rather
  than a bug — but nothing in the code says so, and the docstring cites Eq. (14) directly.
- `v_off` has the same issue and the same fix, but cross-group blocks are not rank-one in the same
  clean way, so its behaviour under the fix is less predictable.
- Scaling changes the residual's units, so any **absolute** residual threshold elsewhere in the
  codebase becomes meaningless and must be re-derived, not rescaled by hand.
- None of P1–P3 is measured yet. This document is a hypothesis with a test plan, not a result.
