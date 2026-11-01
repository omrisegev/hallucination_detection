# Pre-label stop: the declared approximation fails its own recovery test

Date: 2026-11-01. Protocol: `docs/experiments/MARKOV_DS_COUNT_PREREG_2026-11-01.md`.
No real label, no real data and no `PROTOCOL.json` freeze were involved. Everything here is synthetic (section 5, step 1).

## What was built and verified
- `spectral_utils/markov_ds_count.py`: log J_d by two-panel Gauss-Legendre around the mode. Max |error| against
  scipy quad is 4.5e-13 over d = 2..300 and tau = -80..80; it matches the forward recursion to 7e-15 for tau ≥ 0.
- The analytic gradient of the pooled objective matches central finite differences to 4 decimals in all 6 parameters
  (`syn.py`). The fit converges from all 5 starts to the same optimum.

## What failed
Synthetic data were generated exactly by the model in section 3.1: Markov labels conditioned on at least one error,
K = 3–5 voters, and a random per-answer location and scale. The section 3.4 objective uses one configuration per k
(the top k by the fused score). It does not recover θ. Two corrections of it, tried pre-label, also fail:

| objective (one configuration per k) | true (e, ρ) | fitted (e, ρ) | fitted share vs true | source |
|---|---|---|---|---|
| as written: P(y_k) · lik(y_k) | (.15, .5) | (.045, .244) | .153 vs .274 | `viterbi_vs_multiplicity.out` |
| as written | (.10, .3) | (.000, .011) | .104 vs .189 | same |
| + log C(n,k) multiplicity | (.15, .5) / (.10, .3) | (.335, .408) / (.388, .356) | .369 / .383 | same |
| count prior P(count=k) · lik(y_k) | all three settings | e ≈ .73–.83, ρ ≈ 0 | ≈ .44–.47 ("upper half") | `count_prior.out` |

The count rule itself works when θ is known. At the true θ, the top-k approximation gives a count MAE of 1.12–1.14,
against GAP's 1.41 (`viterbi_vs_multiplicity.out`, first setting). The failure is in estimating θ from the
single-configuration objective. That objective is not a likelihood: it keeps only the configurations the fused ranking
proposes.

## The exact model works (enumeration over all 2^n − 1 configurations, n = 4..9, 800 answers)
`exact.py`, `exact.out`:

| true (e, ρ, δ) | fitted δ | fitted (e, ρ) | MAE exact fit | at true θ | GAP | SHARE (given the TRUE share) |
|---|---|---|---|---|---|---|
| (.15, .5, 1.5/1/.8/.6) | 1.52/1.02/.76/.59 | (.114, .468) | **0.884** | 0.858 | 1.127 | 1.055 |
| (.25, .6, 1.2/1/.7/.5) | 1.26/.98/.67/.46 | (.170, .605) | **1.121** | 1.064 | 1.411 | 1.153 |

So the model in section 3.1 is identifiable and counts better than both GAP and an oracle-share SHARE, when the
likelihood sums over all configurations.

## Feasibility of exact enumeration on the real benchmarks (offsets only, no labels)
- **Socratic**, the judging benchmark: median 9 steps, maximum 16. All answers can be enumerated.
- **Hard2Verify:** maximum 21. Feasible, but heavy for the longest answers.
- **PRMBench:** median 11, maximum 107. 78% of answers have ≤ 16 steps. The long answers need a sampler.

## Status
Stopped before the freeze. Changing section 3.3/3.4 from the single-configuration approximation to the exact sum is a
model change, and it is Omri's decision. No hash was frozen and no real cell was fitted.
