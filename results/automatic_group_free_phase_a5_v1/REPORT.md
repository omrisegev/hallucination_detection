# Automatic group-free IU — Phase A5 S1a

## Outcome

- Formal frozen verdict: **CLOSE_NUMERICAL_NONCONVERGENCE**
- S1a nuisance gate: **FAIL**
- S1b and real-data S2 authorized: **no**
- Real source cache accessed: **no**
- Retrospective correctness labels accessed: **no**

The sealed S1a execution ran the exact 100 registered world-8 seeds
`521600..521699` against the previously committed source and runtime boundary.
Ninety-eight repetitions were usable. Seed `521639` closed because no penalty
arm was usable; seed `521691` closed during a held-mixture fit. Both are
registered numerical failures, and there were no implementation-invalid
failures. Under the frozen rule, any unusable repetition closes S1a, so the
formal verdict is `CLOSE_NUMERICAL_NONCONVERGENCE`.

## Adversarial usable-only analysis

The 98 usable repetitions independently reject the scientific premise; this
diagnostic does not replace or repair the formal verdict.

| nuisance-world gate | registered requirement | observed usable result |
|---|---:|---:|
| final direction prefers target to nuisance | at least 90/100 | 62/98 |
| correction prefers target to nuisance | at least 90/100 | 25/98 |
| candidate minus IU AUROC | bootstrap lower bound >= 0 | mean -0.038484; 95% CI [-0.047495, -0.029659] |

Even counting both unusable repetitions as successes gives only 64 final and
27 correction preferences, so the count gates are mathematically unreachable.
Across the usable runs there were eight positive, 54 negative, and 36 exactly
zero candidate-minus-IU changes; none of the 20,000 registered-seed bootstrap
draws had a nonnegative mean.

The failure mechanism is also legible. The label-free likelihood selector chose
`alpha=1` in 46/98 usable repetitions. Those full-correction runs lost
0.080974 AUROC on average and preferred the target for only 11/46 final
directions and 2/46 correction directions. In this planted world the shared
nuisance is stronger than the target, so likelihood frequently escalates the
IU-orthogonal correction precisely when that correction is nuisance-aligned.
This is the semantic non-identifiability the early anti-repackaging gate was
designed to expose.

## Decision

A5 closes at S1a. Do not open the remaining synthetic S1b schedule, download
the 23 real raw caches for A5, or inspect retrospective labels. A numerical
repair cannot rescue the route because the usable evidence independently and
decisively fails all three nuisance gates. The research ladder proceeds to A6,
where target-changing and nuisance-preserving interventions provide the new
self-supervised information absent from `P(X)` alone.

## Provenance

- Frozen source commit: `f3bc9744584aef4d557c2ded4f6aaa8dfa7b73dd`
- Boundary commit: `0ad7f983044ec4f8be762741cd7367420989578a`
- Boundary SHA-256: `45bd4589b325844fe98d836c5b2760dd30be49bf1520e53b0c4aea648ea37dec`
- Repetition aggregate SHA-256: `c47e2563f44bee8ae7e5ec6df332ad95661697afc999d4cf16996f7f4c1cfb29`
- Completion SHA-256: `169e45b92fba1401b38a8ee086708edc5bb537c19454c2a3b9955996e5c93049`
- Pre-run boundary report SHA-256: `7f4d00836f1241ee3fad7aba48478f58b68e354c011b0842586c39f157b80175`

An independent no-edit audit reproduced `verify-nuisance`, matched all 100
ordered checkpoint records to the aggregate, verified every checkpoint's
boundary hash, and recomputed the usable-only statistics above.
