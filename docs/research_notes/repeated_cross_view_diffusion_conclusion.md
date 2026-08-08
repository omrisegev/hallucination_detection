# Repeated cross-view diffusion: conclusion

**Date:** 2026-08-07
**Method:** RCV-AD-IU-PCR
**Decision:** stop the static common-manifold route

## Short answer

Repeated alternating diffusion successfully found reproducible sample
geometry across complementary feature subsets. It did not improve the
hallucination ranking.

The registered dependency-blocked primary scored **0.7741 AUROC**, only
**+0.004pp** above IU-PCR. It improved 10 cells and lost 14. Its equal-family
interval was **[-0.052,+0.029]pp**. It did not beat atomic-random splitting,
family-blocked splitting, or DUFS-LIU. Only 5 of 11 continuation gates passed.

The important result is the separation between convergence and usefulness:

- median partition-to-consensus graph CKA was **0.536**;
- median partition-score-to-consensus Spearman was effectively **1.000**;
- median `T=8` versus `T=16` score Spearman was **1.000**;
- graph CKA had Spearman **-0.240**, `p=0.259`, with AUROC change.

The construction converged, but convergence did not identify correctness-
relevant geometry.

## What was tested

Every partition divided the full fixed-stable feature pool into two disjoint
views. Each view produced a sample k-nearest-neighbour Markov operator. The
registered graph used the symmetric alternating product

\[
S_{A,B}=\frac12(P_AP_B+P_BP_A).
\]

Sixteen frozen partition graphs were averaged and reduced to a sparse
consensus graph. Its Laplacian entered the same final two-dimensional IU-PCR
solve used by DUFS-LIU.

Three partition rules were compared:

1. atomic random partitions;
2. dependency-blocked partitions, where complete-linkage groups above
   absolute Spearman 0.85 could not be split;
3. family-blocked partitions using the six provenance families.

The dependency-blocked method was the registered primary because it prevents
near-duplicate features from creating false cross-view agreement.

## Main results

| method | cell-macro AUROC | change vs IU-PCR | wins / losses | worst cell |
|---|---:|---:|---:|---:|
| IU-PCR | 0.7741 | 0.000pp | baseline | 0.000pp |
| DUFS-LIU | 0.7741 | +0.008pp | 13 / 10 | -0.317pp |
| atomic-random AD | 0.7742 | +0.018pp | 11 / 11 | -0.071pp |
| family-blocked AD | 0.7743 | +0.019pp | 11 / 11 | -0.267pp |
| dependency AD, T=4 | 0.7740 | -0.008pp | 10 / 14 | -0.179pp |
| dependency AD, T=8 | 0.7740 | -0.010pp | 10 / 14 | -0.178pp |
| dependency direct average | 0.7737 | -0.033pp | 8 / 15 | -0.222pp |
| dependency AD, T=16 | 0.7741 | +0.004pp | 10 / 14 | -0.133pp |

The small atomic and family means are not positive findings. Their equal-
family intervals both cross zero, they split wins and losses evenly, and no
registered mechanism gate promotes them.

## What the controls show

### Repeating more partitions is not the missing step

`T=4`, `T=8`, and `T=16` are all effectively tied with IU-PCR. The `T=8`
and `T=16` output rankings are almost identical. More resampling estimates the
same correction more accurately; it does not create a useful correction.

### Dependency blocking did not reveal a cleaner common manifold

At the primary lambda, atomic-random and family-blocked paths were about
+0.018--0.019pp, while dependency blocking was +0.004pp. The difference is
too small and uncertain to claim duplicate leakage, but the registered
anti-leakage rule clearly did not improve the result. Random splitting is not
promoted merely because its mean is slightly larger.

### Alternating products are better than direct graph averaging, but not useful

Dependency alternating diffusion beat the direct-average control by 0.037pp.
This supports the narrow implementation claim that operator composition is
not identical to arithmetic graph averaging. It does not establish an AUROC
improvement over IU-PCR.

### Connectivity is not the main failure

At `k=7`, six of 24 primary consensus graphs were disconnected, especially in
large QA cells. Increasing to `k=11` raised the connected-cell count from 18
to 20 and reduced the worst component count from 245 to 7. Its AUROC change
was still only +0.005pp. Better connectivity did not rescue the method.

### Stronger graph influence is harmful

The complete frozen lambda path contains no hidden dependency-blocked gain:

| lambda | atomic random | dependency blocked | family blocked |
|---:|---:|---:|---:|
| 0.03 | +0.005pp | -0.003pp | +0.007pp |
| 0.10 | +0.018pp | +0.004pp | +0.019pp |
| 0.30 | +0.028pp | -0.005pp | +0.010pp |
| 1.00 | +0.013pp | -0.061pp | -0.045pp |
| 3.00 | -0.005pp | -0.127pp | -0.141pp |

At the safe lambda, partition-specific output rankings are almost identical
because the two-dimensional IU-PCR solution changes little. When lambda is
large enough to force a larger correction, performance becomes worse.

## Scientific conclusion

The experiment rejects this proposed bridge:

> A correctness-relevant latent manifold can be isolated by retaining sample
> relations that recur across random or blocked partitions of the existing
> static feature pool.

It does not reject alternating diffusion as a general method. The needed
multi-view assumption is not satisfied by splitting one already correlated
static feature matrix. Both halves can share confidence, answer length, model
style, or another stable nuisance. Repeated partitions only make that shared
geometry more reproducible.

This result agrees with the previous sequence:

- stable learned micro-views did not improve IU-PCR;
- the atomic-operator proxy converged to the wrong feature operators;
- semantic family smoothing did not identify local reliability;
- repeated cross-view diffusion now shows that reproducibility across feature
  subsets also does not supply the missing target information.

The shared problem is no longer graph estimation. It is **target
identifiability from one static feature matrix**.

## Next research decision

Do not tune the dependency threshold, partition count, k, or lambda on these
labels. Do not promote the random or family split means. Close static
repartitioning of the current feature pool as the next leading direction.

The earlier positive diagnosis remains: family expertise changes across
IU-PCR score regimes. To use it without correctness labels, the next view must
contain information not deterministically derived from the same feature
matrix. Examples include a genuinely independent generation, evidence view,
or controlled perturbation. If the project must preserve single-pass
inference, a multi-pass method should first demonstrate the mechanism and only
then be distilled into a single-pass router on new data.

## Audit trail

- Frozen protocol: `docs/experiments/FROZEN_REPEATED_CROSS_VIEW_DIFFUSION.md`
- Frozen report: `results/repeated_cross_view_diffusion_v1/REPORT.md`
- Figures: `results/repeated_cross_view_diffusion_v1/figures/`
- Source: `spectral_utils/repeated_cross_view_diffusion.py`
- Fit/report: `scripts/repeated_cross_view_{fit,report}.py`
- Known-answer test: `scripts/test_repeated_cross_view_diffusion.py`
- Run fingerprint:
  `75688ad173434232a516d06a9646bd9cd6238ed55581e1808ce47abf0f495923`

The original fit bundle contained 72 arrays and no key containing `label` or
`target`. Every score and registered source hash was verified before labels
were opened. The 24 cells remain retrospective development evidence.
