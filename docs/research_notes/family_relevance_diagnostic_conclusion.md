# Family relevance diagnostic: conclusion and next junction

**Date:** 2026-08-07
**Experiment:** GCFR-U-PCR (Graph-Coupled Family Relevance U-PCR)
**Decision:** stop before a learned mixture

## Short answer

The motivating idea was partly correct:

> Different feature families are useful for different samples or score
> regimes.

The experiment found real evidence for this statement. The best family expert
changes across quartiles of the frozen IU-PCR score. The diagnostic oracle has
an equal-family headroom of **+2.833 AUROC percentage points**, with Holm-
corrected permutation **p=0.006**.

The proposed solution was not correct:

> Within-family agreement, stabilized by a graph of semantically related
> families, identifies which family is reliable for each sample.

The registered graph-coupled method lost **0.135pp** against IU-PCR, won only 8
of 24 cells, and lost **0.243pp** against the same gate with no graph. Every
tested positive graph strength had a negative mean change. The graph did not
collapse; it made active, sample-specific changes, but those changes were not
aligned with correctness.

## What was tested

The method starts from the fixed two-component IU-PCR score. Features are
assigned to the six existing provenance families. For every sample, it gives
each family a local weight. Raw weight evidence is high when the oriented
features inside a family agree on that sample. A six-node Laplacian then makes
related families receive similar weights.

The registered family edges were:

- entropy level -- entropy dynamics -- structural;
- sampled-token energy -- partition energy -- top-k distribution.

The real-data method was frozen at graph strength `beta=3` and replacement
strength `alpha=1` after a 20-seed synthetic study. It was compared with
IU-PCR, deployed U-PCR, DUFS-LIU, no graph, a permuted graph, a global gate,
and sample-permuted local gates. Scores and source hashes were frozen before
the correctness labels were opened.

## Why the synthetic and real results differ

The synthetic study deliberately contained two worlds.

1. Inactive family members had independent noise. Their disagreement exposed
   the inactive family. GCFR improved IU-PCR by **+0.773pp**, with 20 wins in
   20 seeds.
2. Inactive family members shared a coherent nuisance. They agreed with each
   other while being wrong about the target. GCFR lost **9.272pp**, with 20
   losses in 20 seeds.

The real result behaves like the second failure condition, although this does
not prove that the exact synthetic nuisance exists in the real data. Internal
agreement says that a family is coherent. It does not say that the coherent
quantity is hallucination correctness.

## Real-data results

| method | cell-macro AUROC | change vs IU-PCR | wins / losses | worst cell |
|---|---:|---:|---:|---:|
| deployed U-PCR | 0.7735 | -0.053pp | 11 / 13 | -1.842pp |
| IU-PCR | 0.7741 | 0.000pp | baseline | 0.000pp |
| DUFS-LIU | 0.7741 | +0.008pp | 13 / 10 | -0.317pp |
| local family gate, no graph | 0.7751 | +0.108pp | 14 / 10 | -0.692pp |
| registered GCFR | 0.7727 | -0.135pp | 8 / 16 | -1.016pp |

The small no-graph gain is descriptive. Its equal-family interval is
[-0.002,+0.428]pp and it was found inside the registered sensitivity table
after evaluation. It is not a new promoted method and must not be tuned on
these 24 cells.

The parameter sweep diagnoses the graph directly. All `beta=0` paths had a
small positive mean change. Every `beta>0` path had a negative mean change.
This held for all three tested replacement strengths. Therefore the harmful
part is cross-family smoothing, not an unlucky choice of one large graph
strength.

## What the controls rule out

- **Optimization or collapse:** rejected. Gates varied across families and
  samples, and the output ranking changed. The mechanism was active.
- **The exact semantic edges are useful:** unsupported. The correct graph did
  not beat the permuted graph.
- **Only local variation matters:** unsupported for this gate. The primary did
  not beat global or sample-permuted controls.
- **More graph smoothing will fix it:** rejected by the full registered beta
  path.
- **There is no conditional family relevance:** rejected by the IU-PCR-rank
  context test. Conditional specialization exists, but the proposed router
  cannot identify it.

Trace length and raw family disagreement did not pass the corrected
specialization tests. IU-PCR rank did. Across the 23 valid cells, the best
family changed about 2.48 times per cell across the four IU-PCR quartiles.
There was no single universal family-to-quartile rule, so the label-only
winner table cannot be converted into a deployable router.

## Scientific conclusion

The family graph encoded **measurement relationship**, but the needed graph
must encode **shared reliability for the target**. These are different
relations. Two features can come from the same semantic source, or measure
similar behavior, while sharing the same target-irrelevant nuisance. A
Laplacian then spreads the wrong confidence more smoothly.

The current evidence supports this narrower claim:

> Family expertise is conditionally different, especially across IU-PCR score
> regimes, but static feature agreement and semantic family adjacency do not
> identify which expert is correct.

This connects the new experiment to the earlier results. CA-SpecRaGE showed
that sample-specific view agreement did not improve fusion. The atomic-
operator audit showed that stable feature geometry selected reproducible
nuisance. GCFR now shows the same identifiability problem even after adding
the prior knowledge that some feature families are related.

## Next junction

Do not build a more flexible learned family mixture from these labels. It
would have enough capacity to fit the diagnostic oracle without solving the
label-free routing problem.

The next premise should keep the useful discovery and replace the failed
signal:

1. Keep IU-PCR rank as a frozen **regime coordinate**, not as a pseudo-label.
2. Obtain an independent, sample-level self-supervised observation of family
   reliability. Candidate sources are repeated generations, benign prompt or
   decoding perturbations, evidence-conditioned generations, or semantic
   answer consistency.
3. Before fitting a router, test whether that observation predicts which
   family expert helps within held-out cells and feature families.
4. Include coherent, repeatable hallucinations as the explicit failure case.
5. Build a graph or local mixture only if that frozen premise transfers and
   passes absolute tail-safety gates.

This makes the next question precise: not “can a graph assign local weights?”
but “what independent observation tells the graph which locally coherent
family is relevant to correctness?”

## Evidence and audit trail

- Frozen protocol: `docs/experiments/FROZEN_FAMILY_RELEVANCE_DIAGNOSTIC.md`
- Real report: `results/family_relevance_real_v1/REPORT.md`
- Synthetic results: `results/family_relevance_synthetic/`
- Real figures: `results/family_relevance_real_v1/figures/`
- Frozen run fingerprint:
  `3297076c0faa88de042ec586f91dd5288ee16ee2b54579a8fbcc0b8999e4dfa0`

The 24 cells are retrospective development evidence. Any next selected method
needs confirmation on new data or a genuinely untouched dataset/model family.
