---
slug: few-sample-feature-selection-via-feature-manifold-learning
title: "Few-Sample Feature Selection via Feature Manifold Learning"
authors: "David Cohen (Technion), Tal Shnitzer (MIT CSAIL), Yuval Kluger (Yale School of Medicine / Applied Mathematics, Yale), Ronen Talmon (Technion)"
arxiv_id: "not found in extract"
venue: "ICML 2023 (Proceedings of the 40th ICML, Honolulu; PMLR 202)"
year: 2023
source_pdf: papers/Few-Sample Feature Selection via Feature Manifold Learning.pdf
extracted_text: papers/extracted/few-sample-feature-selection-via-feature-manifold-learning.md
last_digested: 2026-08-05
---

## ⚠ TWO CORRECTIONS TO THE STEP-224 RESEARCH BRIEF — READ FIRST

Gemini's `advisors_research.md` presented this paper as **Bracha Laufer-Goldshtein's**, and as
an **unsupervised** method usable as a label-free keep rule. Both are wrong, verbatim from
page 1 of the extract:

1. **Authorship.** The authors are **David Cohen, Tal Shnitzer, Yuval Kluger, Ronen Talmon**.
   Laufer-Goldshtein is not an author.
2. **Supervision.** Abstract, first sentence: *"we present a new method for few-sample
   **supervised** feature selection (FS)."* The method *"first learns the manifold of the
   feature space of **each class**"* and §1 states it *"identifies the meaningful features by
   comparing the underlying geometry of the feature spaces of different classes in a
   **supervised setting**."* Class labels are structural — the whole method is a comparison
   *between* per-class kernels. There is no label-free reduction of it.

This is the third instance of fabricated attribution in Gemini's backfills for this repo
(cf. Steps 176 and 179). Recorded here so it is not re-inherited.

## Summary

A **filter**-type supervised feature-selection method for the few-sample regime. It learns a
kernel-based manifold of the *feature space* separately for each class, then uses **Riemannian
geometry** on the resulting SPD kernels to build a composite kernel that isolates where the
per-class feature geometries differ, and finally scores features by spectral analysis of that
composite kernel. Because the kernels capture **multi-feature associations**, the method is
multivariate by design — its stated advantage over classical filter methods, which *"consider
each feature independently and ignore the underlying feature structure."*

## Method sketch

1. Per class, build a symmetric kernel over the **features** (nodes are features, not samples)
   capturing multi-feature associations.
2. Combine the per-class kernels via Riemannian geometry on the SPD manifold (Pennec et al.
   2006; Bhatia 2009) into a **composite** kernel that extracts the *differences* between the
   learned feature associations.
3. Score features by spectral analysis of the composite kernel; take the top-ranked.

## Connection to our pipeline

**Out of scope as a label-free arm** — it cannot be run in the Step-224 channel, whose whole
premise is that no labels from the target cell are available. Two conceivable uses, both ours
rather than the paper's, and neither currently planned:

- **Pseudo-labelled**: substitute an L-SML/U-PCR consensus pseudo-label for the class variable,
  as `a6_pseudolabel_gates` does for DUFS. But the method needs *two per-class kernels*
  estimated from a split of ~100–300 rows, which is a much heavier ask than a gate objective.
- **Supervised ceiling**: run it with true labels as an upper bound on what feature-manifold
  geometry can reach in this channel. Step 223 already has a cheaper, tighter ceiling — the
  label-handed greedy at +1.88pp — so this would add little.

Its genuinely transferable idea is the **feature-graph** perspective (nodes = features, edges =
associations), which our channel has only ever touched through scalar cohesion statistics. That
idea is available label-free and is what `a9_dpp`'s `det(C_S)` volume criterion and GroupFS's
feature-graph Laplacian already exercise.

## Notes / open questions

- Datasets, baselines and scores were not extracted in this pass — the paper was digested to
  settle its supervision status and authorship, which it did decisively. If it is ever brought
  in scope, re-read §5 before quoting any number.
- Kluger is a co-author here and on `deep-unsupervised-feature-selection-by-discarding-nuisance-a`
  (LS-CAE) and the ℓ0-CCA paper — the Yale side of the advisor network, not the TAU side.
