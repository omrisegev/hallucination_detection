---
slug: local-manifold-explanations
title: "Local Manifold Explanations with Tangent Space Regression"
authors: "Jesse He, Yusu Wang, and Gal Mishne; Halıcıoğlu Data Science Institute, University of California San Diego"
arxiv_id: "not found in extract"
venue: "Proceedings of Machine Learning Research 334:1–19; Topology, Algebra, and Geometry in Data Science"
year: 2026
source_pdf: papers/Local_Manifold_Explanations.pdf
extracted_text: papers/extracted/local-manifold-explanations.md
last_digested: 2026-08-19
---

## Summary

LTSREx explains the local intrinsic geometry of an already chosen or estimated
manifold. At each sample it estimates a tangent space, regresses the local
tangent coordinates on a supplied interpretable feature dictionary with
ElasticNet, and smooths the resulting coefficient vector fields with a
parallel-transport-aware connection Laplacian. LEGO is the paper's robust
tangent estimator for normal noise and curvature.

The method is an explanation method, not a target-aware manifold-selection
method: it assumes a sample manifold/embedding and local dimension, and the
authors explicitly leave dictionary choice to domain knowledge. It can explain
which variables drive a chosen geometry, but does not determine whether that
geometry represents hallucination correctness rather than a nuisance.

## Datasets & models used

- Synthetic: 5,000 points on two randomly rotated Swiss rolls in
  \(\mathbb{R}^3\), with uniform noise normal to the manifolds and a four-function
  dictionary whose two correct functions depend on the roll.
- Rotated MNIST: deskewed training images rotated uniformly between 0 and
  \(\pi\), embedded by UMAP in \(\mathbb{R}^3\). The six-function dictionary is
  rotation, mean intensity, stroke width, diameter, and the two largest
  persistent-\(H_1\) values.
- Drosophila clock-neuron scRNA-seq (Ma et al. 2021, GEO GSE157504): 3,000
  highly variable genes after filtering and normalization, PCA to
  \(\mathbb{R}^{50}\), then UMAP to \(\mathbb{R}^{10}\).
- PBMC3k scRNA-seq: preprocessed ScanPy cell counts embedded by UMAP in
  \(\mathbb{R}^4\), with individual gene-expression values as candidate
  explanations.
- No language model, hallucination dataset, online trace, or downstream
  detector is evaluated.

## Methods it compared itself against

- The controlled synthetic comparison crosses four tangent-space choices:
  no tangent projection, local PCA, weighted local PCA, and LEGO; each is tested
  with and without coefficient denoising.
- BIR, LIME-style t-SNE explanations, LXDR, LIMEADE, ManifoldLasso, and TSLasso
  are discussed as related explanation methods. They are not included in a
  numerical real-data benchmark in this paper.
- LTSREx differs by explaining local intrinsic tangent variation rather than
  extrinsic embedding coordinates or one globally shared feature set.

## Experiments — methodology & scores

The Swiss-roll evaluation computes explanations for 500 random points, averages
over five seeds, uses \(k\in\{8,16,32\}\), and counts a point correct only when
both selected functions equal the two ground-truth functions for its roll.
Figure 6 contains curves but the extract contains no tabulated accuracy values.
The reported qualitative result is that LEGO is most robust at high noise for
\(k=8,16\), LPCA can be robust at \(k=32\), and coefficient denoising improves
all tangent estimators.

| Setup | Metric | Score | Notes |
|---|---|---|---|
| Two Swiss rolls | Exact two-feature explanation accuracy | Not numerically tabulated | 500 explanations/run, five seeds; Figure 6 only |
| Rotated MNIST | Feature-selection frequency | No scalar headline score | 1,000 random points; top two features |
| Drosophila / PBMC3k | Qualitative biological interpretation | No quantitative benchmark | Selected genes visualized in local UMAP neighborhoods |

Across experiments the paper sets \(\lambda_1=\lambda_2\) and \(\gamma=1\).
Dataset-specific choices include \(d=2\), UMAP dimensions, neighborhood sizes,
and ElasticNet strengths; no sensitivity or transfer table is reported for the
real datasets.

## Connection to our pipeline

**Useful, narrow role: manifold audit and explanation.** Given one frozen sample
or token geometry, LTSREx could show locally which registered telemetry or
explicit nuisance variables (for example length, generic difficulty, or model
shape) parameterize it. LEGO could improve this audit if tangent estimation is
actually corrupted by normal noise or high curvature. The connection Laplacian
could make explanation maps less brittle across nearby samples.

**Not a direct DUFS-LIU rescue.** DUFS already rewards predictable, smooth
neighborhood structure. LTSREx then explains that structure by reconstructing
its tangent coordinates; neither objective says whether the structure is
aligned with correctness. The project's broad-28 result already showed a
stable, active DUFS geometry that was less aligned with first-error
localization, and the atomic/family-relevance audits identified the same
target-identifiability failure. LTSREx could therefore explain a nuisance
manifold very cleanly without making it useful.

**Not a drop-in IU-PCR replacement.** IU-PCR produces one global affine score
from unlabeled feature moments under an additive covariance model. LTSREx
produces sample-specific local coefficient matrices conditional on an embedding,
dictionary, graph, dimension, and neighborhoods. Using those coefficients as a
router would be a new nonlinear local-fusion method and would still need an
independent target-relevance signal.

**Low direct relevance to the unified Localization/Early method.** Offline
LTSREx could diagnose which causal streams drive a frozen token-trajectory
geometry. The paper does not define a prefix-only graph, future-leakage-safe
neighborhood construction, first-error objective, warning policy, or online
update. Its neighborhood smoothing may also blur abrupt error onsets. A causal
token-manifold adaptation would be a new method outside the paper's evidence.

## Notes / open questions

- The title can be misread as manifold discovery. LTSREx estimates local tangent
  spaces on an assumed manifold representation and explains them; it does not
  choose the semantically correct manifold from competing representations.
- Required choices remain: ambient representation/embedding, intrinsic
  dimension \(d\), neighborhood size \(k\), feature dictionary, ElasticNet
  penalties, denoising strength, and the graph used for smoothing. The paper
  explicitly says estimating \(d\) is nontrivial and dictionary choice requires
  domain knowledge.
- Real-data evidence is qualitative and conditioned on UMAP embeddings. There
  is no downstream AUROC/F1, online/causal protocol, runtime table, or held-out
  model/dataset transfer test.
- The printed denoising objective and closed form do not match as written:
  minimizing \(\|\widehat W-W\|_F^2+\gamma\|L\widehat W\|_F^2\) yields an
  \(I+\gamma L^\top L\) normal equation, while the paper states
  \((I+\gamma L)^{-1}W\). The released implementation must be checked before
  borrowing this component.
- Elementwise \(\ell_1\) sparsity on a coefficient matrix expressed in a local
  tangent basis is generally not invariant to rotations of that basis. This
  weakens the claim that the selected sparse explanations are wholly intrinsic.
- Best future use in this project: a frozen diagnostic/veto test that asks
  whether a proposed geometry is locally driven by registered nuisances. Do not
  treat smooth or stable explanations as evidence of hallucination relevance.
- A stronger but riskier premise test would ask whether frozen LTSREx
  coefficients predict held-out family-expert improvement across cells. It
  should be attempted only with fresh evidence or an independent
  target-relevance observation, not by tuning on the already opened cells.
