# White-box layer-fusion research record

**Branch:** `codex/whitebox-layer-fusion`

**Status:** `POSTHOC / PRELIMINARY / VALIDATION BLOCKED`

**Last updated:** 2026-08-19

This is the durable index for the white-box work reconstructed from the
`layer-lens-v1` captures. It records what was measured, what was fitted, which
comparators are approximations, and which result is currently the final
candidate. Detailed chronological decisions are in `HISTORY.md`, Steps
243–245b; operational handoff is in `PROGRESS.md`.

## Claim boundary

- The original `whitebox/per-layer-views` branch was not pushed and was not
  recoverable from local Git objects. The benchmark was reconstructed from
  raw generation caches and layer sidecars stored under
  `gdrive:hallucination_detection/cluster_results/`.
- The current roster supports a cross-dataset and limited cross-architecture
  result for the captured models. It is not evidence for all LLM families.
- Corrected live Gate B and the two-cell architecture-fidelity GPU pilot are
  still incomplete. Every report must therefore remain visibly
  `PRELIMINARY / VALIDATION BLOCKED`.
- The depth-metric search, NRM variants, final 13-expert contract, and matched
  white/gray comparison are retrospective. Their intervals describe the
  observed cells; they are not independent confirmation.
- Fit APIs receive label-free matrices and anchors. Score hashes are frozen
  before the evaluator opens correctness labels. Hallucination is always the
  positive evaluation class (`1 = incorrect`).

## Captured information

For every generated candidate and every transformer layer, the sidecars store:

- logit-lens entropy (`lens_H`);
- target-token log probability (`lens_logp_tgt`), converted to target NLL;
- top-1 log probability (`lens_logp_top1`), converted to top-1 surprisal;
- KL divergence to the final layer (`lens_kl_final`);
- residual norm, covariance eigenvalues, and a random hidden projection.

The four lens quantities exist at three hook positions: attention output, MLP
output, and residual stream. Token-length and tensor-shape joins are checked
against `raw[problem]["candidates"][candidate]`; repeated candidates are
grouped by problem for resampling and supervised folds. Geometry performance is
not promoted because the missing capture source leaves its pooling semantics
unverified; only rotation-invariant geometry summaries are admissible later.

The original capture did **not** run TriLens, DoLa, HaloScope, or Spilled
Energy. It cached sufficient internal/output traces to construct local proxy or
reproduction arms offline. Consequently, “TriLens” in these reports means a
grouped supervised L2 probe over registered layer-entropy views; it is not a
score emitted by the capture job itself and not a claim of exact upstream-code
reproduction.

## Feature and fusion progression

### 1. Registered layer benchmark (`whitebox_layer_fusion_v1` and `v2`)

The headline `resid-core-L` contract first averages four standardized residual
metrics within each layer: entropy, target NLL, top-1 surprisal, and
KL-to-final. This produces one expert per layer. The richer `lens-96` contract
keeps four metrics at three hook positions over eight spaced layers.

Registered unsupervised fusers include deployed U-PCR, IU-PCR, DUFS-LIU-PCR,
equal mean, PC1, SU-PCR, L-SML, fixed-band clustered U-PCR, four-band
hierarchies, and module-by-metric hierarchies. Controls include final-layer NLL
and token length. Grouped logistic regression and best-single-layer selection
are label-using diagnostic ceilings and are never presented as unsupervised
methods.

The original registered primary result was negative: basic layer fusion did
not beat final-layer NLL. `lens-96` DUFS-LIU reached 0.7253 macro AUROC while
final-layer NLL reached 0.7298; basic all-layer residual fusion was much weaker.

### 2. NRM transfer addenda

`whitebox_layer_fusion_nrm_v1` applies NRM-CS-IU to four depth bands or twelve
module-by-metric contribution families. Its small LOMO gain reverses under
LODO, and the lens arm is negative. It was not adopted.

`whitebox_layer_organic_nrm_v1` implements the organic grouping hypothesis:
one layer is one group, with entropy, target NLL, and top-1 surprisal as three
atomic features inside it. This is structurally meaningful, but group identity
alone does not regularize within-layer weights. Its gains reverse under the
same-model and same-dataset controls, so it was not adopted.

### 3. Token-tail and distributed-depth search

The later search adds information discarded by token means: q90, maximum,
standard deviation, and tail-average summaries of entropy, target NLL, top-1
surprisal, target-vs-top1 gap, entropy excess, and KL-to-final. It compares
single views, depth bands, organic per-layer groups, hierarchical views, and
forced depth coverage.

The final **pure inner-state** contract contains 13 label-free experts:

1. the earlier depth-consensus score;
2. three maximum-token tail summaries;
3. an organic hierarchical layer expert;
4. a hierarchical `lens-96` DUFS expert;
5. all-layer and banded complementary depth summaries; and
6. depth-spread mean KL-to-final.

These 13 experts are fused by deployed U-PCR. It is not the four-expert
simple-average fallback. A separate hybrid replaces one role with ordinary
generation entropy and is explicitly labeled as an output-assisted variant.

## Final white-box result and original-method comparators

Equal-cell macro over 13 eligible cells, sorted by AUROC:

| Method | AUROC | Hallucination AUPRC | Status |
|---|---:|---:|---|
| Hybrid distributed-depth U-PCR | 0.785538 | 0.652755 | retrospective output-assisted candidate |
| Pure distributed-depth U-PCR | 0.784612 | 0.648128 | final pure white-box candidate |
| Best atomic internal view | 0.784186 | 0.648765 | per-cell label-using oracle |
| Pure distributed-depth DUFS-LIU | 0.783964 | 0.648229 | unsupervised ablation |
| Pure distributed-depth IU-PCR | 0.783645 | 0.647896 | unsupervised ablation |
| Equal mean of the 13 experts | 0.779856 | 0.644336 | unsupervised control |
| TriLens grouped L2 probe | 0.768933 | 0.645543 | supervised local approximation |
| DoLa KL grouped L2 probe | 0.767839 | 0.636814 | supervised local approximation |
| Generation entropy | 0.739872 | 0.615392 | gray/output baseline |
| Final-layer target NLL | 0.729778 | 0.589226 | gray/output baseline |
| DoLa KL equal mean | 0.698101 | 0.583602 | unsupervised local proxy |
| Spilled Energy Eq. 8 mean | 0.659605 | 0.542173 | unsupervised local proxy |
| HaloScope direct projection | 0.557405 | 0.426478 | unsupervised local proxy |

Pure U-PCR is +0.015710 AUROC above the local TriLens probe with a positive
paired interval, but only +0.000426 above the strengthened per-cell atomic
oracle and that oracle interval crosses zero. The honest result is therefore a
strong retrospective candidate, not a robust single-layer-oracle victory.

Comparator names describe the implemented local arm. They must not be cited as
exact paper reproductions unless a separate fidelity record establishes that.

## Exact-row comparison with the 30-feature gray-box system

The final pure white score and frozen gray `mixed-v2` system were recomputed on
the exact 31,440-row intersection of 13 shared cells:

| Method | AUROC | Hallucination AUPRC |
|---|---:|---:|
| Gray mixed-v2 DUFS-LIU | 0.782994 | 0.687731 |
| White pure distributed-depth U-PCR | 0.781690 | 0.677048 |
| Gray mixed-v2 deployed U-PCR | 0.780940 | 0.686236 |
| Exploratory equal-z white + gray | 0.790203 | 0.690580 |

White minus final gray is -0.001304 AUROC
[-0.016931,+0.012300] and -0.010683 AUPRC
[-0.035363,+0.011078]. With deployed U-PCR on both sides, white minus gray is
+0.000750 AUROC [-0.012926,+0.013420]. Thus white-box does not currently have
an aggregate score advantage over the 30-feature gray method.

Its practical advantage is coverage: 42,238 scorable candidates versus 31,467
gray complete cases. Mean per-cell Spearman correlation is 0.8677. The
post-hoc equal-z hybrid gains +0.007209 AUROC [+0.000101,+0.014105] over final
gray, but must be frozen and tested on new data before promotion.

## Artifact inventory

| Directory | Purpose | Canonical output |
|---|---|---|
| `results/whitebox_layer_fusion_v1/` | original six-cell frozen benchmark | `REPORT.html` |
| `results/whitebox_layer_fusion_v2/` | 14-cell cross-architecture benchmark | `REPORT.html` |
| `results/whitebox_layer_fusion_nrm_v1/` | four-band/module-family NRM addendum | `REPORT.html` |
| `results/whitebox_layer_organic_nrm_v1/` | one-layer/three-feature NRM test | `REPORT.html` |
| `results/whitebox_depth_metric_search_v1/` | depth-summary search | `REPORT.html` |
| `results/whitebox_depth_token_search_v1/` | token-readout search | `REPORT.html` |
| `results/whitebox_depth_tail_screen_v1/` | intermediate tail-component screen | machine-readable tables/JSON |
| `results/whitebox_depth_tail_consensus_v1/` | tail-consensus candidate | `REPORT.html` |
| `results/whitebox_depth_tail_organic_consensus_v1/` | organic-tail intermediate | frozen tables/manifests |
| `results/whitebox_depth_consensus_v1/` | initial four-component consensus | `REPORT.html` |
| `results/whitebox_depth_distributed_consensus_v1/` | final output-assisted consensus | `REPORT.html` |
| `results/whitebox_depth_distributed_pure_v1/` | final pure white-box candidate | `REPORT.html` |
| `results/whitebox_vs_graybox_matched_v1/` | exact-row white/gray comparison | `REPORT.html`, `REPORT.md` |

Prepared matrices and the 2.3 GB source cache are intentionally ignored by
Git. Compact frozen score bundles, tables, reports, figures, and manifests are
versioned. Source manifests record the Drive provenance and hashes needed to
reconstruct ignored data.

## Reproduction and next decision

Use the scripts named in `HISTORY.md` Steps 243–245b. Do not tune the final
13-expert registry on the same cells again. The next confirmatory experiment is
a new, preregistered capture that:

1. passes corrected live Gate B and architecture fidelity;
2. evaluates pure white, gray mixed-v2, and the frozen equal-z hybrid on the
   same rows;
3. preserves problem-grouped resampling and hallucination-positive AUPRC; and
4. reports both discrimination and row coverage.
