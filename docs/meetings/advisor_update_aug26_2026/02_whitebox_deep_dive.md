# Deep dive 2 — White-box internal features

## Question

Can layer-wise internal measurements supply information that is unavailable in the final output probabilities, and can they be fused without labels?

The main external reference is TriLens, which uses logit-lens entropy from the attention output, MLP output and residual stream at every transformer layer and trains a supervised probe. Our experiment reused the layer-wise measurement idea but asked a different question: can U-PCR/DUFS-style fusion combine those trajectories without a supervised classifier?

## Feature contract

For each layer and internal pathway, the capture includes:

- logit-lens entropy;
- target-token negative log-likelihood;
- top-1 surprisal;
- target-to-top-1 logit/probability gap;
- entropy excess above top-1 uncertainty;
- divergence from the final-layer distribution.

The captured panel covers 14 dataset-model cells and nine model families. The initial representation grouped internal paths compactly; the later distributed-depth representation adds tail summaries, depth-quartile experts and a hierarchical 96-view lens expert.

## Fusion methods

- U-PCR over depth views;
- DUFS-LIU over layer/path measurements;
- hierarchical groups defined by depth quartile or module/metric identity;
- Family-NRM-style residual fusion;
- supervised TriLens-style and atomic probes as evaluation-only ceilings.

All promoted scoring fits are label-free. The supervised rows are comparators or ceilings, not the claimed method.

## Results

The initial registered white-box v2 compact representation was negative: DUFS-LIU reached 0.6181 macro AUROC versus 0.7298 for final-layer NLL. This showed that internal trajectories contain signal, but the initial compact unsupervised objective did not recover it.

The richer distributed-depth pure U-PCR candidate reached 0.7846 macro AUROC; the hybrid that adds one transparent final-output entropy control reached 0.7855. Against the local supervised TriLens approximation, both are positive. These candidates remain preliminary because the feature registry was selected after inspecting the existing cells and corrected live validation is still open.

The exact matched white/gray comparison is the cleanest summary:

- 31,440 identical answers across 13 cells;
- white pure U-PCR: 0.78169 AUROC / 0.67705 AUPRC;
- gray mixed-v2 DUFS-LIU: 0.78299 / 0.68773;
- white minus gray AUROC: -0.00130, interval [-0.01693,+0.01230];
- white coverage: 42,238 scorable answers;
- gray complete-case coverage: 31,467 answers;
- post-hoc equal-z white+gray: 0.79020 AUROC, exploratory only.

## Interpretation

The current evidence supports three bounded claims:

1. Internal-layer features contain useful hallucination signal.
2. They provide materially broader row coverage than the complete gray-box feature contract.
3. On matched answers, white-box alone does not yet improve aggregate discrimination.

The post-hoc combined score suggests complementarity, but it must be frozen and tested on new cells before promotion. The next clean experiment is a preregistered crossed model-by-dataset validation using the final distributed-depth and white+gray rules without retuning.

## Visuals and reports

- [White-box advisor brief](../advisor_update_aug21_2026/03_whitebox_depth.html)
- [Matched white-versus-gray report](../../../results/whitebox_vs_graybox_matched_v1/REPORT.html)
- [Distributed-depth pure report](../../../results/whitebox_depth_distributed_pure_v1/REPORT.html)
- [Distributed-depth hybrid report](../../../results/whitebox_depth_distributed_consensus_v1/REPORT.html)
- [Registered white-box v2 report](../../../results/whitebox_layer_fusion_v2/REPORT.html)
- [White-box NRM report](../../../results/whitebox_layer_fusion_nrm_v1/REPORT.html)
- [Layer-organic NRM report](../../../results/whitebox_layer_organic_nrm_v1/REPORT.html)
- [All white-box plots and intermediate reports](ASSET_INDEX.md#white-box-plots-and-reports)
