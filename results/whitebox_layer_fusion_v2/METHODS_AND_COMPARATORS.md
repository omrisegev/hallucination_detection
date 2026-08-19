# Methods and Comparator Fidelity

## Our fusion methods

All label-free methods receive the exact same standardized matrix and final-layer target-NLL orientation anchor for a given feature contract.

- **Deployed U-PCR** reuses the repository's frozen deployed configuration: L2 loss, exclusion enabled, difficulty gate disabled, simple-average fallback enabled, recomputation after exclusion, one g² projection component, and scale ratio 0.25.
- **IU-PCR** uses the full expert pool and two PCs.
- **DUFS-LIU-PCR** uses DUFS seeds 11/23/37, 80 epochs, graph k=7, and Laplacian lambda 0.1. Lambda zero is asserted bit-exact to IU-PCR.
- **SU-PCR** is the registered sparse/dependency reproduction control.
- **Continuous L-SML** uses its canonical no-explicit-group form.
- **Clustered U-PCR** uses four fixed architecture-relative contiguous depth bands and requires identifiability.
- **Hierarchical U-PCR/IU-PCR/DUFS-LIU-PCR** applies the same solver inside fixed groups, standardizes/orients the virtual experts without labels, and applies the solver again across groups. Folded weights are retained.
- **Controls** are final NLL, token length, equal mean, and anchor-oriented PC1.

No method may use `max(AUC,1-AUC)`, label-selected sign flips, post-hoc layer selection, or pooled cross-cell scores.

## Literature comparisons

| Name | Saved-data arm | Fidelity | Labels | What is deliberately not claimed |
|---|---|---|---|---|
| **TriLens** | 3×L attention/MLP/residual entropy matrix; token-mean reduction | Feature-faithful approximation | None for fusion; five-fold grouped L2 LR is a diagnostic ceiling | Exact fixed token readout and the paper's 80/20/5-seed supervised protocol are not reproduced |
| **HaloScope** | Fixed middle-layer `k=4` direct SVD membership on 256-D mean-token JL projection | Stage-1 proxy | None | Not the last-token full-state representation, validation-selected threshold/k/layer, pseudo-labeling, or trained MLP classifier |
| **DoLa-style detector** | Residual KL-to-final depth vector | KL proxy | None for fusion; grouped LR ceiling separate | Saved KL is not the comparator's JSD; original DoLa is decoding, not a detector |
| **Spilled Energy** | Eq. 8 reconstructed from raw top-K sampled-token logprob and adjacent `logsumexp`; mean/min full-answer pooling | Equation-faithful token proxy | None | Exact-answer span localization and tokens outside saved top-K are unavailable |
| **INSIDE** | K=10 middle-layer last-token logdet EigenScore, alpha=0.001 | Paper equation without feature clipping | None | Only the rejected CoQA/Llama-1 cell exists; no primary comparison or macro |

## Supervised probes

Balanced logistic regression is a label-using diagnostic ceiling. It uses five-fold `StratifiedGroupKFold`, seed 20260812, `class_weight="balanced"`, and zero problem overlap. AUROC/AUPRC are calculated in each fold and averaged. OOF probabilities from independently calibrated folds are never concatenated.

The TriLens paper also reports a nonlinear MLP. This benchmark does not add it to the primary report because it would be a larger supervised model-selection exercise and cannot answer the question of whether our zero-label fusion works.

## Evaluation protocol

- Candidate-level AUROC is primary; AUPRC and prevalence are secondary.
- Thirteen protocol-eligible cells form the headline equal-cell macro. CoQA/Llama-1 is appendix-only.
- The original six-Llama macro and seven-model GSM8K macro are frozen continuity/mechanism cohorts.
- 2,000 deterministic paired bootstrap draws use base seed 20260812 and resample problem IDs within each cell. Identical draws are reused across methods.
- Paired rows report 95% intervals, per-cell deltas, wins/ties/losses at ±0.001, worst-cell loss, Wilcoxon p, and Holm adjustment.
- A robust-improvement claim requires both registered primary AUROC intervals to be positive and all capture validation gates to pass. Neither condition is met.

## Interpretation boundary

The strong supervised ceilings do not rescue the label-free claim. They show that information exists in the matrices; they do not show that U-PCR/IU-PCR/DUFS-LIU extracts it without labels. Likewise, the high `lens-96` secondary result cannot replace the failed primary after evaluation. It is evidence for a follow-up contract, not a promoted finding.
