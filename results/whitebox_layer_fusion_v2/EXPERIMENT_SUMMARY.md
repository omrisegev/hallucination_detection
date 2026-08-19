# White-box Layer-Fusion Benchmark v2 — Experiment Summary

## Status

**PRELIMINARY / VALIDATION BLOCKED.** The complete offline benchmark has run, but the result is not promoted to `VALIDATED`: corrected live Gate B does not yet cover all 14 cells and the independent two-cell architecture-fidelity pilot has not run. The failure to validate is a capture-evidence boundary, not a missing offline result.

The recovered `whitebox/per-layer-views` branch shows that the original experiment was intended as a broad internal-state telemetry arm. Its capture code is now available locally and explains the saved tensors. The remote branch was at `02bcf09`; the four implementation commits were reconstructed on `codex/whitebox-layer-fusion` as `79fd1b7`, `89e1ffb`, `232b05f`, and `3d75d7c`.

## Bottom line

The registered claim is **not supported**. On the 13 protocol-eligible cells, the headline DUFS-LIU fusion of one residual-core expert per layer is substantially worse than final-layer NLL and slightly worse than IU-PCR on the identical matrix:

| Registered method | Macro AUROC (95% grouped-bootstrap CI) | Macro AUPRC (95% CI) |
|---|---:|---:|
| Final-layer target-token NLL | 0.7298 [0.7189, 0.7396] | 0.5892 [0.5746, 0.6068] |
| Equal mean, residual core | 0.5962 [0.5836, 0.6089] | 0.4563 [0.4451, 0.4732] |
| U-PCR, residual core | 0.6196 [0.6068, 0.6318] | 0.4767 [0.4648, 0.4942] |
| IU-PCR, residual core | 0.6206 [0.6081, 0.6329] | 0.4812 [0.4685, 0.4990] |
| DUFS-LIU-PCR, residual core | 0.6181 [0.6056, 0.6302] | 0.4785 [0.4659, 0.4960] |

Primary paired results:

| Contrast | Metric | Macro delta [95% CI] | W/T/L | Worst cell |
|---|---|---:|---:|---:|
| DUFS-LIU residual core − final NLL | AUROC | **−0.1117 [−0.1245, −0.0996]** | 3/0/10 | −0.2055 |
| DUFS-LIU residual core − final NLL | AUPRC | **−0.1107 [−0.1248, −0.0955]** | 3/0/10 | −0.2891 |
| DUFS-LIU residual core − IU-PCR | AUROC | **−0.00253 [−0.00325, −0.00186]** | 2/4/7 | −0.01563 |
| DUFS-LIU residual core − IU-PCR | AUPRC | **−0.00267 [−0.00381, −0.00158]** | 2/4/7 | −0.01925 |

Holm-adjusted Wilcoxon p-values are supporting evidence only. The primary AUROC values are 0.1538 and 1.0 after adjustment; the directional bootstrap intervals already show that the registered effect is negative.

## What did work

There is useful internal-state signal, but the compact unsupervised residual-core fusion does not extract it robustly.

| Arm | Macro AUROC | Macro AUPRC | Interpretation |
|---|---:|---:|---|
| Generation entropy mean | **0.7399** | **0.6154** | Best descriptive label-free score; output-layer baseline, not a layer fusion |
| Final-layer NLL | 0.7298 | 0.5892 | Strong transparent baseline |
| DUFS-LIU on expanded `lens-96` | 0.7253 | 0.6025 | Nearly matches final NLL in AUROC and exceeds it in AUPRC; secondary arm |
| IU-PCR on `lens-96` | 0.7202 | 0.5960 | Similar evidence that the richer metric/module contract matters |
| DoLa-KL equal mean proxy | 0.6981 | 0.5836 | Useful depth-to-final signal without a learned orientation |
| Spilled Energy Eq. 8 mean proxy | 0.6596 | 0.5422 | Partial paper-equation reconstruction; full-answer rather than exact-span pooling |
| HaloScope direct projection proxy | 0.5574 | 0.4265 | Weak under the saved mean-token JL representation |
| TriLens entropy equal mean | 0.5694 | 0.4329 | Three-position entropy alone is weak without supervised probing |

The expanded lens contract beats the TriLens-only DUFS arm by +0.1601 AUROC [0.1457, 0.1747] and +0.1697 AUPRC [0.1534, 0.1852], with 12/0/1 cell wins/ties/losses. This was a registered secondary comparison, not a post-hoc replacement for the failed primary.

## Supervised diagnostic ceilings

These rows use labels inside five-fold grouped cross-validation and are excluded from every label-free headline and claim. Metrics are averaged per fold; independently calibrated OOF probabilities are never concatenated.

| Diagnostic ceiling | Equal-cell macro AUROC | Equal-cell macro AUPRC |
|---|---:|---:|
| Balanced LR on residual core | 0.7759 | 0.6519 |
| TriLens L2 logistic probe | 0.7689 | 0.6455 |
| DoLa-KL logistic probe | 0.7678 | 0.6368 |
| Evaluation-selected best single layer | 0.7560 | not aggregated |

This gap is the central mechanism result: the captured layer trajectories contain hallucination-relevant information, but the current label-free orientation/dependence objective is not aligned reliably enough across datasets and architectures.

## Architecture replication

The seven-model GSM8K cohort gives the cleanest available architecture comparison, although it is still one dataset:

| Method | GSM8K seven-model macro AUROC | Macro AUPRC |
|---|---:|---:|
| Generation entropy mean | 0.7512 | 0.5307 |
| Final-layer NLL | 0.7358 | 0.5094 |
| DUFS-LIU `lens-96` | 0.7116 | 0.4820 |
| DoLa-KL equal mean | 0.7130 | 0.4815 |
| DUFS-LIU residual core | 0.5656 | 0.3136 |
| DUFS-LIU TriLens entropy | 0.5255 | 0.2754 |

On the original six Llama-3.1-8B cells, `lens-96` DUFS-LIU scored 0.7524/0.7311, reproducing v1. Its drop to 0.7116 AUROC on the seven-model GSM8K cohort is why the broader result must not be presented as a general improvement.

## Decision

- Reject the current `resid-core-L` DUFS-LIU headline as a robust improvement.
- Do not substitute `lens-96` as a new primary after seeing the result.
- Treat the rich module×metric matrix and the supervised probe gap as evidence for the next research question: a label-free objective with explicit view-family structure and cross-cell calibration, rather than more unrestricted depth experts.
- Keep the report `PRELIMINARY / VALIDATION BLOCKED` until the independent validation work is completed.

The complete figures, per-cell scores, paired deltas, weights, gates, graph diagnostics, manifests, and limitations are in [REPORT.html](REPORT.html).
