Subject: Update: IU-PCR extensions, white-box views, and localization

Hi Ofir, Bracha and Amir,

Following up on our July 30 meeting, I worked through the three directions we identified: extending IU-PCR itself, testing whether internal-model information is worth assuming, and moving from answer-level detection to localization in reasoning and RAG.

### TL;DR

- **IU-PCR:** Clustering and DUFS/graph regularization did not materially improve answer-level detection. NRM did: on the reserved PRMBench confirmation it improved response AUROC from 0.7206 to 0.7252, +0.46pp with paired 95% CI [+0.07,+0.84]. Removing NRM's manual feature families failed, leading to the current PTNI experiment.
- **White-box:** The registered compact layer-fusion method failed against final-layer NLL. Richer internal-state views do contain signal, and a retrospective distributed-depth U-PCR candidate reaches 0.7855 macro AUROC, about +1.66pp over our local TriLens approximation, but it still requires independent validation.
- **Applications:** On ProcessBench, the fixed reasoning pipeline reaches 0.3035 F1 versus 0.2496 for the matched label-free Mind the Gap control, +0.0539 [+0.0316,+0.0773]. On RAGTruth, the fixed pipeline reaches 0.7276 answer, 0.6893 sentence and 0.6586 token AUROC.

### 1. The algorithmic line

Bracha's suggestion was to incorporate clustering similar to L-SML or integrate DUFS more naturally into U-PCR. I tested both. Clustering dependent features lost about 4.5pp, while Laplacian IU-PCR, where a DUFS graph regularizes the solve, was essentially tied with IU-PCR. Smoothness and dependence describe real structure, but do not identify which structure corresponds to correctness.

NRM is the useful exception. Inspired structurally by HARP, it decomposes IU into six measurement-family contributions and adds a small residual covariance mode not already explained by IU. It uses no labels, adds no inference pass and remains one affine score. It transferred positively across development, ProcessBench and SemGrad, and passed the PRMBench confirmation above. HLE was positive but missed its confidence-interval gate.

Atomic-NRM then removed the six manual families and consistently hurt performance, although a supervised atomic control showed headroom. The information exists; covariance alone cannot identify its target direction.

This motivates PTNI. It uses reciprocal prompt-response interventions in which each prompt and fixed response appears equally often as correct and incorrect, plus meaning-preserving renderings that isolate nuisance. The target sign is therefore mechanically known rather than inferred from covariance. Its construction/provenance stage passed, but there is no detector result yet; the next stage tests whether simple prompt, length, template or overlap shortcuts already solve the task.

### 2. The white-box line

To test Amir's suggestion of expanding the available information, I collected per-layer residual-stream and logit-lens views across nine model families and 13 eligible cells. The registered compact fusion was negative: 0.6181 AUROC versus 0.7298 for final-layer target-token NLL. Richer views were much stronger, and supervised probes show clear internal-state headroom around 0.77.

The later 0.7855 distributed-depth candidate is promising, but its view registry was chosen after inspecting these cells. I therefore treat it as a hypothesis for independent validation, not a current result or a change to the one-pass gray-box claim.

### 3. The application line

For reasoning, the pipeline detects whether a trace contains an error and then maps token risk to the first erroneous step. A unified DUFS-LIU version reaches 31.72% ProcessBench macro F1 versus 25.71% for Mind the Gap. On PRMBench, trajectory-first IU reaches 0.6711 step AUROC, below the trained PRM ceiling of 0.7983.

For RAG, full-context, no-context and leave-one-evidence-out scores become repeated evidence views. The fixed pipeline is competitive with our exact local GASP reproduction. Evidence removal itself is not novel; the possible contribution is label-free fusion of dependent evidence contrasts and one-pass deployable distillation.

### Current decision

The clean thesis story currently seems to be: IU-PCR is the label-free one-pass anchor; the failed clustering, graph and atomic variants expose a target-identification bottleneck; NRM is a small confirmed exception using a manual family prior; PTNI tests whether controlled self-supervision can replace that prior; and the practical contribution is transfer to reasoning and RAG localization. The white-box branch remains separate and preliminary.

Rather than attach every experiment, I prepared three documents for discussion:

- a technical note on IU-PCR, HARP, NRM, the Atomic-NRM failure and PTNI;
- the fixed reasoning/RAG application report;
- the white-box report, including both the failed registered primary and the preliminary distributed-depth candidate.

I would like to discuss which line should be the main thesis contribution and which should remain supporting or negative evidence. Would Thursday or Sunday work?

Thanks,
Omri
