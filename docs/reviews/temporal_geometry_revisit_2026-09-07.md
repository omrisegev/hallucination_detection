# Revisit IMM, LOCA, Diverging Flows and KalmanNet

Omri explicitly requested all four on 2026-09-07. They remain in the active
research program alongside HMM/BOCPD and graph-based token/window sampling.
This is an implementation/history audit and initial source verification, not
a completed paper digest or a new experiment result.

## What the inspected history actually tested

Sources: `HISTORY.md` Step 151 and the subset-sweep follow-up;
`docs/research_notes/thesis_pivot_assessment.md`; `scripts/pivot_trackB.py`;
`spectral_utils/temporal_models.py`; `scripts/pivot_trackA.py`;
`spectral_utils/anomaly_utils.py`; and `spectral_utils/selectors/a4_antigravity.py`.
This is a scoped audit of these files and current Python sources, not proof
that every branch or external computer has been searched.

| Named direction | What was implemented in the inspected study | What the result can establish | New role worth testing |
|---|---|---|---|
| IMM | Two-state Gaussian-emission HMM, with separate AR/ordinary-Kalman innovation scorers | Those specific regime/innovation features did not beat the final-answer controls. This was not an implemented interacting bank of continuous-state filters. | A switching-filter trajectory readout: separate steady and changing dynamics, track innovation and mode probabilities. Test after freezing feature scores, with ordinary Kalman and HMM controls. |
| LOCA | AE/PRAE-style reconstruction and density precursors; an eigen-ratio feature-count selector was also called LOCA-related | Plain reconstruction/density and intrinsic-dimension selection are not the LOCA training objective. Their failures do not constitute a direct LOCA result. | Local covariance normalization/embedding before grouping or sample selection, if a defensible measurement-burst approximation can be demonstrated within a single trace. |
| Diverging Flows | GMM/KDE and related anomaly scores in feature space | These test simpler density-anomaly hypotheses. They do not reproduce a flow-matching model with a transport-inefficiency mechanism. | Conditional prediction/transport in gray-box telemetry space, fitted without choosing known-correct training traces. Test whether its score adds localization information beyond ordinary prediction error. |
| KalmanNet | Fixed AR/Kalman innovation scores, used as a go/no-go precursor | The chosen fixed filters were highly correlated with entropy level. This does not prove that every learned-gain filter is incapable of improvement. | Unsupervised gain learning from observation prediction, with blocked chronological validation inside the same answer, compared with fixed Kalman and the unchanged IU/Joint readout. |

The recorded ordinary-Kalman result was approximately 0.703 final-answer
AUROC on the 200-answer GSM8K/Llama-8B trace cell, below DeepConf 0.735 and
L-SML-5 0.754. The reported innovation/entropy correlations were 0.93–0.97.
Those are historical final-answer results under that experiment's contract;
they are not new single-answer localization numbers. The broad 29-cell GMM
result was 0.553 versus L-SML continuous 0.651. Preserve those negatives.

The source comment calling a Gaussian HMM the "honest version of IMM" is too
broad. IMM approximates inference in switching continuous-state models through
interaction/mixing of filter hypotheses. A Gaussian-emission HMM does not in
general implement those continuous-state dynamics. Learning model parameters
does not automatically make the two algorithms equivalent. The prior
Gaussian HMM is a useful simpler control, not an IMM reproduction.

## Verified sources and corrections

- **Diverging Flows:** [arXiv:2602.13061v1](https://arxiv.org/abs/2602.13061v1)
  is *Diverging Flows: Detecting Extrapolations in Conditional Generation*, by
  **Constantinos Tsakonas, Serena Ivaldi and Jean-Baptiste Mouret**. The
  [current v2](https://arxiv.org/abs/2602.13061) is titled *Native Extrapolation
  Awareness in Flow-Based Conditional Generation* (revised July 15, 2026).
  The old attribution to Bracha Laufer-Goldshtein is incorrect. The abstract
  describes flow matching with inefficient transport for off-manifold
  conditions; ordinary GMM likelihood is not its implementation. Hidden
  transformer states and a correctness-filtered training pool were choices
  of our old proposal, not requirements established by this paper.
- **LOCA:** [Peterfreund et al., arXiv:2004.07234](https://arxiv.org/abs/2004.07234)
  includes Ofir Lindenbaum and uses repeated-measurement sampling plus local
  normalization to learn standardized coordinates. Neighboring tokens change
  context and are correlated; they must not be labelled valid independent
  measurement bursts without testing the approximation.
- **Unsupervised KalmanNet:** [Revach et al., arXiv:2110.09005](https://arxiv.org/abs/2110.09005)
  appeared in [EUSIPCO 2022](https://eurasip.org/Proceedings/Eusipco/Eusipco2022/pdfs/0001571.pdf),
  not the unverified EURASIP-2024 citation in our old proposal. It trains from
  observation prediction without ground-truth states. The studied setting
  assumes partially known state-space structure. Our unknown telemetry
  dynamics and one-answer fitting budget need their own adaptation and audit.
- **IMM:** Blom and Bar-Shalom, *The interacting multiple model algorithm for
  systems with Markovian switching coefficients*, IEEE TAC 33(8), 780–783,
  1988, [DOI 10.1109/9.1299](https://doi.org/10.1109/9.1299). The publisher
  page was access-challenged during this check; no claim of a full paper read.

## Integration rules

**Latest user scope clarification:** IU-PCR / Joint L-SML remains the central
method. Every idea below is a supporting component. The advisor-facing
contribution is an improvement to our fusion architecture, with measured
continuity to its earlier versions. A standalone HMM, neural filter,
autoencoder or flow detector is at most a diagnostic control.

For each addition compare the same fusion core with/without it, and simple
aggregation with the same addition. This tests both the auxiliary component's
increment and whether learned fusion still contributes. Keep names explicit,
such as `IU + IMM readout` or `Joint + local-geometry preprocessing`;
neither denotes an implemented or successful method yet.

The current representation pilot stays frozen. Add each component in its own
versioned short experiment using the same release IDs, official step spans
and historical anchors. Do not couple LOCA, a neural filter and a new graph in
one first test: their separate contributions would be uninterpretable.

1. IMM/HMM/BOCPD are candidates for the chronological readout of fused
   scores, not evidence
   that an inferred regime literally denotes hallucination. Fit dynamics and
   any regime orientation only from permitted telemetry and declared rules.
2. KalmanNet can support fusion through an innovation/reliability view or
   filtering the fused trajectory. Its prediction target can be the next
   observed telemetry vector;
   it must not use correctness labels, teacher states from a verifier, or a
   pretrained cross-answer gain while being called strictly answer-only.
   Training reconstruction on the same point that is scored can hide errors;
   use blocked/predictive validation and report fitting cost.
3. LOCA-inspired local geometry can affect feature grouping or which token
   positions are sampled. Test burst stability and boundary retention first.
   A plain autoencoder or local whitening adaptation must be named as such.
4. A flow-derived prediction/transport signal can enter fusion as an added
   view or reliability estimate. A flow can operate on output telemetry without requiring hidden states,
   but that is a new design. Its training objective, conditioning, point
   exclusions and test scoring must be specified without label-selected clean
   examples. Low density is not synonymous with a reasoning error.

Short traces may be inadequate for neural fitting; report that limitation and
test declared within-answer regularization or a clearly separate pooled
comparison. Do not silently replace the primary single-answer objective.
The old no-go decisions inform compute priority; they do not prove universal
impossibility for these algorithms under the changed representation.
