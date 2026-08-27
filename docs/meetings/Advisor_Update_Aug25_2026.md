Subject: Reconstruction benchmark results and proposed thesis focus

Hi Ofir, Bracha and Amir,

I have completed the aligned reconstruction benchmark and the application
follow-ups we discussed. The benchmark compares 13 label-free methods on the
same 24 dataset–model cells, with grouped uncertainty estimates and separate
evaluation tracks for localization, early prediction, stopping and RAG.

The main result is not one universal winner. DEEM-B3 has the highest point
estimate in the 24-cell response benchmark, but the paired uncertainty does not
separate it from several simpler alternatives. Across the broader study,
graph- and dependence-aware extensions often recover real structure, but they
do not provide a stable improvement over the simpler IU/U-PCR and averaging
references. This suggests that additional structural complexity is not the
main source of reliable gains on the current response-level measurements.

The application results are more differentiated:

- **First-error localization:** the best ProcessBench adapters improve macro F1
  by 0.33–0.58 percentage points over matched IU, but all paired intervals
  cross zero. On PRMBench, the simpler token-only head is clearly better than
  response–token fusion.
- **Causal prefix prediction:** the selected Step272 score improves AUROC over
  Unified-28 by 3.26 percentage points at 64 tokens and 4.58 percentage points
  at 256 tokens, with both paired intervals above zero. This is the clearest
  positive application result, although it is prediction from saved prefixes
  rather than an adaptive stopping claim.
- **Actual stopping:** LEASH reduces generation length by about 39% overall,
  but lowers pass@1 by about 18 percentage points and loses accuracy in all six
  ready cells. I therefore view it as an informative accuracy–compute tradeoff,
  not a successful deployment rule.
- **RAG evidence:** RAGTruth AUROC is 0.727/0.689/0.659 at the
  answer/sentence/token levels. The local GASP gain over matched IU is small and
  its interval crosses zero. RefChecker behavior changes substantially across
  accurate-, noisy- and zero-context settings, so I kept the seven RAG panels
  separate rather than reporting a pooled score.
- **Internal layers:** white-box depth features provide wider row coverage, but
  do not show a validated aggregate advantage over the final-output gray-box
  score.

My current interpretation is that the strongest thesis story may combine a
general label-free fusion framework with a careful account of where additional
structure does and does not help, followed by the stronger task-specific
prefix result and localization application framework. I would value your
advice on two decisions:

1. Should the main narrative lead with the general fusion framework or with
   localization and early prediction as the clearest application?
2. What should the final prospective test prioritize: new-question/new-model
   localization, calibration, or a smaller confirmation of the internal-layer
   result?

I attached a one-page results map and four short visual briefs. The map also
links to the complete interactive 13-method report and the exact result tables.

Could we meet next week to discuss the paper and thesis structure?

Thanks,
Omri
