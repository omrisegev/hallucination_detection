Subject: Update after our July 30 meeting

Hi Ofir, Bracha and Amir,

Since my August 3 email, I finished the clustering/DUFS, internal-layer and ProcessBench experiments. Conformal remains unstarted.

- Basic fusion: cluster-aware U-PCR lost 4.5 points in 25 experiments. DUFS's answer graph added almost zero. Methods remain near 0.774 macro AUROC. One stable pattern followed length.
- Family-NRM uses disagreement among six hand-defined measurement types to correct IU-PCR. On 6,966 PRMBench math traces, AUROC rose 0.7206 → 0.7252: +0.46 points, 95% interval [+0.07,+0.84].
- Internal layers and final-output scores had similar AUROC on 31,440 identical answers. Their combination still needs validation.
- Across the three certified ProcessBench scorer panels, the point-leading adapters added 0.33 to 0.58 F1 points over matched IU, but every interval crossed zero. At 64 and 256 tokens, Step272 exceeded Unified-28 by 3.26 and 4.58 AUROC points. Certified paper-specified-partial LEASH callback stopping cut tokens but lowered pass@1 in all six ready cells; two Mistral cells were blocked. Certified RAG spans seven unpooled retrospective panels; local GASP shows no superiority.

SU-PCR, the 2022 sparse-error extension, did not clearly help. Our graph-free continuous additive DEEM-B3 adapter is inspired by DEEM, not a reproduction of its published hard multinomial/iRBM method. It completed 24 experiments and was not worse than IU-PCR under the preregistered rules, but was not declared better. Its features and macro differ, so I am not ranking it yet.

For the aligned rerun, I plan new Family-NRM and graph-roughness versions that learn inside each dataset–model experiment, without donor data or labels. Cross-dataset variants will be separate unlabeled-donor or donor-label controls. These local versions are not results yet.

I do not yet have one directly comparable general winner. I would value your view on what should lead, and whether a final test should study new-question localization, conformal calibration or internal layers.

Could we meet next week?

Thanks,
Omri
