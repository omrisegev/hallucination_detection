# Papers referenced in the advisor update

## Core unsupervised fusion and dependence

- Ariel Jaffe, Ethan Fetaya, Boaz Nadler, Tingting Jiang and Yuval Kluger. **Unsupervised Ensemble Learning with Dependent Classifiers.** AISTATS 2016. Used as the dependence/clustering inspiration for the pre-meeting clustered U-PCR experiment; our regression extension is not a paper reproduction.
- Omer Dror, Boaz Nadler, Erhan Bilal and Yuval Kluger. **Unsupervised Ensemble Regression.** 2017. Provides the continuous covariance/PCR basis behind U-PCR.
- Yaniv Tenzer, Omer Dror, Boaz Nadler, Erhan Bilal and Yuval Kluger. **Crowdsourcing Regression: A Spectral Approach.** AISTATS 2022. Supplies the IU-PCR/SU-PCR framing and the low-rank-plus-sparse covariance model.
- Uri Shaham et al. **A Deep Learning Approach to Unsupervised Ensemble Learning.** 2016. Earlier RBM/DNN lineage for dependent unsupervised ensembles.

## Feature selection, graphs and nonlinear energy models

- Ofir Lindenbaum, Uri Shaham, Jonathan Svirsky, Erez Peterfreund and Yuval Kluger. **Differentiable Unsupervised Feature Selection based on a Gated Laplacian.** NeurIPS 2021. Inspired both direct gate-ranking experiments and the continuous DUFS-LIU sample graph; DUFS-LIU is our adaptation.
- Ofir Lindenbaum, Moshe Salhov, Amir Averbuch and Yuval Kluger. **l0-based Sparse Canonical Correlation Analysis.** Inspired the cross-channel gated objective tested in the feature-selection cycle.
- Junchen Yang, Ofir Lindenbaum, Yuval Kluger and Ariel Jaffe. **Multi-modal Differentiable Unsupervised Feature Selection.** Inspired the shared-operator graph control.
- Amitai Yacobi, Ofir Lindenbaum and Uri Shaham. **Generalizable and Robust Spectral Method for Multi-view Representation Learning (SpecRaGE).** Inspired multi-view graph constructions; the repository methods are adaptations.
- Junjie Hu et al. **HARP: Hallucination Detection via Reasoning Subspace Projection.** 2025 preprint. Inspired the target-versus-nuisance subspace architecture used in contribution space. We did not reproduce HARP's supervised hidden-state detector.
- Ariel Maymon, Yanir Buznah and Uri Shaham. **Unsupervised Ensemble Learning Through Deep Energy-based Models.** AISTATS 2026. Inspired the official-package adapter and the later continuous B3/CIW adaptations; those are not paper-exact reproductions.

## White-box detection

- Bohan Yang et al. **TriLens: Per-Layer Logit-Lens Entropy for White-Box Hallucination Detection.** 2026 preprint. Main reference for the three-pathway, all-layer internal trajectory; our main fusion is label-free rather than a supervised TriLens probe.
- Binkowski et al. **Hallucination Detection in LLMs Using Spectral Features of Attention.** Reference for the supervised LapEigvals ceiling; it is not a like-for-like label-free comparator.

## Reasoning localization, early prediction and stopping

- Chujie Zheng et al. **ProcessBench: Identifying Process Errors in Mathematical Reasoning.** Supplies first-error reasoning labels and the official localization metric.
- **Mind the Gap: Catching Hallucinations via Evidence Drop.** ICML 2026. Main label-free localization comparator.
- Yichao Fu et al. **Deep Think with Confidence.** Main conceptual reference for confidence-aware reasoning and compute; some local comparisons are adapted controls rather than paper-exact reproductions.
- **LEASH: Logit-Entropy Adaptive Stopping Heuristic for Efficient Chain-of-Thought Reasoning.** Reconstructed under a paper-specified-partial protocol with observed callbacks and forced closure.
- Oren-Loberman, Azar and Huleihel. **Online Auditing of Information Flow.** IEEE TSIPN 2024. Used for the sequential accuracy-delay formulation, not for a theorem-level reproduction.

## RAG hallucination detection

- Niu et al. **RAGTruth: A Hallucination Corpus for Developing Trustworthy Retrieval-Augmented Language Models.** ACL 2024.
- **GASP.** 2026 preprint. Evidence-removal reference evaluated on the local exact cohort.
- **LettuceDetect.** 2025 preprint. Supervised example-level ceiling.
- Hu et al. **RefChecker: Reference-based Fine-grained Hallucination Checker and Benchmark for Large Language Models.** EMNLP 2024.

For exact local PDFs, digests and artifact provenance, see the project `papers/` index and the [algorithm chronology packet](/Users/osegev/Documents/Codex/2026-08-25/referenced-chatgpt-conversation-this-is-an/outputs/post_july30_upcr_dufs_evidence_packet.md).
