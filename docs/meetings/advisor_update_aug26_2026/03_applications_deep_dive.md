# Deep dive 3 — Applications beyond completed-answer detection

The application work deliberately keeps four different prediction units separate. A response-level AUROC, first-error macro F1, prefix AUROC and token-saving/pass@1 frontier are not entries in one leaderboard.

## A. First-error localization

### Goal and references

The goal is to predict whether a reasoning trace contains an error and, if so, locate the first erroneous step. ProcessBench supplies the official step annotations. Mind the Gap is the main label-free localization comparator; supervised PRMs and large critic models are separate higher-access ceilings.

### Method

The reconstruction crosses all 13 frozen answer-level methods with one shared local head. The response head estimates whether the complete reasoning trace is unreliable. The local head fuses 29 token-level uncertainty trajectories with two-component L2 IU-PCR, and each step receives the maximum token risk inside its span. ProcessBench combines empirical response and step ranks by their geometric mean, then calibrates a threshold that chooses between the highest-risk step and **no error**. PRMBench uses the step risk directly because it labels every step rather than asking for one first-error decision. Step boundaries reduce token scores after fusion; they do not construct the token representation.

### Results

- GL-LIU: 31.36% ProcessBench macro F1.
- Reproduced Mind-the-Gap control: 25.71%.
- Later aligned three-scorer adapters: +0.33 to +0.58 F1 points versus matched IU, with intervals crossing zero.
- ProcessBench component audit: 13 response-only heads span 17.36%-19.20% macro F1; token-only reaches 29.44%; the best response-token combination reaches 31.07%.
- PRMBench component audit: token-only reaches 0.6712 AUROC, above the best response-token combination at 0.6493; the supervised Qwen2.5-Math-PRM-7B ceiling is 0.7983.
- Later CIW transfer: CIW response + frozen token-IU29 reaches 30.91% ProcessBench macro F1 versus 31.02% for CA-DEEM (internal B3), and 0.5811 PRMBench AUROC versus 0.5842 for CA-DEEM. This did not test a token-level CIW model.

The stable contribution is the localization framework and token-first construction, not a robust graph increment. The direct follow-up is to adapt the innovation idea to token trajectories and validate the resulting localizer on new data.

## B. Causal prefix prediction

### Goal and reference

The target is final-answer correctness, but the score may use only the prefix observed at a fixed token budget. DeepConf motivates the confidence-under-compute question, while our single-trace causal score is an adapted common-protocol comparator rather than a paper-exact DeepConf reproduction.

### Results

- 64 tokens: Step272 0.5955 AUROC versus Unified-28 0.5629; delta +0.0326, interval [+0.0035,+0.0625].
- 256 tokens: Step272 0.6572 versus Unified-28 0.6114; delta +0.0458, interval [+0.0147,+0.0765].

This is the clearest positive application result. It is prediction from saved fixed prefixes, not proof of adaptive stopping or realized compute savings.

## C. Actual stopping with LEASH

LEASH is evaluated as an accuracy-compute frontier, not as an AUROC detector. The reconstruction observed the actual callback and forced closure in six eligible cells; two Mistral cells were blocked because the tokenizer did not expose the required chat template.

- Overall token reduction: 38.83%, interval [37.08%,40.54%].
- Overall pass@1 delta: -18.26 percentage points, interval [-21.21,-15.48].
- Pass@1 decreases in every ready cell.

The result demonstrates real compute savings and a real accuracy cost. It does not support a matched-accuracy or paper-exact stopping claim.

## D. RAG evidence panels

### References and units

- RAGTruth: answer-, sentence- and token-level hallucination labels.
- GASP: sentence-level evidence-removal scoring.
- LettuceDetect: supervised example-level ceiling.
- RefChecker: claim-level three-way NLI and binary fixed-claim transfer under accurate, noisy and zero context.

### Results

- RAGTruth test AUROC: answer 0.7274, sentence 0.6892, token 0.6587.
- GASP local sample: 0.6708 versus 0.6597 for matched IU; delta +0.0111, interval [-0.0125,+0.0343]. No superiority claim.
- LettuceDetect supervised example-level F1: 0.7929.
- RefChecker three-way NLI accuracy: 0.6007 accurate context, 0.7620 noisy context, 0.7337 zero context.
- RefChecker fixed binary AUROC: 0.6645, 0.6402 and 0.7506 respectively.

The settings and prediction units are not pooled. RAGTruth also shows substantial task heterogeneity between QA and Data-to-Text, reinforcing the need to report panels separately.

## Visuals and reports

- [Response/token fusion architecture and component result](figures/localization-fusion-summary.svg)
- [Applications advisor brief](../advisor_update_aug21_2026/04_localization_and_early.html)
- [Fair paper-exact comparison report](../../../results/fair_paper_exact_comparisons_v1/REPORT.html)
- [ProcessBench localization report](../../../results/ours_only_localization_v1/REPORT.html)
- [GL-LIU factorial report and plots](../../../results/gl_liu_factorial_v2/REPORT.html)
- [Global/local/online architecture report](../../../results/global_local_online_architecture_v2/REPORT.html)
- [RAG evidence-contrast report](../../../results/ragtruth_evidence_contrast_v1/REPORT.html)
- [RAG mixed-v2 evidence-aware report](../../../results/ragtruth_mixed_v2_evidence_aware_v1/REPORT.html)
- [All application plots and source reports](ASSET_INDEX.md#application-plots-and-reports)
