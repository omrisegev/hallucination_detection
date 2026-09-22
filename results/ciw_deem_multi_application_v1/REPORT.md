# CIW-DEEM multi-application benchmark v1

This report adds CIW-DEEM to the application lanes created by the
reconstruction benchmark. It keeps each task's prediction unit separate.

## What was actually run

- The registered 24-cell completed-response benchmark.
- Twenty-two compatible external completed-response cells covering
  ProcessBench, PRMBench, Evidence-Drop, GPQA, and HLE.
- ProcessBench and PRMBench localization, using frozen CIW response risk plus
  the already-frozen token-IU29 locator. No token model was refitted.
- Causal early detection at token budgets 16, 32, 64, 128, 256, and 512.
- RAGTruth response-level detection on the full original-30 feature contract.

Every new score was fitted without correctness labels and frozen before the
corresponding evaluator opened labels. These application results are
retrospective because the benchmark outcomes were historically visible.

## Main results

The registered 24-cell CIW score remains:

- cell-macro AUROC: `0.7820255514493354`;
- equal-dataset-family AUROC: `0.7492330051057238`.

External completed responses are mixed. CIW improves over B3 on GPQA
(`+0.001424`) and HLE (`+0.005777`), is nearly tied on Evidence-Drop
(`-0.001230`) and ProcessBench (`-0.001846`), and regresses on PRMBench
(`-0.008035`).

Localization does not improve. ProcessBench macro-F1 is `0.309136` versus
`0.310228` for B3. PRMBench step AUROC is `0.581138` versus `0.584218` for B3
and about `0.6004` for DUFS-LIU.

Causal early detection is close to IU28 at most budgets. CIW is slightly
higher at 16 and 64 tokens, but lower at 32, 128, and 256 tokens. At 512 tokens
AUROC is undefined for every method because one subset has one class; CIW
AUPRC is `0.738888`.

RAGTruth response-level transfer is the strongest new application result. On
the same 1,800 test responses, CIW has AUROC/AUPRC `0.771222/0.635797`, versus
`0.760523/0.596613` for Original-30 IU-PCR and `0.762882/0.598308` for
Original-30 DUFS-LIU. Equal weighting of QA and Data2txt gives CIW AUROC
`0.630244`, versus `0.602162` for DUFS-LIU. The pooled and task-macro rankings
differ because Data2txt and QA have different class/task structure.

## Lanes that cannot use registered CIW directly

EDIS prepared inputs contain 26 features and omit all four partition-energy
coordinates. The registered CIW gate requires the complete 3-by-3 core:
entropy, sampled-token surprisal, and partition energy crossed with mean,
sliding variance, and CUSUM. Running a two-source substitute would be a new
method, not CIW-DEEM v1.

RAG sentence, token, GASP sentence, LettuceDetect span, and RefChecker claim
panels use units that do not carry a completed-response 30-feature input.
They require sentence-, token-, span-, or claim-specific models. A constant
response score copied to every sentence or token would not be localization.

Stopping is also not inferred from prefix AUROC. A stopping method needs a
frozen policy mapping prefix risk to stop/continue and must be evaluated on
accuracy versus compute.

White-box layer fusion is a separate access tier. CIW's coordinates are
output-probability and energy summaries, not hidden-layer features. A fair
comparison is possible only after an exact shared-row gray-box adapter is
frozen; inserting the CIW transform into hidden states would define a new
white-box method.

See `COVERAGE.csv` for the complete lane-by-lane status and `METRICS.csv` for
the compact numerical summary.
