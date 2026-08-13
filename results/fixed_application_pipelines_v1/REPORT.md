# Fixed RAG and Reasoning Pipeline Results

The fixed RAG pipeline reaches answer AUROC 0.728 [0.704, 0.751]. The fixed reasoning pipeline reaches ProcessBench eight-cell macro F1 0.307; on the matched Qwen3-8B four-subset protocol it reaches 0.304 versus 0.250 for Mind the Gap, a paired delta +0.054 [+0.032, +0.077]. and PRMBench step AUROC 0.671. The trajectory-first PRMBench result improves clearly over the old step-first adapter. Reasoning is competitive with the label-free Mind the Gap control, but it remains far below supervised PRM and the 72B critic.

## Fixed RAG

- RAGTruth answer AUROC: **0.7276**.
- RAGTruth answer 95% source-group interval: **[0.7041, 0.7506]**.
- RAGTruth sentence AUROC: **0.6893**.
- RAGTruth token AUROC: **0.6586**.
- Dev-calibrated answer example F1: **0.5483**.

## Fixed reasoning

- ProcessBench eight-cell macro F1: **0.3070**.
- Mind the Gap control macro F1: **0.2571**.
- ProcessBench matched Qwen3-8B four-subset F1: **0.3035** versus **0.2496** for Mind the Gap.
- Paired Qwen3-8B F1 delta: **+0.0539 [+0.0316, +0.0773]** across identical calibration/evaluation splits.
- PRMBench step AUROC: **0.6711**.
- Qwen2.5-Math-PRM-7B ceiling AUROC: **0.7983**.

See `REPORT.html` for plots, detailed tables, method flow, and limitations.
