# Entropy sampling: reviewed findings

Same 110 development answers: 24 PRMBench and 86 ProcessBench-Qwen3-8B. These are retrospective matched results, not untouched confirmation. All fitting uses the current answer only.

| Fitting-window selection | IU pooled AUC | IU within-answer AUC | IU PB score | Joint graph pooled AUC | Joint graph PB score |
|---|---:|---:|---:|---:|---:|
| All windows | 0.6813 | 0.7688 | 30.16% | 0.6555 | 30.22% |
| Uniform in time | 0.6764 | 0.7770 | 28.34% | 0.6457 | 26.87% |
| High entropy only | 0.7584 | 0.7789 | 28.35% | 0.7413 | 28.20% |
| Half low + half high | 0.7275 | 0.7717 | 26.27% | 0.7144 | 26.36% |
| Across entropy quantiles | 0.7152 | 0.7920 | 20.77% | 0.6702 | 24.37% |

PB is the macro harmonic score of clean-answer accuracy and exact first-error accuracy, not general classification F1. Pooled PRMB AUROC is not official PRMScore. Within-answer AUROC averages the 16 answers containing both step labels.

Neither new selector replaces the references on both benchmarks. Quantile IU improves within-answer AUC over full by 0.02316 (exploratory 95% interval 0.00213 to 0.05232), but PB falls by 9.38 percentage points (interval -17.51 to -2.56). Compared with high-only, the within-answer difference remains uncertain (-0.00669 to 0.04073). Intervals are not adjusted for multiple comparisons.

Joint graph also loses PB points with both new selectors. Joint results include fallback to IU on 20/110 answers for tails and 19/110 for quantiles; no final outputs are invalid. Among the 72 newly fitted answers the fallback counts are 18 and 17. Dense scores and mixture-gate support stay fixed, but normalization and fusion fitting both change with selection. This experiment does not isolate their causal contributions.

Keep the full matched reasoning benchmark and corrected historical refits as the next priority. Retain quantile sampling as local-ranking evidence for future diagnosis; do not launch a new sweep or promote a winner from these 110 answers.

Scientific review PASS: 220 selections, 1008 linear-score checks, 1540 dense-readout replays, six representative shared-kernel refits, 14 metric bundles. Supplemental review independently checks within-answer AUC by positive-negative pairs, 35 report rows, 31 intervals and six local links. This is same-session review, not external validation.

[Full HTML table and intervals](REPORT.html) | [Scientific review](REVIEW.json) | [Supplemental review](REVIEW_SUPPLEMENT.json)
