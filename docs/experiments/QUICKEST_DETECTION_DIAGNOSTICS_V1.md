# Quickest-detection diagnostics of the frozen first-error locators — 2026-09-22 (Step 428)

Question: when the first-error task is read as a quickest change detection problem
(one latent switch from faithful to erroneous, detect it as early as possible under a
false-alarm budget; Itkin, arXiv 2606.12476), what do the frozen locators and channels
look like, and is the long-chain collapse (SLA 45.5 -> 16.4 % across depth buckets, Step
423) a property of the readout or of the channels?

This is a diagnostic stage, not a candidate. Labels enter only as evaluation targets:
nothing is fitted, selected or thresholded on them. No new inference. Inputs are the
frozen token matrices (`TOKEN_MATRICES.npz`, sha recorded), the readout-family profiles
(`results/readout_family_v1/profiles_full.npy`), the frozen CT7 step scores, the Step 422
token-fusion OOF scores and the Mind-the-Gap replay. Branch
`claude/readout-quickest-detection-v1`, base = consolidation `2e7758158`.

Three measurements, in this order.

B2, Markov persistence (PRMBench, 6,969 answers, v3 per-step flags). P(error at s |
error at s-1) against P(error at s | no error at s-1), the base error rate, the
error-run-length histogram, the fraction of erroneous answers with more than one error
run, and the first-error hazard by step index; overall, by the eleven PRMBench
classifications and by step-count stratum; 2,000 source-question bootstrap draws for the
two conditional probabilities. Reading: if errors persist once they start, the one-switch
premise of the quickest-detection framing holds on math chains and "first error" is the
right target; if runs are short and multiple, sequential rules of any kind are
mis-specified here.

B4, empirical delay floor (ProcessBench, 4,442 erroneous and 2,358 clean answers). For a
locator with score distribution f1 on first-error steps and f0 on error-free steps
(steps before the first error in erroneous answers, plus every step of clean answers),
a Lorden-type bound gives mean delay >= ln(1/alpha) / KL(f1 || f0) at false-alarm rate
alpha. KL and the Bhattacharyya distance are estimated from 48 pooled-quantile bins with
0.5 smoothing, for every channel x readout (7 frozen + 10 extended), the Page statistic at
k in {0.25, 0.5, 1.0}, a prefix-standardised causal variant, CT7, the token fusion,
Mind-the-Gap, the longest-step control and, when present, the readout-family OOF
consensus fusion; raw and answer-standardised; macro, per cell and per step-count
stratum; 1,000 source-question bootstrap draws on the macro and stratum values. The
divergence of post-error steps against error-free steps is recorded beside it as context
for the "conditional tokens" effect (Snel and Oh, arXiv 2507.20836). Reading: a floor that
stays flat across strata for the best readout of a channel says the readout is not the
bottleneck; a floor that grows with depth for every readout of every channel says the
channels carry no early-error signal on long chains and a new view is needed.

B1, delay against false alarm (ProcessBench). The argmax is replaced by a first-crossing
rule: stop at the first step whose answer-standardised score reaches a global threshold,
swept over 200 quantiles. Clean answers give the false-alarm rate; erroneous answers give
the early-stop rate, the detection rate (stop at or after the first error), the exact rate,
the no-stop rate and the mean and median delay in steps and in tokens. Reported per
stratum and per cell for CT7, the token fusion, Mind-the-Gap, the longest-step control, each
channel at top5 and at the Page maximum (k = 0.5, with 0.25 and 1.0 as sensitivity), the
causal variant of both, and the OOF consensus fusion when present; summarised at
false-alarm budgets 5, 10, 20 and 30 % by the lowest threshold within budget. The argmax
rule's SLA / early / late are printed on every row as the reference point. These sweeps are
descriptive; no threshold is promoted.

Anchors that must hold before anything is read: the `Dataset` replay asserts (token L-SML
SLA .3592, token equal .3259, CT7 macro-F1 .41188745848863717, CT7 PRMBench within-answer
AUROC .7723966352864217 on 6,030 eligible answers) and the frozen profile sha
`f932ad81…f5f3` inside `profiles_full.npy`. Sanity checks written into the outputs: the
"always step 0" behaviour of the longest-step control in B1, B2 totals equal to 6,969
answers.

Artifacts: `results/quickest_detection_diagnostics_v1/` — `B2_PERSISTENCE.{json,csv,png}`,
`B4_DELAY_FLOOR.csv`, `B4_BEST_READOUT_BY_CHANNEL.csv`, `B4_DELAY_FLOOR.png`,
`B1_DELAY_CURVES.csv`, `B1_SUMMARY.csv`, `B1_DELAY_VS_FALSE_ALARM.png`,
`B1_DETECTION_VS_FALSE_ALARM.png`, `MANIFEST.json` (input shas, timings). Script:
`scripts/diagnostics/quickest_detection_diagnostics_v1.py`.

Out of scope: any learned CUSUM increment (that is a supervised ceiling, deferred), any
retuning of the Page drift k or the threshold quantile on these results, and any claim
about the paper's own numbers (its RAGTruth setting is not ours).
