# Why C4U and TELE16 behave differently across benchmarks (2026-11-11)

Development diagnosis, CPU only, stored scores and banks, no refit of any frozen recipe. 8 benchmarks x 4 PRMs
(Qwen-PRM-7B, Qwen-PRM-72B, PURE, RLHFlow; Math-Shepherd PRM excluded), PRM-backbone telemetry mode.
C4U = frozen `C4U-parisi-within-jaffe-cross-v1` (`results/rigorous_recipe_v1/c4u_standalone/c4u_new.py`, unchanged);
TELE16 = stored `TELE_N16`. Parity: PRMBench Qwen-7B C4U within-AUC reproduces the frozen 0.8398 exactly.

## Short answer
The ranking of C4U/TELE16 vs the PRM is explained mostly by **what the benchmark's labels reward**, not by the telemetry
signal disappearing. Three mechanisms, in order of size:

1. **Answer-level signal vs per-answer normalization.** TELE16 and C4U work on answer-z-scored channels, so they cannot
   say "this whole answer is worse than that one". Where much of the label variance sits between answers (many clean
   answers: CRV-Arith 92%, CRV-GSM8K 79%, PRM800K 63%, ProcessBench 35%), the PRM wins largely because it keeps a global
   scale. Evidence: z-scoring the PRM itself per answer costs it .11 (ProcessBench) to .22 (CRV-Arith) pooled AUROC but
   only .03 on PRMBench and 0 on DeltaBench. Across the 32 cells, the TELE16 gap correlates with the between-answer label
   share (Spearman -.75; leave-one-benchmark-out range -.63 to -.86) and with the clean share (-.75).
2. **Labelling protocol: onset vs cascade.** Telemetry spikes where the reasoning goes wrong (the first error). Benchmarks
   that label every step after the first error as wrong (CRV-Arith 90% of error answers, Hard2Verify 50%, Socratic 30%)
   reward a score that stays high after the error, which PRMs (trained on prefix correctness) do and telemetry does not.
   On the first-error target with position stratification, the TELE16 gap shrinks: Socratic -.10 -> -.02,
   Hard2Verify -.04 -> -.01, CRV-Arith -.37 -> -.17. ProcessBench and PRM800K stop labelling at the first error, so the
   error is always the last labelled step: position alone gives within-AUC 1.00 / .995 there (within-AUC is not a valid
   localization metric on these two).
3. **PRM strength / in-domain PRMs.** On PRM800K, CRV and ProcessBench the PRMs are near their training distribution
   (pooled AUROC .85-.91); on PRMBench and DeltaBench they are weak (.63-.75), leaving room for telemetry. Supervised
   headroom of adding the 16 channels to the PRM (5-fold answer-grouped LR) is largest on PRMBench (+.076) and smallest on
   CRV-GSM8K (+.001). This is confounded with (1) at the benchmark level: 8 benchmarks cannot separate them.

C4U-specific failures on top of that:
- **Prevalence**: C4U's estimated error prevalence stays at .17-.40 whatever the truth; ratio to the true step error rate
  is 1.08 on PRMBench but 2.3 (ProcessBench), 3.1 (CRV-GSM8K), 4.4 (PRM800K), 11.9 (CRV-Arith). Its own-count flags
  therefore over-flag on low-error benchmarks; this is why PB F1 collapses.
- **Orientation**: 4 of 32 cells have C4U pooled AUROC < .5 (CRV-Arith/Qwen-72B .25, PRM800K/RLHFlow .34, Hard2Verify
  PURE .49 and RLHFlow .48), all on benchmarks with few errors or few answers.

What is **not** the driver:
- **Per-channel signal exists everywhere.** First-error, position-stratified AUC of single channels is .6-.8 on every
  benchmark (fig4). On CRV-Arith the best channels reach .70-.73 while fused TELE16 is .54 pooled.
- **Conditional dependence on the PRM is similar everywhere.** Mean |Spearman(channel, PRM)| within correct steps is .17-.26
  and within error steps .19-.34 on all benchmarks; it does not track the gap (Spearman .14 / -.31). It is slightly higher
  within error steps on CRV-GSM8K (.34) and CRV-Arith (.31), where the supervised headroom is smallest.
- **Which channels carry the signal changes.** Channel-effect profiles correlate .97 between PRMBench and Socratic (Socratic is
  built from PRMBench) but only .11-.56 with the others; digit channels lead on PRM800K/ProcessBench, surprisal/derivative
  channels on PRMBench. TELE16 was selected on PRMBench, so its fusion is tuned to a profile that does not transfer.

## Files
| file | what |
|---|---|
| `extract.py` | builds every cell with `c4u_tele16_all_benchmarks_v1/run_all.py` code + `c4u_new` and saves step arrays to `cells/` (gitignored, ~regenerable) |
| `analyze.py` | T1 label structure, T2 arm quality (pooled, within, first-error position-stratified, answer-level), T3 channels, T4 LR headroom |
| `report.py` | T5 attribution (+ T5b leave-one-out), T6 class-conditional dependence, T7 per-cell table, T8 error-answers-only, T9 regression, figures, `SUMMARY.md` |
| `fig1..fig7_*.png` | label structure, quality per target, gap vs answer-level signal, channel effects, CI vs headroom, C4U prevalence, gap by target |

Definitions: first-error target = steps up to and including the first labelled error (all steps of clean answers), label
= "is the first error", AUROC computed inside each step-index stratum (cap 12) and pair-weighted. Between-answer label
share = var(answer-mean label) / var(step label). Norm penalty = PRM pooled AUROC minus pooled AUROC of the PRM's
per-answer z-score.

Caveats: 32 cells but only 8 benchmarks, so the factors are confounded and correlations are descriptive. Hard2Verify has
177 answers. Hard2Verify C4U is built with the Socratic R0 construction (no frozen H2V cell). Channel columns are the C4U
inputs (answer-z of rank-normal raw for R1 benchmarks, stored answer-z for Socratic/H2V). Standalone (`sa`) mode not run here.

Reproduce:

    cd results/bench_gap_diagnosis_v1
    COLLECTION_DATA=C:/Users/omris/TAU/hd_collection_data python extract.py   # ~1.5 h, all 32 cells
    python analyze.py && python report.py

Next directions this suggests: (a) evaluate onset detection separately from cascade labels when comparing to PRMs;
(b) give C4U/TELE16 a global (between-answer) term or answer-level gate, e.g. keep the PRM's raw scale for the
"does this answer have an error" decision; (c) a label-free prevalence check for C4U on low-error benchmarks.
