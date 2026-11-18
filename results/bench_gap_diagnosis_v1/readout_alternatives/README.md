# Length-robust step readouts of the telemetry channels (2026-11-12)

Development diagnosis, CPU only, no model re-run. Data were already inspected, so no readout here is a confirmed method.
Labels are used only for evaluation.

**Question.** On Socratic and Hard2Verify the final step is very short. Under the current readout (mean of the 10
highest token values in the step, then answer-z), the final step sits near -1 answer-z whether it is right or wrong.
Does a readout that does not depend on step length fix this, and does it close the gap to the PRM?

**Short answer.** No. The length-robust readouts remove the final-step artifact: final-step level goes from about -1.2
to about 0 on Socratic. They do not create a correct-vs-wrong difference at the final step, because there is almost none
in the token signal. They also lose first-error detection on PRMBench and Socratic. The current top-10 readout stays the
best or tied-best everywhere except Hard2Verify, where every arm, the PRM included, is near chance. Dropping each step's
first token changes almost nothing (≤ .006 AUC, except -.024 within-answer on ProcessBench).

## Data and gate
- Raw token telemetry: the PRM-backbone `external_telemetry.pkl` / `prmbench_telemetry.pkl` files, restored with local
  rclone from the Drive backup `gdrive:hallucination_detection/cluster_results/{prm_boost_external_v1,
  prm_boost_processbench_v1, prm_boost_v1, prm_boost_gen_v1}`. AIRCC was unreachable (two ssh timeouts), but it was not
  needed. Raw pickles were deleted after extraction. The step-level outputs (`steps/*.npz`) are kept outside git.
- Token values: the same mask1 and empty-step handling as the bank builders, plus
  `fusion.validate_telemetry` (11 bank11 channels, risk-oriented). Channels not covered: realized_z, realized_drv
  (step statistics, not token readouts) and the 3 digit channels (excluded).
- **Gate PASS** (`T_gate.csv`). My top-10 matches the stored values in all 16 cells:
  - raw bank: max abs diff 0 to 1.8e-13 for Socratic, Hard2Verify and PRMBench, and ≤1.9e-6 for ProcessBench (float32 cell storage);
  - stored answer-z: about 1.2e-7.
  - One exception: one Hard2Verify answer (PURE and RLHFlow), where `top50_js` is constant at ln 2. My answer-z sets
    it to 0; the stored file z-scored float noise (diff 2.65). This is a zero-variance edge case, not a readout difference.

## Readouts (all on risk-oriented token values; then answer-z per answer)
- `top10`: current readout.
- `top10_lenres`: current readout with pooled, label-free linear removal of answer-z log(token count).
- `top10_drop1`: current readout after dropping each step's first token.
- `mean`, `median`, `p90`: within-step mean, median and 90th percentile.
- `top20pct`: mean of the top 20% of tokens (at least 1).
- `frac_above`: share of tokens above the cell's global 80th percentile of that channel.

Composite = plain average of the 11 channels (`avg11`, fixed bank signs). `avg3` (q15_H1, chosen_surprisal,
logprob_margin) is in `T_summary_avg3.csv`. Per-channel rows are in `T_cells.csv`.

## Key numbers (avg11, mean of 4 PRMs; `T_summary_avg11.csv`)
| bench | readout | within | pooled | first-err strat | final step correct / wrong | pairs a / b / c / d |
|---|---|---|---|---|---|---|
| PRMBench | top10 | .754 | .713 | .782 | -.17 / +.17 | .78 / .75 / .64 / .67 |
| | mean | .655 | .630 | .698 | +.25 / +.27 | .70 / .58 / .51 / .68 |
| | p90 | .717 | .682 | .752 | +.11 / +.24 | .75 / .67 / .56 / .69 |
| | PRM | .755 | .747 | .808 | +.37 / +.97 (PRM answer-z) | .78 / .69 / .60 / .75 |
| Socratic | top10 | .633 | .613 | .762 | -1.24 / -1.16 | .76 / .78 / .64 / .48 |
| | top10_lenres | .658 | .639 | .729 | -.14 / -.05 | .73 / .69 / .57 / .64 |
| | mean | .639 | .626 | .647 | +.52 / +.54 | .67 / .50 / .47 / .75 |
| | PRM | .746 | .740 | .783 | +.33 / +.72 | .80 / .69 / .59 / .80 |
| Hard2Verify | top10 | .529 | .526 | .638 | -.91 / -.93 | .70 / .65 / .52 / .49 |
| | median | .559 | .551 | .613 | -.21 / -.24 | .65 / .67 / .59 / .55 |
| | PRM | .524 | .561 | .672 | -.57 / -.46 | .60 / .78 / .56 / .49 |
| ProcessBench | top10 | (.718, not localization) | .717 | .700 | -.57 / +.28 | .72 / - / - / - |
| | mean | - | .617 | .600 | +.15 / +.41 | .69 |
| | PRM | - | .873 | .880 | -.25 / +1.29 | .80 |

Pair types: (a) first error vs correct steps before it, (b) first error vs correct steps after it, (c) later errors vs
correct steps after the first error, (d) later errors vs correct steps before the first error.

## Figures
**`fig1_gap_to_prm_by_readout.png`**: 11-channel average minus PRM, per benchmark and metric.
- No readout closes the Socratic gap. Within-answer AUC is -.09 to -.13 below the PRM for every readout. Linear length
  removal is the least bad there (-.088 vs -.113 for top10), but it costs .03 to .05 on first-error detection.
- On PRMBench and ProcessBench, every alternative loses to top10: by .04-.10 within-answer on PRMBench, and by .05-.10
  on pooled and first-error AUC on both.
- Hard2Verify is the only benchmark where mean, median or frac_above move above the PRM on within-answer AUC (+.02 to
  +.035). There, the PRM itself is at .52, there are 177 answers, and the per-PRM spread is large (.45-.65), so this is noise-level.

**`fig2_final_step_level.png`**: answer-z level of the final step, correct vs wrong.
- Mean, median, frac_above and p90 move the Socratic and Hard2Verify final step from about -1 to about 0 (or positive).
  The artifact the other machine found is a property of the top-10 readout of very short steps.
- Even after that fix, wrong and correct final steps stay at the same level on Socratic and Hard2Verify (differences
  ≤ .09). On PRMBench the top-10 readout separates them by .34. The PRM separates them by .39 (Socratic) and .60 (PRMBench).
- So removing the artifact does not add final-step signal. Telemetry has no information about whether a bare final
  answer is wrong. The PRM does.

**`fig3_pair_types.png`**: which comparisons each readout wins.
- Moving from top10 toward mean or median trades (b) for (d). On Socratic, (b) falls .78 to .50 and (d) rises .48 to
  .75. The averaging readouts score whole erroneous suffixes higher (cascade-like), but lose the onset spike that
  separates the first error from the correct steps after it.
- Top10 already beats the PRM on (b) on PRMBench (.75 vs .69) and Socratic (.78 vs .69). Its deficit is (d), later
  errors vs correct steps before the first error. That is where the remaining gap sits, and it is a labelling-protocol
  (cascade) effect more than a readout effect.

## Files
`extract_readouts.py` (raw pickle -> step readouts + gate), `evaluate.py` (alignment to `../cells`, gate vs stored
values, metrics), `report.py` (summaries, figures). Reproduce: run `extract_readouts.py` per cell (about 13 min
for a 3,000-answer cell on 7 workers), then `READOUT_STEPS=<dir> python evaluate.py && python report.py`.
