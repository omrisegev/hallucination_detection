# View selection stage 2 pre-registration: telemetry-only L-SML, exhaustive subset search (2026-10-26)

Omri, 2026-10-26 (Hebrew, verbatim): "אני רוצה לעשות ניסוי דומה שבו אני פשוט מריץ את אותו אלגוריתם (כמו במאמר) רק
בלי הprm בכלל אלא רק על בסיס הטלמטריה שלו. כלומר l-sml על subsets שונים שיכולים להכיל את כל הקומבינציות החל מ4-16,
ושנמצא מי מצליח גם על prmbench וגם על socratic. ... בסוף אשמח להשוואה בין הsubsets שנבחרו בשתי הריצות שלנו. אם יש
subset שעובד בשתי המטרות - עם הprm ובלעדיו - זה יהיה מעולה. אשמח לראות לבסוף את התוצאות האלה בגרפים".

Translation:
- Run the same algorithm (as in the paper), but without the PRM at all, on its telemetry only.
- That is, L-SML on every subset of 4 to 16 channels, to find which ones succeed on both PRMBench and Socratic.
- At the end, compare the subsets selected in the two runs. A subset that works for both goals, with the PRM and
  without it, would be ideal.
- Show the results in charts, not only in words.

Background:
- Stage 1: `PRM_VIEW_SELECTION_STAGE1_PREREG_2026-10-25.md`; results in Step 479.
- Proposed stage 1b, not run: `PRM_VIEW_SELECTION_STAGE1B_PROPOSAL_2026-10-26.md`.

## What changes relative to stage 1

Everything else (cells, pool, answer-z, GAP votes, group detection, SML, DS-EM, continuous scoring, metric) is
unchanged.

1. **No PRM anywhere.** The PRM is not a view, not a group, not an anchor and not a tie-breaker.
2. **Orientation.** Channels keep their stored orientation, i.e. the frozen bank definition used by TELE_N16 and
   CHAIN (`ABL5_CHANNELS_*.npz` / `SCORES_*_CHANNELS.npz`); none is flipped.
   - This is the paper's assumption that every classifier beats random in its stated direction (Jaffe 2016, Lemma 4).
   - The orientation is not re-derived here. As a diagnostic, the number of channels that anti-correlate with the
     plain mean of the 16 channels is reported per cell.
3. **Between-group label switching** is resolved by the paper's assumption (iii): Y is flipped if fewer than half of
   the virtual views have ψ + η > 1. Weights are clipped at 0 as before.
4. **Tie of a virtual vote** (weighted group vote exactly 0): it takes the GAP vote of the plain mean of S, the
   PRM-free analogue of stage 1's anchor.
5. **Minimum structure:** K ≥ 3 telemetry groups in every cell, as in stage 1. The between level then has at least 3
   conditionally independent views, which is enough for a binary latent-class model (Kruskal). Subsets with
   4 ≤ |S| ≤ 16: 64,839.
6. **Fused score:** Σ_g w_g · answer_z(Z_g w_g), then GAP.

## Comparators per cell

- **The plain mean of the same channels** (primary), then GAP.
- **The PRM alone** under the same GAP rule (secondary).
- **Context:** TELE_N16 (continuous L-SML on all 16) and MEAN_N16.

## Success

- **Primary:** valid in all 10 cells and fused > the same-set mean in all 10. The question: does label-free
  weighting of telemetry beat its plain average on both benchmarks?
- **Secondary:** fused > max(PRM alone, same-set mean) in all 10. Telemetry alone beating the PRM; stage 0 makes
  this unlikely on Socratic.

## Analyses (pre-specified)

1. Counts for the primary and the secondary; cells-won histogram; the min-margin distribution; per-cell win rates;
   counts by size.
2. Channel frequency among primary successes, or among the top 1% by min-margin if there are none, against the
   frequency among valid subsets.
3. Cross-benchmark selection (PRMBench → Socratic and back) and leave-one-PRM-out, on the primary margin.
4. **Comparison with stage 1, over subsets valid in both runs:**
   - Spearman correlation of the two min-margins (stage 1: fused − max(PRM, mean); stage 2: fused − mean);
   - overlap of the two top-1% sets (Jaccard), and the subsets in both;
   - the **joint subset**: the one maximizing the minimum of all 20 margins (10 cells × 2 runs);
   - the number of subsets that satisfy stage 1's goal and stage 2's primary at the same time, reported for 10
     cells and for the 8 cells without Math-Shepherd. The 8-cell version is pre-registered here as a secondary
     analysis; the reason is stage 1's position diagnosis of Math-Shepherd;
   - the channel frequency in both runs side by side.
5. Every subset reported by name is recomputed with the evaluator's own loops.

## Charts (pre-specified; PNG files in `results/prm_view_selection_v1/stage2/figures/`)

1. Cells-won histogram, stage 1 against stage 2.
2. Per-cell win rate, both runs (10 cells).
3. Scatter of the stage-1 against the stage-2 min-margin over common valid subsets, with the top subsets and the
   joint subset marked.
4. Channel frequency in the top 1% of each run against the base rate.
5. The per-cell margins of the joint subset and of each run's best subset, in both runs.

## Predictions, before the run

- **T1.** Telemetry-only L-SML beats the same-set mean on PRMBench for most subsets.
  - On Socratic it rarely does.
  - Stage 0 found the label-free estimators rank channels wrongly on Socratic RLHFlow and Math-Shepherd. Primary
    success in all 10 cells is rare or absent.
- **T2.** No telemetry-only subset beats the PRM alone on the strong PRMs on Socratic (secondary ≈ 0).
- **T3.** The stage-1 and stage-2 min-margins correlate positively but weakly. Subsets good with the PRM keep the PRM
  dominant on Socratic, which stage 2 cannot do.

## Labels

Labels enter only the metric, i.e. the selection. As in stage 1, the PRMBench pool excludes the label-known control
answers. This is in-sample development; held-out evidence comes from the cross-checks and from later datasets.

## Not in scope

GPU; new inference; any change to stage-1 or frozen results; stage 1b.
