# Notes for Codex: advisor page after v10 (2026-10-11)

Read with `docs/meetings/advisor_meeting_2026-10-08/PAGE_REVIEW_TASKS_2026-10-11.md`
(branch `codex/lsml-external-generalization-v1` @ 8996379). The page builder is on a
different branch: `claude/thinkfix-rescore-v1`, `docs/reviews/thinkfix_rescore_review_20261011/build_presentation.py`.

## 1. Do not lose the v10 additions (blocking)

The tasks doc lists "method x bank tables with mean rows/columns" and "the PRMBench-style
leaderboard table" under "already good, keep". These are NOT made by `build_presentation.py`.
Claude added them to v10 with a post-build patch:

- Script: `results/advisor_page_v6_2026-10-11/add_advisor_views.py` on branch
  `claude/advisor-page-v6-official-prmscore` (v10 source beside it: `advisor_review_v10.html`).
- Usage: `python3 add_advisor_views.py <repo_root> <official_prmscore_dir> <in_html> <out_html>`.
- It adds: (a) "All banks x all methods" within-AUC table (bank mean, method mean, LR column) and
  Q80 PRMScore table; (b) section `#leaderboard` "05b Where we sit on the PRMBench leaderboard":
  radar (NR, NCL, MS, DR, PS, CI, DC, SC, ES) plus a Table 3-style table with 8 paper rows.
- Leaderboard numbers come from `official_prmscore_v1.py` and `official_prmscore_thinkfix_mask1/`,
  which are on branch `claude/advisor-page-v6-official-prmscore`
  (`results/advisor_page_v6_2026-10-11/`), not on `claude/thinkfix-rescore-v1`.

If you rebuild with `build_presentation.py`, either port these views into the builder (preferred),
or run the patch after the build. Otherwise the next publish drops them.

## 2. Task 3 (top step is an error) can be filled now from an existing file

`results/thinkfix_rescore_v1/PRMBENCH_COMPARISON.csv`, metric `prmb_top1_on_error`, N16
(old replay -> thinkfix_mask1):

| Scorer | Old | Now |
|---|---|---|
| LSML_CONT | 0.622 | 0.599 |
| IU_PCR | 0.609 | 0.595 |
| MEAN_MOM_ga | 0.625 | 0.606 |
| DS_FILTER_EQ | 0.611 | 0.600 |
| ALL_EQ | 0.611 | 0.588 |
| LR (supervised) | 0.627 | 0.579 |
| Qwen2.5-Math-PRM-7B | 0.578 | 0.578 |
| Step index alone | 0.189 | 0.189 |

The claim still holds, but the margin over PRM-7B fell from about +0.04 to about +0.02.
No CI exists yet; add a paired bootstrap by source group (task 10) before calling it a win.

## 3. Consistency items in v10 that the tasks doc does not list

- **Sensitivity convention.** The PRMBench paper averages Sensitivity over PS, DR and MS.
  The review text uses a partial Sensitivity without MS (0.598; 0.547 elsewhere). The leaderboard
  table uses the paper convention (L-SML 69.3, IU-PCR 69.5, PRM-7B 75.5). Use one convention
  in the text, and name it.
- **PRM-7B AUC.** The page shows "≈0.801". The exact value is 0.80118
  (`PRMBENCH_COMPARISON.csv`, `prmbench_panel_b`). Show 0.801 without "≈".
- **Hard2Verify.** The external section does not say that step index alone (0.483 Balanced F1,
  ANSWER_Z_Q80) is higher than all methods on Hard2Verify after the fix (best panel cell 0.433,
  N16 TCI_ONLY_ga; locked L-SML 0.421). On Socratic the methods win (step index 0.580, locked
  L-SML 0.628). Add one line: the issue is Hard2Verify only (200 answers). Sources:
  `external/EXTERNAL_PANEL_COMPARISON.csv`, `external/EXTERNAL_LSML_COMPARISON.csv`.
- **Footer.** v10 still has "Local replacement artifact; published Claude artifact and research
  results unchanged" (task 6). Confirmed in the live source.
- **Rule names.** Use the protocol names (`ANSWER_Z_Q80`, `R2_COUNT`, `ENTROPY_GATE_Q80`).
  `results/decision_rule_v1/SUMMARY.md` uses R0/R1/R2/R1F, and its "R2" (R1 count with
  within-answer placement) is not the same definition as protocol `R2_COUNT` (count from Q80 on
  the raw-channel mean). Do not mix the two in task 5a.

## 4. Length

The tasks doc asks for a smaller page. If space is needed, keep the N16 row and the bank-mean /
method-mean summaries in the body, and move the full bank x method matrices to the appendix.
Keep the radar and the leaderboard table in the body.
