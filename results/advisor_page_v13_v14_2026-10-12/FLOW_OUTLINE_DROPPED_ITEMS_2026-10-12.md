# What the flow outline left out or moved to the appendix (2026-10-12)

Compared against: the 8 Oct plan (`docs/meetings/advisor_meeting_2026-10-08/README_HE.md`, sections 1-6),
the task list `PAGE_REVIEW_TASKS_2026-10-11.md` (tasks 1-13), the 8 Oct page, and page v12.
Outline: `FLOW_OUTLINE_2026-10-15.md`. Numbers are copied from v12 or project memory; nothing new was run.

## The three items Omri asked about

| Item | Where it was in the plan | Where it is in the outline | What exists (no new runs) | Proposal |
|---|---|---|---|---|
| External benchmarks (Socratic, Hard2Verify) | Plan §2 "generalization". Socratic first, because it uses the PRMBench taxonomy | Appendix A only | Locked L-SML, official metrics. Socratic (2,995): L-SML 0.628, equal weights 0.600, CT7 0.566, step index 0.580. Hard2Verify (200): L-SML 0.421, but step index alone 0.483 is higher than all our methods. 4B: 0.631 / 0.433. Caveat: the locked weights were fitted before the think-block fix and cannot be refitted. | New section "Does it transfer?" after section 3. One graph: per benchmark, L-SML vs equal weights vs CT7 vs step index (8B and 4B). Socratic is the main result; Hard2Verify is shown with the step-index warning. Published PRM-7B on Socratic (68.0) as context only. |
| Run without teacher forcing (the model's own answers) | Plan §4 "teacher-forcing comparison" | Dropped (only the withdrawn "equivalence" claim in the v12 appendix) | B16, after the fix: own answers 0.757 vs teacher-forced ProcessBench 0.742, difference +0.015 [−0.021, +0.052]. The GRP arm is not stable (PB reference 0.720 → 0.606). Exact first-error hit drops (26% vs 44%, pre-fix). 70% of natural errors are types that PRMBench does not insert. This is on GSM8K/MATH answers, not on PRMBench. | One short box in section 1: "the signal also works on the model's own answers; we do not claim equivalence". No graph, because it is not PRMBench. |
| Qwen3-4B | Plan §2 "generalization" | One bullet in section 1, tables in appendix A | N16 AUC 0.779 → 0.775; all 10 banks change by less than 0.01. PRMScore 65.4 → 65.0; category profile the same. LR drops in all 10 banks (−0.003 to −0.010). | Keep the bullet in section 1. Use the per-bank Δ graph (already in v12) in section 4: "LR depends on the model more than our methods". |

## Other items from the plan that the outline left out or moved

| Item | Source | In the outline | Proposal |
|---|---|---|---|
| How PRMBench was built and why Simplicity is hard | Task 9, plan §0b ("critical") | Only the NR/NCL examples, collapsed | Add 3 lines at the top of section 3: errors inserted by GPT-4o into correct PRM800K paths; NR/NCL can keep the final answer correct; the "all ~40 literature rows" claim is withdrawn (10 of 15 rows). |
| LR: where each one is right (four-way top1 by error type) | Task 7, Omri's request | Appendix B | Move into section 4 as graph 2. It links to the channel × type heatmap: we win on redundancy, domain, circular; LR wins on confidence, deception. |
| Conditional correlations (error vs valid steps) | Plan §3, task 8 | Appendix C | Keep in appendix, with one bullet in section 4: features correlate more on error steps (|r| 0.30 vs 0.24), so the independence assumption holds only roughly. |
| Estimated vs true accuracy and error rate (DS/MoM) | Plan §3, task 8 | One line in section 6 | No change. |
| Mind the Gap (the only label-free competitor with the same access) | 8 Oct page, competitors table | Not in the outline and not in v12 | The 8 Oct PRMBench number (AUC 0.714) is on the old input and was not recomputed after the fix. Recomputing needs a new run (freeze). Proposal: leave it out, or one line "pre-fix context". |
| Pipeline variants (tail votes, _cc/_ga, HEM/TCI) | 8 Oct page | Columns of the appendix matrix | No change. |
| Digits: N13 vs N16 | Plan §6 item 7 | Visible in the section 0 heatmap | Add one bullet in section 0 if wanted: digit channels add about +0.01 AUC (N13 0.766 → N16 0.779). |
| Q80 provenance; registered rules | Tasks 5, 5a | Appendix E; rules in section 6 "what we tried" | No change. |
| Cost: one pass, no training, vs a trained PRM | Plan §6 item 4 | Not in the outline | One sentence in section 2. No runtime numbers exist; do not invent them. |
| Our channels from the PRM's own base model, fused with the PRM | Task list, "not for Thursday" (Omri, 2026-10-11) | Not in the outline | Keep out of the main story. Optional one line in the closing "next directions" box. |
| ProcessBench | 8 Oct page | Removed (Omri, 2026-10-12) | No change. |

## Decisions Omri gave on 2026-10-12 (to apply in the next outline version)

- Old 8 Oct values: appendix only. The message in section 0 is "the picture did not change" plus the next steps.
- Channel × error-type heatmap: main graph in section 3 (which features catch which error type).
- ProcessBench: out of the page this week.
- The next outline version is a separate file, so the two versions can be compared.
