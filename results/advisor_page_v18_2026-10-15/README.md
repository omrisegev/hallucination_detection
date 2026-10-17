# Advisor meeting: v18, with the restored visuals

This is a second complete standalone presentation alongside v17. Open `published_v18_2026-10-15.html` locally. It includes 16 main slides, seven appendices, Hebrew speaker notes and slide navigation. Main content: 34 minutes; discussion: six minutes.

Omri requested three elements from v16 back in the main story:

1. The original PRMBench construction and nine-category diagrams: Simplicity (NR/NCL), Soundness (ES/SC/DC/CI), Sensitivity (PS/DR/MS). These form new slide 2, after the task introduction.
2. The two actual NR/NCL pencil-price examples, openly displayed side by side on new slide 3. They retain original step labels and source IDs. The introductory slide no longer duplicates the full NCL answer.
3. The original category-profile polygon/radar plot, restored as the default view of the error-profile slide (now slide 8). The v17 matched line profile, including LR, remains available as a second view.

All other findings, metric definitions, fixed-threshold diagnostics and same-checkpoint L-SML placeholders come from the unchanged v17. No score, fit or inference was recomputed. The radar reuses v16's three original curves and flag rules; the separate LR profile is explicitly under Q80.

## Version choice for Claude

Read `CLAUDE_COMPARE_V17_V18.md`. Compare the two complete decks and recommend what to present, respecting Omri's explicit preference for these visuals. v17 keeps the lean 31-minute story and nine-minute discussion; v18 uses three additional minutes for the restored taxonomy and paired examples.

The original v17 is published in `results/advisor_page_v17_2026-10-15/` on the same branch `codex/advisor-v17-2026-10-15`. Its HTML hash remains `0117d91120c6e2c96f23dfd9f26c87fc1f5835d116318ebe79a9a7404f03c522`.

## Rebuild

From the repository root, use the existing project environment:

```sh
/Users/osegev/Desktop/hallucination_detection/.venv/bin/python results/advisor_page_v18_2026-10-15/build_v18.py
```

The builder reads the preserved v17 HTML/mapping and pinned v16 HTML inherited on this branch. On the original checkout it can use the local pinned v16 snapshot. It performs no numerical analysis. `BUILD_RECEIPT.json`, `SOURCE_MANIFEST.json`, `SLIDE_MAP.json` and `SPEAKER_NOTES_HE.md` identify exactly what was reused and added. The final HTML requires no other files to view.
