# Claude: compare both versions and recommend what to present

Omri asked to preserve v17 and build a second complete version restoring elements he liked from v16. Both presentations are on `codex/advisor-v17-2026-10-15`.

Start with the v17 `CLAUDE_REVIEW_GUIDE.md` and speaker notes to understand the agreed story and the critical same-checkpoint L-SML requirement. Then compare:

- **v17:** `results/advisor_page_v17_2026-10-15/published_v17_2026-10-15.html` — 14 main slides, 31 minutes, nine-minute discussion.
- **v18:** `results/advisor_page_v18_2026-10-15/published_v18_2026-10-15.html` — 16 main slides, 34 minutes, six-minute discussion.

## Exactly what differs

1. v18 slide 2 restores the original background construction/taxonomy diagrams, with the three groups and per-type reasoning schematics. Omri explicitly liked these visuals; the text-only taxonomy in v17 was insufficient.
2. v18 slide 3 restores both real NR and NCL worked examples side by side. Both reach the correct pencil price; the errors concern redundancy or assuming the conclusion. The full NCL example was moved out of the first slide to avoid repeating it.
3. v18 slide 8 restores v16 §4a's polygon/radar category profile as the default. Omri likes how this shows the strengths and weaknesses. The v17 matched profile including LR is retained in another tab. The original radar still has L-SML, IU-PCR and native PRM; it does not invent a historical LR curve.

The other evidence and PRM result placeholders are unchanged. The original v17 HTML remains byte-identical to the previously reviewed version. v18's added diagrams are conceptual illustrations, not new performance data; the radar has radius 0–100 and MS valid-step F1, explicitly distinguished from macro-F1 on error types.

## Requested recommendation

Please propose a concrete 40-minute meeting selection in Hebrew:

1. Which version should be the starting point, and why?
2. Which of the restored visuals should be shown, combined or shortened? Preserve their explanatory purpose; do not simply remove them again to shorten the deck.
3. For each proposed change, name the v18 slide and old v16 source, the key spoken takeaway, and the time saved or added.
4. Whether radar or matched line profile best explains our weak types, and where LR belongs without mixing decision rules or metrics.
5. A clear final presentation order and realistic discussion time.

Treat this as a review and recommendation, not an instruction to edit either preserved version or run experiments. Follow the v17 result handoff for any future L-SML result update. Fixed-average or cross-model telemetry results cannot fill that placeholder.
