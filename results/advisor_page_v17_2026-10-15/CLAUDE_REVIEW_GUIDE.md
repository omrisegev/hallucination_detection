# Claude: review v17 and propose improvements

## What Omri asked for

Review this exact presentation for the advisor meeting on 15 October 2026. Understand the decisions and story below, then propose focused improvements. This is a request for suggestions, not permission to run new experiments, edit the agreed deck, submit cluster jobs, or publish a new version automatically.

Read in this order:

1. `README.md` — delivery, source and rebuild boundaries.
2. `SPEAKER_NOTES_HE.md` — slide-by-slide reasoning, source mapping and limits, in Hebrew.
3. `published_v17_2026-10-15.html` — the complete standalone presentation, 14 main slides and 7 appendices.
4. `CLAUDE_RESULT_HANDOFF.md` and `PRM_UPDATE.json` — the two result placeholders and the exact intended experiment.
5. `DATA.json`, `SOURCE_MANIFEST.json` and relevant CSVs — if a numerical or metric question arises. The display JSON is about 4.3 MB; inspect targeted sections rather than dumping the entire file into context.

The original v16 is retained at `results/advisor_page_v13_v14_2026-10-12/published_v16_2026-10-13.html`, commit `2cf1209cda253a50b53897fa14c99b4431cb0fcd`. New-to-old mapping is in `SLIDE_MAP.json`. The initial local v17 validation is in `VALIDATION.json`; its HTML hash is preserved by packaging.

## Agreed story and why this order

We have signals that detect different reasoning-error types. Label-free fusion is competitive, but expanding the feature bank does not give consistent improvement. Supervised LR improves with the additions. Ranking inside an answer and deciding which steps to flag are different problems. We diagnose this distinction, show external-benchmark evidence, then introduce same-model PRM fusion as the next question.

The audience must see a real question and step-labelled answer, then understand AUC, PRMScore and PRM, before interpreting results. Therefore the old `0 / 1 / 2 / 3` opening was replaced with background → metrics → the familiar methods table → nested-bank comparisons → leaderboard. Further method/error/decision analyses follow the comparison table rather than interrupting the introduction.

The bug slide moved to the appendix: it consumes scarce time and breaks the research narrative. If asked about it, the appendix gives the concrete reconstructed conditioning prompt, same fixed answer, first-token diagnostics and the actual correction. Teacher-forcing correction, self-generated-answer transfer, and 4B/8B robustness are separate questions. The old withdrawn-claims section is removed entirely.

The meeting budget is 31 minutes of main content plus 9 minutes of discussion. Suggestions must fit this budget. Do not restore every old diagnostic to the main story.

## User decisions to preserve

- Begin the method comparison with the familiar table from last week, using corrected inputs.
- Official PRMScore is primary; AUC explains ranking. Do not present AUC-only method comparisons.
- LR must use a matched flag rule when compared on PRMScore. The new deck computes Q80 PRMScore from saved fresh source-group OOF predictions, and its LR AUC/Top1 come from those same predictions. Historical LR TOP1 PRMScore is not a Q80 score.
- Bank expansion uses only `N14 ⊂ N16 ⊂ N25 ⊂ N43 ⊂ N56`. All three digit channels remain; expansions add 2, 9, 18 and 13 channels without removing any.
- Preserve the N25→N43 research question: which additions help LR and hurt L-SML/IU-PCR, and why? The bank contrast alone is not feature attribution.
- Show 4B and 8B, while saying their point estimates are close rather than claiming equivalence.
- Keep all 56 heatmap channels, but guide the advisors with three clear findings. Add only one extra visualization, the H-style selected-feature PRMScore profile. Do not add F/G to the main deck.
- Keep all six count/ranking diagnostic bars. Define exactly what ranking/count is preserved and what truth is supplied; gains are not additive.
- Explain every axis and legend. The old 6b measures Top1-on-error counts, not PRMScore, whole-answer accuracy or first-error accuracy, and is now in the appendix.
- Q80 is already one global threshold after per-answer normalization. The exact fixed `z>0.5` comparison and its identical flags-by-length curve were added. This score is not a probability. Best global cuts are label-selected diagnostics.
- Show both Socratic-PRMBench and Hard2Verify. Their official metrics differ; results are mixed. Target-fitted label-free transfer, PRMBench-trained LR transfer and frozen-weight transfer must remain distinct.

## Critical PRM fusion correction

Omri explicitly requires continuous L-SML of the PRM head and signals from **that same PRM checkpoint's own backbone**. The three fixed channels are `energy_level`, `hist_entropy_cusum_abs_series`, and `realized_drv`.

The earlier result around 69.38 used a fixed half-PRM / half-feature-mean composition. It is not L-SML. Six-channel results with Qwen3 features from another model are excluded. Do not relabel either as the requested experiment.

Slides 12–13 intentionally reserve the graph and result. The requested arms and update interface are specified in `CLAUDE_RESULT_HANDOFF.md`. Three-feature selection used labels; fitting the fusion without labels does not make the whole selection pipeline unsupervised. Preserve source-group separation and report failures/partial coverage explicitly.

The archived generalization specification describes fixed-average S3. Cluster runs may supply useful raw telemetry, but they only answer the revised question after the intended same-checkpoint L-SML evaluation. Omri said that if suitable results finish before the meeting, they can be shared; otherwise show a dated running/pending status. The current deck does not claim L-SML generalization has succeeded.

## Review requested

Please return a short Hebrew review with:

1. Up to five high-value proposed changes, each naming the v17 slide and old source, why it helps the story, and what exactly should change.
2. Any remaining metric/protocol ambiguity or inaccurate claim, with its evidence. Separate confirmed errors from presentation preferences.
3. One-sentence main message and spoken transition for any slide where the present wording is weak.
4. A prioritized cut list if the 31-minute content budget is still unrealistic.
5. Status of the two PRM placeholders: which exact compatible results exist, which are running or pending, and whether they actually implement L-SML.

Do not invent missing results or silently replace the intended fusion. Distinguish the Git-readable evidence from raw NPZ/PKL replay inputs catalogued outside Git.
