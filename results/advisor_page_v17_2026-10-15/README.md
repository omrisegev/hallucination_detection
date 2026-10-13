# Advisor meeting v17 — 15 October 2026

Open `published_v17_2026-10-15.html`. It is a standalone local HTML file, with 14 main slides (31 minutes), 7 appendices, embedded charts and Hebrew speaker notes. No server or internet access is required.

- **Present** switches to one slide at a time; left/right arrows or Page Up/Down navigate.
- **הערות דובר** or **N** toggles the Hebrew notes.
- Slide 7 has two views: the complete 56-channel heatmap and the selected-feature PRMScore plot. Enlarge opens the complete heatmap.
- Print / PDF uses the browser's print dialog. Enable notes first to include them.

## Sources and mapping

Start with `CLAUDE_REVIEW_GUIDE.md` for the agreed story, user decisions and requested review. `SLIDE_MAP.json` and `SPEAKER_NOTES_HE.md` map every slide to the old v16 sections, its evidence, reading instructions and limitations. `DATA.json` holds the exact display data. `SOURCE_MANIFEST.json` records the numerical source hashes; `PUBLICATION_CONTENTS.json` records this Git reading delivery.

The pinned v16 source is commit `2cf1209cda253a50b53897fa14c99b4431cb0fcd`, HTML Git blob `9aaab92066600edb593b7050cd7f042acf7f1719`. Local source HTML is preserved under `docs/reviews/advisor_v16_walkthrough_20261013/`. CSVs in `sources/` were read from that pinned commit. The archived generalization specification is from `claude/prm-boosting-generalization-v1`; it specifies an older fixed-average experiment and does **not** establish results for the revised L-SML experiment.

New derived values use saved predictions only:

- LR official Q80 PRMScore is evaluated from `lr/OOF_N*.npz:LR_FULL`. Both AUC and PRMScore refer to this fresh source-group OOF fit, rather than combining historical LR with a different fit.
- Exact numeric threshold `z > 0.5` is evaluated on the same answer-normalized N16 scores as Q80. Its length-count curve uses those identical flags.
- STEP_INDEX in the H-style plot is newly derived with positive step order, per-answer normalization and source-fold donor Q80, using `>=` to match the gallery protocol. The other individual-feature PRMScores are preserved gallery exports.
- The error-type-only count diagnostic is the rounded 68.9 value preserved from v16; it has not been re-run here.

No model inference, model refitting or package installation was performed. The initial build was local. This reading delivery is prepared for publication on `codex/advisor-v17-2026-10-15`, based on the pinned v16 commit; it does not deploy a hosted website or change a Claude Artifact.

## Rebuild

From the repository root, with the existing project environment:

```sh
/Users/osegev/Desktop/hallucination_detection/.venv/bin/python results/advisor_page_v17_2026-10-15/derive_data.py
/Users/osegev/Desktop/hallucination_detection/.venv/bin/python results/advisor_page_v17_2026-10-15/build_v17.py
```

The first command requires the local saved arrays and evaluator modules listed in the source manifest; large NPZ/PKL inputs are not part of this Git delivery. Copies of the exact two evaluator/helper modules used for derivation are under `sources/runtime/` for inspection. The second command can rebuild from this branch: the frozen JSON, CSVs and correction assets are included, and the pinned v16 HTML is inherited from the base commit. On the original dirty checkout it falls back to the preserved local v16 snapshot. The final HTML has no external file dependencies.

`VALIDATION.json` records checks against the original complete local inputs and the exact delivered HTML hash. `validate_v17.py` also verifies those raw-input source hashes and therefore needs the original local artifacts. It is not a claim that a Git-only checkout can replay the numerical experiment.

## Claude result update

Read `CLAUDE_RESULT_HANDOFF.md`. Populate `PRM_UPDATE.json`, then run only `build_v17.py`. Do not replace the L-SML placeholders with fixed-weight average results or with features from another model.
