# Advisor deck vs competitors (2026-10-29)

Thread "הערכת מצב מול מתחרים וכיוון התזה". Sources of the published artifacts. No new experiment here: every number comes from the result branches below.

| Folder | Artifact | What |
|---|---|---|
| `competitor_deck_v2/` | https://claude.ai/artifact/T412tuQ6tEqTW33yMzYjvj | Focused deck (11 slides): TL;DR, PRISM, CROP, SLIDER (intro + result each), generalization (within-AUC and PRMScore heatmaps), Bracha's LR question. AUTO and TELE16 in every comparison. ARES removed. |
| `competitor_deck_v1/` | https://claude.ai/artifact/F97WWXk2cthq7qyrtDZa97 | First deck (15 slides), superseded by v2. |
| `artifact_sources/` | earlier scorecard / dashboard pages | Page sources as published. |
| `data/generalization_within_auc.csv` | | Within-answer AUC of PRM, AUTO, TELE16 standalone per benchmark x PRM (from bracha_lr_transfer_v1/RESULTS.csv). |

Number sources (branches):
- `claude/telemetry-only-vs-competitors-2026-10-28`: results/telemetry_only_vs_competitors_v1/TABLES.md (PRISM, CROP, SLIDER).
- `claude/bracha-lr-transfer-2026-10-28`: results/bracha_lr_transfer_v1/ (LR on PRM800K; generalization).
- `claude/ares-protocol-2026-10-29`: results/ares_protocol_v1/, results/generalization_prmscore_form_v1/.
- `claude/status-report-verification-2026-10-29`: verification of the 21 Oct status HTML (1,445 of 1,446 numbers pass, 0 errors).

Key decisions in the thread (Omri, 2026-10-29):
- ARES removed from the deck.
- SLIDER compared in plain accuracy (their metric), class-balanced LR threshold; GAP balanced accuracy as a footnote.
- CROP: every row uses CROP's label-calibrated threshold; only the step score differs (theirs trained on labels, ours label-free).
- PRISM: AUTO+GAP dominates PRISM's operating point on Qwen-7B/72B (higher TPR, same or lower FPR); not on PURE/RLHFlow; no CI possible.
- Correction found: CROP-slide AUROC values were replaced by TABLES.md values (e.g. ProcessBench PRM 0.883, not 0.954).

`rfull.js` / `rcheck2.js`: Playwright render checks of slides (run from the folder that contains `deck2/project`).
