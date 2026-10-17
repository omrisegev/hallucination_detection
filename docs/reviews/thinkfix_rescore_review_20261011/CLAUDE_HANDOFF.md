# Revised PRMBench presentation — 11 October 2026

The revised meeting presentation is [advisor_review.html](advisor_review.html).
The importable copy for claude.ai is [advisor_review_claude.html](advisor_review_claude.html); it has the same content and figures, with supporting-file links pointing to this GitHub branch.
Read [REVIEW.md](REVIEW.md) for the audit, [VALIDATION.json](VALIDATION.json) for checks and limits, and [PRESENTATION_DATA.json](PRESENTATION_DATA.json) for the plotted values.

## Requested emphasis

The October 8 meeting had little time for PRMBench. Present the complete corrected PRMBench results, with Continuous L-SML and IU-PCR as the main methods. Compare them visually with plain averaging, DS-filter averaging, supervised LR and PRM-7B. Keep the newer tail-vote pipelines as supporting comparisons. The prefix correction belongs in the appendix.

Masking the first answer token is a separate ablation, not a necessary part of the empty-think-block correction. On N16, masking changes Continuous L-SML AUC by -0.000749 and IU-PCR by -0.000135. The bank explorer includes corrected unmasked results separately. Keep the campaign's masked input clearly labeled; no preprocessing policy has been promoted.

## Evidence and claim boundaries

- Reviewed research commit: `080877eb9a8369341d647f4115a3b475c040b46e`.
- N16 masked within-answer AUC: Continuous L-SML 0.779193; IU-PCR 0.780459; DS-filter average 0.780080; plain average 0.770797. These were independently checked on saved scores.
- Supervised LR 0.801489 is a reported reference, not independently refitted here. PRM-7B AUC 0.801 is an artifact-reported rounded value; its official PRMScore comparator comes from the saved manifest.
- Per-type N16 AUCs were reconstructed from saved scores. Their weighted aggregate matches overall AUC within 1e-12. Sensitivity category comparisons cover only two subtypes and must retain that qualification.
- Keep the unproved self-generation equivalence, restricted-LR selection leakage, stale resume identity and old-input entropy gate findings visible in supporting material.
- No new model inference, fusion refit, experiment-code changes or frozen-result changes were made for this presentation.

## Update the original Claude Artifact

Original URL: https://claude.ai/artifact/NQNTYpFqVyx6xRnRS4NENV

Use `advisor_review_claude.html` as the revised source in the original artifact's authoring conversation. Review it against this handoff and the evidence files before replacing the source. Preserve the corrected numbers, method emphasis, metric definitions and claim boundaries. Do not treat an unchanged viewer page or a new duplicate as a verified update of the original URL.

Published to GitHub in commit `f05f10b808a533ed776b34f17a08cc52f7e6ee16`. The school-profile Claude account had viewer access; the personal Chrome profile held the author account. From the original authoring session, Claude published the exact committed import copy as **version 9 at the original URL**, retaining version 8 in history. Codex independently reloaded the public URL and verified the revised title, method emphasis, figures and displayed values. Claude reported that its downloaded live HTML matched the committed file byte for byte. See [publication record](PUBLICATION.json).

## Rebuild locally

From the main checkout, use its existing virtual environment:

```sh
MPLCONFIGDIR=/private/tmp/thinkfix-mpl-cache .venv/bin/python docs/reviews/thinkfix_rescore_review_20261011/build_presentation.py
```

The main HTML embeds its figures and data and requires no CDN. Supporting audit files are included in this directory. The Claude import copy changes only supporting hyperlinks; regenerate that copy after rebuilding the main HTML.
