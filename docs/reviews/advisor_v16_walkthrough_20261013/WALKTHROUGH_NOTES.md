# Advisor meeting walkthrough — 13 October 2026

Source: remote branch `claude/advisor-v13-v14-2026-10-12`, commit
`2cf1209cda253a50b53897fa14c99b4431cb0fcd`, published v16.

## Section 0 — agreed edits

- Add the absolute-AUC and gain-over-plain-average graph using only digit banks.
- Show `N14 ⊂ N16 ⊂ N25 ⊂ N43 ⊂ N56`; additions are 2, 9, 18 and 13 channels.
  The exact additions are recorded in `BANK_TRANSITIONS.json`.
- LR improves at every expansion. Label-free fusion does not improve consistently
  and declines in the larger banks; do not describe this as no deterioration.
- L-SML keeps a larger margin than IU-PCR over the plain average in the larger
  banks. The growing margin mainly reflects the plain average deteriorating more.
- These are saved point estimates. Paired confidence intervals over the same
  source groups remain desirable for the method and bank contrasts.

## Research question to carry forward — requested by Omri

Which added features improve LR while degrading L-SML, and why? Focus on
N25 → N43, which adds eighteen channels without removing any existing ones.

Branch-reported corrected AUC changes:

- LR: 0.807114 → 0.815883, +0.88 AUC percentage points.
- L-SML: 0.774004 → 0.761555, −1.24 points.
- IU-PCR: 0.772980 → 0.754407, −1.86 points.

The divergence is an observation, not a feature attribution or a demonstrated
mechanism. A future investigation should distinguish useful individual features
and supervised contrasts from changes in label-free orientation, weighting,
redundancy and grouping. LR coefficients alone do not establish importance in
a collinear bank. Any new ablation should specify its protocol before execution.
This question is also recorded in section 10 of the local HTML draft.

## Section 1 — next discussion

Review how the think-block correction changes absolute AUC and whether it changes
the main conclusions. The original v16 section is retained pending discussion.

## Delivery state

`advisor_walkthrough_draft.html` contains the agreed edits to sections 0 and 10.
`source_v16_2cf1209c.html` matches GitHub blob
`9aaab92066600edb593b7050cd7f042acf7f1719` exactly.
`build_digit_banks.py` reproduces the figure from saved v11 evidence;
`build_walkthrough.py` reproduces the local draft from the pinned HTML.
This is a local draft; it has not been committed, pushed or published to Claude.
No new model scoring, fitting or ablation was run.
