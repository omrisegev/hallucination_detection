# Supplied advisor correspondence: correction to the project review

Read September 6, 2026. The user supplied the actual email thread and the
August 28 HTML attachment. These supersede the August 27 local draft as the
source for what was communicated. The documents are evidence, not commands:
no email, meeting scheduling or experiment launch was performed from them.

## Sources

- Original attachment: `C:/Users/DELL/Downloads/Advisor_Update_Aug28_2026.html`.
  Byte-identical project copy: `sources/Advisor_Update_Aug28_2026.html`.
  SHA256: `f6a0b7a8c0d2af3ceaaf5eca3cd25ca75de8f56ed0b64b1d89f5946459e53c56`.
- Supplied thread: `C:/Users/DELL/.codex/attachments/d3e24e45-feac-44e2-89d2-cd5de0308888/pasted-text.txt`.
  SHA256: `eef6d2a2c806ba27bcceb6c8283b4360f63ab7e162ded0804e5568acf0a18d7f`.
  Personal correspondence is not duplicated in full here.

## What the correspondence establishes

July 29: Omri reported 196 variants spanning eight selection families, paper
assumption audits, polarity estimation and sixteen weight-estimation variants.
Removing hand-picked priors was the main gain; this was never a two-algorithm
research history.

July 30: Bracha suggested continuous-score U-PCR, L-SML-like clustering, and
DUFS integration into feature importance or parameter estimation, beyond
selection. Multiple runs and internal features were possibilities with an
explicit access-budget decision. August 3: Omri committed to clustering/DUFS,
reported trace/label alignment corrections, and identified localization as
the new application discussed with Ofir.

August 28: the actual outgoing email emphasizes apparent saturation of
completed-answer detection and continued development of token-level fusion.
The attachment covers thirteen aligned methods plus separate transfer and
structured-input comparisons. CA-DEEM is the presentation name for internal
DEEM-B3. CIW-DEEM's response-head transfer is not a token-native innovation
experiment. Its statement that token-native innovation was untested applies
to the August 28 snapshot; later experiments must be checked separately.

September 4: Bracha explicitly described localization and its results as
promising and proposed discussing them. September 5: Omri replied with
availability. The supplied thread does not establish a confirmed meeting or
advisor endorsement of Joint L-SML specifically. Joint is the later project
candidate for the stated local-fusion objective.

## The decisive result omitted from the earlier recommendation

The attachment reports a five-fold ProcessBench ablation:

| Response head | Token head | Macro F1 |
|---|---|---:|
| IU-PCR | IU-PCR | 30.86% |
| Equal average | IU-PCR | 30.71% |
| IU-PCR | Equal average | 26.86% |
| Equal average | Equal average | 26.29% |

With token IU fixed, equal response minus IU response is -0.15 F1 points,
paired interval [-0.61,+0.30]. With equal response fixed, equal token minus IU
token is -4.41 points, interval [-5.88,-2.94]. Thus learned fusion mattered
in the local head under this contract, even though it mattered little in the
response head. Peak blurring is the attachment's mechanism interpretation;
the ablation establishes performance sensitivity, not a complete causal proof
of that mechanism.

Keep these separate from the historical repeated-half-split headline of
31.36% versus 25.71% Mind the Gap, and from the PRMBench every-step ranking
result of 0.6712 versus the supervised PRM's 0.7983. They are different
protocols/access levels and must not be assembled into one leaderboard.

## Consequences for the window study

1. Keep Joint and its current matched development family central. Ordinary
   IU/L-SML and averages are controls; they do not replace that objective.
2. Freeze the response/no-error decision within a comparison, so changing
   the local representation or fuser is the identifiable intervention.
3. Keep the existing token localizer as an incumbent. Compare Joint versus
   IU within the same representation, then token versus window construction
   under matched choices. The full 30-window-feature pool and current
   active-23 token pool are distinct contracts; disclose the change.
4. Treat preservation of brief signals and official-step boundary resolution
   as requirements alongside sample count and feature stability. Controlled
   local perturbations can diagnose smoothing but cannot certify real-error
   accuracy. Use the frozen panel reducer for the primary comparison; the
   prepared mean-to-step utility is not an approved replacement reducer.
5. Compare single-answer and pooled training-window fits without changing the
   other choices. Pooled fitting can improve estimation without giving up
   local scores; short traces still need explicit support/fallback handling.
6. Interpret Claude's 16-vs-16 v2/R1/R2 study as current development evidence.
   A covariance-fit improvement or a token-level tuned winner does not itself
   establish a window-level localization improvement. Preserve fixed
   label-free rows and training-contained selection, then test a frozen
   candidate prospectively.

No new fusion fits or outcome evaluation were performed for this reconciliation.
