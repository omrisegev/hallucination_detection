# Independent result replay

Status: **PASS**

- Replayed all four localization methods from saved step scores.
- Replayed five answer-level scores in all 24 historical cells.
- Coverage is complete, no fitting fallback occurred, and every learned vector has 17 finite coefficients.
- All reported 97.5% intervals are present and finite; the HTML has the required comparisons and clean UTF-8 text.

## Findings audit

- The added inputs do not improve the frozen one-answer localization route.
- A small complete-answer signal is visible, especially with equal weights.
- The combined run cannot tell whether selected-token surprisal, tail mass, or both caused that signal.
- Current IU-PCR weighting does not turn the signal into the strongest method; Joint Shrinkage remains weak.
- No automatic promotion threshold was used.
