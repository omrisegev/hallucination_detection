# Digit alternative probability: one frozen source-data experiment

2026-09-27. User explicitly authorized this new digit experiment and a dedicated
branch from bef0b1de3. Historical exclusion remains in force for production methods;
this is a research exception, not reinstatement of the old digit-dependent method.

## Question and fixed definition (before evaluation)

Can probability assigned to a competing digit provide useful localization without
depending on which token was supplied/generated? At each token position, take the
second-highest ORIGINAL probability among the ten verified ASCII single-digit
tokens (0..9). No conditioning/renormalization on digits, selected-token mask,
temperature transform, or threshold tuning. Aggregate by top-two mean per step.
Higher means more risk. No new inference, GPU work, fitting or feature sweep.

The saved top-50 distribution censors small digit probabilities. The operational
candidate is the second-highest digit probability PRESENT in top-50, or zero if
fewer than two digits are present. Where censored, true second-digit probability
lies between zero and the probability of the 50th token (up to stored precision).
Save both lower/upper step summaries and report certified peak stability, censor
coverage and an upper-endpoint sensitivity row. That endpoint is a diagnostic,
not another selected candidate and does not bound the accuracy of all completions.

## Population and comparison

All 13,769 matched source answers / 145,597 steps from the frozen corrected
localization v3 JOINED bundle: eight ProcessBench cells and PRMBench/Qwen3-8B.
Read raw inputs from the main checkout, never modify them. Verify tokenizer IDs,
source hashes, record/token/span alignment and raw labels against frozen labels.
The historical `inspect_cell.py` supports candidate-list generation caches, not
these step-telemetry rows; use the localization audit/schema checks instead.

Comparators: old digit disagreement top2, old digit token-clock innovation top1,
entropy top2 (matched readout), entropy top10 (established baseline), and uniform
random-step expectation. All use the same rows and labels. Primary endpoints:
PB macro exact first-error localization on erroneous answers (NOT F1 or PRMScore),
PRMB mean within-answer AUC on mixed-label answers (benchmark-defined estimand).
Return per-cell metrics, all coverage/failures, constant scores, peak ties and
tie-uniform exact-location diagnostics. Standard benchmark first-argmax is primary.
No gate or threshold is learned. No result implies complete-detector improvement.

Pre-evaluation correction: the first extraction was interrupted before any scores
or quality evaluation were saved. Historical digit innovation emits only at
provided-digit opportunities after token zero. Readout excludes inactive tokens;
steps with no opportunity retain the historical bank's zero storage value, with
availability counts reported. This auxiliary diagnostic is not a replay of the
trained historical fusion or its native mask-aware standalone decoder. The old
digit top2 primary comparator is all-token and has no such missing-evidence mask.

Pre-evaluation alignment audit: extraction stopped on an added disjoint-token-span
assertion. The existing producer deliberately maps character intersections and
uses assert_alignment(strict=False); 3/6,969 PRMB answers contain 10 one-token
boundary overlaps. Re-tokenizing all three from their original step text reproduces
every token ID and span exactly. SPAN_AUDIT.json records this evidence. Preserve the
frozen spans; require exact audited identity for any overlap and fail on any new
one. Report a secondary exclusion sensitivity for these three answers, not a new
benchmark or a silently repaired result. All PRMB labels replay against v3 before
resuming. No quality result had been computed when this implementation assertion
was corrected.

Paired source-question bootstrap, 5,000 replicates, seed 20260927. Two primary
candidate-minus-control contrasts (digit top2, entropy top2) on two endpoints:
98.75% intervals (Bonferroni family of four). Entropy top10 contrasts descriptive.
Check null by permutation of step scores within answer, fixed seed, report pooled
mean expected behavior over 20 permutations. No outcome-dependent sign reversal.

Evidence is promising for this standalone formulation if it improves the primary
matched controls consistently, with paired uncertainty. Complementary successes
alone do not prove independent errors or a fusion gain. This is exposed development
data. These caches are teacher-forced: the formula is decoding-independent but the
experiment cannot establish generated-answer quality or an empirical TF/generation
equivalence. No external benchmark selection.

## Verification and artifacts

Meaningful mechanism tests: hand examples, selected-token invariance, greedy old
signal degeneracy, missing-digit bounds, malformed inputs, causal innovation and
step pooling. Independently recompute token features with scalar predicates over
all tokens and evaluate saved step scores with separate pairwise AUC calculations.
Input hashes, code hashes, command, timing, scores, metrics and report are saved
under results/digit_alternative_probability_v1/. Fail loudly on contract mismatch.
