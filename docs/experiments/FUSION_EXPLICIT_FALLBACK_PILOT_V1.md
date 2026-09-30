# Explicit answer-local fusion fallback pilot v1

Date: 2026-09-07. Adaptive development, not untouched confirmation.

## Question and frozen scope

Does a declared second feature bank rescue failed Joint fits and improve
localization beyond the original Joint + IU fallback and matched simple fusion?
Fusion remains the method. Routing is a reliability component, not an oracle
choosing the best detector for an answer.

Use exactly the 58 answers of `localization-cached-v1-20260907` and frozen
`fusion_context_bank_pilot_v1` scores. Retain its 17 arms as unchanged anchors.
No new inference, feature fit, graph, lambda, grouping or threshold search.
The context-bank outcomes and labels have already been inspected in earlier
stages. Freezing this policy before this evaluation does not make it held out.

## Two policies, fixed before evaluating composites

The legacy K roster is used for routing. Expanded-K arms remain references.
An eligible Joint bank satisfies the existing fit contract: admissible groups,
convergence, multistart PASS, full-rank profiled global Jacobian and condition
at most 1e8. Reuse the audited frozen validity flag, and assert it is identical
for the lambda-zero, graph-0.1 and permuted-graph arms within each bank.

1. `single`: moment Joint if eligible, otherwise moment IU.
2. `dual`: moment Joint if eligible; otherwise context Joint if eligible;
   otherwise moment IU.

Apply each policy to Joint lambda 0, graph lambda 0.1 and the node-permuted
lambda-0.1 graph: six composite arms. No answer ID, length threshold, score,
gate output, correctness label or measured performance chooses the route.
Current fit counts suggest 43 moment, 13 context and two IU routes for `dual`;
these counts must be independently checked, not assumed by the implementation.

Two supporting controls use the exact `dual` bank route: equal aggregation
and IU on that bank. On the final IU route both use the moment bank. The
equivalent `single` controls are exactly the existing moment equal and IU
anchors; do not duplicate them. The routed equal/IU controls still depend on
Joint's eligibility calculation, so do not claim they avoid its fitting cost.

The result table has 25 arms: 17 inherited anchors plus eight composites.
The runner registers 30 paired contrasts before any composite is evaluated.
No context-first policy, learned router or label-based best-of-bank rule.

## Score and failure contract

Copy the selected source's window scores, official-step scores, native GMM
decision and peak unchanged. There is no cross-bank score averaging or
post-routing renormalization. This retains the existing pooled PRMB endpoint;
within-answer AUC is also mandatory because cross-answer score scales matter.
All scores retain the declared negative-entropy orientation anchor and
within-answer normalization. This is offline, one-answer fitting, not online
causal detection and not anchor-free learning.

Eligibility uses fit validity only. A valid fit predicting no error stays
selected. A selected source's failed readout remains a failed readout; it
does not trigger another fallback. If the final IU fit also fails, preserve
an explicit failed fit. Never convert failures into a successful clean answer.
Keep all pure Joint failures visible in the reference table. Report routed
coverage separately from pure Joint coverage and show each route's counts.

Native gate + peak is the primary ProcessBench prediction. Also retain the
previous stage's same parent-IU binary gate for every method as a diagnostic:
this gate is fitted on the same answer, but is not a newly validated no-error
rule. Gate-open uses the selected source's peak. Do not select between native
and fixed-IU decisions based on evaluation outcomes.

## Comparison and review

Primary evidence: `dual` versus `single`, original IU, and equal aggregation
under the same routing. Graph attribution also requires graph versus lambda
zero and the permuted graph. PRMB pairs use common valid IDs, with counts;
ProcessBench includes all 46 answers and counts invalid predictions as failures.
Report pooled and mean within-answer PRMB AUC, PB macro harmonic mean of clean
and exact-error accuracies, per-cell results, fit/readout coverage and route
counts. Use the existing paired source-group bootstrap, 1,000 draws, seed
2026090706, retaining undefined draws. Intervals are exploratory, unadjusted
for multiple comparisons and repeated adaptive development.

Persist a hash-bound manifest, score freeze, evaluation, contrasts, independent
review, and an HTML report. Routing code must not accept labels. Review the
truth table, no-error and readout-failure behavior, exact source inheritance,
direct label joins, all metrics using separate calculations, matched subgroup
comparisons, and representative explicit bootstrap reconstructions. Verify
all 17 inherited metric bundles exactly. The reviewer can reuse the previously
audited fits; it must state that no full refitting was performed.

One CPU process; scoring is cached composition, not fresh end-to-end latency.
Report this runtime limitation. Finish and review this bounded stage before
choosing the next research action. No publication winner is promised.
