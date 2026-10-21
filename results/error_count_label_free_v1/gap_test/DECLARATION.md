# Gap significance test before cutting (declared 2026-10-21, BEFORE any labelled metric of this study)

Written after reading code/structure only (score shapes, step counts n, the gold JSON field names). No labelled
metric of this study (including the parity values) had been computed when this file was written. Rules below are
not changed after seeing results; anything added later is marked POST HOC in the report.

## Question (Omri)
(1) Can one dataset-wide minimal gap say "this answer has an erroneous part and a correct part"?
(2) Can we test whether the answer's sorted scores really split into two classes, and cut at the largest gap only
if they do? Is GAP's success predicted by the absolute size of the chosen gap or by its size relative to the
answer's other gaps?

## Arms (ranking fixed; only the per-answer count changes)
PRMBench (6,969 answers, PRMScore on 6,211 noncontrols; classification == 'correct' are synthetic controls):
- B_CHAIN_N16: answer_z(CHAIN_N16) from BOOST_SCORES.npz (Qwen2.5-Math-PRM-7B boosting refit).
- T_LSML_CONT: thinkfix_mask1 bank_N16 S__LSML_CONT (already answer-z; used as is, like run.py).
- T_ALL_EQ: thinkfix_mask1 bank_N16 S__ALL_EQ_ORIENTED (same).
Socratic-PRMBench (2,995 answers), qwen_prm and qwen_prm_72b SCORES (signed): CHAIN_N16 and TELE_N16. Scores live on
non-empty steps; answer-z over the answer's non-empty steps; empty steps are always predicted incorrect (as in
eval_socratic_ablation5.py). Official metric = PRMScore from confusion()/summary(), ×100 here.

## Per-answer label-free quantities (on answer-z z of the n scored steps)
s = z sorted descending; all gaps g_j = s_j - s_{j+1}, j = 1..n-1; window K = min(ceil(n/2), n-1);
window gaps d_k, k = 1..K. G1 = max d; k* = first argmax + 1 (exactly run.py gap_count); G2 = second-largest d;
Gmed = median of all n-1 gaps g; R1 = G1/max(Gmed,1e-12); R2 = G1/max(G2,1e-12).
Null (per n): M = 50,000 draws of n iid N(0,1), each standardized (mean 0, population sd 1), same statistics.
p_null = (1 + #{G1_null >= G1}) / (M + 1); p_R1 and p_R2 likewise; q95_G1(n), q95_R1(n), q95_R2(n) = null 95th pct.
n <= 2 is untestable (n = 2 standardizes to +-1 always): such answers count as "not passing" and use the fallback
(3 PRMBench answers, 4 Socratic answers).

## Label-free policies (primary family, 2 tests x 3 fallbacks)
Test T1: pass iff n >= 3 and p_null < 0.05.   Test T2: pass iff n >= 3 and R1 > q95_R1(n).
Passing answers: flag the top k* steps (GAP). Failing answers: fallback
F0 = flag nothing ("no error"); FQ = the number of steps Q80 flags in that answer (placed by the same ranking);
F1 = flag the top 1 step.  Policies P1-F0, P1-FQ, P1-F1, P2-F0, P2-FQ, P2-F1.
A-priori primary policy (literal reading of the idea: no significant split -> say no error): P1-F0.
"Best policy" = highest PRMScore among the six per arm; this choice USES labels and is labelled as such.
Secondary (declared, not eligible for "best"): T3: R2 > q95_R2(n), with the same three fallbacks.
Label-free dataset-wide threshold (Omri's question 1 without labels): tau_null = 95th percentile of the null G1
pooled over the benchmark's n distribution (every n >= 3 answer of the benchmark, PRMBench including controls,
weighted by its number of answers); policy DW-F0/FQ/F1: G1 >= tau_null -> top k*, else fallback.

## References
Q80 (PRMBench: z > global 80th pct of answer-z; Socratic: az_q80 '>=' as in its official eval), GAP (always cut),
constant k=2 (PRMBench construction prior), ORACLE (true count, LABEL ceiling).
LABEL reference (Omri's question 1 with labels): tau swept over the 0,2,..,98th percentiles of G1 among scored
answers with n >= 3 (PRMBench noncontrols), each with F0/FQ/F1; PRMScore curve and best tau. Label-chosen.

## Metrics
PRMScore (PRMBench official pooled macro-F1 on noncontrols; Socratic official), count MAE, share of answers that
pass (noncontrols / all), zero-count accuracy; per-category PRMScore for P1-F0 and for the best policy.
Paired source-group bootstrap (PRMBench, groups from load_population, 2,000 draws, seed 20261021, as analysis.py)
for P1-F0 and for the best policy vs GAP and vs Q80. Socratic: answer-level paired bootstrap, 2,000 draws.

## Diagnostics (labels; GAP's own flags)
Population: answers with n >= 3 (PRMBench noncontrols). SUCCESS_SET = GAP flag set == true error set on scored
steps (Socratic: non-empty steps with include = True); SUCCESS_COUNT = k* == true count; per-answer step precision
and recall of GAP's flags. Primary on all such answers; secondary on erroneous answers only (count >= 1), because
GAP never flags 0 and cannot succeed on a clean answer.
For G1, R1, R2, -p_null (and -p_R1): medians in success vs failure; AUROC for SUCCESS_SET (oriented so > .5 means
"bigger gap -> success"); n control by (a) exact-n stratified AUROC (strata with >= 20 answers and both outcomes,
weighted by stratum size; within an exact n, G1 and p_null are monotone so their stratified AUROC is identical),
(b) partial Spearman correlation with success given n, (c) logistic regression success ~ log n + z(G1) + z(R1).
Paired bootstrap (1,000 draws; PRMBench source groups, Socratic answers) for stratified AUROC(G1) - AUROC(R1).
Deciles of each quantity: pooled flag precision, mean recall, success rate, median n.
Verdict rule: "absolute" if stratified AUROC(G1) > AUROC(R1) and AUROC(R2) with the bootstrap CI of G1-R1 above 0;
"relative" if the reverse; otherwise "no clear difference".
Clean-answer detection (PRMBench, 176 clean of 6,211): AUROC of p_null (higher -> clean), and of -G1, -R1, p_R1,
pooled and exact-n stratified; pass rate of each test among multi_solutions vs erroneous answers.

## Amendment A (written after stage 1, which is label-free, and still BEFORE any labelled metric)
Stage 1 showed R1 and R2 are heavy-tailed (PRMBench R1 99th pct ~21-28, max up to 226; R2 max up to 586).
The logistic regression in diagnostic (c) therefore uses z(log G1) and z(log R1) (and a second fit with z(log R2))
instead of raw z(G1), z(R1). AUROC/rank-based diagnostics are unaffected (monotone transform). Null calibration was
self-checked label-free: on 4,000 fresh Gaussian answers with PRMBench's n distribution, T1/T2/T3/DW pass 4.9% /
4.7% / 5.2% / 4.9%. Label-free pass shares on the real scores (all answers): PRMBench T1 7.0% (B_CHAIN_N16),
12.3% (T_LSML_CONT), 13.5% (T_ALL_EQ); Socratic T1 2.5-4.3%. tau_null = 1.616 (PRMBench), 1.610 (Socratic).
