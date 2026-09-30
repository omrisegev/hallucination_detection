# High/low versus full-range entropy sampling

User-authorized parallel experiment, 2026-09-07. The full matched reasoning
benchmark remains running under its original frozen protocol. This separate
study uses the existing110 development answers (24 PRMB,86 PB-Qwen3-8B),
corrected v3 labels/v2 groups. It is not full-dataset or untouched confirmation.

## Question and fixed design

Does covering low as well as high entropy improve the existing answer-only
fusion, relative to high-only sampling, uniform-in-time sampling and full fitting?
Entropy is an unlabeled uncertainty proxy, not a correctness class.

Keep the original moment/context bank route, width8, N original nonoverlapping
fitting windows, and m=min(N,max(32,ceil(N/2))) budget. Two new selectors:

- `entropy_tails`: choose floor(m/2) lowest-entropy windows first; then choose
  the remaining ceil(m/2) highest from the unused indices. Both rankings break
  ties in favor of earlier time indices. This guarantees disjoint selections.
- `entropy_quantiles`: sort all windows by ascending entropy, ties by time;
  choose rank floor((2*j+1)*N/(2*m)), j=0,...,m-1. These are evenly spaced
  mid-quantiles of the empirical distribution. For N100/m50, this gives five
  selected windows per equal-count entropy decile. This is not random sampling
  or inverse-probability correction.

Restore chronological order. When m=N, replay the exact full selector output.
The existing72 eligible/38 replay split stays fixed. Selecting two tails still
underrepresents the middle; neither selector is assumed to improve accuracy.

Reuse the exact `fit_bank`/`apply_readout` implementation from the prior sampling
replication: selected-row normalization, orientation, feature removal/grouping,
IU/Joint fits, DUFS gates, graph and condition100 heads. Same graph seed namespace
and identity, not new per-selector seeds. Seven cores: equal, IU, Joint lambda0,
Joint graph0.1, permuted Joint graph, equal graph, equal permuted graph.
Only Joint FIT failure triggers selected-row IU fallback in the same bank.
Other numerical/readout failures remain explicit. Score ALL original windows;
GMM uses all original fitting-window risks. Official spans and peak/no-error
rule unchanged. Dense features remain computed: no inference-saving claim.

Both normalization and fusion fitting change with selection, as in high-only.
No causal weight-only interpretation. Report pooled and within-answer AUROC,
clean/no-error and exact first-error components, coverage and native/fallback
counts. These expose scale versus local-ranking differences without treating a
post-hoc affine diagnostic as another candidate.

## Comparators and evaluation

Keep all176 frozen current110 entries unchanged, including the strong
risk-selected equal+permuted control. Add14 outputs,190 total. Display the
five-selector by seven-core table: full, uniform, high-only, tails, quantiles.
Primary families: IU, Joint graph and equal-permuted control. For each new
selector compare each of those3 against full/uniform/high-only and compare
the two new selectors to each other. Within new selectors retain graph versus
lambda0/permutation/IU/equal-graph and IU versus equal:31 fixed paired contrasts.

Use existing audited source-group bootstrap (1000 draws), all110 and the fixed
72 eligible population reported separately. Registered contrasts use all110;
PRMB deltas use common valid answers, PB keeps every row with invalid decisions
counted as failures. These are exploratory intervals, not multiple-comparison
corrected confirmation or a new guarantee of a two-task winner. Never compare
AUROC to native PRMScore. No label-driven selector/budget tuning after results.

## Execution and review

One worker, one BLAS thread, Windows below-normal process priority when available;
one-hour scheduling cap with resumable per-answer atomic files. No edits to
Claude's worktree, full-benchmark sources, previous frozen artifacts, or Drive.
No additional model inference. Expected144 new bank fits (2 x72);38 rows replay.

Preflight tests: disjoint tails/ties/odd budgets, empirical quantile positions,
chronological order, all-row replay and invalid inputs. Before new scoring,
replay high-only fitting/readout on representative eligible bank routes against
saved outputs; freeze inputs/imported code/protocol/tests. Freeze all outputs
before evaluation. Verify source label/group joins and unchanged176 metrics;
review new selection, linear risk reconstruction, dense mixture decisions and
representative actual bank refits. Reused numerical kernels/metrics are disclosed.
Produce JSON, paired contrasts and a simple-English HTML report with statuses.
