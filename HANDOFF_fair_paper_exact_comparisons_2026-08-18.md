# Handoff: consolidate fair paper-exact hallucination comparisons

**Date:** 2026-08-18

**Status:** next-session planning brief; do not execute before user approval

**Primary goal:** use the already acquired cluster assets to produce the
fairest possible direct comparison of our frozen methods with published
competitors.

## 1. Session boundary

Start by reading `CLAUDE.md`, `PROGRESS.md`, `Research_Directions.md`, and this
file completely. Also read:

- `docs/research_notes/early_online_detection_canonical_status_2026-08-19.md`;
- `HANDOFF_paper_exact_cluster_acquisition_2026-08-16.md`;
- `papers/PAPER_EXACT_SOURCES.md` and the relevant paper digests;
- `results/global_local_online_architecture_v2/REPORT.md`;
- `results/local_online_comprehensive_v1/REPORT.md`;
- the Unified Causal IU-PCR protocol, implementation, subset-search reports,
  and final decision artifacts available in the current checkout/local result
  directories;
- all committed manifests for ProcessBench, DeepConf, REFRAIN, LEASH,
  Streaming Hallucination Detection, Mind the Gap, uPRM, PRM/critic, and the
  24-cell Global benchmark.

The first response must be a plan, not an execution. Read-only inspection of
local files and `rclone lsf/lsjson/size/cat` is allowed to establish inventory.
Do not download large data, launch cluster jobs, alter Drive, modify code, or
open labels for a new confirmation cell until the user approves the plan.

## 2. Frozen method position

Treat ordinary **Unified-28** as "our unified method" for the comparison
program. Its roster is seven causal base streams crossed with:

- `level`;
- `ewma16`;
- `positive_area`;
- `persistence`.

Frozen Llama transfer:

| task | Unified-28 | matched incumbent | delta |
|---|---:|---:|---:|
| Global AUROC | 0.6629 | 0.6870 | -0.0241 |
| Localization macro-F1 | 0.2880 | 0.2419 | +0.0461 |
| Early AUROC | 0.5587 | 0.5777 | -0.0189 |

Localization improves significantly; Global and Early fail their frozen
non-inferiority margins. Unified-28 is the best unified candidate, not a
claim that one head has replaced the dedicated methods. Keep the dedicated
incumbents in every relevant direct table. Do not re-tune Unified-28, DUFS
lambda, task weights, signs, feature roster, or accumulator on evaluation
rows.

Corrected Qwen development values are 0.6914/0.3040/0.5301. Never quote the
withdrawn pooled values 0.7012/0.3278/0.5435. The separate 24-cell Global
result is ordinary-36 0.7591 versus mixed-v2 DUFS-LIU 0.7766; Math ties and QA
regresses. It is a separate panel, not a replacement score for Unified-28.

## 3. Why this cycle exists

The cluster campaigns successfully acquired expensive model-dependent assets,
including full ProcessBench critic/PRM/control outputs and several other
localization panels. The project deliberately separated acquisition from
offline analysis so that feature construction, calibration, joining,
bootstrapping, and reporting could run locally. That offline integration was
never completed as one canonical comparison package. New local method
development then made the gap more visible: the method-of-record changed, but
it was not replayed across all newly acquired competitor populations.

The next cycle closes that gap. More generation is a last resort, not the
default.

## 4. Non-negotiable comparison lanes

Never create a single leaderboard across these lanes.

### A. Global final-answer detection

Common population, final-answer wrongness, common AUROC/AUPRC and fixed-FPR.
Include Unified-28 where telemetry permits, the frozen mixed-v2 IU-PCR and
DUFS-LIU heads, max entropy, and eligible paper baselines. Keep the 24-cell
Math/QA breakdown visible; do not let QA macro weighting hide the Math result.

### B. First-error Localization

Use the official ordered 3,400-row ProcessBench population and official
earliest-error-or-clean evaluator. Include Unified-28, family-six Local,
max-entropy+top-5, GL-LIU, Mind the Gap common-protocol replay, PRM, critic,
uPRM's cheap control, and full uPRM only if it genuinely exists. Report error
accuracy, clean accuracy, per-subset F1, macro-F1, access tier, and paired
question bootstrap. Preserve Mind-the-Gap native SLA in a separate subtable.

### C. Causal prefix detection

On identical generated traces, report AUROC/AUPRC at absolute budgets
16/32/64/128/256/512, time to reliable warning, ever-warning FPR, and fixed
trace-level false-alarm targets. Include Unified-28, IU28, the selected
dedicated Online head, max/mean entropy, and an exact DeepConf-derived scalar
only after its saved confidence is verified against pinned official code.
Streaming hidden-state probes belong here only if official rows, labels,
splits, layers, and checkpoint/code are available.

### D. Stopping and adaptive compute

For REFRAIN, LEASH, DeepConf, fixed budgets, and any declared policy based on
our frozen score, report pass@1/accuracy against total generated tokens,
latency, forced-closure rate, parser failures, and full accuracy-compute
frontiers. A detector AUROC is not comparable to token savings. DeepConf
multi-trace results must not be merged with single-trace stopping.

## 5. Fidelity and access labels

Every table row must carry exactly one fidelity label:

- `official-exact`;
- `paper-specified`;
- `paper-specified-partial`;
- `adapted-common-protocol`;
- `published-context-only`;
- `blocked-assets`.

Also record model/checkpoint, training/labels, number of model passes, hidden
state/logit access, dataset revision, ordered-ID hash, prompt/decoding hash,
evaluator hash, and run commit. Published values are regression targets and
context, never substitutes for a direct replay.

## 6. Required planning output before execution

The next agent must return a concrete plan containing:

1. A row-by-row asset inventory: local path, Drive path, size, completion,
   schema, row IDs, telemetry, labels, and manifest/hash health.
2. A competitor coverage matrix by lane and fidelity, clearly distinguishing
   complete, CPU-scoreable, incomplete, and blocked.
3. The exact shared population and metric contract for every direct table.
4. A join audit proving which methods can be compared on identical IDs now.
5. The smallest CPU-first execution DAG that produces publishable tables from
   existing data before requesting any new GPU work.
6. Explicit new-compute gates for missing DeepConf, REFRAIN, LEASH, Streaming,
   or full-uPRM assets, with estimated cost and scientific value.
7. Verification tests: split isolation, suffix causality, no final-length
   leakage, deterministic joins, evaluator unit tests, threshold calibration
   isolation, grouped bootstrap, and result-hash reproducibility.
8. A deliverable map: machine-readable common tables, per-question records,
   native-paper panels, confidence intervals, coverage/blocker report, and one
   advisor-facing HTML whose default view contains direct comparisons only.

The plan should maximize fair comparisons from assets already paid for. It
must not tune toward published values or silently substitute a different
model/prompt/metric when exact reproduction is blocked.

## 7. Definition of done

This program is complete only when:

- every claimed win/tie/loss points to identical row IDs and one metric;
- Unified-28 and the dedicated incumbent are both present wherever the saved
  telemetry supports them;
- external methods are either replayed fairly or visibly marked as context or
  blocked;
- metric-incompatible paper results are separated rather than normalized into
  one score;
- every table row links to its data manifest, method hash, evaluator, and
  fidelity/access declaration;
- a new reader can answer "where are we competitive?" without reading caveats
  from several historical reports.

## 8. Prompt for the next conversation

Copy the prompt below verbatim into a new conversation:

> Read `CLAUDE.md`, `PROGRESS.md`, `Research_Directions.md`, and
> `HANDOFF_fair_paper_exact_comparisons_2026-08-18.md` completely. We are
> freezing Unified-28 as our unified method-of-record and concentrating the
> next cycle on fair apples-to-apples comparisons against the published
> competitors using the cluster data already acquired. Do not execute runs,
> modify code, download large artifacts, mutate Drive, or launch cluster jobs
> yet. First perform a read-only inventory of local and `gdrive:` manifests and
> produce a detailed, approval-ready plan. The plan must separate Global
> detection, ProcessBench first-error Localization, causal prefix detection,
> and stopping/adaptive-compute into distinct lanes; specify identical row IDs,
> metrics, evaluators, calibration, grouped uncertainty, fidelity/access labels,
> missing assets, CPU-first execution order, and explicit gates for any new GPU
> work. Include Unified-28 and the dedicated incumbent in every eligible direct
> table. The goal is to turn the completed acquisition into precise,
> publishable comparisons—not to start another feature or DUFS search.
