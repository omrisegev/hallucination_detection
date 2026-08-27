# Working synthesis: methods and data for the next benchmark

**Status:** planning only. No new experiment was run.

## Short answer

We now have enough data to ask two different questions:

1. which response-level fusion method is strongest on the same 24 cells; and
2. which ideas transfer to other models, datasets, and prediction tasks.

These questions need one method registry but several leaderboards. A complete
answer AUROC, first-error F1, prefix AUROC, and token-saving policy do not
measure the same thing and should not be averaged.

## Methods to keep for now

The main 24-cell rebuild should retain:

- continuous L-SML and the DUFS/GroupFS selector-to-L-SML pipelines;
- U-PCR, U-PCR with estimated feature polarity, and IU-PCR;
- DUFS-LIU, SU-PCR, and balanced-atomic CA-SpecRaGE;
- DEEM-B3;
- the new within-cell Family-NRM and PGRD variants; and
- simple means and entropy as floors.

GOOD_5/GOOD_6, LOCO_5, and the best single feature are useful references, but
labels influenced their selection. Mixed-v2, contribution-balanced IU,
Atomic-NRM, Family-residual Graph-LIU, DEEM-B1/B2, and PGRD graph controls stay
as sensitivity or mechanism rows. Residual-Graph DEEM remains a synthetic-gate
failure and is not the same method as DEEM-B3.

Family-NRM and PGRD should always show three regimes: A uses the target cell
only and no labels; B uses unlabeled donor cells; C allows donor-label model
selection. The new A variants enter the main table but are still unrun.

## Data and task map

| lane | main population | main output |
|---|---|---|
| response core | 48,607 answers / 24 dataset-model cells | cell-macro and every cell |
| response transfer | ProcessBench global, SemGrad, PRMBench-response, HLE, Evidence-Drop, AQuA, S2-GSM8K | one table per feature contract |
| published comparison | paper-aligned suite and six identity-proven legacy cells | access/fidelity-labelled anchors |
| localization | ProcessBench Llama + Qwen panels; PRMBench-step separately | first-error F1 or step AUROC |
| early detection | causal ProcessBench prefixes | AUROC/AUPRC by token budget |
| stopping | AQuA/GSM8K × Qwen/Llama/Phi | pass@1 versus tokens |
| RAG | RAGTruth answer/sentence/token; RefChecker claims | separate unit-specific tables |
| white-box | 31,440 exact-common answers | accuracy plus coverage |
| repeated generation | one 200-question MATH-500 cell with five runs | separate high-cost panel |
| negative scope | GPQA, old LCiteEval RAG, rejected pilots | limitation appendix |

The Evidence-Drop four-cell panel was easy to miss. It contains 5,638 Qwen
responses on GSM8K and the full MATH test, but its raw LFS files need retrieval
and a common matrix rebuild. Seven of the 24 core cells also repeat each source
question ten times, so row-level intervals need the original question IDs.

## What “generalization” can mean here

All current external outcomes have already been opened somewhere in the
project. They are useful retrospective transfer tests: they can show that a
gain survives, disappears, or reverses. They are not a clean confirmation.

A real confirmation requires a later population whose labels stay sealed until
the method, adapter, rows, and scores are frozen. The current benchmark should
help us choose that test; it should not claim that it already exists.

## Next decision

Before running anything, choose:

1. one common 24-cell feature contract, plus a separate 16-feature transfer
   contract if broad coverage is more important than using all measurements;
2. which soft candidates remain full runs versus cheap controls;
3. which leading response methods receive localization, prefix, and RAG
   adapters; and
4. the uncertainty and promotion rules.

The detailed protocol is
`docs/experiments/MULTI_POPULATION_METHOD_BENCHMARK_V1.md`. Its registries are
under `configs/multi_population_benchmark_v1_*`. The 24-cell method definition
remains in `docs/experiments/GLOBAL_24CELL_METHOD_BENCHMARK_V2.md`.
