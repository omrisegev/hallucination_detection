# Rule C transfer to Hard2Verify and DeltaBench: frozen protocol

I wrote this on 2026-11-22, before any child telemetry existed and before reading any label of rule C on these
benchmarks. Omri approved it on 2026-11-21 ("yes, set up cycle4 and run H2V + DeltaBench").

## What is frozen

- **Rule C**, unchanged from agentS2 (2026-11-19):
  - Location: GAP flags on the method-of-record L-SML step score s0. The fit is R0 per-answer z on 16 channels
    (hard top10 + 5 step channels).
  - Filter: keep a flag only where sg > tau.
  - Score sg = RankGauss(pool) @ Wg, oriented by sign corr(sg, s0).
  - tau = (1 - beta) quantile of sg over the GAP-flagged pseudo-clean steps.
  - beta is chosen on the grid 0, .01, ..., 1 by the maximum of the pseudo-metric. A tie goes to the largest beta.
  - An answer with no surviving flag has 0 flags.
- **Fit scope:** the full C pool of each benchmark. There is no dev/test split, because every fit is label-free.
  - Hard2Verify: the 181 answers with standalone telemetry. The other 19 answers have no telemetry (18 are longer than
    4,096 tokens, 1 failed); they are not rerun and not scored.
  - DeltaBench: all 1,236 answers. These are native-context rows plus RoPE dynamic NTK x3 rows, as in the stored ext
    cell. The children use the same two passes.
- **Pseudo pairs:** a parent's steps 0..k (pseudo-clean) against the child's steps 0..k (pseudo-wrong), for every
  child with local k >= 1.
  - Hard2Verify: V3 + V4.
  - DeltaBench: V3 only. This is a deviation: only 19 of 1,217 questions have a second answer, so V4 is not possible.
- **Pseudo-metric:**
  - Hard2Verify: PRMScore with keep = ones (pseudo.py, unchanged).
  - DeltaBench: the official section macro-F1 with target = {k + 1}. A pseudo-clean prefix scores 1 if no section is
    flagged, else 0 ("empty=1"). This is a deviation: the official script drops samples with an empty target, which
    would remove every pseudo-clean prefix.

## Evaluation (once, `code/eval_ext.py`)

- **Hard2Verify:**
  - Metrics: official step Balanced F1 (x100) and the PRMScore form.
  - Populations: own (181 answers) and common (177 answers).
  - Arms: rule C; method of record (GAP, no gate); the 5 PRMs (stored PRM-mode scores) under the native rule and
    under GAP.
- **DeltaBench:**
  - Metric: official section macro-F1. The target is unuseful + error sections (official script); error-only is also
    reported.
  - Arms: rule C; method-of-record GAP; method-of-record s0 under mu-sigma; 4 PRMs under mu-sigma and under GAP.
- **Primary contrast:** rule C minus the method of record (GAP, no gate). It is the H2V Balanced F1 on the own
  population, and the DeltaBench macro-F1 on the union target.
- **CI:** paired 95% CI from a 2,000-draw group bootstrap over question groups (seed 20261122).
- **Checks:**
  - The method-of-record arm must reproduce H2V 47.2247 (common population).
  - The DeltaBench mu-sigma PRM-7B arm must reproduce 29.34 (stored scores).
- After the evaluation, no arm, threshold or endpoint changes.

## Amendment 1 (2026-11-22, written before any rule C output or label read)

- **C pool = CR.** CR is the C answers re-run in the same nvcr container as the children (jobs 372169 / 372170). It is
  not the stored C cell, which was made in the native numpy-2 env.
- **Why:** smoke S3 (report only) on H2V gives top15_turnover Spearman 0.91 / 0.86 between a fresh container run and
  the stored cell (12 rows). Pseudo pairs compare parent and child prefixes, so both must come from one environment.
  The 11-18 SSL runs also re-ran C.
- **The stored cell stays a reference.** The method of record is reported on both pools: stored C must give
  47.2247 (H2V common), and CR is reported next to it.
- **Hard2Verify:** the 181 answers with stored telemetry are re-run (all fit 4,096 tokens).
- **DeltaBench:** all 1,236 answers are re-run, native pass + RoPE x3 pass, merged.
