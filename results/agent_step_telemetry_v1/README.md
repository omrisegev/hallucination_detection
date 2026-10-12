# Agent step telemetry v1 — CPU review delivery

**BLOCKED_BY_CPU_GATES**: the additive implementation is ready for review,
but G1/G2 fail and prevent model loading. No Qwen/GPU run or benchmark
scoring/evaluation occurred. Third benchmark: AFTraj-2K.

Start with [the handoff](../../docs/HANDOFF_agent_step_telemetry_v1.md) and
[the runbook](../../docs/experiments/AGENT_STEP_TELEMETRY_V1_RUNBOOK.md).

## Counts

| Benchmark/subset | Items | Steps | Source cases or split |
|---|---:|---:|---|
| aft:agentic | 875 | 7332 | 405 safe / 470 unsafe |
| aft:coding | 608 | 6113 | 361 safe / 247 unsafe |
| aft:math | 793 | 12861 | 396 safe / 397 unsafe |
| apb:bfcl | 250 | 2590 |  |
| apb:gaia_dev | 250 | 1628 |  |
| apb:hotpotqa | 250 | 734 |  |
| apb:tau2 | 250 | 3557 |  |
| tprm:GTA | 236 | 236 | 118 |
| tprm:ToolSandbox | 260 | 260 | 130 |
| tprm:bfcl | 222 | 222 | 111 |
| tprm:tooltalk | 172 | 172 | 86 |

Total: **4,166 items / 35,705 steps**. APB: 1,000 / 8,509;
ToolPRMBench: 445 pairs / 890 items and steps; AFTraj: 2,276 / 26,306.
AFTraj test split: 332; unsafe indices in range, safe indices -1, zero exceptions.
Unsafe sources: 728 injected / 386 diagnosed. No annotation values are used for collection.
AFTraj release/paper: 2,276/2,272; ToolPRMBench release/paper: 445/984.

## CPU gates

| Gate | Result | Checked | Failures |
|---|---|---:|---:|
| G1 | FAIL | 910 | 6 |
| G2 | FAIL | 4166 | 2277 |
| G3 | PASS | 445 | 0 |
| G4 | PASS | 4166 | 0 |
| LENGTH | PASS | all subsets | 0 |
| G5 | PASS | release counts/ranges/test IDs | 0 |
| G6 | PASS | 9 CPU fixture records, interrupted/resumed/fresh | 0 |
| G7 | NOT_RUN | requires approved GPU smoke | — |

G1: six TPRM variants fail because native history-prefix rendering changes
the final unscored assistant think block. G2: all 2,276 AFTraj items plus
`apb:gaia_dev:287` have tokens crossing the prescribed character boundaries.
All failed UIDs are in `CPU_GATES.json`. Strict spans stay unchanged; no
failed item was silently removed, truncated or assigned a relaxed gate.
See the runbook for the contract decisions Claude must review.

Sixteen focused tests, eight unchanged external regressions and both CPU
collector smokes pass. The fixture collector saves 4/9 before stopping,
resumes to 9/9 and reproduces an independent fresh run exactly.
Empty scored turns retain zero-length spans and empty top-50 arrays;
a dedicated test asserts one forward pass. No mask1 is applied.

## Lengths

| Subset | Median | P95 | Maximum | Excluded >32768 |
|---|---:|---:|---:|---:|
| aft:agentic | 551 | 5404.6 | 14268 | 0 / 875 |
| aft:coding | 909 | 1845.8 | 3523 | 0 / 608 |
| aft:math | 848 | 2260.0 | 11912 | 0 / 793 |
| apb:bfcl | 14987 | 19040.2 | 20215 | 0 / 250 |
| apb:gaia_dev | 4743 | 27703.7 | 1.51592e+06 | 8 / 250 |
| apb:hotpotqa | 1958.5 | 8654.7 | 62186 | 2 / 250 |
| apb:tau2 | 8628.5 | 13871.0 | 27717 | 0 / 250 |
| tprm:GTA | 628 | 1206.5 | 4144 | 0 / 236 |
| tprm:ToolSandbox | 1383 | 4014.0 | 5044 | 0 / 260 |
| tprm:bfcl | 5252 | 7968.8 | 9254 | 0 / 222 |
| tprm:tooltalk | 3452 | 3569.4 | 3596 | 0 / 172 |

Eight GAIA (3.2%) and two HotpotQA (0.8%) items are overlength.
No other subset has exclusions; no subset breaches 5%. Maximum is
1,515,925 tokens. `EXCLUDED_OVERLENGTH.json` lists all ten IDs.
There are 42 retained empty steps. These length exclusions do not solve G2.

## Provenance and data boundary

- Dataset pin: `11191169d3d99298df098eee8d83db425117faba`.
- Source manifest SHA256: `f856bc9b3bfa265cd6e1b64bc44083a38c390d3fc090057bfe87d52d1f749f2b`.
- Protocol SHA256: `78633e6834212d8ad08491782be65643c0464265fffb175f2bdd5d1e5d00e346`.
- `SOURCES.json` catalogs every source/reference/tokenizer file and hash.
- `INPUTS.json` contains per-benchmark answers hashes; `GPU_PREFLIGHT.json` binds code and inputs with verdict FAIL.
- Only compact reports, fixture examples and source manifests are in Git.
- Private sources, answers, evaluator-only JSONL and model records stay outside Git.
- No Drive transfer or AIRCC access was attempted.
- The main checkout and its earlier dirty changes were preserved; the worktree initially had two unrelated CRLF checkout differences, which are excluded from this branch.
