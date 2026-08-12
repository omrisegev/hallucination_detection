# Frozen CB-CS-IU transfer to Llama ProcessBench

**Status:** new scorer-family confirmation on the same underlying ProcessBench examples used in the Qwen3 study.

CB-CS-IU changed cell-macro reasoning-error AUROC by **+1.263pp** versus ordinary IU (4W/0L; worst +0.476pp). The paired four-subset interval is [+0.708, +1.692]pp.

| contrast | delta | paired subset 95% interval | W/L | worst |
|---|---:|---:|---:|---:|
| `cardinality - iu` | +1.263pp | [+0.708, +1.692] | 4/0 | +0.476pp |
| `leverage - iu` | +0.965pp | [+0.531, +1.405] | 4/0 | +0.364pp |
| `dufs_liu - iu` | +0.149pp | [+0.062, +0.218] | 4/0 | +0.013pp |
| `cardinality - leverage` | +0.298pp | [+0.134, +0.546] | 4/0 | +0.112pp |
| `cardinality - dufs_liu` | +1.114pp | [+0.654, +1.485] | 4/0 | +0.463pp |
| `cardinality - uniform` | +0.716pp | [+0.327, +1.105] | 4/0 | +0.282pp |
| `cardinality - reverse_cardinality` | +3.273pp | [+2.223, +4.191] | 4/0 | +1.800pp |

## Frozen gates

- **PASS — positive mean subset delta:** 1.263068090964642
- **PASS — positive paired subset interval:** 0.7078113205430059
- **PASS — at least three wins:** 4
- **PASS — tail safety:** 0.47560995836858355
- **PASS — numerical invariants:** 8.881784197001252e-16

## Boundary

Fit accessed neither per-row target key; data, scores, source, and upstream manifest hashes were verified before report-time label access. Aggregate class counts were visible in the upstream manifest before the run.

This confirms transfer across telemetry/scorer model families. It is not independent-example confirmation because the reasoning chains and labels are shared with the earlier Qwen3 ProcessBench study.
