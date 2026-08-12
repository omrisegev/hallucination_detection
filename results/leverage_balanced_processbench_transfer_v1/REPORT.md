# Frozen leverage-balanced IU transfer to ProcessBench

**Status:** external-task transfer; formula frozen before ProcessBench evaluation, but ProcessBench is not a historically untouched benchmark in this repository.

## Primary confirmation result

Across the six confirmation/model-transfer cells, leverage-balanced IU changed reasoning-error AUROC by **+0.340pp** versus ordinary IU (5W/1L; worst -0.018pp). The equal-subset interval is [+0.045, +0.582]pp.

Frozen DUFS-LIU changed the same baseline by +0.221pp.

## Primary-target table

| slice | method | AUROC | delta vs IU | equal-subset 95% interval | W/L | worst |
|---|---|---:|---:|---:|---:|---:|
| confirmation | `leverage_balanced` | 0.7989 | +0.340pp | [+0.045, +0.582] | 5/1 | -0.018pp |
| confirmation | `dufs_liu` | 0.7977 | +0.221pp | [+0.164, +0.291] | 6/0 | +0.096pp |
| confirmation | `uniform` | 0.7953 | -0.019pp | [-0.567, +0.254] | 3/3 | -0.583pp |
| confirmation | `cardinality` | 0.8044 | +0.895pp | [+0.721, +1.101] | 6/0 | +0.570pp |
| confirmation | `reverse` | 0.7826 | -1.284pp | [-1.507, -0.872] | 0/6 | -1.752pp |
| development | `leverage_balanced` | 0.7827 | +0.366pp | [-0.151, +0.884] | 1/1 | -0.151pp |
| development | `dufs_liu` | 0.7812 | +0.213pp | [+0.201, +0.225] | 2/0 | +0.201pp |
| development | `uniform` | 0.7826 | +0.353pp | [-0.069, +0.776] | 1/1 | -0.069pp |
| development | `cardinality` | 0.7863 | +0.725pp | [+0.564, +0.886] | 2/0 | +0.564pp |
| development | `reverse` | 0.7682 | -1.088pp | [-1.294, -0.882] | 0/2 | -1.294pp |
| all | `leverage_balanced` | 0.7948 | +0.346pp | [+0.094, +0.611] | 6/2 | -0.151pp |
| all | `dufs_liu` | 0.7936 | +0.219pp | [+0.167, +0.272] | 8/0 | +0.096pp |
| all | `uniform` | 0.7921 | +0.074pp | [-0.210, +0.305] | 4/4 | -0.583pp |
| all | `cardinality` | 0.7999 | +0.853pp | [+0.676, +1.088] | 8/0 | +0.564pp |
| all | `reverse` | 0.7790 | -1.235pp | [-1.462, -1.040] | 0/8 | -1.752pp |

## Transfer gates

- **PASS — confirmation cell-macro improvement:** +0.340pp
- **PASS — positive equal-subset interval:** [+0.045, +0.582]pp
- **PASS — confirmation wins:** 5/6 wins
- **PASS — tail safety:** worst=-0.018pp
- **PASS — beats frozen DUFS-LIU:** LB=0.7989; DUFS=0.7977
- **PASS — numerical invariants:** max=1.78e-15

## Boundary

Maximum effective-weight reconstruction / orthogonality / trust-scale errors were 1.776e-15 / 5.000e-16 / 2.776e-17.

Final-answer incorrect is saved as a secondary target in `summary.csv`; it did not select the method or gates.
