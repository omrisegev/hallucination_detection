# Graph Geometry Selection Research V1 — frozen external transfer

**Retrospective stress test on historically opened outcomes; not independent confirmation.**

Every telemetry-derived score and row identifier was physically isolated, frozen, and hash-verified before this report opened an outcome field. The fixed canonical full and matched-cross scores reproduce the prior frozen external arrays exactly in every cell.

| domain | canonical | label-free | supervised one-SE | supervised max-mean | Family-NRM |
|---|---:|---:|---:|---:|---:|
| `processbench_llama` | +0.588 | +0.711 | +1.330 | +1.483 | +1.580 |
| `processbench_qwen` | +0.137 | +0.277 | +0.800 | +0.630 | +0.557 |
| `semgrad` | +0.257 | +0.404 | +0.537 | -0.245 | +1.310 |
| `prmbench` | -0.420 | -0.374 | -0.120 | -0.535 | +0.460 |
| `hle` | +0.912 | +0.625 | -0.419 | +0.531 | +0.345 |

Matched-cross scores were frozen for every selector as a separate actuator control; no outcome-facing selector chose between full and cross.

The score fit was physically target-free, but the preceding isolation process necessarily unpickled historically opened ProcessBench, SemGrad, and PRMBench caches. It accessed only the registered telemetry, identifier, and non-target eligibility fields.
