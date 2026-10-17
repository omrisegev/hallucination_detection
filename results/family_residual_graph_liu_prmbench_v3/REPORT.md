# Family-residual graph LIU v3 bug-repair sensitivity — PRMBench

**FAIL**: finalist vs IU -0.007pp (source-group 95% CI [-0.010, -0.004]pp).

Family-NRM changed AUROC by +0.460pp; the finalist recovered -1.5% of that point gain. `D_0.5`=-0.237pp.

| method | AUROC | AUPRC | delta vs IU |
|---|---:|---:|---:|
| `iu` | 0.720602 | 0.220771 | +0.000pp |
| `finalist` | 0.720533 | 0.220770 | -0.007pp |
| `fixed_default` | 0.720900 | 0.220614 | +0.030pp |
| `cardinality` | 0.711966 | 0.222382 | -0.864pp |
| `family_nrm` | 0.725206 | 0.228811 | +0.460pp |

This outcome was known before v3 was specified. It is a retrospective bug-repair sensitivity, not transfer confirmation.
