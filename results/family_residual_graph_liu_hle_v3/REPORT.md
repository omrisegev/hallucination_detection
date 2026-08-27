# Family-residual graph LIU v3 bug-repair sensitivity — HLE

**FAIL**: finalist vs IU -0.019pp (stratified 95% CI [-0.047, +0.008]pp).

Family-NRM: +0.345pp; recovery -5.5%; `D_0.5`=-0.194pp.

| method | AUROC | AUPRC | delta vs IU |
|---|---:|---:|---:|
| `iu` | 0.516775 | 0.034325 | +0.000pp |
| `finalist` | 0.516585 | 0.034319 | -0.019pp |
| `fixed_default` | 0.510336 | 0.033651 | -0.644pp |
| `cardinality` | 0.505819 | 0.033403 | -1.096pp |
| `family_nrm` | 0.520229 | 0.034886 | +0.345pp |

HLE contains only 68 judged-correct answers, uses the interim Codex judge, and its outcome was known before v3. This is a retrospective bug-repair sensitivity only.
