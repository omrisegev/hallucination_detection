# Tail15 readout head-to-head v1 — report

Status: **COMPLETE / REVIEW PASS**

| Readout | Math q | Math answer F1 | PB answer F1 | PB AUROC | PB localization | Clean accuracy | Error exact |
|---|---:|---:|---:|---:|---:|---:|---:|
| `tail15_mass__token_top10` | 0.40 | 0.632415 | 0.697932 | 0.799571 | 36.6736% | 0.681510 | 0.238181 |
| `tail15_mass__token_mean` | 0.45 | 0.625927 | 0.691500 | 0.792172 | 36.6064% | 0.727735 | 0.228726 |

Primary Top10-minus-mean localization delta: +0.067pp; family-wise 98.333% interval [-0.984, +1.149]pp.

Point-preferred readout: `tail15_mass__token_top10`. This is a development decision, not external confirmation.

The historical tail15-mean q=.3 result is retained only as a PB-developed diagnostic and did not enter this selection.
