# Broad50 without digit features — development result

Full 13,769 answers. Source-fold pooled fitting; fixed non-digit tail15 gate.

| Arm | PB macro % | PRMB within-answer AUC | Native answers |
|---|---:|---:|---:|
| entropy_H1 | 36.3476 | 0.730111 | 13769 |
| equal50 | 37.8963 | 0.748902 | 13769 |
| continuous | 37.7632 | 0.747926 | 13769 |
| joint | 38.0485 | 0.746070 | 13769 |
| joint_balanced | 37.9903 | 0.746046 | 13769 |
| original4 | 37.4749 | 0.753436 | historical |
| innovation5 | 39.8314 | 0.760293 | historical |

Primary paired contrast (balanced minus ordinary Joint), 97.5% source-group intervals:

```json
{
  "pb": {
    "point": -0.0005810646197833558,
    "low": -0.002072419195904974,
    "high": 0.0005852824258137246,
    "confidence": 0.975,
    "draws": 10000
  },
  "within": {
    "point": -2.3820399899121547e-05,
    "low": -0.00022866867435803056,
    "high": 0.00017205455712520878,
    "confidence": 0.975,
    "draws": 10000
  }
}
```

Invalid fits use the predeclared H1 fallback; inspect native coverage before attributing performance to Joint.
This single broad-bank test is development evidence, not untouched confirmation or an exact redundancy-invariance test.

Independent audit: PASS (`AUDIT.json`): all fold weight maps replay exactly;
independent sklearn AUROC and PB formulas match; both non-digit historical
anchors reproduce. All 13,769 answers have native fits; no fallback or missing
step inputs occurred. Six feature/optimizer numerical checks passed.

The block-balanced variant does not improve either endpoint in this run.
Both intervals include zero; all broad50 methods trail innovation5 at the point
level on both metrics. This result does not close broad banks or the Joint family.
Joint weight-map cosines exceed .9998 between losses. Group discovery chose K=3
in three folds (outer averaging guard) and K=4 in two; the group selection and
readout remain possible sensitivities, not established explanations of errors.

Gate wording correction: the executed frozen gate is whole-answer Tail15 Top10
mean, **without subtraction of the answer mean**. The frozen protocol's word
"prominence" was wrong; see `GATE_DEFINITION.md` and `GATE_PROVENANCE_CHECK.json`.
PRMB within-answer AUROC is ungated. No scores or choices changed after evaluation.
