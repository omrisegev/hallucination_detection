# Gate feature and readout selection v1

Status: **COMPLETE / REVIEW PASS** on all 6,800 frozen ProcessBench answers.
This is development selection; the selected gate is evaluated cumulatively in
the separate integration replay, not claimed as model-transfer confirmation.

## Result

With the q15 finalist locator and q=.3 threshold rule fixed, the development
winner among 33 feature/readout candidates is whole-answer mean probability
mass missing outside top-15: `tail15_mass__token_mean`.

| Gate detector | PB all-8 | PB q4 | PB q8 | Clean accuracy | Error exact | Mean detector AUC |
|---|---:|---:|---:|---:|---:|---:|
| **tail15 mass / token mean — selected** | **36.8818%** | 36.0180% | **37.7456%** | **57.9304%** | **27.9604%** | **.792172** |
| tail50 mass / token mean | 36.8176% | 36.8374% | 36.7977% | 56.0221% | 27.6677% | .790979 |
| entropy / token mean — baseline replay | 36.6107% | **37.4205%** | 35.8009% | 50.0424% | 27.6677% | .742301 |
| q15 H1 / token mean | 36.6107% | **37.4205%** | 35.8009% | 50.0424% | 27.6677% | .742301 |
| q15 VE1 / token mean | 36.4845% | 37.3383% | 35.6307% | 50.2545% | 27.7127% | .750285 |
| q15 Hinf / token mean | 36.3743% | 37.3288% | 35.4197% | 49.5759% | 27.5326% | .736988 |
| raw -log p1 / token mean | 36.3594% | 37.2950% | 35.4238% | 49.5759% | 27.5101% | .738163 |

`q15 H1` and the native entropy detector yield the same decisions and metrics
under this contract, so H1 does not add a distinct gate. `Hinf` and raw
`-log p1` are both weaker than entropy at q=.3.

## Readout finding

| Readout family | Best feature | PB all-8 | Mean detector AUC |
|---|---|---:|---:|
| Whole-answer token mean | tail15 mass | **36.8818%** | .792172 |
| Mean of per-step Top10 | q15 VE0 | 36.2343% | .772636 |
| Whole-answer token Top10 | tail50 mass | 34.5840% | **.798257** |

Top10 raises detector separability for several features but does not improve
the final PB macro under one global other-fold q=.3 threshold. In particular,
tail50 Top10 has AUC .7983 but PB only 34.5840%. The answer-level distribution
and cell calibration matter, not only clean/error rank separation. Therefore
the earlier success of Top10 as a *step locator* does not transfer to the
whole-answer gate.

## Selected contrast and boundary

The selected gate improves the recalculated entropy baseline by +0.271 PB
points. The post-selection descriptive 95% group-bootstrap interval is
[-1.079,+1.594] points and crosses zero. It produces 1,137 different final
predictions: 474 are corrected only by tail15 and 275 only by entropy.

The current finalist's historical 36.6201% used the previously frozen entropy
thresholds. Recomputing the q=.3 thresholds on the complete current valid
locator roster gives 36.6107%; this 0.0094-point difference is calibration-
roster bookkeeping, not a score change. The cumulative integration replay uses
the exact historical thresholds for its entropy baseline and the exact
selection thresholds for tail15.

## Decision

`SELECT_TAIL15_TOKEN_MEAN_AS_NEXT_GATE_CANDIDATE_ON_DEVELOPMENT`.

The point result is positive and the detector AUC improvement is substantial,
but the final-F1 interval is inconclusive. Tail15 is therefore carried into the
cumulative integration check and a possible later q/calibration experiment; it
is not yet frozen as a confirmed replacement for entropy.

## Integrity

- 9 mechanism checks PASS; 33 candidates built by a label-free API.
- Real smoke covers every 8-cell x 5-fold combination.
- Full extraction covers 6,800/6,800 PB answers with finite detectors.
- Native entropy mean is reproduced bitwise for every PB row.
- Detector archive was frozen before target evaluation at SHA256
  `e3ccc87503df2629834348ab0e65727f8a43ccefd784eccf76588a21ce99a36a`.
- Same detector/readout/quantile rule in every cell; no package, GPU, cluster,
  Drive, commit or push action occurred during this experiment.
