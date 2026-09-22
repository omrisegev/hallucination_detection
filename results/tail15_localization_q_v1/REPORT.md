# Frozen Renyi + tail15 development method v1

Status: **COMPLETE / REVIEW PASS — DEVELOPMENT FROZEN**

## Final method

- Locator: q15 `H0lim/VE0/VE0.75/VE1`, per-view Top10, natural-unit equal mean.
- ProcessBench gate: missing top-15 mass, whole-answer Top10, one uniform q=0.33.
- q objective: maximum official ProcessBench all-eight exact-localization macro-F1.
- PRMBench: unchanged frozen locator and nested PRMScore q=.8 contract; no no-error gate.

The selected point lies on a shallow development plateau: PB all-eight is
37.4597% at q=.31, 37.4697%
at q=.32, 37.4749% at q=.33,
37.4174% at q=.34, and
37.3969% at q=.35. The exact q=.33 maximum is a
development choice, not evidence that the hundredth is stable externally.

## Cumulative ProcessBench comparison

| Method | PB all-8 | PB q4 | PB q8 | Clean accuracy | Error exact |
|---|---:|---:|---:|---:|---:|
| Starting static locator + entropy mean q=.3 | 36.1674% | 36.9200% | 35.4148% | 0.500848 | 0.270599 |
| Locator update only | 36.6201% | 37.4205% | 35.8196% | 0.500848 | 0.276677 |
| Gate update only | 36.8759% | 36.7183% | 37.0335% | 0.603478 | 0.253940 |
| **Complete frozen development method** | **37.4749%** | **37.3703%** | **37.5795%** | **0.603478** | **0.260018** |
| Complete method at math q=.40 | 36.6736% | 36.2207% | 37.1265% | 0.681510 | 0.238181 |
| Historical tail15 mean PB q=.3 diagnostic | 36.8818% | 36.0180% | 37.7456% | 0.579304 | 0.279604 |

Final-minus-start delta: +1.307pp; family-wise
99.00% paired whole-source-group interval
[-0.484, +3.067]pp.

## PRMBench locator comparison

| Locator | Within-answer AUROC | Fold AUROC | Pooled OOF AUROC | PRMScore q=.8 |
|---|---:|---:|---:|---:|
| Starting static fusion-before-Top10 | 0.751115 | 0.721815 | 0.721407 | **0.634805** |
| Current q15 per-view-Top10 locator | **0.753436** | **0.722708** | **0.722305** | 0.634412 |

The ProcessBench q was selected on these development labels. The complete
method is frozen for external/new-model confirmation, not claimed as an
unbiased ProcessBench generalization result.
