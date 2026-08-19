# γ̂3 corrected recompute — conclusion (2026-08-16)

**Verdict: the channel does not survive its own correction. Stop; Step B's
orientation premise is the number that failed, and Step C is closed.**

Scope: Step A of the atomic (group-free) NRM plan — close the technical debt
on the b-coupled cubic orientation channel established in Step 252
(`atomic_orientation_reply_2026-08-13.md`) before anything is built on it.

## What was owed

Step 252 reported two load-bearing numbers for γ̂3:

- atomic level: pooled `cos(γ̂3, g*) = +0.76`, 13/17 sign agreement;
- family level: sign-bit margin `≈ 0.56`, "eight times the all-ones margin",
  the basis for replacing the deployed family-NRM sign rule.

Both were computed with the cubic probe orthogonalized against `{1, b}` only.
The same memo's §6 states the correction any future b-coupled estimator owes:

> any future b-coupled estimator must Gram-Schmidt φ3 against {1, b, φ2} (the
> {1,b}-only version leaks a balance-dependent quadratic term that can flip
> its sign) and winsorize b.

## Design

- **Fidelity control first.** The `{1,b}` probe was reproduced verbatim from
  `atomic_orientation_diag.py:hermite3_moment` and had to hit the frozen
  numbers before anything else was believed.
- **Corrected probe.** `φ2 = GS(b² | {1, b})`, `φ3 = GS(b³ | {1, b, φ2})`,
  both computed numerically (not through the unit-variance closed form) so
  orthogonality is exact in-sample after winsorization. `b` winsorized at a
  symmetric quantile then re-standardized. Registered primary: 1%.
- **Attribution.** The two ingredients of the correction were crossed
  (φ2-orthogonalization × winsorization ∈ {0, 0.5, 1, 2.5, 5, 10}%) so the
  verdict can name which one moves the number.
- **Artifact controls.** Every cell pooled under three conventions — raw
  moment (the original), unit-RMS probe, and direction-only — because removing
  φ2 shrinks ‖φ3‖ by a cell-dependent factor that reweights the n-weighted sum.
- Labels enter only through the Fisher reference directions and the cosine
  readouts built from them, exactly as in Step 252. No estimator reads a label.

## Results

Fidelity control reproduces Step 252 exactly: pooled cos **+0.7617**
(ref +0.7617), signs **13/17** (ref 13/17), per-cell median **+0.5129** at 87%
positive (ref +0.5129 / 0.870).

### Atomic, 23 source cells / 17 frozen atoms

| probe | pooled cos(γ̂3, g\*) | signs | per-cell median |
|---|---:|---:|---:|
| `{1,b}` only (Step 252) | **+0.7617** | 13/17 | +0.5129 |
| corrected, no winsorization | **−0.0806** | 7/17 | +0.1535 |
| corrected, winsor 1% (primary) | **+0.3350** | 12/17 | +0.4470 |
| corrected, winsor 5% | +0.4648 | 11/17 | +0.5612 |

### Family sign bit — the component Step C would have replaced

The teacher says the deployed bit is already correct:
`cos(v_neutral, family g*) = +0.90`.

| probe | 6-family margin | 5-family margin | sign correct? |
|---|---:|---:|:--:|
| all-ones (deployed) | +0.0650 | +0.0594 | yes |
| `{1,b}` only (Step 252) | +0.4889 | **+0.5532** | yes |
| corrected, winsor 1% | **−0.1702** | **−0.1903** | **no** |
| corrected, winsor 2.5% | +0.1018 | +0.1058 | yes |
| corrected, no winsorization | −0.1689 | −0.1743 | no |

The memo's quoted ≈0.56 is the **5-family** restriction (+0.5532); the
deployed calibration spans all 6 provenance families (+0.4889). The collapse
is identical under both bases, so it is not a basis artifact.

### Attribution: it is the φ2 removal, not the winsorization

| φ2 removed? | winsor | raw pooling | unit-RMS | direction-only |
|:--:|---:|---:|---:|---:|
| no | 0.000 | +0.7617 | +0.7837 | +0.7254 |
| no | 0.050 | +0.7064 | +0.7171 | +0.6810 |
| no | 0.100 | +0.6830 | +0.6845 | +0.6517 |
| **yes** | **0.000** | **−0.0806** | **−0.1254** | **−0.1261** |
| yes | 0.010 | +0.3350 | +0.2757 | +0.2141 |
| yes | 0.050 | +0.4648 | +0.4317 | +0.3793 |

- Winsorization alone is **harmless**: the original probe holds +0.68…+0.76
  across the whole 0–10% range.
- Removing φ2 at zero winsorization takes the alignment to −0.0806, and it
  stays negative under all three pooling conventions — **not** a scale artifact.
- Winsorization then partially restores the corrected probe *monotonically in
  the knob* (+0.24 → +0.34 → +0.43 → +0.46), never reaching the original. That
  is tail suppression buying alignment, not a recovered channel.
- Median `cos(φ3_baseline, φ3_corrected)` across cells is **+0.56** — the two
  probes are only about half the same measurement.

## Interpretation

Most of the reported +0.76 was carried by exactly the balance-dependent
quadratic leak the Step-252 memo itself identified. The genuine cubic
b-nonlinearity channel is real but much weaker (~+0.34 at the registered
winsorization), and **at family level its sign is not determined by the data**:
it flips between winsor 1% (−0.17) and 2.5% (+0.10).

This is not a marginal miss. At the registered primary setting the proposed
Step-C replacement bit would have **flipped the deployed family NRM into the
wrong orientation** — a component whose confirmation rests on a CI floor of
+0.07pp. The all-ones bit's 0.065 margin remains thin, but it is at least
correct and knob-free; the γ̂3 replacement is neither.

## Consequences

- **Step C (family sign-bit robustification): closed by measurement.** The
  "8× better margin" claim inverts once the leak is removed.
- **Step B (retrospective kill-test): not started.** It selects
  `j* = argmax_j |⟨v_j, γ̂3⟩|` and orients by `sign(⟨v_j*, γ̂3⟩)`. Both the
  selection and the sign are functions of the vector that just lost half its
  alignment and all of its family-level sign stability. Running it as
  specified would test a premise already known to be broken.
- **What is *not* claimed**: this does not close b-coupled orientation in
  general. It closes the *reported magnitudes*. A cubic channel at cos ≈ +0.34
  may still carry usable information; what it cannot support is a single
  transported sign bit or a direction selection, which is what both remaining
  steps needed from it.

## Open question for Omri

The honest reading is that the atomic-orientation program lost its instrument.
Given the Step-252 transport wall (any global direction is worth ~+0.1pp max
on the heterogeneous pool), and now the loss of the orientation channel that
made the atomic route look reachable, the question is whether the atomic
(group-free) NRM route should stay open at all, or whether the corrected
γ̂3 at +0.34 is worth one narrower test on the homogeneous domains
(ProcessBench) only — where Step 252 measured its single positive transported
result (+0.39pp, 4/0 on PB Llama, at the *uncorrected* estimate, so that number
is now also in doubt).

No further work was done past this gate.

## Reproduction

```
python results/gamma3_corrected_2026-08-15/gamma3_corrected_recompute.py
python results/gamma3_corrected_2026-08-15/gamma3_decomposition.py
python results/gamma3_corrected_2026-08-15/family_margin_basis_check.py
```

~80s each; artifacts `RESULT.json`, `DECOMPOSITION.json`,
`FAMILY_BASIS_CHECK.json` plus logs in the same directory.
