# Length axis, attack 1 — the length-calibrated readout is closed

Runs `scripts/diagnostics/length_axis_by_subset_v1.py`; output
`results/token_probability_fusion_v1/LENGTH_AXIS_BY_SUBSET.json`. Development-only.

## The question, and the decision rule fixed before the run

Step 420 measured that neutralising the step-length coupling costs 8.16 PB points
(CT7 41.19 → LX7 33.03) and that restoring an explicit log-length view recovers
only 3.35 (LX8 36.38). Both are **macro** numbers; the per-subset decomposition was
never computed, and the mean is exactly what hid the length asymmetry.

Decision rule, stated in advance:

- **cost uniform** → the calibrated readout is a dead tool, close it;
- **negative on short chains, positive on long ones** → it is a length-*conditional*
  mechanism, and the deficit against Chen et al. has a candidate repair.

## Rebuild gate

The five Step-420 arms are rebuilt from the frozen banks restored from the Drive
backup. The script refuses to print a single new number unless all five published
macro values reproduce within 5e-4. They do — max |diff| **0.0354 pp**.

Past the gate, CT7 is reported from the *frozen* score vector, not the rebuild: the
rebuild differs at 1.15e-06 (a float32 round-trip), enough to flip an argmax on a
near-tied step and move a per-cell SLA by a few tenths. With the frozen vector every
CT7 number here is identical to Stage A's, and macro is exactly 41.19.

## Result — gate-free SLA by subset

| cell | CT7 | LX7 | LX8 | CT7+LEN | LEN | LX7−CT7 | LX8−CT7 |
|---|---|---|---|---|---|---|---|
| GSM8K 4B | 48.31 | 39.61 | 44.44 | 48.31 | 41.06 | −8.70 | −3.86 |
| GSM8K 8B | 47.34 | 32.85 | 40.10 | 45.89 | 41.06 | −14.49 | −7.25 |
| MATH 4B | 35.69 | 26.77 | 28.45 | 37.37 | 33.16 | −8.92 | −7.24 |
| MATH 8B | 36.87 | 25.59 | 29.63 | 39.56 | 33.16 | −11.28 | −7.24 |
| OlympiadBench 4B | 39.03 | 27.23 | 32.07 | 38.73 | 24.81 | −11.80 | −6.96 |
| OlympiadBench 8B | 37.67 | 28.14 | 32.68 | 36.91 | 24.81 | −9.53 | −4.99 |
| Omni-MATH 4B | 37.15 | 26.09 | 30.04 | 37.55 | 28.06 | −11.07 | −7.11 |
| Omni-MATH 8B | 37.02 | 27.80 | 30.70 | 37.68 | 28.06 | −9.22 | −6.32 |
| **SHORT** (gsm8k, math) | 42.05 | 31.21 | 35.66 | 42.78 | 37.11 | **−10.85** | **−6.40** |
| **LONG** (olympiad, omni) | 37.72 | 27.31 | 31.37 | 37.72 | 26.44 | **−10.41** | **−6.35** |
| macro PB (gated) | 41.19 | 33.03 | 36.38 | 41.40 | 35.14 | | |

## The interaction — paired source-group intervals, 10,000 draws

| arm | short chain | long chain | **long − short** |
|---|---|---|---|
| LX7 − CT7 | −10.85 [−14.24, −7.54] | −10.42 [−12.83, −8.03] | **+0.43 [−3.68, +4.60]** |
| LX8 − CT7 | −6.41 [−9.41, −3.52] | −6.35 [−8.40, −4.32] | **+0.06 [−3.57, +3.65]** |
| CT7+LEN − CT7 | +0.74 [−0.62, +2.10] | −0.00 [−1.06, +1.06] | −0.74 [−2.44, +0.94] |

## Verdict — attack 1 is closed

**The cost is uniform.** Length calibration hurts by ~10.4–10.9 pp on short and long
chains alike, and the interaction is +0.43 pp with an interval straddling zero. With
the log-length view restored the cost is ~6.4 pp, again equal on both. The hypothesis
that neutralisation hurts GSM8K while helping exactly where we lose to Chen et al. is
**falsified**: it hurts OlympiadBench and Omni-MATH by the same amount.

By the rule fixed before the run, the tool is dead and the line closes.

**Honest limit.** The interaction interval is ±4 pp, so this establishes "no
detectable length-conditional benefit", not "provably identical". What rules the
mechanism out is not the width but the location: a readout that selectively rescued
long chains would show a large positive interaction, and the point estimates are
+0.43 and +0.06 — centred on zero, not a suppressed positive.

**Two things worth keeping.** `LEN` alone scores 41.06 on GSM8K and 37.11 averaged
over the short subsets, against 26.44 on the long ones — step length by itself is a
strong locator on short chains, which is Step 420's "length is evidence, not only
bias" caveat showing up per subset. And `CT7+LEN` is neutral everywhere
(+0.74 / −0.00), so the +0.21 macro gain Step 420 reported for it is not concealing
a subset where the explicit length view earns its place.

This leaves **attack 2, the derivative channel**, as the only live route on the
length axis.
