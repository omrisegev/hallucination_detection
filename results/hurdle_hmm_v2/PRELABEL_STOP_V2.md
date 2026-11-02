# Hurdle HMM v2: pre-label stop (2026-11-02)

The v2 synthetic stop rule (PREREG v2, section 7) failed. No real-data fit was run and no real label was read.

Rule: median HH-BAYES macro-F1 >= max(GAP, SHARE, GATE_GAP, THR) - .005 in >= 5 of 6 (pi, delta) settings,
and <= 1 of 12 fits at a bound. Result: 4 of 6 settings pass; 1 of 12 fits at a bound (pass).

| pi | delta | HH-BAYES | best control | difference | pass |
|---|---|---|---|---|---|
| .10 | 1 | .536 | .561 (THR) | -.025 | no |
| .10 | 2 | .738 | .712 (GATE_GAP) | +.026 | yes |
| .60 | 1 | .628 | .639 (THR) | -.011 | no |
| .60 | 2 | .823 | .790 (THR) | +.033 | yes |
| .95 | 1 | .675 | .670 (GATE_GAP) | +.005 | yes |
| .95 | 2 | .850 | .809 (THR) | +.040 | yes |

Source: SYNTHETIC_V2.csv (2 seeds per setting, N = 1,500, m = 200).

Interpretation:
- At delta = 2 the model adds 2.6-4.0 macro-F1 points over the best calibrated control.
- At delta = 1 it does not beat a global threshold (THR) or a gated GAP when pi is low or medium.
- Real data are at delta about 1 (D3: d = .8-1.15). So the expected real gain is zero or negative,
  even though the data come from the model itself here.
- With 2 seeds the per-setting medians are noisy (seed spread up to .04 at pi = .95, delta = 1).
