# Matched controls and innovation mechanism analysis

All 13,769 development answers; 10,000 source-group bootstrap draws.

| Method | PB % | Within | PRMScore |
|---|---:|---:|---:|
| mean__H0lim_VE0_VE075_VE1 | 37.4749 | 0.753436 | 0.634412 |
| append_innovation__H0lim | 39.8314 | 0.760293 | 0.638830 |
| append_innovation__VE075 | 37.7799 | 0.755932 | 0.635784 |
| innovation__H0lim | 35.6929 | 0.747636 | 0.587641 |
| RBM12_logit | 37.1253 | 0.745204 | 0.622215 |
| append_duplicate_H0lim | 37.0518 | 0.751985 | 0.634290 |
| append_centered_H0lim | 37.0518 | 0.751985 | 0.633722 |
| append_shuffled_prefix_H0lim | 36.9548 | 0.752373 | 0.634723 |
| baseline_crossfold_gate | 37.4761 | 0.753436 | 0.634412 |

True-prefix innovation was independently reconstructed with a scalar running-sum loop for every token.
Centered/duplicated H0lim isolate reweighting and answer offset. The shuffled-prefix control breaks chronology.
Control design follows the initial development finding. This is mechanism evidence, not independent confirmation.
