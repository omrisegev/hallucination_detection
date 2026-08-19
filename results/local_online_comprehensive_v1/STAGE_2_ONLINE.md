# S2 causal Online feature screen

**Verdict: `PARITY_WITH_DIRECT_COMPETITOR`.**

Every prefix score was rebuilt from truncated telemetry. No full-trace broad curve was sliced.
Direct Tier-A reference on the same development rows: `step272_twohead`.
Frozen S2 selection: `o_family6__fast_slow`.

| method | AUROC@64/128 | delta vs direct bar | 95% CI |
|---|---:|---:|---|
| o_family6__fast_slow | 0.6020 | +0.0121 | [-0.0262, +0.0478] |
| o_broad28__level_fast_slow_area_persistence | 0.6020 | +0.0121 | [-0.0399, +0.0651] |
| o_raw9__fast_slow | 0.6010 | +0.0111 | [-0.0280, +0.0490] |
| o_broad28__fast_slow | 0.6007 | +0.0108 | [-0.0275, +0.0491] |
| o_family6__level_slow | 0.5979 | +0.0080 | [-0.0251, +0.0426] |
| o_broad28__slow_area_persistence | 0.5961 | +0.0062 | [-0.0553, +0.0681] |
| o_family6__level_fast_slow_area_persistence | 0.5923 | +0.0024 | [-0.0463, +0.0497] |
| o_broad28__level_slow | 0.5917 | +0.0018 | [-0.0349, +0.0384] |
| o_raw9__level_fast_slow_area_persistence | 0.5904 | +0.0005 | [-0.0529, +0.0532] |
| step272_twohead | 0.5899 | — | — |
| o_raw9__level_slow | 0.5887 | -0.0012 | [-0.0415, +0.0373] |
| o_raw9__slow_area_persistence | 0.5836 | -0.0063 | [-0.0744, +0.0623] |
| o_family6__slow_area_persistence | 0.5824 | -0.0074 | [-0.0644, +0.0498] |
| o_raw9__shortlong_innovation_recovery | 0.5766 | -0.0133 | [-0.0636, +0.0371] |
| iu28_registered | 0.5746 | -0.0152 | [-0.0447, +0.0131] |
| deepconf_w32 | 0.5719 | -0.0179 | [-0.0646, +0.0295] |
| deepconf_w64 | 0.5666 | -0.0233 | [-0.0674, +0.0224] |
| mean_entropy | 0.5661 | -0.0238 | [-0.0684, +0.0204] |
| o_broad28__shortlong_innovation_recovery | 0.5558 | -0.0341 | [-0.0979, +0.0301] |
| max_entropy | 0.5542 | -0.0357 | [-0.0683, -0.0018] |
| o_family6__shortlong_innovation_recovery | 0.5407 | -0.0492 | [-0.1104, +0.0109] |

Warning thresholds are calibrated separately per family on clean calibration traces. Warnings are one-sided and non-withdrawable; `potential_tokens_remaining` is not a realized-savings claim.
