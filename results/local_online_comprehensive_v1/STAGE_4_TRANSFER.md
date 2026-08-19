# S4 scorer-transfer and robustness audit

**Verdict: `REGRESSES_DIRECT_COMPETITOR`.**

Qwen3-8B and Llama-3.1-8B scorer copies are paired by source question in every interval.
Local direct reference: `max_entropy__step_top5mean`. Online direct reference: `iu28_registered`.

| task | method | primary | delta vs direct | grouped 95% CI | tier |
|---|---|---:|---:|---|---|
| local | qwen_prm | 0.7280 | — | — | B |
| local | qwen72b_critic | 0.5895 | — | — | B |
| local | finalist_global_detector_local_locator | 0.3662 | +0.0048 | [-0.0264, +0.0375] | A |
| local | max_entropy__step_top5mean | 0.3614 | — | — | A |
| local | gl_liu_v1_replay | 0.3364 | -0.0250 | [-0.0669, +0.0155] | A |
| local | max_entropy__persistent_q90_3 | 0.3280 | -0.0335 | [-0.0721, +0.0064] | A |
| local | max_entropy__peak | 0.3111 | -0.0503 | [-0.0783, -0.0202] | A |
| local | step272_twohead | 0.3078 | -0.0536 | [-0.0899, -0.0189] | A |
| local | mind_the_gap | 0.2646 | -0.0968 | [-0.1333, -0.0582] | A |
| local | qwen3_judge_control | 0.0913 | — | — | B |
| online | iu28_registered | 0.6104 | — | — | A |
| online | step272_twohead | 0.6082 | -0.0022 | [-0.0180, +0.0132] | A |
| online | mean_entropy | 0.5926 | -0.0178 | [-0.0496, +0.0144] | A |
| online | deepconf_w64 | 0.5922 | -0.0182 | [-0.0508, +0.0138] | A |
| online | max_entropy | 0.5921 | -0.0183 | [-0.0333, -0.0027] | A |
| online | finalist_global_detector_local_locator | 0.5882 | -0.0222 | [-0.0502, +0.0042] | A |
| online | deepconf_w32 | 0.5853 | -0.0251 | [-0.0575, +0.0063] | A |

Tier-B rows are same-question compute ceilings, not same-access deltas. Potential tokens remaining are not realized savings.
