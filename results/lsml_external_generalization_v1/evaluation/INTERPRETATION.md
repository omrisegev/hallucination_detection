## Interpretation and decision

**Keep frozen step-level bank11 L-SML as the leading learned method.** Its source-fitted weights add measurable value on the external data. On Socratic, the improvements over its ordinary equal and learned-partition equal controls are positive after correction across 18 primary contrasts for both Qwen3-8B and QwQ-32B. The observed-disjoint panel preserves these gains. This is evidence for transferable fusion value, not a claim of benchmark leadership.

Hard2Verify is a qualified positive result: frozen L-SML reaches 43.670 Balanced F1 versus 37.751 CT7 and 39.758 partition equal. The corrected intervals support both gains. Its +2.787pp gain over ordinary equal has interval [-0.222,+5.839]pp; that particular claim remains inconclusive. The global label-shuffle null also includes this gain. The calibrated prediction prevalence contributes to null differences, so a positive raw delta alone is insufficient.

**Do not promote the current answer-local token L-SML implementation.** It loses to its ordinary equal control on both Socratic backbones: -1.285pp [-1.894,-0.665] and -1.087pp [-1.854,-0.313]. Native coverage is 2984/2995 per backbone, and the matched-native panels preserve the ordering; eleven short-answer fallbacks per cell do not explain the losses. This rejects this implementation as the primary method, not all CPU runtime learning. Its higher descriptive within-answer AUC than frozen L-SML on Hard2Verify (.6664 vs. .6343) and Socratic/QwQ (.7094 vs. .7011) does not establish superiority on the registered decision metric.

A cross-family qualification matters: local equal reaches 63.247 on Socratic/Qwen3, effectively tying the frozen-L-SML point estimate 63.221. Frozen L-SML is therefore the leading registered *learned* variant; it is not universally better than every averaging recipe. Its evidence for learned weights comes from its own matched step-level controls. Averaging remains a diagnostic control, not the proposed final method.

### What limits absolute performance

Frozen L-SML identifies 29.23% of error steps on Hard2Verify and 37.60%/39.39% on the two Socratic backbones. Correct-step recall remains 86.30%/87.83%/87.84%. The policy calls 20.22%/20.88%/21.49% of steps incorrect, while the gold error fractions are 41.94%/34.27%/34.27%. Both ranking and decision calibration deserve investigation; the current experiment does not identify a single cause.

The most concrete Hard2 diagnostic is **41/42 entirely correct answers receiving at least one error flag**. If every answer must keep its observed number of error predictions, even an oracle that reassigns those predictions using gold cannot exceed 52.673 Balanced F1. The weaker global-prevalence-only ceiling is 65.052. These are independently verified optimistic bounds on the observed decision budgets, not bounds on other thresholds or all L-SML methods. They motivate testing source-only alternatives that retain information about whether an entire answer is correct, including the effect of final answer standardization. They do not justify target-tuned thresholds or claim that calibration alone fixes the method. See `CALIBRATION_DIAGNOSTICS.json`, `BOUND_MATH_TEST.json`, and `independent_null/DECISION_BUDGET_CHECK.json`.

### Diversity already available in the collected traces

Frozen L-SML decisions disagree between Qwen3 and QwQ on 1751/26055 Socratic steps (6.72%); mean within-answer Spearman is .9003 over 2991 eligible answers. There is some complementary information, with substantial shared behavior. Answer-local decisions differ on 12.96% of steps, but that extra variation does not make the local method stronger on the primary metric. No cross-backbone fusion was fitted or evaluated here. `BACKBONE_COMPLEMENTARITY.json` records the descriptive analysis; its gold-assisted accuracy bounds are not PRMScore or deployable selectors.

### Next research boundary

Preserve this frozen bundle as the external-transfer baseline. Investigate calibration/answer-level information and any residual or cross-backbone extension on source development data first, using the retained telemetry and CPU computation. External labels have now been examined: subsequent optimization on these results is exploratory and needs a fresh held-out evaluation for a new generalization claim. MedPRMBench remains deferred; published critic/PRM reproduction remains outstanding. The literature context below still exceeds our Socratic scores and includes substantially higher Hard2 results, so no SOTA claim is supported.
