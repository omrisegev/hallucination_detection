# Teacher-forcing repeatability: existing smoke/full overlap

Question: could rerunning identical inference provide complementary inputs to L-SML?
The existing runs already provide two executions for each smoke example. A read-only
comparison found exactly equal saved telemetry values for all 36 answer/backbone
pairs (51,637 token observations). No new inference or quality evaluation was run.

| Cell | Paired answers | Tokens | Exactly equal telemetry pairs |
|---|---:|---:|---:|
| Hard2Verify / Qwen3 | 12 | 29,277 | 12 |
| Socratic / Qwen3 | 12 | 11,180 | 12 |
| Socratic / QwQ | 12 | 11,180 | 12 |

All saved telemetry fields were compared, including top50 token IDs/logprobs,
actual-token logprobs, entropy, logsumexp and spans. Measurement/runtime fields
and run IDs were excluded. Model/checkpoint, precision, attention implementation,
input hash, protocol and prompt/template identities match. The collector calls
model.eval(), fixes all input tokens and samples no generated continuation.

Evidence: [REPEATABILITY_SMOKE_VS_FULL.json](../../results/lsml_external_generalization_v1/REPEATABILITY_SMOKE_VS_FULL.json).
Script: [review_external_repeated_telemetry.py](../../scripts/review_external_repeated_telemetry.py).
This is two executions per example on the 12-example timing selection in each cell,
not three repetitions or a full-population repeatability claim. Equality concerns
saved values; the complete vocabulary distributions were not persisted or compared.

Identical repeats currently offer no observed diversity. Numerical differences can
occur across kernels, platforms or batch shapes, but are not automatically useful
uncertainty or independent evidence. See [PyTorch numerical accuracy](https://docs.pytorch.org/docs/main/notes/numerical_accuracy.html).
The earlier QwQ prefix-layout discrepancy compared different computation shapes;
it did not demonstrate variability between these repeated full-answer runs.

A more promising existing source is Qwen3 versus QwQ on the same Socratic answers.
Their complementary localization value remains untested. Freeze the analysis before
external quality evaluation; use development data to select new variants. If more
views are needed, test fixed, semantics-preserving prompt variants while scoring the
same original steps. Distinguish these changed conditionings from identical repeats.
Temperature transforms or duplicated feature columns do not supply independent
model evidence. Test improvement of learned fusion against matched fixed-weight
controls, not merely increased variance. No such new experiment is launched here.
