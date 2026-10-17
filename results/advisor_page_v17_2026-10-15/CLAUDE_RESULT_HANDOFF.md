# Results to add to v17: same-checkpoint continuous L-SML

Omri's required research question: **can continuous L-SML combine a PRM's own reward head with signals extracted from that same PRM checkpoint's backbone?**

The existing small-bank result around 69.38 used `0.5*z(PRM) + 0.5*mean(z(features))`. It is not continuous L-SML. Do not insert it in slide 12. Six-channel results containing Qwen3 telemetry are excluded. No additional feature-generating model is allowed in the displayed experiment.

## Slide 12: required result and graph

One fixed PRM checkpoint, matched examples/steps, with these three backbone channels:

1. `energy_level`
2. `hist_entropy_cusum_abs_series`
3. `realized_drv`

Show six arms: PRM alone; each of the three channels alone; continuous L-SML of those three channels; continuous L-SML of PRM risk plus those three channels. Use the project canonical continuous L-SML, not a new weighting formula. Report risk orientation, normalization and the prediction rule. Keep donor fitting and supervised feature selection outside evaluation source groups. The chosen three features were label-selected, so report that provenance.

Report official PRMScore (0–100 display scale) and within-answer AUC (0–1), matched row coverage, checkpoint, rule and evidence path for every arm. Prefer a common Q80 comparison and include native PRM as a clearly separate baseline. Present paired source-group intervals if supplied. Fit degeneracy or incomplete coverage must be shown explicitly; do not silently substitute equal averaging or select a better arm after evaluation.

The builder creates aligned PRMScore/AUC panels from `PRM_UPDATE.json:boosting.rows` when `status` is `complete`. Each row needs `label`, `checkpoint`, `rule`, `source`, `prmscore`, `auc`. `fusion` must stay `LSML_CONT`; `same_checkpoint` must stay true. Optional interval fields can be documented in the row/source; the present automatic plot displays point estimates only. If intervals are added to the visual, update the caption to match them.

If the intended fusion does not produce a usable result, set `status` to `failed`, retain the evidence and explain failure in `source`; the deck displays a failure state. If incomplete, leave `pending` and explicitly retain the placeholder.

## Slide 13: cross-PRM generalization

For each new PRM, use its own reward head and its own three backbone channels. Keep feature definitions and the continuous L-SML procedure fixed. Do not re-select features, directions or thresholds using the target evaluation labels. Validate reward and telemetry parity before accepting the result.

The archived cluster specification describes fixed-average `S3_own`. Those runs can supply telemetry but cannot be labelled L-SML results without computing the requested fusion. If they finish before the meeting, first confirm the recipe and evidence. Otherwise keep the dated running/pending status. Do not claim all planned models are currently running without job evidence.

`generalization.rows` needs `checkpoint`, `rule`, `source`, `prm_prmscore`, `lsml_prmscore`, `prm_auc`, `lsml_auc`, `same_checkpoint:true`, `fixed_features:true`. Use `status:complete` only for verified matched results. Scores use the same display units as slide 12. The builder renders paired native-vs-L-SML points per model in separate official-score/AUC panels, plus an explicit rule table.

After updating, rebuild with the repository's existing virtual environment. Keep v16 and frozen experiment packages unchanged. No commit, push, inference run or cluster mutation is authorized by this presentation handoff alone.
