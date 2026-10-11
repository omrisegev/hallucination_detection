# Small verified analysis inputs for the advisor v10 review

Retrieved read-only on 2026-10-11. No cluster connection, GPU run, package install, or project source change. Inputs were size-inspected with `rclone lsjson` before downloading from the configured `gdrive:` remote.

## Corrected score population

The corrected frozen N16 export is `/private/tmp/thinkfix-review-080877eb/results/thinkfix_rescore_v1/prmscore_inputs/bank_N16.npz`: 6,969 answer IDs and 94,203 steps. Metadata is beside it in `prmbench_meta.json`. These IDs, answer step counts, classification, and raw 1-based `error_steps` are the join contract. Some published raw error-step indices are out of range; use the repository's official evaluator semantics, without silently repairing them.

## Authentic source-question groups

`CANONICAL_GROUPS.json`, 4,336,280 bytes, SHA-256 `63c767ffd373b37fdf08ff9ca7711260daa8ed527266ad5e619b7275015a5d94`.

Drive source: `gdrive:hallucination_detection/consolidated_results/local_backup_2026-10-01/main_checkout_results/results/localization_source_group_audit_v1/CANONICAL_GROUPS.json`. Modification time: 2026-09-07T02:55:03.520Z.

The SHA-256 matches the downloaded historical `BRIDGE_MANIFEST.json`. This is grouping version `question_source_components_v2`. Its `rows` contain `row_id`, `source_seed`, question hashes, and `canonical_group_id`. The rule connects official source-seed suffixes and exact whitespace-normalized observed question text; modified questions retain their source seed. `correctness_fields_used` is false.

All 6,969 corrected N16 IDs match exactly one row. There are **707 canonical source groups**; every group is present in the existing local `FOLDS_V2.json` at `/Users/osegev/Desktop/hallucination_detection/.worktrees/fusion-independence-atlas-v1/results/localization_source_group_audit_v1/FOLDS_V2.json`. Do not replace this equivalence relation with IID answer IDs or raw classification/source IDs.

`QUESTION_METADATA.json`, 2,614,622 bytes, SHA-256 `ef721699f4357512928e75912f6c06b70c5f4b95f54458c10f37cfc02c873570`, downloaded from the same Drive directory; it matches both the bridge hash and `CANONICAL_GROUPS.metadata_sha256`. It records old telemetry raw SHA-256 `b934afad0889ffacf0f4420f885ad52ddcf08f2124e506a21cd24f216bd170be`. It contains question hashes, **not question or step text**.

`BRIDGE_MANIFEST.json` (3,069 bytes) and `ARTIFACT_VALIDATION.json` (524 bytes) are accompanying historical provenance. Their historical PASS state is not a new full audit of the corrected pipeline. The new alignment check confirms the current export's exact answer identity and the inherited source equivalence relation, not raw model telemetry.

## Supervised PRM reference

`prmbench_prm.pkl`, 2,053,470 bytes, SHA-256 `95be425c66d4dadc6a8e03567ad66ef2933aee27eb7967bf7200139c56518232`.

Drive source: `gdrive:hallucination_detection/cluster_results/prmbench_qwen25math7b_full/prmbench_prm.pkl`; modification time 2026-08-10T22:37:58Z. The hash exactly matches the tracked `results/data_readiness_2026_08_11/validations/prmbench_qwen25math7b_predictions.json`.

The pickle is a dictionary of 6,969 records containing `idx`, `source_idx`, `classification`, `category`, `n_steps`, `error_steps`, `rewards`, `labels`, and timing. Join by `idx`, never dictionary iteration order. All corrected export IDs match; all step counts, error-step lists, and classifications agree (zero mismatches). It contains no question/step text.

The 94,203 scalar rewards are finite validity scores in [0.0003452301025390625, 1.0]. The original runner uses `labels = 1 if reward >= 0.5 else 0`, where 1 means VALID. Every cached label matches this rule. There are **40 rewards exactly equal to 0.5**. Native error flags therefore use `reward < 0.5`; with error risk `1 - reward`, the equivalent boundary is strictly `risk > 0.5`. Ranking comparisons can use `-reward` or `1 - reward`.

## Corrected LR cache (additional alignment gate required)

`LR_N16_thinkfix_mask1.pkl`, 4,909,901 bytes, observed SHA-256 `e73260d33db3a289766bcce44f681d0428bbdb5088d90eeb41e408ad3a7ac083`.

Drive source: `gdrive:hallucination_detection/cluster_results/thinkfix_rescore_v1/thinkfix_mask1/lr_reference/cache/N16.pkl`; modification time 2026-10-10T22:08:36Z. An independent historical expected hash was not found. The observed hash pins the downloaded copy but must not be described as frozen-manifest verification.

The cache declares bank N16, 6,969 answers, 94,203 steps. It stores per-answer `auc`, `top1err`, `c1`, and `argmax` for LR and shared controls, plus fold coefficients/diagnostics. LR within-answer AUC is 0.8014887597254806. It has no answer IDs or full per-step LR scores. **Ordering was independently verified** against the corrected NPZ for ALL_EQ, DS_FILTER_EQ, and MEAN_MOM_ga: all 6,969 argmax positions match, all top1 error indicators match, per-answer AUC differences are exactly zero, and AUC undefined masks agree. This supports the paired top-step disagreement analysis. Reconstructing full LR scores requires corrected FEATURES plus saved fold diagnostics.

## Inputs still queued

Corrected full FEATURES exists at `gdrive:hallucination_detection/cluster_results/thinkfix_rescore_v1/source_bank_thinkfix_mask1/FEATURES.npz`, **53,985,301 bytes**; not downloaded under this task's 20 MB small-asset limit. Its tracked manifest declares SHA-256 `f14ab5d68360cd30819077051329678944bc631b94f2510bc67c9b697bf0948f`.

Channel-by-type, class-conditional channel/group correlations, reconstructed LR per-step scores, group-vote MoM/TCI prevalence, and corrected entropy gates need additional features/votes/raw inputs. Position-only controls and paired source-group comparisons of frozen scores are feasible from the inputs already available. Text overlap and step length in characters/tokens require actual step text or a verified span export; question hashes alone cannot support them.
