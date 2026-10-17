# Clustered IU-PCR on PRMBench — October 12, 2026

Open `index.html` for the complete comparison. The specified hybrid loses to both
matched baselines on both banks. This is a bounded development result on already
inspected PRMBench data, not external confirmation or a rejection of all possible
hierarchical fusion methods.

The order is continuous L-SML clustering, IU-PCR relevance selection and survivor
fusion within clusters, then IU-PCR relevance selection and survivor fusion across
cluster scores. The exact donor L-SML baseline partition is reused; no clustering
parameters or signs are chosen with labels. The registered relevance rule is
`rho_hat >= max(0.05 * var_y, max(rho_hat)/3)`. Selection happens once per level,
and survivor refits use the maintained two-component L2 IU-PCR configuration.
Single inputs pass through; pairs use continuous signed L-SML. These exceptions
also apply after selection. No top-three minimum is restored. Initially
undersized groups cannot estimate rho and therefore skip relevance selection.

All 10 bank/fold fits cover all 6,969 answers / 94,203 steps, with the unchanged
source-group folds and corrected answer-z feature inputs. Labels are read by the
evaluation stage only after the prediction seal. Main within-answer AUC values:

- N16: IU-PCR 0.780467; continuous L-SML 0.778901; hybrid 0.771874.
- N56: IU-PCR 0.764067; continuous L-SML 0.770411; hybrid 0.735463.

The hybrid trails both baselines within every individual fold. All four paired
primary intervals remain negative after Bonferroni adjustment across the two
banks and two baselines. Q80 pooled noncontrol macro-F1 also declines. The Q80
rule uses the training .8 quantile of answer-standardized fused scores and
applies that frozen threshold to each held-out answer's standardized score.
This matches the recent count-study readout. Scores are not threshold-tuned.
PRMScore in this package means pooled noncontrol macro-F1; it does not mean an
average of category-level scores. Type breakdowns are separate.

N16 retains all inputs at both levels in four folds; fold 3 retains 15 features
within clusters and 4/6 clusters, leaving 9 active features. N56 retains 47/56
features and 8/8 clusters in every fold. This shows that the N16 decline is not
confined to folds with deletion. Separating the effects of hierarchical weights,
normalization and feature deletion would require a distinct registered ablation;
none was added to this comparison.

`PROTOCOL.json` freezes the design before fitting. `INPUT_FREEZE.json` hashes
inputs and runtime sources. `PREDICTION_SEAL.json` authenticates every prediction
and fitted model before evaluation. `RESULTS.json` contains metrics, source-paired
intervals, types and retention telemetry. `checkpoints/*/MODELS.json` contains
feature names, exact source splits, rho estimates, thresholds, survivor masks,
small-cluster modes and affine coefficients. `sources/` preserves runtime bytes.
`AUDIT.json` verifies all seals, unchanged frozen inputs/runtime, baseline replay,
fresh hybrid replay, full two-stage holdout reconstruction and Q80 flags.

From the main repository, run the numerical replay with:

```sh
/Users/osegev/Desktop/hallucination_detection/.venv/bin/python scripts/audit_clustered_iu_prmbench_v1.py
/Users/osegev/Desktop/hallucination_detection/.venv/bin/python -m unittest discover -s tests -p test_clustered_iu_prmbench.py -v
```

The fitting entry point is `scripts/run_clustered_iu_prmbench_v1.py`; it refuses
an existing destination. Five scientific unit tests passed; pytest is not
installed, so tests used Python's standard unittest runner. Fresh hybrid replay
and held-out stage reconstruction pass in all ten folds, within 1e-9; both
baselines reproduce saved advisor v11 scores within 1e-9.

Local CPU work only, uncommitted and unpushed. No installation, GPU inference,
cluster job or remote/Drive mutation occurred. Earlier dirty work and frozen
advisor/count-study evidence remain preserved.
