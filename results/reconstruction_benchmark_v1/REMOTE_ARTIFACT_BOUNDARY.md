# Remote artifact boundary for the science runs

This branch preserves the code snapshot and a compact evidence layer for the application-oriented reconstruction runs. The complete local result tree is approximately 15 GB and is not suitable for GitHub.

Included in Git:

- A/B preparation, fit, score, and evaluation certificates;
- stage and release manifests needed to identify the executed inputs and code;
- aggregate metrics, contrasts, coverage, frontier, bootstrap-summary, and panel-status tables;
- compact unified-reporting tables and reviewed winner-reference contrasts.

The included releases cover EDIS, external final-answer evaluation, first-error localization, fixed-prefix prediction, LEASH stopping, RAG evidence evaluation, and the certified unified-reporting bridge.

Intentionally excluded from Git:

- `private_control/` and private labels;
- multi-gigabyte `FIT_INPUT` snapshots and copied source assets;
- per-example predictions, per-question traces, policy-execution logs, and localization-decision relations;
- large fitted arrays, bootstrap draw arrays, and duplicate A/B payloads whose identities remain recorded in the manifests.

The omitted artifacts remain in the local/Google Drive artifact workflow. Their manifests and cryptographic digests are retained here so that Claude or another reviewer can understand exactly what ran without treating GitHub as the raw-data store.
