# Remote worktree handoff — 2026-08-27

This file is the entry point for reviewing the local research work that was published to GitHub after the reconstruction and application runs.

## Start here

1. **Full reconstruction code and compact evidence:** branch `codex/reconstruction-benchmark-v1`, commit `ead5cdf9e029d4a42cfc1293bdc59eb82e786b5a`.
2. **Application science releases:** branch `codex/reconstruction-science-results-v1`, commit `d9827a835bd2b43fc107d297834c8e190224b9bd`.
3. **Visual reconstruction report:** `results/reconstruction_benchmark_v1/releases/2026-08-24_frozen24_v1/reporting_v2/2026-08-24_frozen24_v1/07_reports/REPORT.html` on the reconstruction branch.
4. **Advisor narrative and deep dives:** `docs/meetings/Advisor_Update_Aug26_2026.md` and `docs/meetings/advisor_update_aug26_2026/` on the reconstruction branch.

The two `REMOTE_ARTIFACT_BOUNDARY.md` files under the result roots explain which large/private artifacts were deliberately left out of Git.

## Published method and experiment branches

- `codex/graph-geometry-selection-v1` — `483d4150c71fb7c3e8d32be7416d1490193be1af`: graph geometry, Family-NRM/PGRD, selection and post-report experiments with their compact result snapshot.
- `codex/iu-graph-smoothing-ablation-v1` — `8829fd8e3304d5a608a7fc23dc687d005fd1ba38`: IU graph-order/smoothing ablations, including score arrays and plots.
- `codex/deem-b3-moe-gating-v1` — `8779197346a5ecc4424f0c20bdff473742e04584`: experimental B3 routing, MoE, local-descent and residual/PGRD challengers. These are preserved experiments, not promoted methods.
- `codex/residual-graph-deem-24cell-v1` — remote `6d9f9b8b8f63570fda8501a75fd7cff6355ce8df`: residual-graph/DEEM 24-cell development.
- `codex/ciw-multipop-benchmark-v1` — `66abed7a5ab0c336f4fcc3558c6a5a8b3b53cd67`: CIW multi-population benchmark worktree.
- `codex/stg-su-input-v1` — `66abed7a5ab0c336f4fcc3558c6a5a8b3b53cd67`: STG/SU input study worktree.
- `codex/fair-paper-exact-comparisons-v1` — `bc296ff499773403be400c9e112740bd0af35ce5`: fair and paper-exact comparison infrastructure.
- `codex/heterogeneous-regressor-threeway-v1` — `ef3154e39eebf22fc67a92d50f18c7400342ff08`: three-way heterogeneous regressor experiments.
- `codex/unified-causal-iu-v1` and `codex/unified-causal-subset-search-v1` — `d3ca3a4a5b44360c20e6b68105969d7c50536aab`: unified causal/IU and subset-search development.
- `codex/whitebox-layer-fusion` — `7cdc39a81004c8ba1a536c019177da3525d3b24c`: white-box layer-fusion work.
- `codex/su-pcr-24cell-audit-v1` — `0a631b28c61496cffb06b32972506cbadfc2cec1`: the audit worktree had no unpublished code delta; its conclusions remain in the reconstruction documentation.

## Reconstruction-run coverage

The detached prefix, localization, LEASH, RAG, unified-reporting and winner-contrast run commits are all ancestors of `codex/reconstruction-benchmark-v1`. The formerly detached science snapshot `b6fdddac5a70d2d55f43a295b50dd25d7fc846f6` is preserved as the parent of `codex/reconstruction-science-results-v1`.

The compact science branch includes aggregate evidence for:

- EDIS and external final-answer evaluation;
- first-error localization;
- fixed-prefix prediction;
- LEASH stopping;
- RAG evidence evaluation;
- the certified unified-reporting bridge and winner-reference contrasts.

## Deliberate exclusions

- The local LFS backup branch was not pushed because it requires uploading 23 large LFS objects and the GitHub LFS budget is blocked.
- Raw source overlays, private controls/labels, multi-gigabyte fit inputs, per-example predictions and very large diagnostics remain in the local/Google Drive artifact workflow.
- The local residual-graph worktree was older than the existing remote branch, so the newer remote head was preserved rather than force-pushed.
