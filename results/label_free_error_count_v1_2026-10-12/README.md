# Label-free error-count study — GitHub reading delivery

Branch: `codex/label-free-error-count-v1`, based on advisor v11 commit `07cb7a45fb01ab62ade51fa5e59717c1f27fd8ff`.
The A–D family is complete. No primary candidate meets the combined success gate.
All evidence remains internal development validation on previously inspected data.

Start with [Claude review guide](CLAUDE_REVIEW.md), [Hebrew findings](FINDINGS_HE.md),
[reading HTML](report/index_github.html), and [independent review](review/FINAL_REVIEW.md).
The [original HTML](report/index.html) is preserved byte-for-byte.

## Delivery scope

All original CSV/JSON results, fitted-model JSON, checkpoints' identity/model
records, source code, logs, scientific SVG/PNG figures and review receipts are
included. NPZ/PKL payloads remain outside Git and are catalogued with sizes and
SHA256 in [RAW_DATA_CATALOG.md](RAW_DATA_CATALOG.md). The inherited advisor v11
reading delivery similarly retains its full numerical inputs outside Git.
**Git alone is a reading/audit delivery, not a complete numerical replay bundle.**

`PACKAGE_MANIFEST.json`, prediction seals and original review receipts describe
the full original local study, including omitted arrays. They remain unchanged;
they are not hashes of a complete clone. `GITHUB_DELIVERY_MANIFEST.json` hashes
this separate publication. [README_ANALYSIS_SNAPSHOT.md](README_ANALYSIS_SNAPSHOT.md)
preserves the exact original reproduction guide and local-publication status.

Verify the reading delivery using the configured local Python runtime:

```sh
.venv/bin/python scripts/package_label_free_error_count_v1_for_git.py --verify-delivery results/label_free_error_count_v1_2026-10-12
.venv/bin/python -m unittest discover -s tests -p 'test_error_count*.py'
```

The synthetic tests do not require the omitted real-data arrays. Worktrees reuse
the main checkout's `/Users/osegev/Desktop/hallucination_detection/.venv/bin/python`.
Full replay requires the original complete study, the frozen advisor package,
its expected runtime, and a new output directory. No new inference, fit,
threshold search, or external confirmation accompanies this publication.
The Claude Artifact was not modified and no review was sent to an external service.
