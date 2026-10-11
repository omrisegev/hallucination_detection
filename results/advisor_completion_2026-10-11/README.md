# Advisor v11: GitHub reading delivery

Branch: `codex/advisor-v11-completion`. This is a reading delivery of the independently reviewed
PRMBench analysis, plus the next-agent count-estimator prompt. The Claude
Artifact itself has not been replaced by this publication.

Start here:

- [GitHub reading HTML](advisor_review_v11_github.html): identical scientific
  content to v11, with a publication banner and large-data links redirected
  to the catalog. It embeds all figures and opens locally in a browser.
- [Original reviewed HTML](advisor_review_v11.html): preserved byte-for-byte;
  its relative binary-data links require the full local bundle.
- [All sixteen Claude assignments](TASK_STATUS.csv).
- [Hebrew findings](FINDINGS_HE.md) and [independent review](review/INDEPENDENT_REVIEW.md).
- [Next-agent experiment prompt](../../docs/experiments/PRMBENCH_COUNT_ESTIMATOR_AGENT_PROMPT_2026-10-12.md).
- [Large-file catalog](RAW_DATA_CATALOG.md).

## Integrity and reproduction

The original analysis was completed locally on 11/12 October 2026. Its
historical `PUBLICATION_STATUS.json`, task statuses and review receipts remain
unchanged and describe that snapshot. They do not describe the later Git
publication. `README_ANALYSIS_SNAPSHOT.md` preserves its original README.

`PACKAGE_MANIFEST.json` describes the complete local bundle, including binary
files intentionally absent from Git. `GITHUB_DELIVERY_MANIFEST.json` separately
hashes this reading delivery and identifies the presentation-only transformations.

All CSV/JSON results, fitted-model JSON, scientific SVG/PNG figures, archived
source code and protocols are included. NPZ/PKL payloads are outside Git.
Downloaded inputs retain inspected Drive paths, sizes and hashes; generated
large outputs have no verified cloud copy. **Git alone is not a full runnable
numerical bundle.** See RAW_DATA_CATALOG.md before attempting reproduction.

For full replay use the matching full local bundle and the original README's
identity-checked commands. Never weaken its checksum guards merely to reuse
a partial checkout. Fresh computation belongs in a new results directory.

The evidence remains internal validation on previously inspected data. The
one failed N11 arm remains undefined. No new GPU experiment or external
confirmation is claimed. The estimator prompt is proposed future work;
its experiments have not yet been run.
