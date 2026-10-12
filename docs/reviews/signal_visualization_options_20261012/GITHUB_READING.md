# Interactive signal gallery — Git reading copy

This delivery contains the six visualization alternatives for advisor section
4c, their display data, builder, template, source hashes and validation receipt.
The existing advisor v11 presentation remains unchanged.

Download/open `index.html` in a browser. It is a self-contained interactive
page with no external libraries or network requests. GitHub's file view shows
the source; this commit does not deploy a hosted website or update Claude.ai.

The source arrays used to build family intervals are intentionally outside Git.
Their hashes are listed in `MANIFEST.json`, and the frozen advisor v11 reading
delivery includes the large-file catalog. Existing `DATA.json` is sufficient
to inspect every plotted value without those arrays. Rebuilding from original
per-answer evidence requires the local frozen inputs described in `README.md`.

`MANIFEST.json` is the original construction receipt: its `published: false`
records the state at construction, not the later Git delivery. Publication is
established separately by the branch/commit and verified remote state.

Reviewers should inspect family means versus fusion scores, fixed channel
orientation, changing raw/position eligibility, member IQR versus confidence
intervals, and the paired STEP_INDEX contrasts. Numeric validation and browser
checks are documented in `REVIEW.md`.
