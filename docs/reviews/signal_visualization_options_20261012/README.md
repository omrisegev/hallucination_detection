# Signal visualization alternatives for advisor section 4c — A–H

Open `index.html` directly in a browser. It is self-contained and makes no
network requests. The original advisor v11 page and frozen results are unchanged.

- A: separate family radars, with member-channel IQR bands.
- B: up to three individual feature profiles, with STEP_INDEX reference.
- C: point estimates and source-group intervals, or paired delta versus STEP_INDEX.
- D: individual channels within each family, with mean and IQR.
- E: descriptive family rank paths across error types.
- F: individual channel specialization across two selected error types.
- G: specialization plus conditional binary-error dependence as bubble size,
  or an explicit dependence Y axis; all, outside-family, or single-channel peers.
- H: raw AUC/PRMScore paths by type, for families or the features selected in B.

Shared controls select all 56 channels or N16, and unrestricted within-answer
AUC or within-position-bin AUC. Wide charts scroll horizontally on narrow screens.
Each chart can export SVG. Choice buttons only record a local preference; send
the chosen letter in chat to request integration into the advisor page.

## Interpretation

Family profiles average the AUCs of their member channels. They do not measure
the AUC of a fused family score. IQR shows variation between member channels,
not statistical uncertainty. Family confidence intervals resample 707 entire
canonical source groups, using 5,000 draws with seed 20261011, on per-answer
mean-member AUC. STEP_INDEX deltas share answer eligibility and bootstrap draws.
Intervals are pointwise and do not adjust for multiple comparisons.

G uses empirical phi correlation of classification-error indicators separately
within valid and erroneous steps, with donor-fitted per-feature Q80 thresholds.
The display takes the larger of the two mean absolute conditional correlations,
using the same eligible peers in both classes. Constant margins and missing
classes stay undefined. This is a descriptive, step-weighted pairwise association
diagnostic, not proof of joint conditional independence, an independence test,
or a measure of expected fusion benefit. Anticorrelated errors are dependent too.
Error rates and eligible-peer counts are in each tooltip.

H's family PRMScores average member-channel PRMScores, not fused-family outputs.
MS uses valid-step F1 and is visually separated from the eight macro-F1 curves.
Position control changes AUC eligibility only; feature PRMScore stays on all
steps with fixed source-OOF decisions. No confidence intervals are plotted in H.

Frozen label-free channel signs are retained, including sub-chance values.
MS is omitted because no erroneous steps are available to define AUC.
Raw and position-controlled results use different eligible answer populations;
their differences should not be read as a same-population causal intervention.
These remain development results from previously inspected data.

## Reproduction

From the repository root, using the existing environment and frozen local inputs:

```sh
.venv/bin/python scripts/build_signal_visualization_gallery.py
```

`MANIFEST.json` records source, builder, template, and HTML hashes. `DATA.json`
contains all display values; `FAMILY_SUMMARY.csv` provides family summaries.
Large source NPZs remain in the existing frozen results package.

G/H numerical inputs were created by `scripts/build_signal_gallery_extensions.py`
in `results/signal_gallery_extensions_2026-10-12/`; the gallery builder embeds
that sealed `extension.json` and copies it as `EXTENSION_DATA.json`. See the
extension package README for fresh-destination reproduction. The original A–F
display payload is byte-identical to the published base; `BASE_V1_RECEIPT.json`
records its hashes and commit. The A–H Git reading delivery adds G/H to that
base. Display JSON, CSV summaries, code and review accompany the HTML; raw
NPZ arrays remain local and are catalogued in `GIT_DELIVERY.json`.

See `REVIEW.md` for independent arithmetic and browser validation results.
