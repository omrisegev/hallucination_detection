# Signal visualization alternatives for advisor section 4c

Open `index.html` directly in a browser. It is self-contained and makes no
network requests. The original advisor v11 page and frozen results are unchanged.

- A: separate family radars, with member-channel IQR bands.
- B: up to three individual feature profiles, with STEP_INDEX reference.
- C: point estimates and source-group intervals, or paired delta versus STEP_INDEX.
- D: individual channels within each family, with mean and IQR.
- E: descriptive family rank paths across error types.
- F: individual channel specialization across two selected error types.

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

See `REVIEW.md` for independent arithmetic and browser validation results.
