# Methods for the top-50-plus-tail protocol correction

The complete mathematical definition is in
[`../ragtruth_evidence_contrast_v1/METHODS.md`](../ragtruth_evidence_contrast_v1/METHODS.md).

This directory is the formula-faithful correction of the approved contract.
Its entropy features use the saved top-50 token categories plus one aggregate
tail category. The original blind run used the separately saved
full-vocabulary entropy at these two positions.

The correction was executed after the original test labels had already been
opened. The formula came from the approved plan and was not chosen by looking
at labels, but this run is not a new blinded confirmation. Both versions are
kept so the deviation and its effect remain auditable.

