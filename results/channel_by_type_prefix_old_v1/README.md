# Single channel x PRMBench error type (PRE-FIX input) — 2026-10-11

Which single channel catches which PRMBench error type. Mean within-answer AUC per error type for each of the 56 channels
(label-free sign = sign of correlation with the cell LEVEL mean, as in `results/monday_types_singles_v1/run.py` part B),
plus the step index as a reference. PRMBench cell only; 745-758 answers per type; point estimates, no intervals.

**Input is the OLD telemetry (before the empty-think-block prefix fix).** The corrected per-channel features exist only on
AIRCC (`source_bank_thinkfix_mask1/MANIFEST.json` in `results/thinkfix_rescore_v1`). Rerun on the corrected FEATURES before
using these numbers in a report. Inputs live in the `ssl-pseudolabel-residual-v1` and `readout-quickest-detection-v1`
worktrees (via `results/monday_nested_banks_v2/dev_run.py`); do not delete those worktrees.

An AUC below 0.5 means the label-free sign rule points the wrong way for that channel. CPU only; labels used for evaluation only.
