# Reproduction

Use the frozen data bundle and Step379 ridge NPZ/JSON pairs named in MODEL_AUDIT.json.
Verify MANIFEST.json and model hashes before use. Large scores remain local.

```powershell
python scripts/run_residual_moment_synthetic.py
python scripts/run_residual_moment_real.py
python scripts/evaluate_residual_moment_real.py
python scripts/report_residual_moment_fusion.py
```

The real scorer does not load correctness labels. The evaluator loads the main
repository evaluation contract. Five outer score archives plus ten pair-exclusion
archives provide complete nested calibration. The ten added triple-exclusion ridge
fits use the unchanged16384-sample deterministic fit rule. Existing single/pair
models and all original results are immutable. NPY data and frozen source models
must be available at their manifest paths; Git summaries alone cannot reproduce
raw scores without those local/Drive artifacts.
