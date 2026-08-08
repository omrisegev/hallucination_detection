# GL-LIU v1 reproducibility runner

This directory preserves the exact analysis path used to create
`results/ours_only_localization_v1/`. Only the temporary absolute paths were
replaced with paths relative to the repository. The method, candidate grid,
development cells, selection rules, and output calculations were not changed.

## Input contract

`--cache-root` must contain:

```text
pb_qwen3_4b/processbench_gsm8k.pkl
pb_qwen3_4b/processbench_math.pkl
pb_qwen3_4b/processbench_olympiadbench.pkl
pb_qwen3_4b/processbench_omnimath.pkl
pb_qwen3_8b/processbench_gsm8k.pkl
pb_qwen3_8b/processbench_math.pkl
pb_qwen3_8b/processbench_olympiadbench.pkl
pb_qwen3_8b/processbench_omnimath.pkl
```

Every cache row must contain the generated trace, aligned token telemetry,
ProcessBench step spans, and the benchmark label. The score constructors do not
read the label. Labels are opened for development selection and evaluation.

## Reproduce into a new directory

Do not overwrite the frozen official result. Run:

```bash
.venv/bin/python scripts/gl_liu_v1/run.py \
  --cache-root cache/localization/processbench \
  --out-dir results/ours_only_localization_v1_reproduction
```

On the cluster, use the configured experiment environment instead of `.venv`.
The required packages are NumPy, SciPy, and scikit-learn, plus this repository.

Then build the advisor report from the frozen official CSVs:

```bash
.venv/bin/python scripts/build_gl_liu_report.py
```

The runner writes component rankings, per-cell metrics, the selected detector
and locator, diagnostics, and pre-label score hashes. The plots are generated
by `scripts/plot_ours_only_localization_v1.py`.

## Important boundary

This is a research reproduction runner, not a deployment API. It includes the
Mind the Gap control so that the comparison is calculated from the same rows
and splits. GL-LIU itself does not consume the Mind the Gap score.
