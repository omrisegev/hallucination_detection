# Runbook

Run from the repository root with the project environment.

```bash
.venv/bin/python scripts/fixed_application_pipeline_experiment.py rag --out results/fixed_application_pipelines_v1
.venv/bin/python scripts/fixed_application_pipeline_experiment.py reasoning --out results/fixed_application_pipelines_v1
MPLBACKEND=Agg .venv/bin/python scripts/fixed_application_pipeline_experiment.py report --out results/fixed_application_pipelines_v1
.venv/bin/python scripts/test_fixed_application_pipelines.py
```

Or run all three stages:

```bash
MPLBACKEND=Agg .venv/bin/python scripts/fixed_application_pipeline_experiment.py all --out results/fixed_application_pipelines_v1
```

Large raw caches remain outside Git. The output contains score hashes and the
machine-readable metrics used by the report.
