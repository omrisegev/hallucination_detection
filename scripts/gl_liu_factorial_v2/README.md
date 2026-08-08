# GL-LIU factorial v2 runner

This runner compares the frozen global IU-PCR and DUFS-LIU scores with three
local heads:

- temporal LIU on the five-view core;
- DUFS-LIU on the five-view core;
- DUFS-LIU on 28 token-resolved curves.

It reports two controlled 2x2 matrices. It does not select a winner after
evaluation.

## Input

The cache root must follow the frozen GL-LIU v1 structure:

```text
cache/localization/processbench/
  pb_qwen3_4b/processbench_{gsm8k,math,olympiadbench,omnimath}.pkl
  pb_qwen3_8b/processbench_{gsm8k,math,olympiadbench,omnimath}.pkl
```

## Run

```bash
.venv/bin/python scripts/test_gl_liu_factorial_v2.py

.venv/bin/python scripts/gl_liu_factorial_v2/run.py \
  --cache-root cache/localization/processbench \
  --out-dir results/gl_liu_factorial_v2_reproduction

MPLBACKEND=Agg .venv/bin/python scripts/gl_liu_factorial_v2/report.py \
  --results-dir results/gl_liu_factorial_v2_reproduction

.venv/bin/python scripts/gl_liu_factorial_v2/verify.py \
  --results-dir results/gl_liu_factorial_v2_reproduction \
  --frozen-dir results/ours_only_localization_v1
```

Use a new output directory when reproducing the run. Do not overwrite the
recorded `results/gl_liu_factorial_v2` artifacts.

## Label boundary

All detector scores, token curves, and locator predictions are fitted and
hashed before the runner reads `row["label"]`. Labels are then used for
component metrics and the repeated calibration/evaluation threshold protocol.
The broad feature constructor accepts a cache row but never reads the label or
step spans.

## Expected headline

On the recorded eight cells:

- frozen GL-LIU v1: 31.36% ProcessBench F1;
- unified five-view DUFS-LIU: 31.72%;
- unified broad-28 DUFS-LIU: 29.03%;
- Mind the Gap control: 25.71%.

Read `results/gl_liu_factorial_v2/REPORT.md` before interpreting these values.
