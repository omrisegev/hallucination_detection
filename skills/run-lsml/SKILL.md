---
name: run-lsml
description: Run L-SML v2 fusion locally on downloaded feature pkls, save results to archive, and optionally render an HTML report. Use when you want to evaluate the current feature subset without going to Colab.
---

# run-lsml — Local L-SML Runner

Runs `scripts/run_lsml_local.py` on locally cached feature pkls, saves results to `results/archive.jsonl`, and optionally renders an HTML report via `scripts/render_html.py`.

## Prerequisites

Five pkl files must exist in `local_cache/` (download once from Drive):
```
MyDrive/hallucination_detection/consolidated_results/math500_res.pkl
MyDrive/hallucination_detection/consolidated_results/gsm8k_res.pkl
MyDrive/hallucination_detection/consolidated_results/gpqa_res.pkl
MyDrive/hallucination_detection/consolidated_results/rag_feats_all.pkl
MyDrive/hallucination_detection/consolidated_results/qa_res.pkl
```

## How to invoke

When the user types `/run-lsml [args]`, do the following:

### Step 1 — Parse args from the user's message (if any)

Supported optional args:
- `--features feat1 feat2 ...` — override the 5-feature default subset
- `--label "some label"` — human-readable tag for this run (default: `{N}feat-lsml-v2`)
- `--html` — also render HTML report after the run
- `--compare` — render comparison HTML across all archive runs (skips running fusion)

### Step 2 — Check that local_cache/ has the pkls

Run:
```
python scripts/run_lsml_local.py --help
```
If `local_cache/` is missing or empty, stop and tell the user which files to download from Drive and where to put them.

### Step 3 — Run the fusion script

Construct the command from the parsed args and run it:
```
python scripts/run_lsml_local.py --data-dir ./local_cache/ [--label "..."] [--features ...]
```

Capture and print the full output including the results table.

### Step 4 — Render HTML (if --html or --compare was passed)

- `--html` → `python scripts/render_html.py --latest`
- `--compare` → `python scripts/render_html.py --compare`

Tell the user the output path of the HTML file.

### Step 5 — Report

Print the AUROC table from the script output. Highlight:
- Any domain/cell with AUROC ≥ 0.80 (strong signal)
- Any domain/cell with AUROC < 0.55 (near-chance, may be structurally incompatible)

If this is a comparison run, note which feature subset performed best per domain.

## Output files

| File | Description |
|------|-------------|
| `results/runs/{timestamp}_{label}.json` | Full structured result for this run |
| `results/archive.jsonl` | Append-only log of all runs — diff feature subsets here |
| `results/latest.csv` | Always-overwritten summary table |
| `results/report_latest.html` | Latest run rendered as HTML (if --html) |
| `results/report_compare.html` | All runs side-by-side (if --compare) |

## Default feature subset (Step 121-123 ablation)

```python
GOOD_FEATURES = ['epr', 'low_band_power', 'sw_var_peak', 'cusum_max', 'spectral_entropy']
```

All 16 features and their signs are defined in `FEATURE_SIGNS` in `run_lsml_local.py`.
