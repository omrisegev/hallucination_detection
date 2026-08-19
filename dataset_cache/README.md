# dataset_cache/

Raw per-cell inference cache metadata and, for the older published cache set, payloads tracked
via **Git LFS** (`*.pkl` under this directory — see `.gitattributes`). This is a deliberate,
curated exception to the `cache/repgrid/` "not for git" policy in the root `.gitignore`.

The 23 payloads added during the 2026-08-19 consolidation under
`four_localization/` and `repgrid/pb_llama31_8b/` are the narrow exception: GitHub rejected
their LFS upload after the repository exhausted its LFS budget. On 2026-08-20 every payload
was checksum-matched to its canonical `gdrive:hallucination_detection/cluster_results/`
copy, then removed from the still-unpublished integration history. Their manifests and full
SHA-256 inventory remain in Git; exact remote mappings and the verification result are in
`DRIVE_BACKUP_2026_08_20.json`. Local copies may remain present but are ignored.

Each `dataset_cache/repgrid/<cell>/` directory holds one (dataset, model, temperature) cell:
a `manifest.json` (config + provenance) and one or more `raw_<dataset>_T<temp>.pkl` files. Each
pkl is a dict keyed by sample index; per CLAUDE.md's rich-save schema, entries carry
`question`, `full_text`, `label`, `token_entropies`, `token_spilled_energies`,
`top_k_logprobs`, and `gen_token_ids` (schema varies slightly by cell vintage — some older
cells only have `token_entropies`).

## What's included, and why

| Category | Cells | Status |
|---|---|---|
| 24 in-scope grid (`scripts/inscope_cells.py:INSCOPE`) | QA + math cells behind the headline U-PCR / GOOD_6 / L-SML numbers | canonical |
| GPQA | `gpqa_llama70b`, `gpqa_llama8b`, `gpqa_mistral7b`, `gpqa_qwen72b`, `gpqa_r1distill8b`, `trace_gpqa_r1qwen7b` | out of scope for the thesis (Step 191: features at chance) but archived |
| RAG | `rag_2wikimultihopqa_*`, `rag_hotpotqa_*`, `rag_narrativeqa_*`, `rag_natural_questions_*` (5 models each) | out of scope for the thesis (Step 191) but archived |
| EDIS / math-competition | `edis_amc23_*`, `edis_gsm8k_*`, `edis_math500_*` pilots, `cache/edis_aime24` → `dataset_cache/edis_aime24` | EDIS replication pilots (Steps 116-183) |
| ProcessBench / localization | `pb_qwen3_4b`, `pb_qwen3_8b`, `ars_*`, `evdrop_*` | Extension F, Track A — step-level error localization ("Mind the Gap" reproduction), see `experiment/step-localization` branch |
| Four-front localization/RAG validation | `four_localization/*` | Canonical raw telemetry for ProcessBench, PRMBench, HLE, RAGTruth/GASP/Lettuce, and RefChecker validation |
| ProcessBench Llama-3.1-8B | `repgrid/pb_llama31_8b` | Canonical four-dataset localization cache used by the local/online application analyses |

Not included: `cache/_backup/` (stale duplicate snapshot), `cache/_incoming/` (transient staging
area), and `inside_coqa_llama7b*` (Step 216 — REJECTED cell, degenerate generation from a chat
template applied to a base checkpoint; kept locally for reference, not archived here).

`INVENTORY_2026_08_19.json` records the size, mtime, and SHA-256 of the 36 payload and metadata
files considered during the consolidation (3,351,678,118 bytes). The 13 metadata files remain
ordinary Git objects; the 23 pickle payloads are recovered from the Drive mappings above.

## Loading

After restoring a Drive-backed payload to its manifest directory, point any existing loader at
the path instead of `cache/repgrid/`:

```python
from spectral_utils.io_utils import load_cache
load_cache("dataset_cache/repgrid/<cell>/raw_<dataset>_T<temp>.pkl")
```

`scripts/inspect_cell.py` and `scripts/labelfree_standing_report.py` etc. take a directory
argument, so pass `dataset_cache/repgrid/<cell>` in place of `cache/repgrid/<cell>`.

## Split files

GitHub's Git LFS caps individual objects at 2GB. Two GPQA raw pkls exceed that
(`gpqa_r1distill8b/raw_gpqa_T1.0.pkl`, `trace_gpqa_r1qwen7b/raw_gpqa_T1.0.pkl`) and are stored as
`<name>.pkl.part-00`, `.part-01`, ... (also LFS-tracked, see `.gitattributes`). Reassemble before
loading:

```bash
cat raw_gpqa_T1.0.pkl.part-* > raw_gpqa_T1.0.pkl
```

```python
# Windows / no cat available
import glob
parts = sorted(glob.glob("raw_gpqa_T1.0.pkl.part-*"))
with open("raw_gpqa_T1.0.pkl", "wb") as out:
    for p in parts:
        with open(p, "rb") as f:
            out.write(f.read())
```
