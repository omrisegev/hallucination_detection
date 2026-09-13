# CLAUDE.md — MV_EPR Spectral Hallucination Detection

## Matched supervision clarification - 2026-09-11

The matched token-Top10 coefficient-update experiment is complete in this
worktree. Both arms share the same frozen answer-local RBM and externally
fitted13-coefficient correction family; only the training loss/label access
differs. Neither corrected arm is strictly answer-only. Do not describe the
experiment as from-scratch supervised versus unsupervised training, or a
performance ceiling. See PROGRESS.md for outcomes and the next unlaunched
objective-alignment proposal. First_near_max remains closed for this RBM path.

## Omri RBM readout clarification - 2026-09-11

Use original argmax for the active RBM research path. The first_near_max
isolation experiment is closed; do not tune or adopt it for RBM. Historical
near-max scores for other methods remain archived references. The tested
position-conditioned correction is not adopted after full-data regression;
see PROGRESS.md for the completed supervised position diagnostic and its limits.


## Omri update - 2026-09-10: temporal probability fusion authorized

In this isolated worktree the active stage is
`docs/experiments/DIRECT_PROBABILITY_TEMPORAL_V3.md`. The user approved revisiting
all fusion mechanisms on direct probabilities, fixed settings, and adding
ordered lagged inputs, level/change and two-axis fusion. First complete the
18-arm temporal representation comparison on all13,769 localization rows;
do not claim that its shrinkage arms are full Joint or B3. Those mechanisms and
historical24 transfer remain explicit continuing work. Chat-first results;
no new HTML reports or performance pass/fail thresholds. Preserve old outputs.
This current authorization supersedes older no-new-grid restrictions below.

## Omri decision update - 2026-09-08: consolidate before new experiments

User approved `docs/experiments/RESEARCH_CONSOLIDATION_20260908.md` after the
pause. Complete frozen historical Joint first, then full sampling, plus an
independent audit of Claude's fixed gate. Return a Hebrew HTML reflection and
machine-readable ledger BEFORE a new improvement experiment. No expanded grid.
PB is the first improvement target; keep PRMB and historical comparator panels.
Freeze mean raw entropy q0.3 as candidate: fusion answer-local, gate externally
calibrated without labels, detector/q selected using development outcomes.
Preserve source results. Do not infer current liveness from old RUN_STATE flags.

## Omri decision update - 2026-09-07: subsets are feasibility checks only

This supersedes earlier short-cycle instructions about drawing research
conclusions from small cohorts. Small runs may check implementation, numerical
stability, runtime and feasibility; they must not establish improvement, rank
research directions, or promote a candidate. The pilot/full regression in
Step327 showed that the previous subset was not a reliable full-population
performance estimate. Do not repeat subset sweeps as evidence of progress.

Use the complete matched development benchmark for comparative findings,
with fixed labels/source groups, historical leaders, simple controls, per-cell
and aggregate metrics, failures/coverage, and paired uncertainty. Freeze the
candidate and selection rule before a separate untouched confirmation.
Full cached evaluation is development evidence, not publication confirmation.
Evaluating every answer does NOT authorize pooling their fitting data: the
primary method still fits each answer alone; pooled/calibrated comparisons
must declare their different access. Preserve running frozen experiments.

## Omri decision update - 2026-09-07: reasoning benchmark first

The full matched REASONING benchmark and corrected historical-leader refits
are priority1 and a research-method requirement, not an optional final report.
Analyze existing experiments in parallel to identify real algorithmic gains
and application gains (local ranking, first-error/no-error decisions, coverage
and runtime). Do not open another sweep merely because a pooled AUC improved.
LOCA, Diverging Flows, KalmanNet and Shlezinger-inspired extensions are LOW
priority supporting backlog. Fusion remains central; answer-only fit is primary.

ProcessBench and PRMBench are the active core in separate metric panels.
RAG/grounding, claim/span and agent tasks are outside the active claim; preserve
historical artifacts. MR-GSM8K is the first external candidate to audit after
method lock, including GSM8K source overlap; ReTraceQA/GR-Ben are later transfer
candidates pending executable data contracts. A new scorer on previously seen
PB answers is not untouched confirmation. Preserve historical24 as a separate
later transfer with predeclared reasoning strata and its unchanged full macro.

Risk selection means highest mean-entropy8-token fitting windows, not known
errors: m=min(N,max(32,ceil(N/2))). Refit on selected rows and score ALL rows.
TOKENS and TRAJECTORY refer to the same observation axis: distinguish row
selection, chronological processing and combination of multiple fusion curves.

Reasoning, source conversation and initial gain analysis are recorded in
`docs/research_notes/reasoning_benchmark_decisions_2026-09-07.md`.
This amendment does not modify the frozen Step321 scoring protocol or run.

## Active priority update from Omri - 2026-09-07

Full matched localization benchmarking now takes priority over additional
small development sweeps. See `docs/experiments/LOCALIZATION_FULL_BENCHMARK_V3.md`
and `results/localization_full_benchmark_v3/METHOD_REGISTRY.json`. The unchanged
answer-only anchor pass has started on13769 model-answer rows; this is not
the completed shortlist/historical comparison. Keep IU/LIU/L-SML/Joint and
historical dedicated localizers in the continuing registry. Corrected-label
and source-fold refits, coverage and within-answer evidence are required.
See PROGRESS.md for live handles. Full cached data remain development data.

## Session start
**Always read `PROGRESS.md` before doing anything else.** It has the current experiment status, what's running, what's fixed, and what to do next. Do not rely on git log alone — PROGRESS.md is the handoff document.

**Review [SUPERVISED_ORACLE_CORRECTION.md](file:///C:/Users/omris/TAU/hallucination_detection/SUPERVISED_ORACLE_CORRECTION.md)** to understand the ML evaluation guidelines (specifically regarding class weight balancing and avoiding the `cross_val_predict` calibration pitfall) established after correcting the Step 142 Logistic Regression baseline.

Shortcut: type `/session-start` to run the full initialization sequence automatically.

---

## Slash commands available

| Command | When to use |
|---------|-------------|
| `/session-start` | **Start of every session** — reads PROGRESS.md, git status, last HISTORY steps, prints priority action |
| `/update-docs` | After completing work — drafts HISTORY.md Step N entry + PROGRESS.md update, then commits |
| `/new-cell` | Adding an analysis/inference cell — generates correct three-branch pkl reload template |
| `/nadler-audit` | Before/after Nadler fusion — validates all 4 invariants (views, z-score, ρ, sign) |
| `/colab-setup` | New notebook — generates Cell 1 + Cell 2 + gptqmodel stub for the requested model |
| `/notebook-audit` | Before committing a notebook — spawns sub-agent to check for 8 common bugs |
| `/aircc-setup` | One-time AIRCC cluster bootstrap (first login is manual; config/dirs/prefetch automated) |
| `/aircc-submit` | Submit an inference job to the AIRCC cluster (sync code + sbatch + job id) |
| `/aircc-status` | Check AIRCC job state — squeue/sacct + log tail + verdict |
| `/aircc-fetch` | Fetch finished cluster results + validate the rich-save pkl schema |
| `/paper-digest` | Read/re-read a paper under `papers/` — checks the cache (`papers/index.md`) first, only extracts + digests if not already cached |

---

## Sub-agent patterns

Spawn sub-agents for these recurring tasks to avoid context pollution:

**Cache Explorer** — Use when user asks "what's in the cache?" or "which phases are done?":
```
Spawn subagent_type=Explore:
"List all .pkl files under /content/drive/MyDrive/hallucination_detection/[path].
For each pkl, report: filename, size, top-level keys, and whether any values are None.
Classify each as VALID / STALE (all-None) / PARTIAL (some None). Report in a table."
```

**Notebook Reviewer** — Use before committing any notebook with new cells (or just use `/notebook-audit`):
```
Spawn subagent_type=Explore with the notebook path + the 8-bug checklist from /notebook-audit.
Returns findings without modifying files. Synthesize before deciding what to fix.
```

**Results Extractor** — Use when user asks "what are our current numbers?" without re-running:
```
Spawn subagent_type=Explore:
"Read consolidated_results/results_all.pkl (or results_summary.csv).
Print a table: domain | model | Nadler AUROC | CI | best_subset.
Sort by Nadler AUROC descending. Flag any None results."
```

**Rule**: Brief the agent with exact file paths + what to return. Synthesize the result yourself before acting on it. Never delegate the decision — only the data gathering.

---

## Project layout

| Path | Purpose |
|------|---------|
| `spectral_utils/` | Core package — all helpers live here |
| `HISTORY.md` | Step-by-step experiment log (append only, see format below) |
| `PROGRESS.md` | Handoff summary — update at end of each session |
| `Research_Directions.md` | Full thesis roadmap — directions, hypotheses, decision gates, priority order |
| `Experiments_Report.md` | Clean summary for advisors |
| `*.ipynb` | Colab notebooks — run on Colab A100, never locally |
| `*.pdf` | Research papers — read when referenced or newly added |

---

## spectral_utils: the non-negotiable rule

**NEVER inline helpers in notebooks.** Every feature extractor, fusion function, model loader, and grading function lives in `spectral_utils`. If something is missing from the package, add it to the right module and commit *before* using it in a notebook.

| Module | Contents |
|--------|---------|
| `feature_utils.py` | `compute_spectral_features`, `compute_stft_features`, `compute_time_domain`, `extract_all_features`, `sw_var_peak_with_window`, `sw_var_peak_adaptive`, `FEAT_NAMES` |
| `fusion_utils.py` | `zscore`, `boot_auc`, `nadler_fuse`, `simple_average_fusion`, `best_nadler_on` |
| `model_utils.py` | `load_model`, `fmt_prompt`, `generate_full`, `token_entropies_from_scores`, `free_memory` |
| `data_loaders.py` | `load_gsm8k/math500/gpqa/hotpotqa/trivia_qa/webq` + prompt + grading functions |
| `io_utils.py` | `load_cache`, `save_cache` |

---

## Colab setup — standard Cell 1 for every new notebook

```python
import os, sys, shutil

# Set BEFORE any torch import. Expandable segments let the allocator reclaim
# physical pages from a freed model, which is what makes a 70B BNB load after
# unloading a smaller model possible at all. Without this, fragmentation OOMs.
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Persist HuggingFace cache to Drive — saves a 36 GB AWQ re-download (~15 min)
# on every runtime restart. Set BEFORE any HF import.
os.environ['HF_HOME'] = '/content/drive/MyDrive/hf_cache'

REPO_DIR = '/content/hallucination_detection'

# Remove stale clone if spectral_utils is missing
if os.path.exists(REPO_DIR) and not os.path.exists(os.path.join(REPO_DIR, 'spectral_utils')):
    shutil.rmtree(REPO_DIR)

if not os.path.exists(REPO_DIR):
    os.system(f'git clone -b master https://github.com/omrisegev/hallucination_detection.git {REPO_DIR}')
else:
    os.system(f'git -C {REPO_DIR} pull -q')

if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

# autoawq is safe here. gptqmodel is NOT — it rewrites numpy/pyarrow .so files
# during install and corrupts them mid-session. Defer gptqmodel to the cell that
# loads the model (see "Model loading rules" below).
os.system('pip install -q "transformers>=4.40" accelerate datasets bitsandbytes autoawq scipy')

from spectral_utils import (
    load_model, generate_full, free_memory,
    extract_all_features, sw_var_peak_with_window, sw_var_peak_adaptive,
    FEAT_NAMES, load_cache, save_cache,
    zscore, boot_auc, nadler_fuse, simple_average_fusion, best_nadler_on,
)

# Force-load datasets (and through it pyarrow + pyarrow.parquet) into memory
# BEFORE the later gptqmodel install. C extensions can't be unloaded from a
# running Python process, so freezing them in memory now makes on-disk rewrites
# inert. Without this, the first lazy pyarrow import after gptqmodel install
# fails with `IpcReadOptions size changed`.
import datasets  # noqa: F401 — imported for side-effect

print('spectral_utils imported OK')
```

**NEVER use `pip install git+https://...`** — it ignores the branch, conflicts with Colab's module cache, and has failed repeatedly in this project.

---

## Model loading rules

- **AWQ / GPTQ models** (ID contains `awq` or `gptq`): `load_model(model_id, quantize_4bit=False)` — package auto-detects, uses `dtype=torch.bfloat16`. `device_map="auto"` is safe because AWQ weights are already quantized on disk (~36 GB for 72B). **Requires both `autoawq` AND `gptqmodel`** — gptqmodel provides the `AwqMarlinLinear` (Marlin fp16) kernel; without it, AWQ will fail or use a slow fallback.
  - **CRITICAL — install order + flags**: `autoawq` goes in Cell 1 (safe alone). `gptqmodel` MUST be installed in the model-load cell with `--no-deps`, AFTER `import datasets` has frozen pyarrow in memory. Installing both together in Cell 1 corrupts numpy and pyarrow on disk (`cannot import name '_center'`, `IpcReadOptions size changed`); installing gptqmodel without `--no-deps` ALSO corrupts transformers (`cannot import name 'divide_to_patches' from 'transformers.image_transforms'` — partial upgrade where new `image_processing_backends.py` references symbols missing in the still-old `image_transforms.py`). All three are unrecoverable in-session — only a runtime restart fixes them. The model-load cell should look like:
    ```python
    # gptqmodel's logger does `import pcre`. The PyPI package is pypcre (C extension
    # over libpcre2, no Py3.12 wheel — needs apt libpcre2-dev to build from source).
    # gptqmodel only uses pcre.compile()+.sub() on a trivial ANSI-escape pattern,
    # so stub pcre with stdlib re — bulletproof, no system libs, no C build.
    import re as _re, types as _types
    _pcre = _types.ModuleType('pcre')
    for _fn in ('compile','match','search','findall','sub','split','fullmatch'):
        setattr(_pcre, _fn, getattr(_re, _fn))
    _pcre.error = _re.error
    for _flag in ('IGNORECASE','MULTILINE','DOTALL','VERBOSE','UNICODE','ASCII'):
        setattr(_pcre, _flag, getattr(_re, _flag))
    sys.modules['pcre'] = _pcre

    # gptqmodel runtime deps that --no-deps skips. All pure-Python:
    # - device-smi (zero deps, eagerly imported by gptqmodel/utils/device.py)
    # - tokenicer (would otherwise pull transformers>=5.x and break Colab's 4.x)
    # - defuser (depends on pypcre — our stub covers it at runtime; --no-deps
    #   avoids pip trying to build pypcre from source against libpcre2-dev)
    # - logbar (zero deps; safe with full pip)
    os.system('pip install -q --no-deps device-smi tokenicer defuser')
    os.system('pip install -q logbar')
    os.system('pip install -q --no-deps gptqmodel')
    mdl, tok = load_model(MODEL_ID, quantize_4bit=False)
    ```
    `--no-deps` is safe because gptqmodel's heavy runtime deps (torch, numpy, transformers, accelerate, safetensors, pyarrow, datasets) are already in Colab. The four packages above are the only gptqmodel-specific runtime deps not in stock Colab Py3.12. **Do not** try `pip install pcre` (different obsolete package) or `pip install pypcre` (needs libpcre2-dev, slow source build) — the stdlib `re` stub is the reliable path.
- **BNB 4-bit (70B+)**: `load_model(model_id, quantize_4bit=True)`. Never pass `torch_dtype` alongside `quantization_config` — bitsandbytes owns dtype internally; passing it bypasses BNB and loads full FP16 → OOM.
- **BNB + 72B on A100**: `device_map="auto"` reads the *pre-quantization* FP16 size (~145 GB for Qwen-72B) and dispatches layers to CPU → BNB raises `ValueError: modules dispatched on CPU`. Fix: use `device_map={"": 0}` to force all layers to GPU before BNB quantizes them.
- **70B BNB on A100 (Llama-3.3-70B etc.)**: 4-bit quantization peaks around 80 GB. With `expandable_segments:True` set in Cell 1 it usually fits on a fresh runtime, but after any other model has been loaded and freed the load still OOMs. Gate the load behind a freshness check (`torch.cuda.max_memory_allocated() < 5 GB`); if not fresh, refuse the load and tell the user to restart and run Cells 1–6 + the 70B driver cell only. Checkpoints from completed (model, dataset) pairs persist to Drive and reload automatically, so the user loses no work.

---

## Google Drive cache — don't trust the HF default

Drive's FUSE doesn't support real symlinks. HF's hub cache (`HF_HOME=/content/drive/...`) stores blobs as real files but tries to symlink them into `snapshots/<rev>/<file>` — those symlinks become 0-byte broken stubs on Drive, and HF re-downloads the full model every session despite the blobs sitting there.

**Fix**: bypass the cache. Use `snapshot_download(local_dir=...)` to a flat directory on Drive (real files, no symlinks), then pass the local path to `from_pretrained` instead of the Hub repo ID. Standard helper to put in a notebook setup cell:

```python
from huggingface_hub import snapshot_download
FLAT_CACHE = '/content/drive/MyDrive/hf_cache_flat'
os.makedirs(FLAT_CACHE, exist_ok=True)

def ensure_flat_dir(repo_id, token=None):
    """Download repo to flat dir on Drive (real files, no symlinks). Idempotent."""
    local_dir = os.path.join(FLAT_CACHE, repo_id.replace('/', '__'))
    sentinel = os.path.join(local_dir, 'config.json')
    if os.path.exists(sentinel):
        return local_dir
    kwargs = dict(repo_id=repo_id, local_dir=local_dir, token=token)
    try:
        snapshot_download(**kwargs, local_dir_use_symlinks=False)
    except TypeError:
        snapshot_download(**kwargs)  # newer hf_hub removed the kwarg; default is copies
    return local_dir
```

`load_model()` still auto-detects AWQ from the path string (looks for "awq"/"gptq"), so passing `/content/drive/MyDrive/hf_cache_flat/Qwen__Qwen2.5-72B-Instruct-AWQ` works the same as the Hub ID.

---

## AIRCC cluster (Slurm GPU allocation)

**Live allocation verified 2026-09-13:** `sdata` reports account
`cycle3_tau_averbuch_prj`, QoS `owner_940` on `power-gpu` (sandbox QoS
`sandbox_owner_940`). Older `owner_880` examples are historical; rediscover before
submission. Noninteractive SSH commands need `SLURM_CONF_SERVER=controller-primary`.
The documented project workspace below remains accessible. Current jobs use
Pyxis; the old rootless-Docker description below is historical. The dedicated
answer-position analysis job was accepted without requesting a GPU.

Second GPU backend besides Colab: national AIRCC cluster, 8× NVIDIA B200, ssh alias `aircc`
(omrisegev1@slurm-login.iucc.ac.il, **TAU VPN required** — a hanging ssh means VPN is down).
Full reference: [cluster/README.md](cluster/README.md). Rules that must never be violated:

- Work only under `/shared/cycle2_tau_averbuch_prj/omrisegev1`, never `$HOME`.
- B200 = sm_100: jobs run inside `nvcr.io/nvidia/pytorch:25.01-py3` (rootless Docker).
  Never pip-upgrade torch inside it; `cluster/requirements.txt` deliberately omits torch/numpy.
- Preemption: SIGTERM → 15 min → SIGKILL → auto-requeue. Every long job must use the
  `cluster/run_inference.py` checkpoint/resume pattern (atomic saves via `save_cache_atomic`,
  SIGTERM trap, idempotent restart).
- **A clean `exit 85` is NOT auto-requeued.** `--requeue` covers preemption; a job that catches
  SIGTERM, checkpoints, and exits 85 on its own simply ends as `FAILED 85:0` and nothing resumes
  it. Any job that will not finish inside its wall needs an explicit chain submitted up front:
  `sbatch --dependency=afterany:$PREV ...`, one wall per link. **Chain LINEARLY** — two jobs both
  depending on `afterany:$SAME_JOB` become eligible together and will race on the same output
  file (atomic replace prevents corruption, not lost rows). Fix a fan-out with
  `scontrol update jobid=<later> Dependency=afterany:<end-of-chain>`. Cost this: jobs 176043,
  176044 and 177759 all died this way before the pattern was written down.
- Code reaches the cluster via `bash cluster/sync_code.sh` (tar-over-ssh) — never rely on
  Claude pushing to GitHub (credentials are not available in-session).
- **Results reach Google Drive via `rclone` ON THE CLUSTER — never via the Drive MCP tools and
  never through a local hop.** rclone lives at
  `/shared/cycle2_tau_averbuch_prj/omrisegev1/bin/rclone` (not on `$PATH`, so always use the full
  path) with its config at `~/.config/rclone/rclone.conf` and one remote, `gdrive:`. The
  destination mirrors the cluster's own `results/` layout one directory per job:

  ```bash
  RC=/shared/cycle2_tau_averbuch_prj/omrisegev1/bin/rclone
  R=/shared/cycle2_tau_averbuch_prj/omrisegev1/results
  $RC copy "$R/<job_dir>" "gdrive:hallucination_detection/cluster_results/<job_dir>" \
      --transfers 4 --checkers 8 --drive-chunk-size 64M --stats-one-line --stats 60s -v
  ```

  For a multi-GB upload, run it detached (`nohup ... > upload.log 2>&1 &`) so it survives the ssh
  session, then poll the log. `rclone copy` is idempotent and resumable, so re-running it after a
  job finishes more subsets is the correct way to top up a partial directory.

  **Why not the Drive MCP tools**: `create_file` needs the file's bytes as a `base64Content` /
  `textContent` parameter, i.e. through the conversation — fine for a manifest, impossible for a
  1 GB pickle. And there is no Google Drive mount on the Windows machine (only OneDrive/BGU), so
  a local copy is not an option either. Uploading from the cluster also avoids pulling 2+ GB down
  the VPN just to push it back up.
- Workflow: `/aircc-setup` (once) → `/aircc-submit` → `/aircc-status` → `/aircc-fetch`.
- **All cluster polling / log-tailing goes through `/aircc-status` or the `cluster-ops` sub-agent — never raw `ssh aircc "squeue/sacct/tail"` loops in the main context.** Each raw ssh re-prints the login banner and dumps full logs into context; the sub-agent returns a one-line verdict. (Step-163 retro: inline ssh polling was the single biggest recurring token sink.)
- **A new `cluster/presets.py` preset MUST pass `python scripts/smoke_preset.py <id>` (CPU-only) before it is submitted.** It runs the preset's real prompt/grader/judge helpers on fixtures — catching prompt / grader / judge-parse bugs offline instead of via a GPU round-trip (4 of the 6 Step-163 pilot bugs were this kind). Gate order: **local smoke → N=30 pilot → full N.**

---

## Raw inference data — save everything, derive later

**Rule**: save the richest raw form of each inference result to Drive. Never discard information during the inference loop that could be useful for any future feature or baseline.

Per-sample pkl entry must include:

| Key | Type | What it is |
|-----|------|------------|
| `full_text` | `str` | Generated answer string |
| `token_entropies` | `list[float]` | H(n) — Shannon entropy per token (top-K=15) |
| `token_spilled_energies` | `list[float]` | ΔE(n) = −log p(sampled token) per token |
| `top_k_logprobs` | `{'ids': int32 [T,50], 'logprobs': float32 [T,50]}` | Top-50 log-probs per token as a numpy-array pair (~3.5× smaller than the old `list[list[tuple]]` form; both schemas are valid in old caches) |
| `gen_token_ids` | `list[int]` | Sampled token IDs (needed for ΔE and attention features) |
| `label` | `int/bool` | Correctness label (graded at inference time) |
| `question` | `str` | Original question text |

**Why `top_k_logprobs`**: H(n) and ΔE(n) are derived quantities that fix K=15 and the sampled token at generation time. Saving top-50 logprobs lets you recompute entropy at any K, compute probability mass features, implement token-level confidence, or compute any future feature — without re-running the model. Storage cost: ~200 KB/sample at 500 tokens × 50 top entries (numpy form).

`generate_full()` returns `top_k_logprobs` and `gen_token_ids` since the AIRCC onboarding commit (param `logprob_top_k=50`, set 0 to disable). The standalone extractor is also exported as `spectral_utils.extract_top_k_logprobs(scores, top_k=50)` for recomputation from cached scores.

Old cached pkls that only have `token_entropies` are still valid for H(n)-based features. The spilled energy and logprob features simply won't be available for those runs.

---

## Analysis-result persistence (Colab `background_save` survival)

Long-running analysis cells (Nadler subset search, length-controlled, PCA, SE baseline) MUST persist their output dict to disk. Colab's `background_save: true` lets the cell finish printing after a kernel disconnect, but in-memory variables (`NADLER_RES`, `LEN_RES`, `PCA_RES`) are gone. Downstream cells then `NameError`.

Standard pattern at the top of each analysis cell:

```python
RES_PATH = os.path.join(RES_DIR, 'foo_res.pkl')
FORCE_RECOMPUTE = False

if not FORCE_RECOMPUTE and 'FOO_RES' in globals() and FOO_RES:
    print('already in memory; skipping')
elif not FORCE_RECOMPUTE and os.path.exists(RES_PATH):
    with open(RES_PATH, 'rb') as f: FOO_RES = pickle.load(f)
    print(f'loaded from {RES_PATH}')
else:
    FOO_RES = {}
    # ... compute ...
    with open(RES_PATH, 'wb') as f: pickle.dump(FOO_RES, f)
    print(f'saved to {RES_PATH}')
```

Same pattern as Cell 6's `run_inference_for_cell` checkpoint logic. Apply it to every cell that takes more than ~30 seconds.

**Local offline scoring — token/time economy** (Step-163 retro):
- **Scoring or feature-extracting a cell >100 MB or K≥10 runs in the background** (`run_in_background: true`) with a generous timeout — never a foreground short timeout. The 857 MB losnet cell timed out at `timeout 400` and had to be re-run; K=10 FFT extraction on a big pkl needs minutes.
- **Inspect any cell's schema with `python scripts/inspect_cell.py <pkl|preset_dir>` before scoring** — it prints N/K, label dist, trace lengths, key-presence (base + energy + judge keys), and the extractable feature set + valid-rate. This replaces ad-hoc `python -c` pkl spelunking.

**Known package gap**: `model_utils.py` still uses `device_map="auto"`. Safe for AWQ and small models. For 72B BNB, override in the notebook or patch the package before the run.

**Colab C-extension corruption — recovery**: If you see `cannot import name '_center'` (numpy) or `IpcReadOptions size changed` (pyarrow) mid-session, do NOT try `pip install --force-reinstall` to recover — C extensions can't be unloaded from a running Python process and the `.so` you're trying to replace is held open. The only fix is `Runtime → Restart runtime`. Prevention is the Cell 1 pre-import + deferred gptqmodel install above.

---

## Notebook standard cell sequence

1. Clone + pip install + imports (`spectral_utils`)
2. Config (`MODEL_ID`, `TEMP`, `MAX_NEW`, `CACHE_DIR`, `N_SAMPLES`, feature subset)
3. Mount Google Drive + create cache dirs
4. Load model (`load_model`)
5. Inference loop with checkpointing (save every 25 samples to `.pkl`)
6. Unload model (`del mdl, tok; free_memory()`)
7. Feature extraction (`extract_all_features` → aligned labels + valid_results)
8. Window ablation for `sw_var_peak` over `WINDOW_SIZES`
9. Individual feature AUC table
10. Nadler fusion — `best_nadler_on` for new experiments; `apply_fixed_subset` for validation phases
11. Gate/decision cell (explicit pass/fail thresholds, printed summary)
12. Save results dict to `.pkl`
13. Plots (feature AUC bar, results landscape, entropy trajectories)

---

## HISTORY.md format

Append new steps at the bottom of `## Steps`. Number sequentially from the last step.

```
### Step N — <one-line title>

**What**: What was done or discovered.
**Why**: Why this matters / what triggered it.
**Result**: Outcome, numbers, or status.

---
```

For debugging sessions with multiple iterations, group under one step with sub-headers (`#### Sub-step` or `**Attempt N**`). Don't create a new step for every failed attempt — one step per logical investigation.

### Two working lines share the numbering — tag, never renumber

This project is worked in two repositories at once (Claude here, Codex on
another machine), so the same step number gets claimed by two unrelated pieces of
work. **More collisions are expected.** When one happens:

- **Keep both blocks.** The log exists so that no attempt is overwritten by
  another; a merge that drops one side has destroyed the record it was meant to
  preserve.
- **Mark the line in the heading**, e.g. `### Step 269 [localization] — …` and
  `### Step 269 [A6/PTNI] — …`. Tag both sides, so neither reads as "the" 269.
- **Do not renumber.** The numbers carry no meaning; the content does. Renumbering
  invalidates several hundred cross-references in `PROGRESS.md`,
  `Research_Directions.md`, `GLOSSARY.md` and commit messages that no tool checks,
  and the file already held duplicate numbers (32, 54, 75, 142, 193, 228) long
  before a second repository existed, so uniqueness was never there to preserve.

**Merging `HISTORY.md` or `PROGRESS.md` needs care.** They are append-only prose
logs and git's text merge does not know that: it interleaves blocks, or drops
conflict markers *inside* a step's prose where they read as an ordinary paragraph
instead of as breakage. Never resolve them with `--ours`/`--theirs`, which
silently discards one side entirely. Resolve by union and then verify by counting
blocks on both sides against the result — see the merge commit `cd423ab` for a
worked example.

---

## Research_Directions.md — the thesis roadmap

This is the single source of truth for what to do next and why. It contains:
- 7 directions with hypotheses, proposed experiments, feasibility/novelty/risk ratings, and supervisor connections
- Per-direction status (`Active`, `Completed`, `Not started`, `Pending`)
- Per-experiment status (✅ COMPLETED, ← NEXT PRIORITY, etc.)
- Decision gates with explicit thresholds (e.g. G3: beat LOS-Net 72.92% on HotpotQA)
- Priority order at the bottom, updated as experiments complete

**When to read it**: before planning a new experiment or notebook, when the user asks "what's next", or when a paper changes the roadmap.

**When to update it**: after an experiment completes, update the status field and add results to the relevant Phase/Experiment section. When a new paper shifts priorities, update the priority order section.

Do **not** duplicate information between Research_Directions.md and HISTORY.md. Research_Directions.md holds the plan and aggregated results tables; HISTORY.md holds the step-by-step narrative of what happened and why.

---

## Research papers

Papers live in `papers/`. **Check `papers/index.md` first** — if a paper is already
`digested`, read `papers/digests/<slug>.md` (and `papers/extracted/<slug>.md` for exact
quotes) instead of re-reading the PDF from scratch.

If it's not cached yet: follow `skills/paper-digest/SKILL.md` (extract → digest → index),
or just run `/paper-digest`. Same procedure whether you're Claude Code or antigravity/Gemini
— the skill is mirrored to `.gemini/skills/paper-digest/` for exactly that reason.

If the paper changes the roadmap, still update `PROGRESS.md` and `Research_Directions.md`.
A substantive new read is worth a HISTORY.md step (title + pointer to the digest file); a
cache hit is not.

---

## Nadler fusion invariants (never violate these)

- Requires **≥ 3 views** — 2-view fusion collapses to near-random AUC.
- Apply **z-score normalization** before fusion (`normalize=True` in `best_nadler_on`) — required for short-trace QA, negligible on long math traces.
- Correlation filter: skip any subset where pairwise Spearman |ρ| ≥ 0.75.
- Feature sign: orient each feature so higher score → more likely correct before fusing.

---

## Best results (reference)

| Setup | Nadler AUC | Notes |
|-------|-----------|-------|
| MATH-500 / Qwen-7B / T=1.0 | **90.0%** | spectral features work on long reasoning |
| MATH-500 / Qwen-1.5B / T=1.5 | 88.3% | |
| GSM8K / Llama-3.1-8B | 76.0% | vs LapEigvals unsupervised 72.0% |
| GPQA / Mistral-7B / T=1.0 | 65.4% | Phase 4 best — beaten by 72B (Phase 8) |
| HotpotQA / Mistral-7B | 59.5% | spectral doesn't transfer to multi-hop QA |

**Thesis scope (updated Step 191 — Omri, 2026-07-20)**:
- **In scope, and the focus going forward = reasoning (math: MATH-500 / GSM8K) + QA** (single-answer factual — TriviaQA / SQuAD / NQ-Open / etc.; several QA cells score well, e.g. `spilled_triviaqa` 0.93, `se_squad_v2` 0.80).
- **Out of scope = multi-hop RAG accuracy detection (`lciteeval`) and GPQA (science MCQ).** Step 191's honest-ceiling check (30-view pool, split-half oracle) is the evidence: GPQA features are **uniformly at chance** (every feature 0.51–0.55, no signal to orient), and RAG signal is **confined to one sub-dataset (hotpotqa) and bottlenecked by feature sign, not pool size** — adding the energy/logprob views moved the honest ceiling only +3.6pp (RAG) / −0.5pp (GPQA). This **supersedes** the earlier "science MCQ works / short factual QA structurally incompatible" note above (that predated the cluster-data cells; it had GPQA backwards and was too pessimistic on single-answer QA).
- **Reopened, as a different task (Step 235, 2026-08-08): RAG citation/evidence grounding is now an active application priority.** This is not multi-hop retrieval accuracy — it is evidence-contrast detection/localization on a *fixed* answer (does it contain unsupported content, and where), tested via the Evidence-Contrast U-PCR/DUFS-LIU direction. See `Research_Directions.md` "Application priority 2" and `docs/research_notes/rag_localization_methods_and_benchmarks_2026.md`.

## Which method to evaluate — ASK, never assume

**Frozen exception — Fair Paper-Exact Comparison Package v1 (Step 279).** For
`fair_paper_exact_comparisons_v1`, Omri has already fixed the method of record:
ordinary **Unified-28**, exactly seven registered causal streams crossed with
`{level, ewma16, positive_area, persistence}`, two-component L2 IU-PCR, and the
Identity accumulator. Do not ask to reselect it, change its roster/signs/task
weights/accumulator, reopen feature or DUFS search, or omit the lane's dedicated
incumbent. The complete contract is
`docs/experiments/FAIR_PAPER_EXACT_COMPARISONS_V1.md`; accepted outputs are in
`results/fair_paper_exact_comparisons_v1/`, including access/fidelity labels,
missing-assets status, and explicit gates for any future GPU work. The general
ASK rule below still applies to every other evaluation.

**Do not infer "the method we use" or "our best variant" from this file, from a results table, or from whatever a recent report happened to headline. Ask Omri explicitly which method to score, before writing any evaluation code.**

The leading arm changes as the work moves, and the ranking is close enough that a stale assumption produces a plausible-looking table of the wrong thing. Two failure modes have already happened:

- **Naming a fixed subset as "the headline".** Hand-picked subsets (`GOOD_5`, `GOOD_6`, `LOCO_5`, …) exist as **reference/compatibility rows**, not as the contribution. Reporting one as the method misrepresents the thesis.
- **Calling the right method by name but running the wrong implementation.** Several methods have more than one entry point, and the obvious one is often the legacy path. Example: if U-PCR is the method asked for, the maintained arm goes through `spectral_utils.upcr.upcr_fit` over the full pool with a fitted config and `sign(ρ̂)` polarity (`scripts/labelfree_standing_report.py:upcr_rho_oriented`) — **not** `fusion_utils.upcr_pipeline` / `eval_subset_flex(fusion='upcr')`. Once Omri names a method, find and mirror the script that actually produced the last reported number for it (`feedback_read_canonical_scorer_first`).

**The research direction, which does not change**: the thesis is about **label-free methods that carry no hand-picked prior knowledge** — deriving orientation, subset size, and feature selection from the data's own structure rather than from a curated feature list, a chosen anchor, or a fixed K (Extension H / Step 199). Any new evaluation should default to that family. A hand-picked subset may appear **beside** it as a reference, clearly labelled, never as the result.

When in doubt about which arm, which subset, or which fusion entry point: **stop and ask.**

## Benchmark continuity and comparator coverage — Omri, 2026-09-06

**Standing user preference:** our experiments must allow comparison with all
relevant candidates among historical and current leading methods. Do not limit
the comparison to the newest project variants or silently forget an earlier
incumbent. Localization is the current application priority.

- Maintain a comparator registry covering established published baselines,
  current leading relevant methods, strong simple controls, and the project's
  historical contenders. Check existing paper digests and official sources;
  do not claim that an old literature inventory is a current, complete survey.
  Record the exact paper/version, implementation entry point, target, input
  access, training/calibration labels, compute budget, and inclusion status.
  A missing asset or incompatible target needs a visible reason and follow-up,
  not silent omission. Paper-reported numbers are context until reproduced
  under a matched contract; adaptations must be identified as adaptations.
- Give each benchmark release a stable identity, independent of experiment
  names. Freeze example/source-group IDs, label version, fold assignments,
  preprocessing fit scope, feature/access contract, score orientation, step
  mapping, no-error rule, metrics, aggregation, tuning budget and evaluator
  source. New experiment names must not silently generate new folds.
- Every new candidate joins the relevant existing comparison table with
  mandatory incumbent and simple-reference rows. Distinguish fixed label-free
  recipes from label-selected recipes; keep selection inside training folds
  with matched budgets. Preserve each frozen historical release.
- If a necessary correction changes the benchmark, issue a new version and
  document the change. Run common anchor methods under the old and corrected
  contracts when meaningful and possible, and show the bridge. Mark invalid
  old results as superseded; never preserve a bug for comparability or compare
  unmatched scores as an algorithmic gain.
- Use separate panels for genuinely different prediction targets or access
  levels, while keeping their context visible. Within localization, support
  both a fixed-representation fusion comparison and an end-to-end comparison
  on shared raw inputs, targets and evaluation IDs. Window and token methods
  can be compared end to end; different internal matrices alone do not make
  them incomparable. Report coverage, failures, runtime and paired uncertainty.
- Reuse frozen scores only when their input, fitting and evaluation contracts
  match. Keep already inspected data labelled as development/retrospective;
  publication confirmation requires an explicitly held-out, untouched test.
- User choices and authorization already given in the conversation persist.
  The general ASK rule above does not require asking again about selected
  candidates. The approved Joint window family and continuation dependencies
  are recorded in the plan below. Registry preparation is not permission to
  launch an unbounded method sweep or alter a running experiment.

Current continuation plan:
`docs/experiments/LOCALIZATION_BENCHMARK_CONTINUITY_PLAN_20260906.md`.

## Latest fusion gate mechanism check - Step319, 2026-09-07

**Step319 completed: controlled mixture-gate mechanism check.**
`results/fusion_gate_null_v1/REPORT.html`; scientific and artifact review PASS.
768 synthetic trials: N16/64/256, AR rho0/.6/.9,64 replicates per cell,
plus rho0/+3SD mean-jump controls. Raw, ordinary Kalman cold/warm and IMM
cold/warm;3840 valid outputs, no failures. Warm processes256 preceding
synthetic points and is a startup diagnostic, not a free-data candidate.
No current110 score/prediction/metric changed; all176 anchors remain current.

One stationary Gaussian source can trigger the gate after filtering:
N256/rho0 raw0/64, IMM cold26/64, warm26/64; N64/rho0 raw0, cold11, warm12.
The effect survives this startup control. N16/rho0 raw already opens15/64,
so short-sample behavior also matters. At N64 with the fixed mean jump,
raw opens21/64 while both IMM variants open64/64: preserve sensitivity in
any correction. These are simulation frequencies, not real-answer false-
positive rates. BIC has no declared5% alarm guarantee; nonlinear filtering
need not preserve a Gaussian marginal. No numerical GMM bug is claimed.

Three tests;768 source paths,1536 independent scalar Kalman trajectories,
3840 normalization/mixture-algebra checks,48 direct vector IMM trajectories
and120 actual GMM refits pass. Simulation106.04s, review12.06s. Artifact
validation1551 hashes,11 local links/images,137 static rows,five ASTs. Both
PNG plots inspected; no browser rendering or external review claimed.

Next bounded stage: verify one fixed calibration procedure on independent
synthetic calibration/evaluation replicates, carrying the source through the
SAME filtering/normalization/GMM procedure. Include mean-jump sensitivity
and nuisance-parameter uncertainty; do not treat an AR Gaussian source as a
proven model of correct reasoning. Then a gate-only current110 comparison
must keep frozen fusion curves/peaks and all176 anchors. A synthetic fix
alone is not an achieved localization improvement. The full research scope
remains active: Joint/feature/IU, named supporting methods, corrected multi-
answer refits, full comparators, sparse/short-error sampling, untouched
confirmation and historical24 transfer are still open.

## Latest trajectory fusion - Step318, 2026-09-07

**Step318 completed: full-trajectory fusion and supporting IMM.**
`results/fusion_trajectory_imm_v1/REPORT.html`; scientific review PASS.
Same110/v3 labels/v2 groups, original banks/fits/windows and149 unchanged
anchors;27 additions,176 total,50 contrasts. All final outputs valid110;
three inherited Joint fallbacks collapse to one IU observation in paired
families. No new inference, feature refit or causal-online claim.

Primary mean/GLS/IMM PRMB/PB: .669437/31.98298%, .683384/30.34837%,
.693254/16.69023%. Original IU .681306/30.15985%, Joint graph100
.655451/30.22293%; strong risk equal-permuted .768422/33.92003% retained.
GLS vsIU intervals[-.02612,+.02882] AUC/[-8.7608,+9.3277]pp include0.
No winner. Static mean/GLS are feature-weight combinations; IMM introduces
chronological state evolution, not semantic correct/error states.

Primary IMM vs static hold gains4 PB successes/loses13. Raw exact peaks
18->17, final exact errors12->7, clean successes15->11. Post-evaluation
fixed-component exchange: hold peak/hold gate30.34837%; hold peak/IMM gate
23.27899%; IMM peak/hold gate27.43170%; IMM peak/IMM gate16.69023%.
These are diagnostics, not new candidates or causal mediation. Clean median
lag1 rises .28649->.63670, median BIC1-BIC2 .95381->6.44959; false alarms
18->22. This motivates checking the mixture gate under dependence, but does
not establish a semantic-error null or prove an effective-N correction.

Seven tests,880 independent R/GLS and990 direct vector-IMM replays,2970
outputs,176 metric bundles,50 paired comparisons and five explicit1000-draw
bootstraps pass. Post-evaluation audit adds48 metric/component checks and1376
lag replays. Scoring52.80s; review113.13s. Same-session/shared-kernel scope
is disclosed. Full HTML, all comparisons and two inspected PNG/SVG plots
are saved; no browser rendering or external review claimed.

Artifact validation PASS:719 hashes,14 links/images,198 static rows,12 Node-DOM cases/256 numeric rows, seven ASTs and44 guide IDs. All handles terminal.

Next bounded priority: audit existing no-error methods and use controlled
unimodal serially dependent trajectories to check whether smoothing alone
can trigger this mixture gate. Then decide one gate-only comparison with
unchanged fusion trajectories/peaks. No broader temporal/graph sweep from
these point estimates. Full comparators, corrected multi-answer refits,
actual KalmanNet/LOCA/Flows, sparse/short-error sampling, untouched confirmation
and historical24 transfer remain open; full research goal active.

## Latest sampling replication — Step317, 2026-09-07

`results/fusion_sampling_replication_v1/REPORT.html`: same corrected110,
original banks/routes, six selectors/seven cores,149 entries including107
anchors,93 contrasts; review PASS.72 can reduce fitting rows,38 copy anchors.
The GMM still sees ALL original fit windows. Dense extraction/scoring remains;
only a failed Joint MODEL fit falls back to sampled IU in the same bank.

Risk IU .758352/28.34853% versus full .681306/30.15985%; risk Joint graph
.741328/28.20441% versus .655451/30.22293%. All ten sampled IU/Joint graph
PB points weaken. Risk equal+permuted graph .768422/33.92003% has higher
points than its full reference, but PB improvement CI includes0. No winner.
Post-evaluation positive affine checks recover much of the pooled jump
without changing original within-answer ranks (IU .748761; graph .735854).
Sampled IU restored to original location/scale gives .689378; graph .665641.
Do not call this a .758 localizer or treat diagnostic transforms as candidates.
Risk IU gains one exact PB success/loses two. No <=32-token first errors among
38 eligible erroneous PB answers; short-error retention is still untested.

Next audit existing full-trajectory fusion and fitting scope, then one bounded
IU/Joint whole-trajectory combination with simple controls and separate
peak/gate evidence. Existing-peak choice and whole-trajectory combination are
different. Preserve149 anchors and strong risk equal-permuted control;
do not expand a graph/window sweep from pooled-AUC effects. The broad mandate,
corrected multi-answer refits, named tracks, complete comparators, untouched
confirmation and historical24 transfer remain open.

## Latest historical correction — Step316, 2026-09-07

`results/localization_history_bridge_v3/REPORT.html` repairs the original58
unique representation/readout/sampling/regularization/context entries:131
methods,199 original comparisons, seven gate diagnostics,25 corrected fallback
references. Current110/107 entries are displayed separately. No score, fit,
failure, prediction or PB point changed; v3 labels/v2 groups now cover these
early lanes. Other source formats and multi-answer refits are not repaired
by this scope. Review PASS, including raw annotations and original arrays.

No winner: entropy-risk IU .668498/19.05242% versus full .647527/17.70833%
has intervals including0 and lower within-answer ranking. PB loses one
success/gains one; total clean/error successes unchanged. Joint+DUFS .763333
on4 valid PRMB must be compared with .750000 original Joint on those same4,
not .625788 on7. Answer-offset IU .692399 pooled AUC is not improved local
ranking or decisions. The actual earlier HMM/Kalman/IMM/BOCPD readouts and
larger-lambda penalties establish no consistent gain. Named methods are not
closed by a different precursor or by these fixed settings.

Next bounded direction: matched observation-selection replication on the
fixed current110, after target-free budget feasibility; original banks/routes,
full/uniform/risk/graph/permuted controls, IU/Joint/equal, short-error retention
and explicit failure/within-answer/exact-success reporting. Freeze a small
design and keep107 anchors; no broad label-guided selector sweep. The full
research mandate, corrected multi-answer refits, comparators, untouched
confirmation and historical24 transfer remain active requirements.

## Latest supporting representation test — Step315, 2026-09-07

`results/fusion_token_gap_v1/REPORT.html`: same110 corrected-label development
answers; replace three surprisal coordinates with provided-token preference
gap, keeping P27, width8, original81/29 bank route and exact graph seeds.
No consistent gain: gap-IU .680187/28.82295% versus original .681306/30.15985%;
gap-Joint graph .649936/30.22293% versus .655451/30.22293%. Native Joint100/110
versus107, ten explicit IU fallbacks; all nine final outputs valid. The two
scalar controls falsely flag all33 clean PB answers under the fixed GMM.
107 total rows,25 comparisons and review PASS. Keep original references and
the stronger permuted equal-graph control visible. This does not close Joint,
graph or provided-token confidence as supporting ideas.

Original raw arrays verify71,057 retained provided-token logprobs exactly;
328 positions outside top50 cannot be reconstructed independently from that
list. All330 scalar/660 top-K feature streams and110 raw targets replay.
This extends downstream fidelity checks, not model-forward/logit alignment.
Next bridge the unique earlier readout/sampling/regularization score sets to
v3 labels/v2 groups before relying on their old PRMB conclusions. Existing
98/25 bridges do not cover every old arm; multi-answer refits remain separate.
Full research mandate, untouched confirmation and historical24 stay open.

## Source-question grouping correction - 2026-09-07

**Current label release is v3, not v2.** Use
`results/localization_prm_label_audit_v1/RELEASE_V3.json`
(`localization-cached-v3-prm-onebased-20260907`) for new work. It retains
the source groups and `FOLDS_V2.json` from
`results/localization_source_group_audit_v1/` and the original graph seed
namespace. Step313 found that v2's PRMB label writer used `flags[step]` for
one-based raw `error_steps`. Use the tested
`spectral_utils.prm_label_contract.prm_error_flags` conversion; out-of-range
annotations stay inert. ProcessBench's zero-based/-1 contract is separate.

All6969 PRMB labels are corrected, while all8 PB files stay exact. The
current110/98-arm and old58/25-arm frozen-score bridges plus207 registered
comparisons are reviewed in `results/localization_prm_label_audit_v1/REPORT.html`.
Historical PRMB metrics and two-task conclusions below that used v1/v2 labels
are superseded, including the near-tie and all21-below-both-anchors claims.
Corrected routed IU .68131/30.16%, graph100 Joint .65545/30.22%; no winner.
Keep the permuted equal-graph control .69226/31.32% visible. Older unique
score sets need bridges. Multi-answer fits/selection need both corrected
labels and corrected source-group folds, not just re-evaluation. Original
files remain frozen. Raw annotation verification must be independent of the
derived label NPZ: agreeing with that NPZ alone did not catch this defect.
The first text/peak audit stopped on its raw-label assertion. Its v3 follow-up
is now complete: `results/fusion_localization_forensics_v3/REPORT.html`.
All110 raw token/span/label joins and1760 downstream trajectory maps pass;
original model logit-position and full top-K fidelity are not established by
that audit. Shared-window top ties are real, but perfect tied-step choice
with the original gate only changes IU PB30.16->31.77% (label-using oracle).
IU has20/53 raw exact peaks, eight hidden by its gate, and15/33 clean false
alarms. Graph100 has18/53 raw peaks, five hidden and16/33 clean false alarms.
Both peak locations and the no-error gate remain bottlenecks. The next
bounded direction is an audit of realized-token versus distribution-confidence
evidence, then one supporting fusion/readout change with matched controls.
Do not promote an oracle or infer that boundary refinement alone will fix it.

The historical source-group release is
`localization-cached-v2-sourcegroups-20260907`. V1 PRMB `source_idx` groups
were perturbation IDs; different versions of the same source question crossed
folds. PB also repeats identical questions under different answer IDs. The
verified audit and score bridge are in that directory's `REPORT.html`.

- Corrected groups connect PRMB source-seed identity and exact whitespace-
  normalized question text, including PB text matches. Preserve train/test
  and source partition suffixes; do not strip IDs speculatively or normalize
  mathematical content. The map is versioned and does not detect all paraphrases.
- Load the corrected group map before using corrected folds. The old telemetry
  NPZ group IDs remain unchanged; do not replace a folds file alone.
- Keep the old scoring identity namespace `localization-cached-v1-20260907`
  when replaying the existing fusion recipes. The grouping-only correction
  must not silently change the graph-permutation random seed.
- All 25 answer-only fallback point-metric bundles replay unchanged; 32
  interval bridges use corrected groups. Earlier v1 question-group intervals
  are historical and need their own bridge when used. The new folds do not
  repair Claude's already trained multi-answer models: those fits, selection
  and scores require rerunning before source-question-disjoint claims.
- The documented pilot exclusion inventory is not a complete project exposure
  inventory. Cached v2 data were already evaluated; a disjoint pilot is still
  development. Truly untouched confirmation remains a separate requirement.
- Describe the current PRMB/PB access accurately: one teacher-forced model pass
  over a provided official answer, then offline answer-only fusion. No new
  answer generation took place. This remains gray-box and one-pass scoring.

## Active research mandate and persistent ideas — Omri, 2026-09-07

This section supersedes the earlier recommendation to stop Joint/graph work
and the requirement to seek a new discussion before each already-authorized
stage. Omri has authorized Codex to lead and execute the research program in
bounded, reviewable stages. Keep reporting evidence and reviewing code; do not
turn this authorization into an unconstrained label-guided sweep.

- Primary goal: consistent localization improvement on BOTH PRMBench and
  ProcessBench, under gray-box, one-pass model scoring, unsupervised access. Fit
  from the current answer alone whenever viable. Offline processing of the
  full trace is not the same as causal online detection.
- **Fusion remains the core — explicit clarification from Omri, September 7.**
  Develop the existing IU-PCR / Joint L-SML method and its feature/trajectory
  fusion architecture. IMM, LOCA, Diverging Flows, KalmanNet, HMM/BOCPD and
  sampling ideas are supporting components: improve inputs, observation
  selection, grouping/weights, or the readout of fused scores. They must not
  become a replacement standalone detector presented as our method. Each
  addition needs the same fusion core with/without the component on matched
  IDs, plus an ablation showing whether learned fusion adds value over simple
  aggregation under that component. Standalone auxiliary scores can be
  diagnostic controls, not promoted replacements. Advisor reports must show
  continuity from the historical fusion method and attribute each measured
  improvement to the tested addition; do not claim a fusion gain when only
  the auxiliary component explains it.
- Keep Joint L-SML and its graph variants active. Test whether a feature bank
  suited to the window representation, stable grouping, and label-free
  hyperparameter choices improve them. A negative test on one roster and
  lambda does not close the family. Do not manufacture groups by duplicating
  features or force inadmissible partitions.
- Improve IU-PCR in the same representation, retaining equal fusion and the
  relevant historical incumbent in every matched comparison. Preserve both
  FEATURE and TRAJECTORY fusion axes and the tested combinations.
- Preserve Omri's TOKEN/WINDOW sampling idea: graph-based observation
  selection inspired by DUFS, compared with full-grid, uniform, and risk-based
  selection at matched budgets. This is different from feature selection or
  graph regularization of fusion weights. It is not already tested by either.
- Revisit HMM, BOCPD and Shlezinger-inspired state estimation as chronological
  readouts of the new single-answer matrix. Audit the old experiments first;
  prior negative final-answer scalar-feature results are not evidence that
  this new use has already failed. Investigate Nir Shlezinger's task-based
  sampling work, with explicit supervision/assumption checks before adapting.
- Additional explicit user request on September 7: revisit **IMM, LOCA,
  Diverging Flows and KalmanNet** as potential components of the new
  architecture. Preserve these alongside token/window sampling. Distinguish
  actual implementations from precursor tests (Gaussian HMM, ordinary Kalman,
  AE/GMM/KDE or eigen-ratio selection). A negative precursor is not a measured
  failure of the named method. Audit:
  `docs/reviews/temporal_geometry_revisit_2026-09-07.md`.
- Temporal-source fidelity: distinguish a boundary before the current
  observation from one after it. Constant reset mass under constant hazard
  is not by itself a BOCPD bug. Do not reuse the old mixed-convention
  `temporal_models.bocpd_gaussian` as a verified new localizer; see
  `docs/reviews/bocpd_boundary_audit_2026-09-07.md`. LOCA's burst assumptions
  and supporting-fusion adaptation are now grounded in the full paper digest
  `papers/digests/loca-local-conformal-autoencoder.md`.
- Stability is a surrogate, not correctness. The first full-row sensitivity
  regularization pilot is complete: `results/fusion_reliability_regularization_v1/REPORT.html`.
  Larger Joint graph lambdas and a label-free selection rule did not establish
  a consistent two-task gain. Keep no-error gating separate from feature
  ranking; a pooled unlabeled gate is a hybrid scope, not answer-only fitting.
  This does not close feature-bank/group discovery. The Shlezinger
  graph-compression paper now has a full digest in
  `papers/digests/task-based-graph-signal-compression.md`.
- Normalization/gate audit: `results/fusion_gate_interface_audit_v1/REPORT.html`.
  Preserve the registered pooled PRMB endpoint, but report within-answer
  ranking and exact localization beside it. An answer-specific constant can
  improve pooled AUC without changing any local ordering; do not describe
  that alone as better localization. A shifted free-mean GMM has the same
  decisions. Gate and location are separate bottlenecks; the next small stage
  returns to feature-bank/Joint-grouping work with fixed-parent/native gate
  diagnostics. Audit the existing causal DSP code before reuse: its historical
  fitted pipeline/subset search explicitly used supervised development.
  Borrowed signs, rosters or references cannot silently become answer-only.
- Context-bank evidence: `results/fusion_context_bank_pilot_v1/REPORT.html`.
  EMA context changes Joint coverage (13 rescued, six lost) and PB point
  estimates, but does not establish a consistent two-task advantage. The two
  Joint banks share only four valid PRMB answers; preserve common-ID evidence
  and full-population failure accounting. The explicit answer-local fallback
  is now tested: `results/fusion_explicit_fallback_pilot_v1/REPORT.html`.
  Original Joint -> IU has promising points relative to IU, but always-context
  equal is higher on both headline endpoints and paired uncertainty does not
  establish a winner. Keep both equal-fusion banks in comparator coverage.
  The dual-bank route changes coverage but adds no PB exact successes.
  No hidden fallback or label-chosen per-answer method is allowed. Preserve
  pure failures and native versus diagnostic fixed-IU gates separately.
  The source-group exposure audit and fixed-recipe development replication
  are now complete: `results/fusion_replication_v1/REPORT.html`. Single
  Joint0 -> IU's earlier advantage did not recur. Dual IU and dual Joint
  graph have encouraging points, but no consistent two-task winner. Keep
  same-bank/routed equal controls: matched PRMB comparisons can reverse
  conclusions drawn from pure-Joint rows with different coverage. The
  pair-group audit is now complete: `results/joint_pair_identifiability_audit_v1/REPORT.html`.
  A pair identifies its residual product, not individual loadings. The
  feasible covariance construction raises fit coverage, not yet accuracy.
  Future pair experiments must use
  `spectral_utils.joint_pair_jacobian.fit_joint_pairs_checked`: it includes
  the review amendment that profiles pair products even when zero. The old
  factor-coordinate Jacobian can be singular there and falsely pass global
  identification. Preserve the frozen prototype and historical scores.
  The checked-pair quality comparison is now complete:
  `results/fusion_pair_quality_v1/REPORT.html`. All 33 arms, 63 contrasts
  and review are complete. Improved coverage did not improve localization;
  the new dual Joint graph regresses on both primary endpoints. Its routing
  changes banks on 31 answers, and the IU routing-only control also weakens
  despite unchanged per-bank IU maps. Fit validity is not a quality selector.
  Do not promote this route or infer that the entire Joint family failed.
  The original-fit conditioning test is also complete:
  `results/fusion_native_conditioning_v1/REPORT.html`. Caps 30/100/300 at
  lambda zero retain original minimum-three fits/groups/routes. Condition30
  raises dual Joint's points to 0.63639 PRMB / 28.43% PB, but improvement
  intervals include zero and dual IU stays higher. No optimum or winner.
  All 180 original fits/condition-1000 scores replay; C/v/u are now saved.
  The graph/conditioning interaction is now complete:
  `results/fusion_graph_conditioning_v1/REPORT.html`. It retains original
  fits/routes, tests all three caps and adds matched equal-graph controls.
  Dual graph100 barely exceeds dual IU at the point level (0.63847/30.22%
  vs 0.63797/30.16%); both paired intervals include zero. No optimal cap
  or consistent winner. An explicitly post-evaluation overlap diagnostic
  finds 31/53 error answers missed by both IU and all tested Joint graph
  peaks. This limits selecting among those peaks, not whole-trajectory
  fusion or new views. The expanded prediction-history/feasibility audit is
  now complete: `results/fusion_prediction_view_audit_v1/REPORT.html`.
  Older token B3, Local/Online IU and CIW also contain innovations, with
  donor/calibration-answer fitting; do not describe the history as only
  scalar final-answer tests. The new AR(1) prototype appends nine columns
  from preceding pairs within one answer, preserving all original 27.
  All 110 answers and independent review pass, but residuals remain
  substantially redundant. Its quality follow-up is now complete:
  `results/fusion_prediction_quality_v1/REPORT.html`. All98 arms/74 contrasts
  and review pass. All330 augmented Joint fits are valid, without fallback;
  AR has more than three groups in76/110 answers. Corrected v3 results:
  all21 augmented recipes trail both references on PB;20/21 trail IU and
  9/21 trail graph100 on both point metrics. AR+IU .66220/23.50%, AR+Joint
  graph .64714/18.01%, versus IU .68131/30.16% and graph100 .65545/30.22%.
  The old AR-IU PRMB regression interval includes zero after correction. Fit health,
  more groups and better telemetry prediction did not give better quality.
  Keep the original references. Next inspect actual first-error spans/text,
  near-tied peaks and clean-decision changes using frozen scores before
  another feature/predictor sweep. Supporting named tracks remain open;
  sign/weight-change diagnostics are descriptive, not a proven cause.
  Verify information beyond
  entropy; do not call a simple predictor actual KalmanNet/Flows. Preserve
  the graph references rather than widening the same dose grid to chase
  a tiny lead. The condition diagnostic was post-evaluation and did not
  establish the cause of localization errors.
  The cached v2 release is already evaluated;
  disjoint pilot IDs alone do not make an untouched publication test.
- Every new result must include historical context and a matched benchmark
  table. Keep frozen releases intact; bridge necessary protocol corrections
  with shared anchors. Do not compare unmatched scores as improvements.
- The intended outcome is an advisor-ready method with a clear, reproducible
  advantage. This is a research target, not a promised finding. Freeze the
  selection rule and primary endpoints before confirmation; require paired
  uncertainty on both benchmarks, coverage/failure and runtime reporting,
  and a test genuinely untouched by development. If no candidate meets that
  standard, report that openly and continue from the documented evidence.
- After locking the localization candidate, preserve the requested separate
  transfer experiment on the historical 24 final-answer cells.

Implementation and evidence ledger:
`docs/experiments/LOCALIZATION_RESEARCH_MANDATE_20260907.md`.

## Short research cycles — Omri, 2026-09-06 (historical; scope updated above)

Omri prefers short, bounded stages: investigate one direction, return concrete
answers, then plan the next stage together. Keep the broad continuation plan
as a backlog, not an automatically executed experiment chain. Each active
stage needs one decision question, a small fixed scope, existing comparator
anchors, an execution cap and a clear report of what was learned. Stop and
return the findings at that stage boundary; do not silently expand a pilot
into a full sweep. A diagnostic subset is labelled as such and does not replace
the continuing benchmark or justify a publication claim.

Historical recommendation, superseded by the 2026-09-07 mandate above:
short cycles 1–3 are complete. IU is the current answer-only feature-fusion
reference. Fixed groups improve Joint coverage but not ranking, and the exact
Joint-LIU lambda-0.1 graph has no attributable gain over lambda zero or a
node-permuted graph. The next recommended stage is a fresh, disjoint long-answer
cohort containing only IU/equal confirmation; if it confirms, test a small
frozen trajectory-readout roster. See
`docs/reviews/localization_experiment_review_2026-09-07.md`.

Backlog idea from Omri — **DUFS-style token/window sampling**: apply the
parameter-free DUFS graph mechanism to observations along one answer (tokens,
non-overlapping windows, or candidate step points) to select a small set of
informative sampling locations before feature/trajectory fusion. The idea is
implemented first as fitting-row selection in
`results/fusion_window_sampling_pilot_v1/REPORT.html` (58 development answers;
full-grid, uniform, entropy-risk, transposed DUFS, shuffled DUFS and a direct
window graph). Sparse end-to-end scoring and short-error retention remain
open: all window features were still computed, and there were no <=32-token
first-error examples among the sampling-eligible PB answers. Preserve temporal
coordinates and test against uniform sampling, top-risk sampling and the full
window grid. Fit the graph and gates from the answer's own unlabeled trace in
the strict arm; never use error labels to choose points. Keep graph neighbors
defined by telemetry similarity separate from chronological adjacency, and
report coverage, selected-point stability under block perturbation, boundary
recall, runtime and localization accuracy. A graph selector may discard a
short error peak, so selection must be evaluated as a localization operation,
not only by the downstream scalar AUROC. The first bounded diagnostic is complete; further stages follow the active
research mandate above. Do not interpret this negative graph-sampling pilot
as closing task-aware sampling or the Joint graph-regularization family.

## Localization fitting scope and the two fusion axes — Omri, 2026-09-06

**Primary research objective:** learn the localizer from the current answer
alone whenever its trace supports a stable fit. The matrix is N windows from
that answer by P feature definitions. Pooling other answers is a separately
reported comparison/fallback, not an equal-status replacement for this goal.

In the strict single-answer arm, fit normalization, feature groups, gates,
covariance/fusion weights and any learned trajectory reducer from that answer
only. Use fixed declared engineering rules or within-answer label-free rules
for width, regularization and stability. Borrowed groups, priors, selected
parameters or correctness thresholds require a separate hybrid/calibrated
label. A fixed algorithm can be shared; externally fitted quantities cannot
silently enter the answer-only arm. Define the no-error decision explicitly.
Longer traces improve fitting capacity but do not guarantee stable or useful
error identification.

Keep **feature-axis fusion and trajectory-axis fusion** explicit in plans and
reports, including the tested combinations. The current Module-B grid fuses
top-10 order-statistic views within steps, using many training answers. It is
not the planned chronological single-answer window study, and "Joint" in a
method name does not mean those two axes were jointly optimized.

**Separate follow-up requested by Omri:** transfer the localization-leading
fusion recipe to final-answer hallucination detection on the historical
24-cell benchmark. Freeze the candidate from localization before evaluating
its 24-cell result, retain the historical contract/comparators, and distinguish
fusion-rule transfer from transfer of the complete trajectory-based pipeline.
These already studied cells provide retrospective transfer evidence.

## Borrowing from a paper — tailor, never transplant

**Omri, 2026-08-05 (Step 225).** A published metric is **inspiration, not a specification**. Take
the concept, then develop it into the form this problem actually needs. This holds for every
algorithm and metric we try. *"If we need to run a discussion on each variant — so be it."*

Step 224 ran 21 published unsupervised feature-selection conditions faithfully — a fidelity
reviewer even cut the primary family from eight to five for insufficient fidelity — and all 111
variants lost to the deployed U-PCR keep rule. That closed **transplanting a published keep rule
into this channel**. It did not close the ideas in those papers.

- Fidelity to the paper is **not** the acceptance criterion for a new arm. It remains the
  criterion for anything *labelled* with an author's name: describe the mechanism, cite the idea,
  do not claim the method (convention in `spectral_utils/selectors/a9_dpp.py`'s docstring).
- **Do not batch-build a family and report the table.** One variant, one discussion, then build.
- Worked example of the reshaping: `HANDOFF_FEATURE_SELECTION_AND_FUSE.md` §0 and §4.2.
