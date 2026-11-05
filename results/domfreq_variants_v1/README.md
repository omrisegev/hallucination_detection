# domfreq_variants_v1: continuous non-entropy replacements for `dominant_freq16`

Run on AIRCC, CPU only, 2026-11-05. Findings: `REPORT.md` (718 variants, Step 0 alignment, top variants) and
`THREE_FEATURES.md` (definitions of the 3 candidates, PRMScore inside Z17 L-SML, error dependence on TELE16).
All variant selection here is development-level (labels used on already-seen data).

## Contents

| path | what |
|---|---|
| `code/compute_variants.py` | stage 1, label-free: telemetry pkl -> per-step variant cache (`CACHE_<bench>_<prm>.npz`, not in git) |
| `code/metrics.py` | stage 2: Step-0 alignment + metrics (a)-(e) -> ALIGNMENT/RESULTS/CANDIDATES/TOP_TABLE.json, SCATTER/HIST_TOP.npz |
| `code/prmscore_dep.py` | stage 3: PRMScore (alone, Z17 swap, EQ16 swap) + class-conditional dependence -> PRMSCORE_DEPENDENCE.json |
| `code/plots.py` | PNGs from SCATTER/HIST_TOP/TOP_TABLE (run off-cluster: the CPU env has no matplotlib) |
| `code/run_*.sbatch` | the exact jobs: 310406 (compute, 22 min), 311719 (metrics), 311889 (PRMScore/dependence); `logs/` |

## What is needed besides this code

The three candidate features need **only the token telemetry** of each cell. Per answer they use
`token_spilled_energies`, `top_k_logprobs` (`ids` and `logprobs`, [T, 50]), `gen_token_ids`, `token_logsumexp`,
`token_entropies` and `step_token_spans`. `build_token_feature_matrix` validates all of these, even though F1/F2 use
only the top-50 JS series and F3 only the surprisal. No labels or model are needed to compute them.

| input | used for | where (AIRCC, `/shared/cycle2_tau_averbuch_prj/omrisegev1/`) | size |
|---|---|---|---|
| PRMBench telemetry pkl, one per PRM | features | `results/prm_boost_v1/full_prm/prmbench_telemetry.pkl` (Qwen-7B); `results/prm_boost_gen_v1/full_{qwen_prm_72b,pure_prm,rlhflow_prm}/prmbench_telemetry.pkl` | ~1.0-1.1 GB each |
| Socratic telemetry pkl, one per PRM | features | `results/prm_boost_external_v1/full_{qwen_prm,qwen_prm_72b,pure_prm,rlhflow_prm}_socratic/external_telemetry.pkl` | ~1.1 GB each |
| bank `FEATURES.npz` | answer order, the 16 TELE16 channels, Step-0 reference | `results/prm_boost_v1/bank_prm/`, `results/prm_boost_gen_v1/bank_<prm>/`, `.../full_*_socratic/bank_v2/` | ~54 MB (PRMBench), small (Socratic) |
| `INPUTS_<prm>.npz` (CAT34 bundle) | PRMBench labels, PRM reward, official error steps, categories | `auto_raw_plus_z_v1/bundle/inputs/`; also Drive `cluster_results/auto_raw_plus_z_v1/bundle/` | ~31 MB each |
| Socratic `REWARDS.npz` | PRM reward (Z17 on Socratic) | `.../full_*_socratic/bank_v2/REWARDS.npz` | small |
| Socratic labels (evaluator-only) | Socratic metrics only | NOT in git; local `scratch/external_generalization_private/inputs/evaluator_only/socratic.json` (copied to the run's `labels/` on AIRCC) | 0.8 MB |

Frozen code that the scripts import (the paths are hard-coded to the AIRCC snapshots; both snapshots are
byte-identical to git, checked by sha256):

- `code_thinkfix_bank_553de79/spectral_utils/...` = git commit **553de79c6** (`external_generalization/_bank11/claude_feature_bank_v1.py`, `external_generalization/fusion.py:top10`).
- `code_prm_boost_v1/mask_fixed/spectral_utils/first_token_mask.py` = `spectral_utils/first_token_mask.py` at git commit **94377142b** (mask1 that keeps already-empty steps empty).
- `auto_raw_plus_z_v1/bundle/code/spectral_utils/{fusion_utils,prmbench}.py` (`lsml_continuous`, `prmbench_evaluate`), vendored in the bundle.

On another machine: check out 553de79c6 (for `spectral_utils`) and take `first_token_mask.py` from 94377142b, then
point `S`, `FROZEN`, `MASK` (compute_variants.py) and `S`, `BUNDLE` (metrics.py, prmscore_dep.py) at the local copies.
To reproduce the tables without recomputing, the JSON/NPZ files here are enough. To recompute the features, the
telemetry pkls are required (~8.6 GB for the 8 cells).

## Commands (AIRCC)

```
sbatch -A cycle4_tau_averbuch_spot_prj -p cpu code/run_compute.sbatch      # stage 1 -> cache/
sbatch -A cycle4_tau_averbuch_spot_prj -p cpu code/run_metrics.sbatch      # stage 2
sbatch -A cycle4_tau_averbuch_spot_prj -p cpu code/run_prmscore.sbatch     # stage 3
python code/plots.py .                                                     # locally, needs matplotlib
```
