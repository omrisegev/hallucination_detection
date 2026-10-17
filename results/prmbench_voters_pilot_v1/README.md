# Pilot — PRMBench Step Voters on AIRCC Cluster (2026-10-16)

## Overview
Validation of PRMBench step-level voters on the `aircc` cluster (B200 GPU, `power-gpu`, `owner_940`, `nvcr.io/nvidia/pytorch:25.01-py3`) ahead of full run execution.

Driver: `cluster/run_prmbench_step_voters.py`
Submission: `cluster/submit_prmbench_step_voters.sbatch`
Isolated code directory: `/shared/cycle2_tau_averbuch_prj/omrisegev1/code_prm_voters_v1`
Cluster results directory: `/shared/cycle2_tau_averbuch_prj/omrisegev1/results/prmbench_voters_v1`

---

## Environment Resolution (Offline Cluster)
1. **Container & Python**: Base image `nvcr.io/nvidia/pytorch:25.01-py3` (Python 3.12, PyTorch 2.6.0a0, NumPy 1.26.4).
2. **Missing Dependencies**: Container initially lacked `transformers`, `accelerate`, and `scikit-learn`. Compute nodes have no internet access, causing standard `pip install` to timeout.
3. **Staged Wheelhouse**: Full offline wheelhouse located at `/shared/cycle2_tau_averbuch_prj/omrisegev1/wheelhouse/tf_thinkfix_20261010`.
4. **Fix Applied**: Updated `cluster/submit_prmbench_step_voters.sbatch` to install all staged wheels using `--no-deps "$WHEELS"/*.whl`, preventing dependency solver failures on `hf-xet`.

---

## Pilot Gate Results (N=30, 35 meta rows)

| Voter Kind | Model | Job ID | Rows | Failed | Unscored | Truncated | Elapsed (s) | Gate Criteria | Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `uprm_judge` | `Qwen/Qwen3-8B` | 286481 | 35 | 0 | 0 | 0 | 1.8s | `n_failed == 0`, no recovery error | **PASSED** |
| `uprm_judge` | `Qwen/Qwen2.5-14B-Instruct` | 286483 | 35 | 0 | 0 | 0 | 2.0s | `n_failed == 0`, no recovery error | **PASSED** |
| `critic` | `Qwen/Qwen2.5-72B-Instruct` | 286484 | 35 | 0 | 0 | 0 | 630.2s | `n_truncated <= 2/30`, `n_unscored <= 2/30` | **PASSED** |

- **Tokenizer Check**: Zero marker-tokenization breaks (`failed == None`) and zero `recovery_error` across both Qwen3-8B and Qwen2.5-14B-Instruct.
- **Critic Generation**: Qwen2.5-72B-Instruct loaded all 37 weight shards in 64s on a single B200 GPU. Greedy generation took ~10-15s per critique, cleanly extracting first error steps with zero truncations.

---

## Track B PRMs Status
Track B PRM adapters (`qwen_prm_72b`, `rlhflow`, `pure`, `mathshepherd`) were already executed and verified on Oct 13–14 by Codex:
- Location: `/shared/cycle2_tau_averbuch_prj/omrisegev1/results/prm_boost_gen_v1/full_*`
- Baseline PRM-7B: `/shared/cycle2_tau_averbuch_prj/omrisegev1/results/prm_boost_v1/full_prm/prmbench_telemetry.pkl`
- Status: **Complete**. No re-run required.

---

## Ready for Full Runs
Chains prepared per `docs/experiments/PRMBENCH_STEP_VOTERS_HANDOFF.md` Section 5:
1. `uprm_qwen3_8b_full`: 2-link linear chain
2. `uprm_qwen25_14b_full`: 2-link linear chain
3. `critic_qwen72b_full`: 6 shards (0..5), each a 3-link linear chain
