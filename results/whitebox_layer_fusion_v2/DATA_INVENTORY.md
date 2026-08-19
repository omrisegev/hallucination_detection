# Data Inventory and Capture Audit

## Frozen source roster

The benchmark uses 14 raw-cache/sidecar pairs from `gdrive:hallucination_detection/cluster_results/`. All access was read-only through the configured `gdrive:` rclone remote. The 28 large source files total **8,206,507,656 bytes** (7.64 GiB). `SOURCE_FREEZE_MANIFEST.json` records the exact remote path, modification time, size, Drive SHA-256, local path, and local SHA-256 for every raw cache and sidecar.

| Cell | Dataset / T | Model family | L / hidden | Source | Valid | Groups | Scope |
|---|---|---|---:|---:|---:|---:|---|
| `gsm8k_t1.0` | GSM8K / 1.0 | Llama-3.1-8B | 32 / 4096 | 500 | 496 | 496 | primary |
| `triviaqa_t1.0` | TriviaQA / 1.0 | Llama-3.1-8B | 32 / 4096 | 500 | 500 | 500 | primary |
| `sciq_t1.0` | SciQ / 1.0 | Llama-3.1-8B | 32 / 4096 | 1,000 | 1,000 | 1,000 | primary |
| `truthfulqa_t0.5` | TruthfulQA / 0.5 | Llama-3.1-8B | 32 / 4096 | 8,170 | 8,170 | 817 | primary |
| `squadv2_t0.5` | SQuADv2 / 0.5 | Llama-3.1-8B | 32 / 4096 | 10,000 | 10,000 | 1,000 | primary |
| `nq_open_t0.5` | NQ-Open / 0.5 | Llama-3.1-8B | 32 / 4096 | 10,000 | 10,000 | 1,000 | primary |
| `gsm8k_r1distill_t0.0` | GSM8K / 0.0 | DeepSeek-R1-Distill-Llama-8B | 32 / 4096 | 500 | 499 | 499 | primary |
| `coqa_llama7b_t0.5` | CoQA / 0.5 | Llama-1-7B | 32 / 4096 | 5,000 | 5,000 | 500 | **rejected appendix** |
| `gsm8k_mistral24b_t1.0` | GSM8K / 1.0 | Mistral-Small-24B | 40 / 5120 | 1,319 | 1,319 | 1,319 | primary |
| `gsm8k_nemo_t1.0` | GSM8K / 1.0 | Mistral-Nemo-12B | 40 / 5120 | 1,319 | 1,317 | 1,317 | primary |
| `gsm8k_phi35_t1.0` | GSM8K / 1.0 | Phi-3.5-mini | 32 / 3072 | 1,319 | 1,315 | 1,315 | primary |
| `gsm8k_mistral7b_t1.0` | GSM8K / 1.0 | Mistral-7B-v0.3 | 32 / 4096 | 1,319 | 1,307 | 1,307 | primary |
| `gsm8k_phi3mini_t1.0` | GSM8K / 1.0 | Phi-3-mini | 32 / 3072 | 1,319 | 1,315 | 1,315 | primary |
| `triviaqa_qwen3_t0.6` | TriviaQA / 0.6 | Qwen3-8B | 36 / 4096 | 5,000 | 5,000 | 500 | primary |

Totals: **47,265 source candidates, 47,238 evaluable candidates, 27 explicit token-truncation exclusions**. There are nine captured model families; eight remain in the primary evidence after excluding the CoQA/Llama-1 protocol defect.

## Join and row identity

Each sidecar key `"i:j"` is joined only to `raw[i]["candidates"][j]`. Positional joins are forbidden. Preparation asserts:

- exact raw/sidecar key-set equality and globally namespaced uniqueness;
- exact candidate-count and correctness-label equality;
- exact generated-token count for each evaluable candidate;
- per-token tensor axes equal `n_gen_tokens`;
- model, layer, hidden-size, projection-seed/dimension, and covariance-rank metadata;
- finite core lens, residual norm, and hidden-projection tensors;
- exact final residual KL identity;
- expected problem-group multiplicity.

Labels are inspected at this point only to verify source identity and are then removed from `LayerCell`. Evaluation reopens the raw cache only after score hashes are frozen.

## Exclusions

All 27 exclusions are raw traces longer than the sidecar's 1,024-token capture cap. The exact row IDs and raw/sidecar token lengths are frozen in `EXPECTED_EXCLUSIONS` and `data_audit.json`: four original Llama GSM8K rows, one R1 row, two Nemo rows, four Phi-3.5 rows, twelve Mistral-7B rows, and four Phi-3-mini rows. No row is excluded using its correctness label or evaluation score.

## Geometry capture defect

The recovered generator computes covariance eigenvalues in double precision and casts them directly to float16. Values above 65,504 overflow. After token-length exclusions, non-finite `cov_eigs` entries are:

- Phi-3.5: 733;
- Phi-3-mini: 4,675;
- Qwen3-8B: 47,008.

Core logit-lens tensors and `hid_proj` are finite in these same rows. Therefore the benchmark explicitly disables covariance-geometry performance on the affected cells; it does not impute or silently drop the overflow values. The mean-token hidden-projection HaloScope proxy remains enabled because `hid_proj` is finite and its semantics were recovered from source.

## Validation evidence

Four newly captured reports (`inside_coqa_llama7b`, Mistral-Small-24B, Mistral-Nemo-12B, Qwen3-8B) explicitly record passing Gate B and architecture checks. Several complete sidecars have false/empty reports because the old resume path rewrote the report with `todo=[]` without rerunning gates. This historical evidence is documented but does not satisfy the promotion gate. The benchmark remains blocked until corrected live Gate B is rerun over the full nested-candidate roster and the separate architecture pilot passes.

The CoQA/Llama-1 cell contains the only `hidden_middle_last` K=10 representation needed for paper-shaped INSIDE, but project Step 216 rejected that cell because of a generation/chat-template defect. Its results are appendix-only and excluded from all 13-cell macros and paired tests.
