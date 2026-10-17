# Frozen PRM boosting transfer implementation

Code delivery from `claude/prm-boosting-generalization-v1` (`dada9487`) on
`codex/prm-boosting-generalization-v1`. The dirty main checkout is preserved.
No merge, model-weight download, GPU/cluster job, channel selection or new-model
analysis was performed.

Assumption: a model exposes its final-normalized backbone output and a checkpoint
or locally cached base vocabulary head. Success requires tokenizer parity,
explicit head inventory, historical metric replay, and all three spec gates.
Template/reward mismatch, missing borrowed head, counted context failure, or a
failed gate blocks reporting fused results.

## Implemented

- `spectral_utils/prm_adapters.py`: historical Qwen-7B, official Qwen-72B,
  RLHFlow, PURE, MathShepherd, and inspectable blocked Skywork protocols.
- `cluster/run_prmbench_prm_telemetry.py`: adapter dispatch, same-forward
  backbone telemetry, 1,024-position vocabulary chunks (historical Qwen-7B
  retains its original GEMM), counted context overflow, native labels, head
  provenance, atomic resume identity, checkpoint/SIGTERM exit 85 and T1 comparison.
  A finished run with counted failures writes its manifest and exits 2.
- `results/prm_boosting_v1/fuse.py`: `--prm-pkl`, `--step-evidence`, `--out`,
  portable repository root and environment bindings for the frozen Monday
  adapter. No frozen function body is rewritten. N16/N16P have no `lf__` channels,
  so historical pool-sign data is not needed; missing signs are not invented.
- `evaluate.py`: all seven frozen arms, native/Q80 rules, exact G1/G2/G3 gates,
  complete-population guard, official metrics, paired source-group bootstrap
  (2,000 draws, seed 20261013), per-model reports/scores and `SUMMARY.csv`.
  It refuses to overwrite an existing tag. Only native results are written if
  gates fail. G3 requires the supplied Qwen3 bank to match the hashes in the
  parity builder's companion `MANIFEST.json`.
- `third_party/skywork_prm/`: Apache-2.0 helpers, including their required
  transitive `modeling_base.py`, byte-identical at upstream commit
  `719b56b17447405e0f10e6c0360a581cf4ffa9c1`.

`results/monday_external_panel_v1/score.py`, `results/prm_boosting_v1/build_bank.py`,
the sbatch file, existing result arrays/CSVs and benchmark contracts remain
byte-identical. Validation records their hashes.

## Verified IDs and wrapper differences

Official wrappers are pinned to [PRMBench commit
`a1ae4eab803e14c4c6316bf89d1679b0486faa8c`](https://github.com/ssmisya/PRMBench/tree/a1ae4eab803e14c4c6316bf89d1679b0486faa8c/mr_eval/models).
All requested HF IDs exist. `HEAD_INVENTORY.json` records canonical IDs,
revisions, config architectures/context lengths and tensor-index evidence.
No real head-weight difference was measured locally; that measurement runs
against cached base weights during loading. Missing comparison weights produce
an explicit null plus `base_head_compare_error`, rather than a fabricated zero.
A borrowed head is mandatory for PURE, whose load fails if it is absent.

- **Qwen:** the official Python wrapper uses a single backslash in `"\boxed{}"`,
  which evaluates to the backspace character `\x08` followed by `oxed{}`. The
  72B adapter preserves this literally and applies upstream `remove_step_prefix`.
  The historical 7B adapter retains the existing literal backslash prompt and
  raw steps for T1. Use `qwen_prm_72b` for 72B; the driver rejects using the
  historical adapter with a 72B override.
- **Prefix cleanup:** upstream strips `Step 3.` and `3.`, but retains `Step 3:`.
  PURE does not clean prefixes. MathShepherd preserves an existing Step prefix
  in the input while excluding it from telemetry.
- **BPE boundaries:** a content token may merge the preceding template space.
  The input and token IDs remain exact. Decoding is checked modulo boundary
  whitespace on new adapters, which is counted separately. Tokens merging
  non-whitespace template text are rejected. Historical Qwen retains its exact
  mismatch check. No independent template/tag token is scored.
- **PURE:** the user selected the official wrapper ID
  `jinachris/Qwen2.5-Math-7B-PRM800K`. HF redirects it to
  [`jinachris/PURE-PRM-7B`](https://huggingface.co/jinachris/PURE-PRM-7B),
  the same revision `b0cce6763f2317762c9cb46720c8fe7b07f44a88`.
  The wrapper's `\n\n` separator is preserved; its card's example uses `\n`.
- **MathShepherd:** the requested mistral-named checkpoint's config declares
  `LlamaForCausalLM`/`model_type=llama`; AutoModel honors that config. The card
  describes a Mistral-7B model. The native candidate tokens remain those derived
  by the official tokenizer, rather than guessed hard-coded IDs.
- **Causal PRMs:** RLHFlow and MathShepherd use the official wrappers' default
  fp32 loading; Qwen/PURE use their specified bf16. RLHFlow uses full-vocabulary
  softmax at pattern+3, while MathShepherd uses softmax only over `[+, -]`.

## Blocked models

**Both Skywork sizes are blocked by the user's decision.** The unchanged helper
adds an extra reward tag after the response's trailing newline and produces
n+1 rewards for n steps. Template tests preserve every upstream input ID and
expose that terminal position separately, but `load`, `rewards`, driver dispatch,
and evaluator dispatch reject inference/evaluation until the protocol is
clarified. No reward was silently discarded from an inference run.

**ReasonEval is optional and blocked.** Its borrowed base LM head and native
redundancy-head evaluation protocol have not been validated. The inspected
checkpoint index has `score_head.weight` and no `lm_head`. It is not exposed
as a runnable adapter.

**PURE is conditional on a cached base head.** Its checkpoint has `score`
weights and no `lm_head`; the adapter reads the base head locally, then applies
it to PURE's own final-normalized state. It never performs a base-model forward.

## CPU validation

T2: 21 real-tokenizer cases (7 adapters x 3 fixture rows), including an empty
step and `Step 3:`. Snapshots were generated by executing the pinned official
wrappers' `getitem_function` bodies; historical Qwen uses the base driver's
`prm_inputs` instead. Exact full token sequences and decoded full strings match.

T3: tiny random Qwen2 config verifies final norm, own/unused/borrowed projections,
a tied borrowed embedding read from a tiny local checkpoint, and native reward
rules against a tiny causal model.

T4: reconstruction on all 6,969 existing answers (6,030 eligible within-AUCs):

- S3: within-AUC **0.8508269739**, official PRMScore **69.3845740451**.
- EQ2: within-AUC **0.8404734468**, official PRMScore **68.5338231181**.

Both match the existing CSVs within 1e-6; see `REPLAY_V1.json`. T4 reuses the
stored v1 float32 channel atlas for S3, as `small_bank.py` did, and the stored
float64 MEAN_MOM_ga score for EQ2. It does not reselect or fit channels.

Additional tests cover gate failure, paired bootstrap, counted context overflow,
SIGTERM checkpoint/resume, chunked projection and corruption detection by T1.
See `VALIDATION.json` for the final test count and command.

The verified project interpreter is
`/Users/osegev/Desktop/hallucination_detection/.venv/bin/python` (3.11.9).
After the user authorized installation when missing, pandas, sentencepiece and
protobuf were installed in that existing environment. NumPy/SciPy were unchanged.

```bash
PRM_TOKENIZER_DIR=/private/tmp/prm-tokenizers \
PRM_V1_METADATA=/path/to/existing/prmbench_prm.pkl \
/Users/osegev/Desktop/hallucination_detection/.venv/bin/python \
  -m unittest discover -s tests -p 'test_prm*.py' -v
```

Tokenizers are CPU-only test assets outside Git, in directories named by
`HF_id.replace('/', '__')` (PURE uses its canonical ID). Without the environment
variable, tests require those tokenizers already in the HF cache. No weight
loading occurs in the tests.

## Commands for Claude (not executed here)

Run the project's `/preflight` workflow before any submission. Preserve the
sbatch file and choose time/linear chained dependencies at submission for 72B.
Do not source the native CPU environment in a GPU submission shell.

T1 first-40 GPU pilot (fresh output path, same metadata order):

```bash
python cluster/run_prmbench_prm_telemetry.py --mode prm --adapter qwen_prm \
  --meta-pkl results/prm_boost_v1/full_prm/prmbench_telemetry.pkl \
  --n-samples 40 --out results/prm_boost_gen_v1/qwen7b_t1 \
  --compare-to results/prm_boost_v1/full_prm/prmbench_telemetry.pkl
```

T1 compares first-row identity/order, rewards, token IDs and spans exactly, and
floating telemetry/top-k log probabilities at absolute tolerance 1e-5. The real
GPU T1 remains **pending**; the CPU comparator test does not claim it passed.

For each new model, run the unchanged bank builder on its raw telemetry. Run
Qwen3 `build_bank.py --expect-equal` separately and keep `PARITY.json` together
with its generated `MANIFEST.json`. Then:

```bash
python results/prm_boosting_generalization_v1/evaluate.py \
  --adapter qwen_prm_72b --tag qwen_prm_72b \
  --features /path/to/own/FEATURES.npz \
  --q3-features /path/to/qwen3/mask1/FEATURES.npz \
  --prm-pkl /path/to/own/prmbench_telemetry.pkl \
  --manifest /path/to/own/manifest.json \
  --q3-parity /path/to/qwen3-parity/PARITY.json \
  --answers /path/to/step_evidence_v1/OOF_ANSWERS.csv \
  --labels /path/to/step_evidence_v1/OOF_STEP_SCORES.npz \
  --out /path/to/fresh/evaluation
```

`--answers` and `--labels` must be the frozen v3 full-roster inputs, aligned to
the bank's UID/offset order. Official PRMScore still uses the original
PRMBench `error_steps`; within-AUC uses v3 labels. Existing result files are not
rewritten. This is a cross-model test on previously inspected development data;
the supervised PRM and Qwen-7B label-selected channels retain that provenance.
Only the fusion is label-free.

## File inventory

Changed:

- `HISTORY.md`
- `LESSONS.md`
- `PROGRESS.md`
- `cluster/run_prmbench_prm_telemetry.py`
- `results/prm_boosting_v1/fuse.py`

Added:

- `results/prm_boosting_generalization_v1/HEAD_INVENTORY.json`
- `results/prm_boosting_generalization_v1/README.md`
- `results/prm_boosting_generalization_v1/REPLAY_V1.json`
- `results/prm_boosting_generalization_v1/VALIDATION.json`
- `results/prm_boosting_generalization_v1/evaluate.py`
- `spectral_utils/prm_adapters.py`
- `tests/fixtures/prm_template_snapshots.json`
- `tests/test_prm_adapters.py`
- `tests/test_prm_telemetry_driver.py`
- `tests/test_prm_transfer_evaluate.py`
- `third_party/skywork_prm/LICENSE`
- `third_party/skywork_prm/README.md`
- `third_party/skywork_prm/__init__.py`
- `third_party/skywork_prm/model_utils/__init__.py`
- `third_party/skywork_prm/model_utils/io_utils.py`
- `third_party/skywork_prm/model_utils/modeling_base.py`
- `third_party/skywork_prm/model_utils/prm_model.py`

The two checkout-generated `_bank11` EOL-only differences are excluded from the commit.
