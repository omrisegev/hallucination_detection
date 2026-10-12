# Agent step telemetry v1

Implementation owner: Codex. Review owner: Claude, project conversation
"process reward beyond language". Third benchmark: **AFTraj-2K**, confirmed
by Omri on 2026-10-13. Who&When is optional and is not implemented.

**Current verdict: BLOCKED_BY_CPU_GATES.** The collector refuses to load a
model while G1 or G2 fails. No AIRCC connection/submission, Qwen inference,
benchmark scoring, feature extraction, fitting or evaluation occurred.

## Source freeze and local execution

Branch `codex/agent-step-telemetry-v1` starts at
`da6b2872e8d8acae26db5dd6777b26f4049f9ad2`. Reuse the main checkout's Python
environment in the dedicated worktree; do not create another environment.

```bash
PY=/Users/osegev/Desktop/hallucination_detection/.venv/bin/python
"$PY" -c 'import sys,numpy,scipy,pyarrow; print(sys.executable,sys.version,numpy.__version__,scipy.__version__,pyarrow.__version__)'
"$PY" scripts/agent_bench/prepare_agent_sources.py --out /absolute/private/sources
"$PY" scripts/agent_bench/prepare_agent_inputs.py --sources /absolute/private/sources --out /absolute/private/inputs --gates
"$PY" tests/test_agent_bench.py
AGENT_TOKENIZER_DIR=/absolute/private/sources/tokenizer "$PY" tests/test_agent_bench.py --cpu-smoke /absolute/private/g6
```

`prepare_agent_sources.py` pins APB, ToolPRMBench and AgentForesight code to
the spec's commits, resolves and pins the AFTraj dataset SHA, and copies
only the four APB release files, four ToolPRMBench release files, AFTraj
Parquet/test split, comparison references and tokenizer configuration.
It never runs official evaluators or uses training/AnswerOnly files.
Every source entry has repository, revision, relative path, bytes and SHA256.
All hashes are verified before any source file is parsed. Counts and
annotation ranges must pass before input JSONL is written.

This host's HTTPS certificate configuration needed installed system CA
certificates; TLS verification remained enabled. Qwen model file requests
returned a Cisco block page. The final freeze uses the project's existing
verified **same-revision** tokenizer snapshot, with SHA256 checks against
the earlier tokenizer restoration provenance. `--tokenizer-local DIR`
accepts only those pinned hashes. No model weights are downloaded.
HTML/block pages are rejected. The initial failed source/input attempts
remain local, separately from `sources_verified` and `inputs_final`.

The approved `pyarrow` installation was the only package change. Local
Python is 3.11.9, Transformers 4.57.6, NumPy 2.4.6, SciPy 1.17.1, PyArrow
26.0.0 and Torch 2.13.0. Cluster wheels remain the existing pinned versions;
local PyArrow 26 does not change `install_offline_tf_thinkfix.sh`.

## Recorded source mapping decisions

Omri approved these source-preserving amendments in the chat:

- ToolPRMBench identifiers repeat. Pair IDs append `:case<zero-based-release-row>`;
  A/B allocation uses SHA256 of that unique pair ID. `group` retains the
  original source-case identity across repeated cases. No deduplication.
- Preserve unmatched think tags as literal text. APB's hotpotqa row 29 is a
  literal mention of `<think>`, not a reasoning block. ToolPRMBench has
  opening-only and closing-only content, plus one GTA action containing
  both tags. GTA actions remain verbatim as required by section 5.3.
  Internal `reasoning_content=""` prevents native template parsing from
  dropping literal source text; source `reasoning_content` remains forbidden.
  This internal control is derived from `think_literal`, not from labels.
- Preserve ToolSandbox text-only actions. Its release also contains nine
  actions with top-level `tool_name`/`arguments`; these map to one native
  tool call without changing values. Missing decision-action roles map to
  `assistant`; absent content maps to `""` and is counted.

The APB release contains 145 assistant messages split under the source
reasoning rule, of which 118 have nonempty normalized reasoning. The spec's
146 think-tag count includes one literal mention. Do not replace the
observed counts with the expectation. Invalid argument strings are logged
and preserved, never repaired.

## Gates and exact boundary issues

See `results/agent_step_telemetry_v1/CPU_GATES.json`, `PREPARATION.json`,
`G6.json` and `README.md`. These files contain all failed UIDs and counts.

G1 uses **actual full rendered prefixes** for 20 deterministic random APB
items whose first scored turn has no source reasoning, plus all 890 TPRM
variants. Source reasoning forms are covered separately by rendering tests.
Six TPRM variants fail exact native prefix parity. Their history ends in an
assistant turn: rendering that history alone causes the native template to
add an empty think block to its final assistant; the same unscored turn
in a complete conversation lacks that block. The rule to preserve all
unscored full-conversation turns therefore conflicts with literal G1 for
these prefixes. Earlier interim G1 checks used separately rendered prefixes;
the final check is stricter and is the reported result.

G2 fails for all 2,276 AFTraj items and APB `apb:gaia_dev:287`. Qwen's normal
BPE tokens can contain both the final punctuation and the following `\n\n`
transcript separator. The prescribed char span ends at the last field,
so decoded complete tokens include extra separator characters. The APB
case has a newline token crossing from empty-think-block padding into
leading content newlines. No char span was expanded, no token was split,
no text was changed, no row was discarded and no tolerance was relaxed.
Every item is retained in inputs; the strict collector rejects these spans.
`EMPTY_STEPS.json` lists 42 zero-length steps, retained as such.

Claude must resolve these **rendering/span contract decisions** with Omri
before a CPU-gate PASS can be issued. In particular, decide how to attribute
a separator-crossing token while preserving the full native token sequence,
and define G1 against the deliberate scored-turn conditioning changes.
The current implementation and reports do not silently choose either policy.

G3 checks all 445 pairs. The first implementation's one pair failure was
an omitted top-level ToolSandbox action shape; after the mapping fix every
pair has identical rendered context and different action text.

G4 recursively checks all 4,166 answer rows. The collector imports no
evaluator module and opens no annotation files. Source IDs, as explicitly
specified, can themselves contain `_safe`/`_unsafe`; they are metadata,
never rendered into the model prompt. Future fitting must not derive
features from these IDs. AFT safe/unsafe variants can also share a task
while the specified `group=conv_id` differs; revisit that grouping before
any evaluation. No downstream grouping or scoring is implemented here.

G5 includes source items, steps, pair counts, domains, test split membership,
safe index -1 and every unsafe index in the full turn range. All pass.
The release has 2,276 AFTraj runs versus 2,272 in the paper text, and 445
ToolPRMBench cases versus the paper's 984. Use release counts.

G6 uses the existing random tiny GPT2 architecture (64 vocabulary, one
layer, 16 hidden dimensions), a reversible character tokenizer applying
the pinned real Qwen3 template, and three fixtures per benchmark.
Interruption after four saved records, resume, no-op resume and an independent
fresh run produce identical telemetry. This validates mechanics on CPU,
not Qwen3 numerical or GPU behavior. G7 remains NOT_RUN.

## Storage and resume contract

`*_answers.jsonl` contains only preparation inputs. Annotations go under
`evaluator_only/` and never accompany cluster collector inputs. The reference
LLM-judge outputs and official evaluator code are copy-only material.
Raw data, private answers, annotations, model records and fixture-model
telemetry remain outside Git; compact hashes/counts/gate evidence are tracked.

One native-token forward pass supplies all telemetry after the first
rendered message. Cheap arrays cover every remaining token, including
context and syntax. Top-50 arrays cover only step tokens;
`top_k_token_indices` explicitly maps their rows to `gen_token_ids`.
Empty top-50 arrays are stored as `[]`; the fixed width remains 50.

Char spans address `full_text`. Step/context/sub-span token indices address
`gen_token_ids`; negative context indices refer to conditioning tokens in
`prompt_ids`. `token_offsets` covers the full native token sequence, while
`gen_token_offsets` covers telemetry tokens. `step_details` keeps explicit
message/entry identity, reasoning/content/field spans and source-reasoning
flags. `entry_to_history` retains original AFT turn indices including zero.
The mask1 rule is not applied during collection.

The existing RecordStore provides atomic records and one exclusive writer.
Run identity covers code hashes, input/protocol/tokenizer hashes, model
revision, dtype, attention mode, context limit, mode and selected UIDs.
Changed identities fail before any run reports can be overwritten.
Resume validates completed record arrays and spans. SIGTERM/SIGINT finish
the current record, then exit 85. No requeue and no concurrent resume writer.

## Future AIRCC handoff (do not run from this computer)

There is no permission to start a GPU job. The meeting freeze remains.
Claude reviews the branch and resolves the failed CPU gates first; then
Omri records permission for GPU smoke. G7 has to pass and timing must exist
before `cost_estimate` can produce `ESTIMATE.json`. Full collection additionally
needs `authorize()`'s protocol/estimate hashes, `approved_by="user"`, and
`max_gpu_hours` at least the estimated GPU hours.

The prepared launcher uses account `cycle3_tau_averbuch_prj`, QoS `owner_940`,
partition `power-gpu`, one GPU, a one-hour default, the NGC 25.01 container,
existing offline thinkfix wheels and an immutable
`/shared/cycle2_tau_averbuch_prj/omrisegev1/code_agent_step_v1_<sha>` --chdir.
It accepts only an offline HF snapshot path ending in the pinned model revision.
Only answers go to `$SHARED/inputs/agent_step_telemetry_v1/`.
Outputs go to `$SHARED/results/agent_step_telemetry_v1/<benchmark>_qwen3_8b_<sha>/`.

Before submission, the external reviewer constructs a new preflight with
all G1-G6 and LENGTH statuses PASS, matching `code_identity`, per-benchmark
`answers_sha256`, `tokenizer_sha256`, `protocol_hash`, a `session_id`,
`reviewed_by="Claude"` and `gpu_approved_by="user"`.
The tracked `GPU_PREFLIGHT.json` deliberately has verdict FAIL.
No timing estimate or budget approval is fabricated.

After an approved collection, check item counts excluding only the ten
overlength rows, finite arrays and exact span/top-50 mapping. Transfer
directly from AIRCC with `rclone copy` to
`gdrive:hallucination_detection/cluster_results/agent_step_telemetry_v1/`,
then `rclone check` with zero differences. Never sync or use a local hop.
