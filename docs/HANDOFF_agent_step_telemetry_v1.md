# Handoff to Claude: agent step telemetry v1

**Review pending. CPU verdict: BLOCKED_BY_CPU_GATES. Do not submit to AIRCC.**
Omri confirmed AFTraj-2K as the third benchmark and authorized a dedicated
worktree and branch push. He explicitly instructed Codex not to connect to
or run AIRCC from this computer. The experiment/GPU freeze remains until
his recorded go; no Qwen inference, benchmark evaluation or new fusion ran.

Implementation and evidence commit:
`5c0bb613f1c637f6c9063bd8e4a749a21b4db6f6` on
`codex/agent-step-telemetry-v1`, based on
`da6b2872e8d8acae26db5dd6777b26f4049f9ad2`.
This handoff is a separate documentation commit after that implementation.

## Read first

1. [CPU delivery README](../results/agent_step_telemetry_v1/README.md)
2. [Runbook and decisions](experiments/AGENT_STEP_TELEMETRY_V1_RUNBOOK.md)
3. [Full failed UID lists](../results/agent_step_telemetry_v1/CPU_GATES.json)
4. [Two rendered fixtures per benchmark, marked spans](experiments/AGENT_STEP_TELEMETRY_V1_FIXTURES.md)
5. [Sources and file hashes](../results/agent_step_telemetry_v1/SOURCES.json)

All implementation changes are additive. Existing PRMBench, ProcessBench,
Hard2Verify and Socratic paths/outputs are unchanged in the branch diff.
The original checkout's earlier dirty work remains untouched. The managed
worktree acquired two unrelated CRLF checkout differences in vendored
bank11 files at creation; they were excluded from both commits.

## What was built

- `scripts/agent_bench/prepare_agent_sources.py`: pinned public source/reference
  copies, tokenizer-only download/cached same-pin validation, SOURCES manifest.
- `scripts/agent_bench/prepare_agent_inputs.py`: all-byte hash verification
  before reading, release counts/range checks, blind per-benchmark JSONL,
  evaluator-only sidecars, complete CPU lengths and G1-G4 reports.
- `spectral_utils/agent_bench/{contracts,render_qwen3}.py`: source-preserving
  APB/four TPRM/AFT mappings, native template rendering, scored-turn think
  conditioning, reasoning/content/context/entry-field spans, strict token checks.
- `cluster/run_agent_telemetry.py`: established external numerical collection,
  sparse step top-50 mapping, finite/shape checks, atomic exclusive-writer
  resume, immutable identities, budget/preflight checks before model loading.
- `cluster/submit_agent_telemetry.sbatch`: future-only immutable snapshot,
  owner_940/power-gpu/offline thinkfix setup. No submission was attempted.
- `tests/test_agent_bench.py` and a pinned official Jinja fixture: 16 focused
  tests plus the three-per-benchmark tiny-model interrupt/resume smoke.
- Runbook, rendered examples and compact `results/agent_step_telemetry_v1/`
  reports, manifests and earlier failed-attempt reports. Raw data is not in Git.

## Counts and gates

The README contains the full subset table. Totals are **4,166 items / 35,705
steps**: APB 1,000 trajectories / 8,509 assistant steps; TPRM 445 cases /
890 variants; AFTraj 2,276 trajectories / 26,306 turns.

G1 **FAIL**: 6/910 checked prefixes. The failing variants are BFCL case14
(`multi_turn_base_93`) and ToolSandbox case81/case91, both A and B.
G2 **FAIL**: 2,277/4,166 items. All AFTraj runs and `apb:gaia_dev:287`
fail strict decoded-token/character-span equality. The complete UID lists
and first failing span offsets are committed.
G3 **PASS**: all 445 pair contexts identical and actions different.
G4 **PASS**: all 4,166 JSONL rows recursively scanned; no annotation keys.
G5 **PASS**: release counts/domains/test IDs and AFT index range checks.
G6 **PASS**: 9 CPU fixture records, interruption after 4, 9 after resume,
no-op resume without rescoring, identical independent fresh telemetry.
G7 **NOT_RUN**: requires the later approved GPU smoke.

Sixteen focused tests and eight original external regression tests pass.
The original external five-item CPU smoke also passes. Python/shell syntax
checks pass. A rejected-startup check with a nonexistent model path confirms
that failed gates raise **before model loading or output creation**.
See `VALIDATION.json` and `G6.json`. These are CPU mechanical checks, not
Qwen3 numerical validation or evidence from benchmark inference.

## Decisions requiring review

**G1 contract conflict.** Six complete-conversation prefixes differ from
the native separately rendered history prefix. Native Qwen marks the last
assistant in a history-only rendering with an empty think block; the same
unscored turn lacks it in the full conversation. We retained native full
conversation context and reported literal G1 failure. The final check
uses actual full-render prefixes; earlier weaker prefix checks are preserved
as interim attempts. Define the gate relative to deliberate conditioning
changes before it can become PASS.

**G2 boundary conflict.** Native Qwen BPE tokens contain final punctuation
plus `\n\n` separators in AFT transcripts. Complete decoded tokens therefore
extend past the stipulated end-of-field span. The APB case has a newline
token spanning think-block padding and leading content newlines. No source
text was changed, token sequence segmented, span expanded, row discarded,
mask1 applied or gate relaxed. Resolve token attribution at these boundaries
with Omri, then rerun every CPU gate on a new versioned input directory.
The current collector correctly refuses these records under the strict spec.

**Approved source amendments.** Omri approved unique TPRM row-index pair IDs,
grouping repeated source cases together, literal unmatched think preservation
and text-only ToolSandbox actions. The runbook records exact mappings, the
nine top-level `tool_name`/`arguments` action variants, and the one GTA action
whose literal text has both tags. GTA text is retained verbatim as the
specific GTA mapping requires. Source reasoning split count is 145 (118
nonempty); the expected 146 includes a literal `<think>` mention.
There are 34 preserved invalid JSON argument strings and 2 mapped null contents.

**Metadata limits before future evaluation.** Specified AFT IDs can encode
safe/unsafe, and safe/unsafe task variants have different conv_id groups.
IDs never enter model prompts. Do not use ID-derived features and revisit
task grouping before downstream fitting/evaluation. No evaluation was built.

## Length report

All 4,166 items were tokenized without truncation. Eight GAIA trajectories
(3.2%) and two HotpotQA trajectories (0.8%) exceed 32,768 tokens; no other
subset has exclusions and no subset breaches 5%. Maximum is 1,515,925 tokens.
`LENGTHS.json` contains every UID length and min/Q25/median/Q75/P95/max per
subset; `EXCLUDED_OVERLENGTH.json` lists all ten exclusions.
Forty-two empty steps are retained and listed in `EMPTY_STEPS.json`.

## AFT annotation integrity and source freeze

Dataset revision: `11191169d3d99298df098eee8d83db425117faba`.
Code revision: `af549083fcc6c3eff4ccc558534515e46ca0578e`.
Source manifest SHA256:
`f856bc9b3bfa265cd6e1b64bc44083a38c390d3fc090057bfe87d52d1f749f2b`.
All **56** source/reference/tokenizer entries have hashes and byte lengths.

The pinned official loader sets safe `mistake_step=-1`. Official prompts
enumerate `Step {i}` over `turns` from zero; our entry_to_history preserves
that mapping, including step zero. Every one of the 1,114 unsafe indices is
within `[0,num_turns-1]`; all 1,162 safe indices are -1; zero exceptions.
The 169 safe + 163 unsafe test IDs all exist and total 332.
Unsafe-source counts are **728 injected / 386 diagnosed**.
The release totals 2,276 rather than the paper's 2,272; TPRM release is 445
rather than 984. These discrepancies remain explicit.

The comparison reference files and official metrics code were copied and
hashed, never executed. Qwen tokenizer HTTPS returned a Cisco block page;
we used the existing same-revision tokenizer restoration, checked against
its recorded hashes. Initial source/preparation failures remain preserved.

## Local private artifacts and next action

Private root on Omri's PC:
`/Users/osegev/Desktop/hallucination_detection/local_cache/agent_step_telemetry_v1/`.
Use `sources_verified/`, `inputs_final/`, and `g6_final/`. `inputs_final/`
contains each `<benchmark>_answers.jsonl` and `evaluator_only/` sidecars.
No benchmark telemetry has been collected. Earlier versioned input directories
record failed preparation and weaker interim checks; do not treat them as current.

The committed `GPU_PREFLIGHT.json` has **verdict FAIL**, no Claude approval
and no GPU go. It binds the implementation code hashes, tokenizer and answer
hashes. No timing estimate or budget decision exists.

Next: Claude reviews source-preserving choices and resolves G1/G2 with Omri.
After CPU gates pass, the immutable AIRCC snapshot and answers-only staging
can be prepared on a computer with cluster access. GPU smoke requires Omri's
recorded go; full inference additionally requires measured timing, estimate,
user budget authorization and the existing alignment gate. All future
cluster-to-Drive transfers use direct `rclone copy` plus zero-difference
`rclone check`. Codex performed none of those external actions here.

## Claude review fixes (2026-10-13)

1. AFT label leak. Release `conv_id` values encode the label (for example `gaia_0000_safe`,
   `..._unsafe_injected_reexec`). `uid` and `source_id` are now a sha256 of `conv_id`. The group
   is a sha256 of the whitespace-normalized task, so safe and unsafe runs of one task share a group.
   `conv_id` is kept only in the evaluator annotations. AFT uids changed: regenerate the AFT inputs
   into a NEW versioned directory. Do not reuse `inputs_final`.
2. G2 definition. Qwen BPE merges step-boundary text such as ".\n\n" and newline padding. A step
   span is now the set of covering tokens. A gate failure happens only when the overhang outside
   the step span has non-whitespace characters, the decoded text differs, or one token crosses two
   steps. `boundary_overhang` (chars, spans) is recorded per item.
3. G1 definition. The Qwen3 template gives the last history assistant turn an empty think block
   when it renders only the history. It gives no block when the scored turn follows. We keep the
   full-render context. G1 accepts exactly this one difference and lists it under
   `template_last_turn_block` (6 TPRM variants).

Result here (real Qwen3-8B tokenizer, local clones at the pinned commits): APB + TPRM, 1,890 items:
G1, G2, G3, G4 and LENGTH all PASS. AFT was not checked because HF is blocked in this container.
Codex must rerun prepare and all CPU gates (G1-G6) on the full 4,166 items.
