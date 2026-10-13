# CPU rerun v2 - 2026-10-13

**PASS: G1-G6 and LENGTH.** All 4,166 inputs were regenerated in
`inputs_v2_20261013`, including all 2,276 AFTraj records. The 14 checked
runtime/test/template files are byte-identical to Claude commit
`afae64a4d6826ac1465249f6a002ffde4b20fd37`. No gate definition or
threshold was changed by Codex for this rerun. No Qwen inference,
benchmark scoring, AIRCC connection or job submission occurred.

Claude reviews this CPU evidence next; G7, GPU smoke and the user budget
decision remain future steps. Thursday 2026-10-15 remains PRMBench focused.

## Coverage

- APB: 1,000 trajectories / 8,509 steps.
- ToolPRMBench: 445 pairs / 890 variants and steps.
- AFTraj: 2,276 trajectories / 26,306 turns.
- Total: 4,166 items / 35,705 steps. No item was silently omitted.
- Length exclusions: 8 GAIA (3.2%) and 2 HotpotQA (0.8%), unchanged.
- Zero truncation; no subset exceeds the 5% exclusion stop rule.

## Gates

| Check | Result | Coverage |
|---|---|---|
| G1 | PASS | 20 seeded APB prefixes + 890 TPRM variants; six exact native last-turn-block exceptions listed |
| Additional AFT prefix parity | PASS | 2,276/2,276 real native observer prefixes |
| G2 | PASS | 4,166/4,166 items, all 35,705 steps under unchanged covering-token checks |
| G3 | PASS | 445/445 pairs |
| G4 | PASS | 4,166/4,166 recursive input scans |
| G5 | PASS | source/domain/step counts, all 1,114 unsafe ranges, all 1,162 safe indices -1, all 332 test IDs |
| G6 | PASS | 9/9 CPU fixtures: 3 per benchmark, interrupt after 4, resume, identical independent fresh run |
| LENGTH | PASS | 4,166 full tokenizations, ten exclusions, no truncation |
| Unit tests | PASS | 17/17, including real native BPE and blind AFT IDs |
| G7 | NOT_RUN | Requires later approved GPU check |

G6 is the registered tiny-model mechanical smoke, not Qwen inference on
the full corpus. Full-corpus span validation is G2. `CPU_GATES.json`
retains the six G1 template variants; this is exactly Claude's pinned
rule, not a new exception added during the rerun.

## AFT label blindness

All 2,276 AFT inputs passed independent checks:
- `uid`/`source_id` are the specified 16-hex hashes; original release IDs are absent from complete input JSON.
- No `conv_id`, evaluator label, mistake metadata, `gold_answer`, `unsafe_source` or `split_test` field is present in scoring inputs.
- Flipping correctness/error metadata and test membership for every source record leaves the collector input byte-identical.
- System/tools, verbatim task and complete transcript independently reproduce the source observations for every item.
- All repeated normalized tasks map consistently to 1381 groups; no source/task hash collision observed.
- AFT input order is sorted by opaque UID; no safe/unsafe partition ordering.
- Collector AST/path scan finds no evaluator imports or evaluator-only path; no evaluator sidecar was opened by this audit.

`AFT_LABEL_BLINDNESS.json` records zero failures. Hash IDs are pseudonyms,
not access control: public release lookup is possible, so keep IDs out
of model prompts and fitted feature matrices. No downstream fitting was run.

## Native spans and preservation

Additional AFT audit checked 2276 items / 26306 turns.
Whitespace boundary overhang is recorded for 21612 spans (43224 characters total).
It is accepted only by the exact whitespace rule committed by Claude.
The full native token sequence, step char boundaries and source text are unchanged.
`AFT_TOKEN_SPAN_AUDIT.json` stores the per-item counts and exact prefix-parity checks.
The collector retains offsets/spans but currently does not serialize
the returned `boundary_overhang` aggregate in telemetry; the audit stores
it separately here. This is a metadata note for Claude, not a CPU gate change.

All older inputs and frozen CPU reports remain intact. APB and TPRM
answers are byte-identical to `inputs_final`; every AFT UID is new.
For AFT, only uid/source_id/group differ from the old input objects.
Source messages, tools, step/field spans and entry mapping are unchanged.

## Review and replay

- `AUDIT.json`: counts, code preservation and complete gate summary.
- `CODE_PIN.json`: 14 exact Git blobs and SHA256 values from `afae64a`.
- `SOURCES.json`: 56 verified source/reference/tokenizer files.
- `INPUTS.json`: hashes for the three newly prepared answer files.
- `CPU_GATES.json`, `G6.json`, `AFT_LABEL_BLINDNESS.json`, `AFT_TOKEN_SPAN_AUDIT.json`: checks and coverage.
- `COMMANDS.json`, `RUNTIME_ENV.json`: exact local commands and environment.
- `CPU_PREFLIGHT.json`: same-code/input CPU PASS snapshot, with reviewer/GPU approvals deliberately unset.
- `PROTOCOL.json`: versioned identity of Claude's current contract, not a change to the acceptance code.
- Reproduce with `scripts/agent_bench/audit_agent_cpu_inputs.py --repo ... --inputs ... --sources ... --previous-inputs ... --g6 ... --out <new directory>`.
- Raw sources/answers/annotations/model records are local only, outside Git. No Drive upload was performed.
