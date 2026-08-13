# Data Readiness Phase

## Goal

This phase prepares the collected data for future research. It does not run a
hallucination detector, a localization method, U-PCR, or DUFS.

The raw files are immutable. The audit reads them, validates them and records
their hashes. It does not rebalance, rewrite, stage or commit them.

## Status meanings

- **READY:** the expected files, rows, labels and structural checks passed.
- **READY_WITH_LIMITATIONS:** the package may be evaluated, but only within its
  documented scientific scope and with the limitation attached to every result.
- **INCOMPLETE:** a required file, subset or manifest is missing.
- **BLOCKED:** a required label or integrity condition is invalid.

Class imbalance is not a data error. The audit reports it but does not remove or
duplicate examples.

## Canonical contract

Future data adapters identify every label-free unit with:

- `dataset_id` and `record_id`;
- `source_id`, so related rows cannot leak across splits;
- split, task and model ID;
- evidence condition and parent ID when relevant;
- immutable artifact path and row key.

Large token arrays remain in the original cache. Labels are represented by a
separate sidecar containing the record ID, label space, value and provenance.
This separation makes it possible for future label-free fitting code to receive
telemetry without receiving the answer labels.

## Checks

The audit verifies, where the data type permits:

- expected row and subset counts;
- SHA-256 checksums, including local Git-LFS objects;
- unique IDs and complete condition pairs;
- finite token telemetry and consistent array dimensions;
- identical answer tokens across RAG evidence conditions;
- valid sentence, claim or reasoning-step alignment;
- label availability, balance and grader provenance;
- completeness of competitor prediction packages.

Git-LFS pointer files are not checked out or replaced. If the corresponding
object already exists under `.git/lfs/objects`, the audit reads that object
directly and verifies that its SHA-256 equals the pointer OID.

## Known gates before future evaluation

1. HLE now has a complete interim `gpt-5.6-sol`/`xhigh` judgment sidecar over all
   2,158 rows. It replaces ROUGE-L for interim local evaluation, but it is not
   the original paper's GPT-4o judge and therefore remains a documented
   limitation rather than a paper-faithful label set.
2. The incomplete Qwen2.5-72B ProcessBench critic package needs OmniMath and a
   final manifest if that package is needed later.
3. The three reported PRMBench alignment failures must be identified. They must
   either be corrected from the source data or excluded with explicit IDs and
   reasons.
4. SemGrad's paper-faithful BEM grading is complete. A stratified human or
   stronger-LLM audit of grader disagreements is optional robustness analysis;
   it is not required to reproduce the SemGrad evaluation protocol.

## Reproduction

From the repository root:

```bash
.venv/bin/python -m unittest scripts.test_data_readiness
.venv/bin/python scripts/data_readiness_audit.py
```

The generated package is written to
`results/data_readiness_2026_08_11/` and contains the registry, one validation
file per dataset, a CSV summary, the canonical schema contract and Markdown and
HTML reports.

Two review queues containing benchmark text are written under ignored
`local_cache/data_readiness/`, not under `results/`:

- a deterministic 80-row SemGrad BEM disagreement audit;
- the 2,158-row HLE official-judge queue.

Only their hashes, counts and protocols appear in the report directory. The HLE
queue must never be committed or published because the benchmark requests that
its question and answer text not be redistributed.
