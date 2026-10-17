# Execution-only amendment, 2026-09-12

The frozen V/C/S/D/T model definitions, data, seeds, scoring, bootstrap and
two scoring workers per suite remain unchanged. The machine has four CPU
cores. Original DUFS uses four workers; one new suite runs alongside it.

After original DUFS completes both reviews and its pipeline, permit up to
two independent suites at once. Capacity must complete and pass review before
either stability or depth can start. Temporal has no capacity dependency.
This changes scheduling only; no candidate, parameter or evaluation is added.

The old waiting controller PID2748 is replaced; its currently running capacity
scorer PID17352 is adopted without interruption. Creation-time checked Windows
handles identify jobs and exit status. The old logs remain. One scheduler
file lock and a per-suite active set prevent duplicate writers. A failed
stage stops new launches, lets already active children finish, and records
the failure. Four scheduling/process-handle tests pass before activation.

State: results/rbm_literature_completion_v1/{QUEUE_STATE.json,PROGRAM_STATE.json}.
Adoption: QUEUE_ADOPTION_20260912.json. Complete per-suite reviews remain
mandatory. The queue finishing does not itself prove overall goal completion.
