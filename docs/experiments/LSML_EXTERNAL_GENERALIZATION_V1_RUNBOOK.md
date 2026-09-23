# External telemetry collection runbook

Status: collection implemented; fusion completion/evaluation deferred by user
priority. No external quality has been evaluated. MedPRMBench remains deferred.

## Inputs and preparation

`scripts/prepare_external_sources.py --out scratch/external_generalization_private/sources`
pins the public release code/data and model revisions without model-weight downloads.
The encrypted Hard2Verify `test.jsonl` must be placed in `sources/hard2verify/encrypted_test.jsonl`
from the revision recorded in `SOURCES.json`; its compact input manifest records
the SHA256. Decryption uses the pinned author's utility. Never commit plaintext.

`scripts/prepare_external_inputs.py --sources <sources> --out <private-inputs>`
writes answer-only files separately from `evaluator_only/`. The collector receives
only `answers.json`. Observed release counts: Hard2Verify 200 answers/1860 steps;
Socratic 2995 answers/26055 steps. Three empty Socratic steps retain zero-length
spans. Eighty rows contain out-of-range error indices; retain the official inert
membership semantics. Exact-text question grouping gives 79 and 2286 groups;
these are not claims about semantic question identity. Development overlap still
needs completion before a disjoint-transfer claim, not before label-blind collection.

## Collection

Use `cluster/run_external_telemetry.py --help`. Required: answer file, pinned model
revision, output directory, collection protocol and same-session preflight JSON.
`--mode smoke` tokenizes the entire corpus, selects <=12 equally spaced length
quantiles including the longest, then collects only those rows. No generated
answers, fits, scores, thresholds or quality metrics. `--mode full` additionally
requires the measured estimate and a user-approved budget record bound to hashes.

Output per answer: prompt and answer token IDs; character offsets; original step
character/token spans; raw top50 IDs/logprobs; actual-token logprob (including
outside top50); full-vocabulary logsumexp; full entropy; historical top15 entropy;
timing and peak GPU memory. `token_entropies` deliberately preserves the historical
top15 convention; `token_entropy_full` is the additional full-vocabulary field.
Precision: bf16 model, fp32 quantities, SDPA. Prompt: exact existing math_prompt
plus ` /no_think`, tokenizer-default chat template. QwQ remains a distinct
backbone/template transfer. Empty steps cannot supply a token readout; later
evaluation must report their failures explicitly, never drop them silently.

Per-answer atomic JSON records use an exclusive writer lock, immutable run identity
and resume skip checks. A stale lock after SIGKILL requires an operator to verify
the old job has ended before removing just that lock. SIGTERM completes the current
record and exits85. Never start two writers in one output directory. No automatic
requeue. Smoke and full output identities are distinct.

The fixed-token alignment gate compares parallel teacher forcing with individual
prefix passes at three positions. It is tested on CPU and runs on the actual GPU
model. Historical generated-cache Gate B is supported with `--validate-pkl` but
requires a compatible source template. The found ARS pilot has thinking ON and
degenerate labels; it is NOT an acceptable replacement validation cache for this
no-think protocol. Do not describe the new alignment check as historical Gate B.

## AIRCC

Use `cluster/submit_external_telemetry.sbatch`, one GPU, one hour, no requeue,
NGC `nvcr.io/nvidia/pytorch:25.01-py3`, commit-specific `--chdir` snapshot.
Current account from `sdata`: `cycle3_tau_averbuch_prj`, QoS `owner_940`, partition
`power-gpu`. Data/cache paths under the existing cycle2 shared directory remain
accessible. Do not reuse old `owner_880` submissions. Shared disk has >11 TB free.
Local verified cache cleanup reclaimed 9,971,019,380 bytes; tracked cache objects
were restored to exact HEAD LFS pointers, untracked copies removed. Restore URLs,
MD5/SHA256 and metadata are under `results/lsml_external_generalization_v1/`.

Cluster rclone requires explicit configuration after the home-directory change:
`/shared/cycle2_tau_averbuch_prj/omrisegev1/bin/rclone --config /shared/cycle2_tau_averbuch_prj/omrisegev1/.config/rclone/rclone.conf`.
Use `copy`, never `sync`/delete, from the verified run directory directly to
`gdrive:hallucination_detection/cluster_results/lsml_external_generalization_v1/<run>`.
Verify archive hashes before treating it as backed up. Download only compact
timing/manifests locally. Decrypted Hard2Verify remains private in both locations.

## Validation and next gate

Run `python -B tests/test_external_generalization.py`,
`python -B tests/test_external_generalization.py --cpu-smoke <new-private-dir>`,
and `python -B tests/test_runtime_fusion_protocol.py`.
The real CPU smoke executes five tiny randomly initialized GPT2 fixture forwards,
quantity extraction, serialization, interruption and resume. It is infrastructure
validation, not a miniature scientific result or evidence for Qwen quality.

After timing: estimate full collection GPU/CPU/storage/wall costs, separately
include model loading and retries. Comparator generation has not been timed by
telemetry jobs; do not present that estimate as the cost of the entire approved
evaluation matrix. Ask for the user's full-run budget decision with these costs.
Only then launch full collection. Further source calibration, CT7 reconstruction,
official evaluator parity, comparator runners, prediction seals, full bootstrap
and technical/Hebrew results remain required for scientific completion.
