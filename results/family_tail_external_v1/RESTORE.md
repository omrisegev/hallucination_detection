# Restore and reproduce the family-tail external comparison

The committed protocol is
[`docs/experiments/FAMILY_TAIL_EXTERNAL_V1.md`](../../docs/experiments/FAMILY_TAIL_EXTERNAL_V1.md).
The source method is identified by the byte-exact transfer lock, not a display name.
Do not edit frozen prediction or source-gate artifacts while reproducing a run.

## Compact repository artifacts versus private archive

Code, source lock, input/code hashes, aggregate results, independent audit scripts,
reports and PNG/PDF plots belong in Git. Large arrays and per-answer feature/score
records are excluded by this directory's `.gitignore`.

`ARCHIVE.json` records the final archive's size, SHA256, storage path and verification
status. `ARCHIVE_CONTENTS.json` records every included member's size and SHA256.
An archive is backed up only when the manifest explicitly records a verified
remote copy; a local `NOT_UPLOADED` archive is not a remote backup.

The 190,131,044-byte archive for this run is at
`gdrive:hallucination_detection/cluster_results/lsml_external_generalization_v1/evaluation_archives/family_tail_external_v1_results_a1e2f683cc6932a9.tar.gz`.
Its remote SHA256 was verified against the local SHA256
`a1e2f683cc6932a95522f71c5bb5e5998bfdfb5f7a27a54f056df4ef675395ed`.
There are 6,312 verified members. `ARCHIVE.json` is the authoritative restore
manifest. The independent review remains deferred; see `AUDIT_DEFERRED.md`.

Download only after checking the manifest and available space. Verify the outer
SHA256 before extracting into an isolated replay directory. Verify members against
`ARCHIVE_CONTENTS.json`. The archive includes no benchmark question/answer text,
raw telemetry PKLs, credentials or decrypted Hard2Verify examples.

Its `reproduction/source_pool/` contains the exact original pool, names, source
answer roster and original telemetry-hash manifest. The result tree contains the
full raw-to-feature gate arrays, per-answer external features/scores, seals and
bootstrap arrays. This permits independent source-feature and score/metric replay
without another language-model pass. Fresh raw-to-feature extraction additionally
requires the existing private raw telemetry described below.

## Dependencies already archived by the preceding pipeline

- Source calibration/group metadata: see
  [SOURCE_DEPENDENCY_ARCHIVE.json](../lsml_external_generalization_v1/evaluation/SOURCE_DEPENDENCY_ARCHIVE.json).
  Restore each member to its `repository_relative_path`, checking its SHA256.
- Earlier bank11/CT7 prediction anchors: see
  [EVALUATION_ARCHIVE.json](../lsml_external_generalization_v1/evaluation/EVALUATION_ARCHIVE.json).
  Their per-answer shards are required for the exact matched-anchor assertions.
- External raw telemetry: see
  [FULL_ARCHIVES.json](../lsml_external_generalization_v1/FULL_ARCHIVES.json) and
  [FULL_COLLECTION_VERIFICATION.json](../lsml_external_generalization_v1/FULL_COLLECTION_VERIFICATION.json).
  Restore the three cells beneath
  `scratch/external_generalization_private/evaluation_archives/`.
- Normalized answer-only inputs, separately isolated evaluator annotations and
  pinned official evaluator source files use the preceding pipeline's private
  input/source directories. Their exact hashes are in `IMPLEMENTATION_FREEZE.json`.
  Do not publish Hard2Verify decrypted text when restoring these inputs.
- The nine original source raw telemetry PKLs are named and hashed in
  `source_full_v1/GATE.json` and `CT7_TOKEN_MATRICES.MANIFEST.json`. Their local cache
  paths are dependencies, not a claim that the PKLs were newly uploaded here.

## Replay order

1. Verify `IMPLEMENTATION_FREEZE.json`, the method lock and all private inputs.
2. Use saved source-gate arrays for independent numerical replay, or run
   `verify_family_external_source.py` with the restored source pool and raw PKLs
   into a **new** output directory for a fresh full13,769-answer extraction check.
3. Run `run_family_external.py` only with the complete source gate. A changed
   implementation/input identity must use a new experiment directory, not resume
   a shard with different contents.
4. `evaluate_family_external.py --seal-only` verifies all6,190 records and seals
   all cells before annotations. The full evaluator then computes all registered
   metrics and100,000 paired source-question bootstrap draws.
5. Run the three independent audit scripts, reconcile their raw results, and
   record `RED_TEAM.md` before treating the result as independently reviewed.
   This run was rendered under the user's explicit waiver recorded in
   `AUDIT_DEFERRED.md`; independent review is still outstanding.
   `diagnose_family_external.py` adds label-free constant-channel and CUSUM
   contribution diagnostics.
6. `render_family_external.py` writes the report and standalone PNG/PDF figures.

This is an exploratory follow-up on already inspected external benchmarks.
Reproducibility does not make them unexposed confirmation data.
