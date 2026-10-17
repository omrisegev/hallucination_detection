# Agent B — independent coverage and implementation audit

Reviewed snapshot: `080877eb9a8369341d647f4115a3b475c040b46e` under `/private/tmp/thinkfix-review-080877eb`. No research summaries or other reviewers' outputs used. Read project rules/handoff to comply with session requirements; conclusions below use code, manifests, seals and tracked per-response export. No GPU jobs, network, package installation or repository mutations.

## Verdict

No demonstrated population loss in the **final** manifests and no major masking/index-offset defect in fixtures. The three dropped Socratic answers in the superseded mask package were repaired; current code preserves already-empty steps and current manifests retain all 2,995 answers. However, a full raw-cache population audit and numerical replay cannot be certified from tracked artifacts: raw teacher-forced pkl/record files, source FEATURES.npz, variant PER_ANSWER.npz and external PREDICTIONS files are absent.

## Independently verified

- `prmscore_inputs/bank_N16.npz`: 6,969 distinct answer IDs; exactly the same IDs as tracked metadata; 94,203 steps; each answer's offset length matches metadata n_steps; 15 finite score/G arrays; all 14 `S__*` score arrays have within-answer zero mean to <3e-16. This is the legacy tracked export, not an independent check of every old/thinkfix/mask1 export.
- All 9 development raw-source SHA256 values in `source_bank_thinkfix/MANIFEST.json` map through MASK source/output SHA256 pairs exactly to `source_bank_thinkfix_mask1/MANIFEST.json`.
- Mask source/output counts: PRMBench 6969/6969; ProcessBench Qwen3-8B 3400/3400; Qwen3-4B 3400/3400. Sum 13769 unchanged, 145597 step count in all source bank manifests. These are independently reconciled manifests, not raw-row enumeration.
- External final counts: Hard2Verify 200/200; Socratic 2995/2995. Current Socratic feature manifest reports 26055 total steps, 26052 nonempty and 3 empty excluded. Superseded mask explicitly recorded 2992 answers and 3 failures; do not mistake that historical directory for the final run.
- Mask fixtures: per-token arrays/top-k fields shift exactly; step spans shift correctly; input entry untouched; original empty spans (387,387), (631,631), (230,230) remain empty after shift; newly empty (0,1) raises MaskError.
- Reconstructed patched external LSML runner/evaluator bytes reproduce both ANALYSIS_FREEZE SHA256 values exactly, including Windows-path A2. Available bootstrap/label-sanity SHA values match too. No stale analysis-script hash found.
- 4 of 7 source-bank code hashes could be checked and match snapshot files: rebuild_source_bank_v4, family_external_features, family_hist_features and family_tail_transfer. The other 3 inherited files were unavailable during this check.
- The source rebuild rejects missing raw IDs, duplicate raw IDs, wrong step/token counts and missing seen rows; `safe_row` removes labels before feature extraction; frozen old bank parity is enforced <=1e-6 and digit masks identical.
- `spans_resubmit.sh` explicitly resubmits mask1 with its shifted STEP_TOKEN_SPANS_THINKFIX_MASK1.npz. Therefore the original old-span submission was repaired, not a confirmed final-results defect.

## Actionable issues

### P2 — cache resume does not bind old rows to prompt or model identity

`cluster/run_teacher_forced.py:133-135, 423-424` and `cluster/run_prmbench_teacher_forced.py:97, 219-222`: existing rows are skipped solely by numeric idx. They do not store/compare prompt IDs, checkpoint revision, template hash or think_prefix. The new default empty_block can therefore resume an existing old `none` cache as complete, and a final manifest describes the newly requested prefix although saved rows are old. AST fixtures invoking the production score_rows functions confirm this: an old row plus changed prompt returns complete with stale payload untouched. The campaign's submit_step2.sh fresh-directory check mitigates this specific recorded run; no evidence the current final artifacts were mixed. Bind a run identity before scoring/resume, refuse mismatch, and verify per-row trace identity.

### P2 — external locked scorer's resume identity omits source telemetry binding

`scripts/run_external_local_cpu.py:25, 31-36`: its output-store identity hashes method/BUNDLE/code, while skip decisions use only output filename existence. Replacing the records directory with another telemetry variant under unchanged UID/method leaves stale outputs unchallenged. `scripts/evaluate_external_locked_scores.py:45-69` verifies prediction code/run identity but never compares saved `payload.telemetry_sha256` to the currently selected source record. The recorded setup creates separate, initially absent working directories for thinkfix and mask1, mitigating this run. Include a source records digest in run identity and verify per-record source hash before reuse/sealing.

### P2 scope question — the entropy abstention gate uses frozen old entropy inputs

`results/thinkfix_rescore_v1/thinkfix_mask1/nested_banks/dev_run_patched.py:81-83,146`: ENT is loaded from the original estimator-input ANSWER_FEATURES epr array; only ext.FE is redirected to repaired features. The ProcessBench ENTROPY_GATE_Q80 F1 panel therefore uses repaired step scores with old answer entropy. This does not affect argmax/within-answer AUROC or the primary ANSWER_Z gate. Recompute the answer entropy from each variant, or explicitly label this rule as an old-entropy frozen continuity control and restrict conclusions accordingly.

### P3 — mask metadata retains old trace length

`spectral_utils/first_token_mask.py:91-95` updates step_token_lengths but leaves align_diag.n_tokens and token_coverage unchanged. Fixture confirms a length4 trace becomes length3 while n_tokens remains4. Derived telemetry metadata is inconsistent, though no downstream numerical corruption was demonstrated. Refresh the trace-dependent diagnostics or explicitly mark them source-only.

### P3 — generic replay submission still routes mask1 to old spans

`results/thinkfix_rescore_v1/cluster_scripts/submit_downstream.sh:15` uses STEP_TOKEN_SPANS_OLD for every variant; later spans_resubmit.sh overrides this for the actual mask1 campaign. A fresh replay through the generic helper can reinstate the length-control discrepancy. Select spans by variant in the helper and record the span file/hash in RESCORE_STATUS.

## Evidence

`audit_fixtures.py` and `fixture_results.json` in this directory contain the deterministic tests. Runtime explicitly verified: main repository venv Python3.11.9, numpy2.4.6, scipy1.17.1. No claim of a fresh full-population GPU rerun or fresh external numerical scoring.
