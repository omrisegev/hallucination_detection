# Supervised diagnostic of Varentropy-expansion fusion v1 — correction record (2026-09-13)

Scope: three confirmed audit findings on the completed supervised diagnostic
(`results/varentropy_expansion_fusion_v1/supervised/`). The scientific
definition of the experiment is unchanged: q = 0.8, v2 source groups and outer
folds, step-label rules, benchmark population (13,769 answers), banks B2d_sel /
B2_sel, top-10 token-mean readout, ridge 0.01, L-BFGS-B. The completed run's
files (`CHECKPOINT.sqlite`, `METRICS.json`, `CONTRASTS.json`, `MANIFEST.json`,
`RUN_STATE.json`, `SCORES.npz`) are preserved unchanged; all corrections live in
`results/varentropy_expansion_fusion_v1/supervised/correction_20260913/`.

Code: `scripts/run_varentropy_expansion_supervised_v1.py` (driver, edited),
`spectral_utils/varentropy_expansion_supervised.py` (helpers added; `fit`,
`score_steps`, `step_labels`, banks untouched),
`scripts/run_varentropy_expansion_supervised_calibration_correction.py` (new),
`scripts/test_varentropy_expansion_supervised.py` (new, 11 tests).

## Finding 1 — PRMScore threshold depended on the held fold's labels

Mechanism: `fit_folds` trains model_g on folds != g; `evaluate_arrays` sets q_f
as the 0.8-quantile of the saved scores of training folds g != f, each produced
by model_g, which was trained with fold f's labels. The train/test group
disjointness assertion does not cover this.

Correction (PRMBench only; ProcessBench scores and gate are reused unchanged):
for each bank and held outer fold f, for each inner fold h != f, fit on
outer-training answers with fold not in {f, h} (same `fit()`, labels, ridge),
score fold h. This yields inner out-of-fold calibration scores for every
outer-training answer from models that never read fold f. q_f is the
0.8-quantile over those scores under the exact concatenation/quantile
convention of `evaluate_arrays` (`calibration_quantile`). Fold f itself is
scored by the existing saved outer fit `prmbench_qwen3_8b|{bank}|{f}` (trained
on folds != f; reused, not refit). The corrected evaluation calls
`evaluate_arrays(..., calibration_thresholds=...)` on the saved scores, and
PB / within / pooled metrics are asserted identical to the original.

Held-fold blindness is enforced at index level (`held_fold_blind_check`: no
training row, no scoring row, no training source group from fold f; consumed
label step indices disjoint from fold f's step indices) and demonstrated by a
perturbation: all known step labels of one held fold are flipped and its
calibration models refit in the same process; theta bytes, calibration scores
and q_f must be bit-identical. Synthetic unit test additionally shows the same
flip on a training fold does change the model.

Outputs: `METRICS_CORRECTED.json`, `THRESHOLDS.json` (original q, replayed q,
corrected q, inner-fit provenance and convergence, score-scale summaries of
inner-OOF vs originally used outer-training vs held-fold scores),
`PROVENANCE.json` (per threshold: every contributing model's training folds
and full training source-group list, intersection assertion, perturbation
record), `CALIBRATION_FITS.sqlite` + `CALIBRATION_MANIFEST.json` (binds the
supervised code, the frozen CHECKPOINT.sqlite / METRICS / SCORES hashes,
JOINED / FOLDS / gate / PRMB labels), `CALIBRATION_SCORES.npz`.

Results: see the table below (filled from the run).

| Bank | PRMScore original | PRMScore corrected | delta |
|---|---|---|---|
| sup__B2_sel (138) | 0.629387 | 0.629332 | -0.000055 |
| sup__B2d_sel (33) | 0.629090 | 0.629366 | +0.000277 |

Thresholds q_f (original -> corrected):

| Bank | f0 | f1 | f2 | f3 | f4 |
|---|---|---|---|---|---|
| B2_sel | 0.370880 -> 0.372545 | 0.353433 -> 0.375131 | 0.378053 -> 0.369081 | 0.375099 -> 0.377547 | 0.380007 -> 0.365332 |
| B2d_sel | 0.370750 -> 0.374474 | 0.354262 -> 0.370621 | 0.378746 -> 0.369170 | 0.376418 -> 0.376418 (-4.2e-7) | 0.381711 -> 0.370150 |

All 40 inner models converged (0 iteration-limit, 0 stalled, 0 failures);
inner training sets 4,141-4,209 answers versus 5,545-5,613 for the outer fits;
0 of 40 contributing models share a source group with their held fold. The
original thresholds replay exactly from the saved scores (assertion), and
every non-PRMScore metric of the corrected evaluation equals the original
(assertion). Perturbation: B2d_sel, held fold 0, 19,415 known step labels
flipped, 4 refits, theta bytes / calibration scores / q_f identical. Runtime:
calibration stage 7,669 s of fits + evaluation, 372 s perturbation (8,041 s
total, 1 worker, threadpool limit 1, PRMB memory gate 2.0 GB as in the
driver's default); cache verification 367 s; smoke re-run about 2 min.
Interpretation is left to the main record; the shifts are reported as
measured, without a claim about their direction or size.

## Finding 3 — smoke PASS rule

Old rule: PASS if every fit status was in {FIT, FAILED, STALLED}; an all-failed
run passed. New rule (`classify_fit`, `smoke_status`): each fit is one of FIT
(converged), FIT_ITERATION_LIMIT (finite, stopped on the optimizer budget —
reported explicitly, never as converged), EXPECTED_SMOKE_LIMITATION (declared
fold failure: empty train/held fold or a missing class), STALLED, or
UNEXPECTED_FAILURE. PASS requires >= 1 FIT/FIT_ITERATION_LIMIT per bank and no
UNEXPECTED_FAILURE / STALLED; INCONCLUSIVE if a bank has no successful fit but
only expected limitations; FAIL otherwise. Per-fit reasons are preserved.
`FIT_HEALTH.json` is written beside SMOKE.json / METRICS.json.

Re-run of the smoke (27 answers, 1 worker, output
`correction_20260913/smoke/`): identical 90 fit outcomes to the original
smoke (20 FIT, all converged; 64 expected limitations: 54 empty folds, 10
missing class; 6 STALLED on training sets of 1–2 answers; 0 unexpected
failures; 0 iteration-limit fits). Status under the new rule: **FAIL**
(because of the 6 STALLED fits); the original SMOKE.json reported PASS for the
same outcomes. The extraction assertions of Finding 4 all held on the 27
answers and per-answer provenance was stored.

Full completed run, re-verified from `METRICS.json`
(`FIT_HEALTH_SUPERVISED.json`): 90 fits, 67 converged, 23 at the iteration
limit (all 23 are B2_sel ProcessBench fits; every PRMBench fit converged),
0 stalled, 0 failures.

## Finding 4 — manifest / extraction verification below the benchmark contract

Driver changes: (a) the manifest now hashes the raw source artifacts (8 PB
pickles, PRMB telemetry, PRMB labels) and the test file (schema
`varentropy-expansion-supervised-v1.1`); (b) `extract` asserts, per answer,
the uid/row_id mapping, spans equal in value to the frozen
`BENCH/scores/<uid>.npz` step_starts/step_ends, token count equal to
`len(token_entropies)` and to the top-K logprob rows, alignment with
`token_spilled_energies` (same length, selected surprisal finite), spans inside
the answer, and PB gate detector equality (`verify_answer`); (c) the cached
info stores per-answer provenance (source path + sha256, row_id, spans sha256,
token/step counts, entropy mean); (d) a checkpoint whose stored manifest
differs is refused with the differing keys named (`manifest_differences`).

Existing completed run: `CACHE_VERIFICATION.json` re-derives, for every
cached answer, the spans from the frozen BENCH npz and compares them with the
cached spans (values), and checks uid, step counts against JOINED offsets and
token counts against the benchmark record — **PASS, 13,769 / 13,769, 0
mismatches** (367 s). The current raw-source hashes are recorded with the
explicit statement that extraction-time source state was not recorded by the
original run and these hashes are not proof of it.

## Tests (`scripts/test_varentropy_expansion_supervised.py`, 11 passing)

All-failed run cannot PASS; unexpected exception cannot PASS; STALLED cannot
PASS; expected-limitation-only run is INCONCLUSIVE; iteration-limit fit is
classified FIT_ITERATION_LIMIT and never counted converged; changed boundary
with the same step count is detected; token/logprob/surprisal misalignment,
row mapping and gate mismatch are detected; changed raw-source hash is
detected and named by the driver's `connect` while the stored manifest stays
untouched; calibration plan / blind check; held-fold label flip leaves theta,
scores and q bit-identical while a training-fold flip changes theta; quantile
convention mirrors `evaluate_arrays`.
