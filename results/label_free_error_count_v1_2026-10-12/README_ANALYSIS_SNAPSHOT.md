# Label-free error-count study — 12 October 2026

Open [report/index.html](report/index.html) for the final Hebrew report and
[FINDINGS_HE.md](FINDINGS_HE.md) for the short research handoff.

**No registered candidate passes the combined success criterion.** The complete
A–D family was evaluated with fixed IU-PCR rankings, matched training-only
controls and L-SML as a secondary ranking. This is internal validation on
previously inspected development data, not untouched external confirmation.
The package is local; no GPU, cluster job, installation, commit, push or
publication belongs to this run.

## Population and scientific contract

- Frozen prerequisite: `results/advisor_completion_2026-10-11/`, including its
  original raw/prepared inputs, archived research code, canonical groups/folds
  and unchanged runtime helpers. It remains a separate frozen evidence package.
- 6,969 answers / 94,203 steps / 707 canonical source groups. Overall and primary
  count MAE use 6,211 noncontrols / 83,371 steps. Report 758 synthetic clean
  controls separately. There are 934 true-zero answers, including 176
  noncontrols; 6,035 answers contain an in-range error. The 160 MS answers have
  no erroneous steps under the original membership convention.
- True count is the sum of valid-step annotation membership. Original
  out-of-range annotations are preserved rather than repaired or counted as
  additional steps. ProcessBench first-error labels are not used as total-count
  truth.
- Every normalization, clustering, group weight, threshold and estimator uses
  donor source groups only. Each bank/fold has one immutable IU-PCR ranking
  and one secondary L-SML ranking. Candidate counts are shared across rankings;
  named Q80/entropy controls retain their method-specific counts.
- PRMScore pools noncontrol confusion counts, then averages valid/error F1.
  Overall is not the mean of nine types. MS uses valid-step F1 alone;
  Sensitivity averages PS, DR and MS. Clean-control PRMScore is undefined.
- Seven registered candidates remain in the multiplicity family, including
  failed arms. Success requires complete, promotable N16 fits and complete
  comparisons: MAE improvement over Q80 and LEGACY_MOM, PRMScore improvement
  over Q80, and a bound excluding recall losses greater than one percentage
  point. Each required one-sided bound uses `alpha=0.05/7`.
- Other intervals are pointwise paired 95% source-group bootstrap intervals,
  using 20,000 shared draws and seed 20261012. They condition on saved OOF
  predictions and do not cover the whole learning/search procedure. N16 is a
  historical reference; N56 is a predefined sensitivity check.

## Contents and failure history

- `PROTOCOL.json`, `AMENDMENTS.json`: registered choices and chronological
  amendments. `INITIAL_GIT_STATE.json` records the preserved checkout state.
- `INPUT_FREEZE.json`, `CODE_FREEZE.json`, `PREDICTION_SEAL.json`: input/runtime,
  implementation and prediction identities. Evaluation authenticates the
  sealed input freeze and the label-source bytes before loading labels.
- `checkpoints/`, `predictions/`: ten donor/held-out identities, fitted models,
  vote arrays, count PMFs, posteriors and counts. Final NPZ banks contain shared
  score/rank arrays and method-specific controls. Failed counts stay `-1`;
  failed posteriors stay missing.
- `evaluation/`: per-answer data, flags plus coverage, metrics, paired contrasts,
  nine types, fixed length bins, partial Spearman controlling rank(length),
  ranking checks, folds and saved bootstrap multiplicities. `SUCCESS.json`
  preserves the protected primary decision.
- `diagnostics/`: quota/length/permutation and independent-voter synthetic
  mechanisms; emission/posterior calibration and conditional dependence;
  count/ranking component oracles and all-zero control; median/mean and
  zero-mixture component contrasts, including absolute A versus threshold-only
  LOCAL on their partial common coverage; grouping ARI/weight PR and CPU timings.
  These do not create additional protected candidates or select thresholds
  from utility. Label-based calibration/oracles are evaluation diagnostics.
- `attempts/attempt01_float64_bridge_failure/`: the first run failed the
  inverse coordinate reconstruction audit in all ten folds before new count
  fitting/evaluation. A002 repairs cancellation with compensated float64
  high-plus-residual arithmetic, retaining the `1e-9` tolerance. This numerical
  attempt is separate from the final scientific fit failures.
- Final N16 A fails in four folds because raw emissions are invalid. N56 LOCAL
  also lacks complete coverage. Their available points represent 1,269/6,211
  noncontrols and cannot be compared to complete-population points as if
  denominators matched. Strict fitting never silently clips or replaces them.
- `review/` contains independent mathematical, bridge, sealed arithmetic,
  quota, calibration and oracle checks; `logs/` preserves executions and
  failures. `sources/runtime/` archives the actual source files.
- `report/` contains the final standalone HTML, PNG/SVG figures and receipts.
  `PACKAGE_MANIFEST.json` records delivered files and runtime hashes. This
  manifest is distinct from the earlier pre-label prediction seal.

## Verify the existing delivery

Run from `/Users/osegev/Desktop/hallucination_detection` with the existing
Python 3.11 environment; install nothing.

```sh
.venv/bin/python scripts/verify_label_free_error_count_delivery.py
.venv/bin/python scripts/run_label_free_error_count_v1.py --stage verify
.venv/bin/python -m unittest discover -s tests -p 'test_error_count*.py'
```

The delivery verifier reads hashes without writing. The driver's `verify`
stage checks the full frozen advisor input inventory and records its receipt.
Tests use synthetic fixtures, including sealed evaluation/report fixtures.
Do not run `predict`, `all` or quota fitting over the sealed reference outputs.

## Recompute derived outputs from saved predictions

These entry points write evaluation/diagnostic/report outputs, so use a scratch
copy of the delivery rather than overwriting the reference. Keep the scratch
copy inside repository `results/`; the frozen advisor prerequisite stays at
its original path.

```sh
export COUNT_REPLAY_ROOT=/Users/osegev/Desktop/hallucination_detection/results/label_free_error_count_v1_saved_replay
.venv/bin/python - <<'PY'
import os, shutil
from pathlib import Path
source = Path('results/label_free_error_count_v1_2026-10-12')
shutil.copytree(source, Path(os.environ['COUNT_REPLAY_ROOT']))
PY
.venv/bin/python scripts/evaluate_label_free_error_count_v1.py --package "$COUNT_REPLAY_ROOT"
.venv/bin/python results/label_free_error_count_v1_2026-10-12/diagnostics/evaluate_emission_calibration.py --package "$COUNT_REPLAY_ROOT"
.venv/bin/python results/label_free_error_count_v1_2026-10-12/diagnostics/evaluate_count_oracles.py --package "$COUNT_REPLAY_ROOT"
.venv/bin/python results/label_free_error_count_v1_2026-10-12/diagnostics/evaluate_component_contrasts.py --package "$COUNT_REPLAY_ROOT"
```

The calibration and oracle scripts also support `--self-test`. The main
evaluator's `--report-only` rebuilds its base report without recomputing metrics.
The final integrated report uses the separate renderer below, including the
sealed quota diagnostic evaluation; it performs no fitting.

```sh
.venv/bin/python - <<'PY'
import importlib.util, os
from pathlib import Path
spec = importlib.util.spec_from_file_location('count_report_replay', 'scripts/report_label_free_error_count_v1.py')
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)
builder.OUT = Path(os.environ['COUNT_REPLAY_ROOT'])
builder.main()
PY
```

Original renderer entry point: `.venv/bin/python scripts/report_label_free_error_count_v1.py`.
Its archived source is `sources/runtime/scripts/report_label_free_error_count_v1.py`.
The renderer currently has no `--package` CLI flag; the scratch override above
keeps the delivered reference intact. Recomputed receipts/timings can differ
from delivery bytes; do not refresh the original manifest to conceal changes.

## Fresh computation of the registered family

This rebuilds historical controls, all ten donor fits, predictions, seals,
primary evaluation and the base HTML into a **new, nonexistent** output root.
It is computational replay of the existing specification, not a new external
test or a newly preregistered search.

```sh
.venv/bin/python scripts/reproduce_label_free_error_count_v1.py \
  --destination /Users/osegev/Desktop/hallucination_detection/results/label_free_error_count_v1_fresh_rebuild
```

`--predictions-only` stops after fitting, sealing and verification. The wrapper
refuses any existing destination and copies the original amendment lineage;
it does not overwrite/resume the sealed delivery. Preserve the historical
advisor package and original failed-attempt archive as prerequisites.

The supplementary quota investigation is separate. Its exact CLI stages are:

```sh
.venv/bin/python results/label_free_error_count_v1_2026-10-12/diagnostics/run_quota_mechanism.py --stage prepare
.venv/bin/python results/label_free_error_count_v1_2026-10-12/diagnostics/run_quota_mechanism.py --stage synthetic
.venv/bin/python results/label_free_error_count_v1_2026-10-12/diagnostics/run_quota_mechanism.py --stage fit
```

These document the stage interface, not permission to overwrite the sealed
quota delivery. For a fresh supplementary rebuild, import the unchanged
diagnostic module and redirect `PACKAGE`, `DEST` and `ATTEMPT` to a separate
new output tree with the authenticated first-attempt donor states; retain its
`ROOT`/`FROZEN` paths. Run `prepare()`, `exact_teacher()`, then `fit()`. The
primary fresh-replay wrapper intentionally does not automate this historical
diagnostic lineage or reconstruct the original numerical failure. The existing
quota seal authenticates its saved outputs; `QUOTA_SCIENCE.json` records the
bounded 590-row experiment and its caveats.

## Interpretation

MoM/Bayes requires a chosen latent orientation and conditional independence
of voters; neither small correlations nor bootstrap-stable triples establish
that assumption. B uses one training-selected triple, not an ensemble of
triples. Median counts additionally assume conditionally independent steps
and fixed fitted parameters. Summing posterior expectations does not require
step independence, but still depends on posterior validity. D adds a clean
answer state without guaranteeing that the latent state equals semantic error.

The tested quota construction affects prevalence estimates, but is not an
algebraic prevalence identity or a complete explanation. The simple absolute
zero threshold also fails to provide a useful count estimator. Both allocation
and ranking retain measurable limitations. This closes the specified bounded
experiment family without claiming that all error-count estimation is impossible.
