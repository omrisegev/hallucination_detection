# Advisor v11 completion package

Open `advisor_review_v11.html` for the main presentation and its expandable
sixteen-row Claude assignment table. Every row records its previous state,
executed action, finding, remaining limit and evidence. `TASK_STATUS.csv` and
`TASK_STATUS.json` preserve the same inventory in machine-readable form.
`FINDINGS_HE.md` is a short Hebrew research handoff.

## Scope and state

All requested CPU analyses were executed. One deliberately unsupported
experimental arm remains failed rather than replaced: N11 L-SML fitted on the
LR half-mask produces only two groups in outer fold 4. Its complete-population
score is undefined. The failure does not affect the completed full-bank N16
or N56 analyses.

This package is **local, uncommitted and unpublished**. No Git commit/push or
Claude Artifact replacement occurred during this implementation. Claude
Artifact reachability has not been verified. The public GitHub repository
does not require private membership. `PUBLICATION_STATUS.json` records this
boundary; task 6 retains the unverified external-access item explicitly.

No GPU inference, remote cluster execution or package installation occurred.
The proposed feature extraction from a PRM backbone is excluded.

## Sources, identity and populations

- Frozen v10 source: `claude/advisor-page-v6-official-prmscore`, commit
  `53ef62e36e80b90de30b83d3fee33e1fb85b82c3`.
- Frozen research source: commit
  `080877eb9a8369341d647f4115a3b475c040b46e`.
- Claude assignments: `sources/v10/docs/meetings/advisor_meeting_2026-10-08/`
  and `sources/v10/results/advisor_page_v6_2026-10-11/CODEX_NOTES_2026-10-11.md`.
- `INPUT_MANIFEST.json` records inspected Drive paths, modification dates,
  sizes and SHA256 hashes. Acquired input files also have receipts. The source
  code archive retains immutable Git-object verification and source manifests.
- `PACKAGE_MANIFEST.json` hashes the delivered inputs, outputs, scripts and
  exact local runtime helpers, including **FOLDS_V2.json** and all copied
  prerequisite inputs. Raw telemetry is **1,117,902,946 bytes**, SHA256
  `684ce4c2a72d38e59a8841c8d0c69076f69a5adf707d5ed997b7c48b4bfba30f`.
  It is stored at `inputs/prmbench_telemetry.pkl`, excluded from Git, with its
  Drive origin recorded in the input manifest. Other large raw inputs are
  similarly excluded; see `.gitignore`.
- `POPULATION_GATE.json` verifies 6,969 answers / 94,203 steps / 707 canonical
  source groups. Noncontrols: 6,211 answers / 83,371 steps. Synthetic controls:
  758 answers. AUC: 6,030 eligible answers; top1-on-error: 6,035 error-containing
  answers. Annotation membership preserves the original out-of-range cases.
- All new splits are **internal validation on previously inspected data**.
  Entire canonical source groups are held together. Labels evaluate the
  unsupervised methods; they do not repair their signs. LR and LR-selected
  masks are explicitly supervised diagnostics.

## Protocol and metric conventions

`protocols/PROTOCOL.json` was saved before the new numerical runs.
`IMPLEMENTATION_CLARIFICATIONS.json` records operational clarifications
documented afterward and does not retroactively claim their preregistration.
The exact historical Q80 chronology and its limits are in
`report/Q80_PROVENANCE.json`.

Ranking means within-answer AUC, averaged across eligible answers. We never
use global AUC over concatenated OOF steps. PRMScore means the repository's
pooled macro-F1 on noncontrols. Overall is not the average of nine type scores.
MS uses valid-step F1 only. Sensitivity is the mean of PS, DR and MS. PRM uses
its native reward threshold: below 0.5 is error. L-SML and IU-PCR use historical
Q80 in the headline comparison; the rule names are exactly ANSWER_Z_Q80,
R2_COUNT and ENTROPY_GATE_Q80. TOP1/2/3 and label-oracle ceilings are diagnostics.

Intervals are 5,000 paired whole-source-group percentile bootstrap draws,
seed 20261011. They are pointwise and conditional on fitted predictions;
they do not correct historical selection or multiple comparisons. Coarse
within-answer nuisance matching changes coverage and pair weights. It does
not remove every possible nuisance, and repetition is partly the NR construct.

## Contents

- `inputs/`, `sources/`: raw data, prepared banks, canonical groups and fold
  maps, archived code, protocol, paper and leaderboard evidence.
- `controls/`: all-bank AUC/top1/PRMScore; prefix/mask decomposition; nuisance
  matched pairs and coverage; exact decision flags/counts; budget curves,
  oracles and actual NR/NCL text examples.
- `lr/`: all 50 outer bank partitions and 1,069 saved fitted models, nested
  masks/C losses, OOF scores, weights, four-way outcomes, omission probes,
  intervals, scientific figures and identity-checked reproduction receipts.
- `statistics/`: all 56 corrected channels by error type, conditional
  correlations, DS/MoM estimates, votes and posterior-count predictions.
- `validation/`: half selection/test and leave-one-error-type-out scores,
  source-group identities and checkpoint manifests.
- `report/`: one integrated deterministic HTML builder; original radar SVG
  and nine concept illustrations; source audits and historical evidence.
- `review/`: independent numerical, telemetry, nested-selection and final
  presentation checks. These are separate from the implementation team's
  arithmetic and structural validation.
- `logs/`: acquisition, population preparation, numerical-run and replay logs.

Scientific figures are saved as SVG and PNG. The HTML embeds its figures and
needs no network or JavaScript library. Relative raw-evidence links require
keeping the package together.

## Reproduce without installing anything

Use the existing repository Python 3.11 virtual environment. From the repo
root:

```sh
.venv/bin/python results/advisor_completion_2026-10-11/reproduce.py --verify
.venv/bin/python results/advisor_completion_2026-10-11/reproduce.py --report
.venv/bin/python results/advisor_completion_2026-10-11/reproduce.py --all
```

`--verify` verifies every manifest-listed input/source/runtime identity,
the population gate, completed numerical statuses and required evidence
links. Every mode rejects changed manifest-listed outputs, checkpoints, inputs and
runtime identities before proceeding. `--report` rebuilds the HTML from saved results
and checks its structure. `--all` also replays controls, LR checkpoint
aggregation/reconstruction, channel/estimator analysis and internal
validation. It uses the frozen prepared inputs; it does not regenerate or
silently change them. Changed identities are an error, not a resume signal.
The wrapper logs each executed command and exits on the first failure.

For first-time input acquisition the source scripts are retained, but normal
reproduction uses the archived inputs and source snapshot and requires no
temporary checkout or network fetch. Do not run `prepare_population.py` over
existing LR checkpoints: even value-equivalent NPZ rewriting can alter file
hashes. A fresh preparation belongs in a fresh output package.

The original Git working tree's unrelated changes were preserved. New helper
files are `spectral_utils/advisor_controls.py`, `advisor_lr.py` and
`advisor_statistics.py`, with driver `scripts/run_advisor_v11_lr.py`.
