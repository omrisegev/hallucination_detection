# Requirement audit — Global/Local/Online architecture v2

This audit maps the requested broader research cycle to the frozen protocol and
completed artifacts. All results are retrospective over existing caches. No new
inference, GPU/cluster work, Google Drive mutation, raw-data mutation, staging,
commit, or push was performed.

## 1. Fusion algorithm

**Requested question:** decide whether IU-PCR remains appropriate or whether a
DUFS/Laplacian alternative earns its extra complexity.

**Completed:** after freezing the selected head matrices, the experiment compared
ordinary IU-PCR with uniform, DUFS, and temporal graph variants on the exact same
matrices and reducers. Every path passed exact `lambda=0` identity. Global DUFS
changed AUROC by +0.0014 and Local graph paths changed F1 by about +0.0059; the
paired intervals include zero. Measured DUFS fitting was up to 24.9x slower than
the uniform path.

**Decision:** retain ordinary IU-PCR. Do not promote DUFS or a Laplacian.

**Evidence:** `FUSION_AGGREGATE.csv`, `GROUPED_FUSION_INTERVALS.csv`,
`EFFICIENCY.csv`, and `FUSION_MANIFEST.json`.

## 2. Harness and architecture

**Requested question:** jointly optimize the number of heads, detector, running
threshold behavior, Global metric, locator, and interactions among outputs.

**Completed:** the frozen harness crossed one shared head, two Global+Local
heads, and three independent heads; detector weights 0, 0.25, 0.50, 0.75, and
1.00; and peak versus persistent-onset localization. Head selection was limited
to the frozen development cells, the architecture was then frozen, and all three
selected architectures were transferred to twelve scorer-model/family cells.
Question-grouped intervals resampled each source question once while carrying
the three scorer copies together.

**Decision:** retain two heads, a 0.50/0.50 calibration-standardized
Global/Local detector and causal Online blend, and the Local peak locator. A
third independent Online head does not improve the three-output frontier. The
historical 0.75/0.25 blend is not retained.

**Evidence:** `HEAD_SELECTION.json`, `ARCHITECTURE_SELECTION.json`,
`ARCHITECTURE_SELECTED_AGGREGATE.csv`, `GROUPED_ARCHITECTURE_INTERVALS.csv`,
`DECLARATION_METRICS.csv`, and `ONLINE_BUDGETS.csv`.

## 3. Head-specific feature engineering

**Requested question:** build features from raw token telemetry for each output
and audit mathematical and streaming requirements, including monotonicity.

**Completed:** nine target-blind, risk-oriented token channels were derived from
entropy, spilled energy, log-sum-exp, and top-k log-probabilities. Global compared
mean/tail/extreme reducers with the registered mixed-v2 head. Local compared raw
levels, onsets, their union, and the registered core replay. Online compared
level/EWMA, onset, positive-area/persistence, and registered IU28 states. New
causal recurrences use O(1) state and pass suffix-replacement and tokenwise versus
chunk replay tests. Orientation and monotone operators were checked explicitly;
the onset transform is documented as non-monotone rather than being mislabeled.

**Decision:** Global retains the registered mixed-v2 features; Local selects the
nine raw level channels; the independently best Online sustained-state head is
not needed once the two-head architecture is selected. Drop-one diagnostics are
exploratory and do not authorize post-hoc pruning.

**Evidence:** `CANDIDATE_LEDGER.csv`, `MISSING_CHANNEL_SENSITIVITY.csv`,
`SELECTED_HEAD_DIAGNOSTICS.json`, `AUDIT.json`, and
`scripts/test_multitask_trajectory.py`.

## 4. Three co-primary outputs

- **Global final-answer error:** selected two-head macro AUROC 0.7164 across the
  twelve transfer cells; +0.0271 [0.0085, 0.0449] versus one shared head.
- **Local first-error localization:** macro F1 0.3136; +0.0740
  [0.0458, 0.1013] versus one shared head.
- **Online early prediction of Global:** 64/128 macro AUROC 0.6075; +0.0067
  [-0.0121, 0.0260] versus one shared head. At a 10% calibration target, the
  trace-level warning policy covers 25.0% of wrong traces with 8.1% observed
  false warnings.

The tasks were never averaged into one headline score. Global final-answer
wrongness and Local trace-error presence remain distinct labels.

## 5. Limitations and next gate

- All evaluated populations were previously opened, so selection requires fresh
  confirmation before a deployment or paper-level generalization claim.
- Phase-15 early transfer is weak at 0.5142/0.5555 AUROC at 64/128 tokens despite
  0.8368 final AUROC.
- The historical mixed-v2 Global prefix replay is the runtime bottleneck and
  should receive an exact or validated incremental implementation.
- Local drop-one results motivate a newly frozen subset roster; no feature is
  removed from the selected head based on these opened outcomes.
- No GPU run is justified for DUFS, a Laplacian, or the third head. A future
  fresh-data run requires an explicitly approved, separately frozen protocol.

The canonical narrative and machine decision are `REPORT.md` and
`DECISION.json`.
