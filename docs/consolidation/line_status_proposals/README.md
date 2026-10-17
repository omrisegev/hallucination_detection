# Line status proposals (consolidation, 2026-10-01 / 10-02)

> **PROPOSAL by Claude, not an Omri decision.** Omri's decision (2026-10-01): write proposed statuses explicitly as proposals, with supporting evidence and reopening conditions; this is not blanket approval to close these directions; distinguish 'this implementation failed' from 'the research direction is closed'. The only explicit closure so far is Joint L-SML (discontinued 2026-10-01).

These 14 files cover lines that have no owner session and no owner-written status file. Each file has the
same parts: proposed status, steps on the branch, evidence (`path @ ref`), what failed (the implementation)
against what is still open (the direction), reopening conditions, dependencies, and material outside git.

The status words describe the tested implementations and the branch, not the research direction:

- **SATURATED**: the implementations tested on current inputs gave no measured gain.
- **IMPLEMENTATION-NEGATIVE**: this implementation lost to its matched reference.
- **SUPERSEDED**: a newer line carries the branch's content.
- **TOOL-ONLY**: kept as code or data that other lines use.
- **OUT OF SCOPE FOR NOW** and **DEFERRED**: Omri's words (2026-10-02). Not closed, and not a next step.

None of these words closes a research direction.

Omri's decisions applied in these files (source: `docs/consolidation/DECISIONS_2026-10-01.md`):

- Joint L-SML is discontinued for new arms (item 2). This is the only explicit closure.
- The whitebox answer-gate stage and the whitebox step-length follow-up are out of scope for now, not closed (item 14).
- The V2 and V3 family-tail audits are approved, as verification of existing work (item 16).
- The transfer proxy is approved as one bounded exploratory test. Its protocol is written before it runs (item 17).
- "Why bank11 transfers" and combining the supervised PRM with the label-free method are deferred, not closed (item 18).

## Files

| File | Proposed status (one line) |
|---|---|
| [claude__depth-feature-fusion-v1.md](claude__depth-feature-fusion-v1.md) | Depth summaries added to the step bank: SATURATED for this implementation. The gate-free engine and the bank11 baseline: TOOL-ONLY. |
| [claude__er-generality-v1.md](claude__er-generality-v1.md) | SUPERSEDED by `claude/ssl-pseudolabel-residual-v1` (Steps 456-457). |
| [claude__estimator-provenance-collection-2026-09-29.md](claude__estimator-provenance-collection-2026-09-29.md) | Third-moment estimator on top-20% marks: IMPLEMENTATION-NEGATIVE, direction open. Provenance note and collection map: TOOL-ONLY. |
| [claude__readout-quickest-detection-v1.md](claude__readout-quickest-detection-v1.md) | Readout and decision-rule levers on bank11 (Steps 428-432): SATURATED. Branch: SUPERSEDED by `lsml-ct7-levers-run`. Frozen step files: TOOL-ONLY. |
| [claude__token-axis-fusion-sampling-3i9r2u.md](claude__token-axis-fusion-sampling-3i9r2u.md) | Branch: SUPERSEDED. Cumulative-vote fusion of first-error localizers: SATURATED for the tested implementations. |
| [claude__token-probability-fusion-v1.md](claude__token-probability-fusion-v1.md) | Token matrices and channel files: TOOL-ONLY. Step readouts: SATURATED. Token-level L-SML: IMPLEMENTATION-NEGATIVE, direction open. |
| [claude__whitebox-layer-views-v1.md](claude__whitebox-layer-views-v1.md) | Depth as step locator: SATURATED for the tested reduction. Answer-gate stage and step-length follow-up: OUT OF SCOPE FOR NOW (not closed). Extraction: TOOL-ONLY. |
| [codex__claude-feature-bank-token-lsml-v1.md](codex__claude-feature-bank-token-lsml-v1.md) | Branch: SUPERSEDED. Token-level L-SML as implemented here: IMPLEMENTATION-NEGATIVE, direction open. Bank11 module: TOOL-ONLY. |
| [codex__cumulative-vote-fusion-v2.md](codex__cumulative-vote-fusion-v2.md) | Cumulative-vote fusion: SATURATED. The `cvf_v2` estimation package and frozen CT7 profiles: TOOL-ONLY. |
| [codex__digit-alternative-probability-v1.md](codex__digit-alternative-probability-v1.md) | Branch: SUPERSEDED. The three digit features: TOOL-ONLY inputs, kept by Omri's 2026-09-30 decision. |
| [codex__family15-tail20-transfer-v1.md](codex__family15-tail20-transfer-v1.md) | SUPERSEDED by `codex/lsml-external-generalization-v1`. The tested family/tail recipe: SATURATED. |
| [codex__lsml-external-generalization-v1.md](codex__lsml-external-generalization-v1.md) | Branch: SUPERSEDED. Harness and frozen bank11 bundle: TOOL-ONLY. Transfer proxy: approved bounded test. Why bank11 transfers: DEFERRED. V2/V3 audits: approved. |
| [codex__lsml-group-confidence-v1.md](codex__lsml-group-confidence-v1.md) | IMPLEMENTATION-NEGATIVE, direction open (no branch of its own; committed on the rescue branch). |
| [codex__prmbench-runtime-fusion-v1.md](codex__prmbench-runtime-fusion-v1.md) | Branch: SUPERSEDED by `lsml-ct7-levers-run`. Supervised PRM plus label-free combination: DEFERRED, not closed. |

The eight historical refs named by the consolidation review are covered separately, in
[../HISTORICAL_INDEX_2026-10-01.md](../HISTORICAL_INDEX_2026-10-01.md).
