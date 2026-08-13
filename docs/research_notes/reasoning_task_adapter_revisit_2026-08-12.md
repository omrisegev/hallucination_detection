# Reasoning localization should preserve the full token trajectory

**Date:** 2026-08-12

**Status:** Deferred design decision recorded during the benchmark review. Keep
the existing PRMBench and ProcessBench results as baselines, but do not treat
their current task adapters as the final unified reasoning method.

## Core observation

U-PCR's distinctive information is strongest on long model-output traces, where
entropy, energy, CUSUM, spectral, and other trajectory features can expose
changes and harmonics. A reasoning solution is naturally one long ordered token
stream whose step or claim boundaries are known. Reducing every short step to a
small independent vector before fusion discards much of that advantage.

The current PRMBench adapter does compute its five token views over the full
trace, but then reduces each view to a mean and maximum within every step and
fits the fusion at the step-vector level. The resulting graph connects steps
with similar ten-dimensional summaries; it does not encode their order or
logical dependence.

## Proposed unified reasoning representation

For solution \(i\), retain the full token trajectory:

\[
X_{i,t,f},
\]

where \(t\) indexes tokens across all concatenated reasoning steps and \(f\)
indexes token-resolved counterparts of the original feature pool. Step
boundaries are metadata, not points at which the feature computation restarts.

The preferred order of operations is:

1. compute all valid positional feature streams over the complete response;
2. fuse the feature streams into one continuous token-risk stream
   \(r_{i,t}\);
3. detect persistent changes, peaks, or state transitions in that fused stream;
4. map the detected token region back to the reasoning step or claim containing
   it;
5. derive task outputs from the same frozen stream:
   - every-step risk for PRMBench;
   - first erroneous step or no-error for ProcessBench;
   - answer-level risk by aggregating the same trajectory.

For a step with token set \(T_{i,k}\), a simple declared adapter is:

\[
s_{i,k}=\operatorname{aggregate}_{t\in T_{i,k}} r_{i,t}.
\]

However, the primary object remains the continuous trajectory. Step-level
aggregation should not replace token-level fusion.

## Relation to prior work in this repository

This proposal is close to the historical positional-view localizer and the
frozen GL-LIU ProcessBench system: positional series were computed at native
trace scale and then mapped to steps. It is not equivalent to the current
PRMBench reporting adapter, which summarizes each feature within each step
before fusion. The unified experiment should make this distinction explicit
and compare the two orders of operation:

\[
\text{fuse tokens, then aggregate to steps}
\quad\text{versus}\quad
\text{aggregate features to steps, then fuse}.
\]

## Design requirements for the future experiment

- Use one frozen feature contract for both PRMBench and ProcessBench wherever
  telemetry permits.
- Preserve exact step boundaries and the original token order.
- Do not recompute long-trace features independently on short steps.
- Separate a global answer-error detector from the conditional error locator
  only when diagnostics show that the two-head decomposition is necessary.
- Compare temporal/state-transition structure with an unordered similarity
  graph; do not call the latter a reasoning graph.
- Evaluate whether broad token-resolved counterparts of mixed-v2 add stable
  information beyond the exact five historical positional views.
- Freeze the token-to-step aggregation and no-error rule without test-label
  tuning.
- Report failure modes by step length, error position, number of steps, and
  error type.

## Current interpretation

The weak PRMBench result does not reject trajectory-based U-PCR. It rejects the
claim that an unordered fusion of ten per-step summaries is already a strong
reasoning-error model. ProcessBench contains a partial implementation of the
more natural trajectory-first idea and should be interpreted separately.
