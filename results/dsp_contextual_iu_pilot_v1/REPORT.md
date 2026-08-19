# DSP-Contextual IU Router pilot v1

**Verdict: `STOP_NO_ROUTING_SIGNAL`.**

The CPU-only S0 falsification gate failed before any real ProcessBench labels were opened. The implemented router therefore did not proceed to S1--S4.

## Synthetic result

| world | IU AUROC | contextual AUROC | delta | wins | target-active mass | worst delta |
|---|---:|---:|---:|---:|---:|---:|
| informative | 0.8422 | 0.8161 | -0.0261 | 8/20 | 0.589 | -0.1019 |
| null | 0.9656 | 0.9652 | -0.0004 | 5/20 | 0.500 | -0.0023 |
| coherent_nuisance | 0.7394 | 0.6002 | -0.1392 | 0/20 | 0.341 | -0.1650 |

The failure is substantive, not merely low power. In the informative world the router placed 58.9% of its mass on the three target-active families (50% is neutral), yet that weakly correct preference still worsened AUROC. Under coherent nuisance the target-active mass collapsed to 34.1% and the score suffered a large loss. The context-independent null was approximately inert, so the problem is the target alignment and safety of adaptation rather than uncontrolled numerical drift.

Failed gates: `coherent_nuisance_mean_safety`, `coherent_nuisance_tail_safety`, `informative_gain`, `informative_wins`.

## Mechanical audit

- Exact global fallback: `True`.
- Question-duplication score delta: `0.000e+00`.
- Question-duplication weight delta: `0.000e+00`.
- Observational-equivalence identity: `True`.
- Covariance-entry IU is regression-tested against ordinary `upcr_fit`.
- The initial impossible `k=32`/`n_eff>=32` combination was corrected before the intentional run by adding eight neighbour questions of headroom.

## Interpretation

DSP states do contain regime structure, but this implementation cannot turn that structure into target-aligned family reliability.  Improving local manifold estimation with LPCA, LTSREx, or LEGO would make the same local geometry more stable; it would not supply the missing correctness contrast exposed by the coherent-nuisance failure.

This closes the registered DSP-contextual covariance router.  It does not rule out a router fed by an independent target-relevant observation or a verified intervention.  No fresh inference run is requested.

## Scope and audit trail

All evidence is retrospective/synthetic premise evidence.  S1--S4 were skipped by the frozen gate; no real-data score, label, cache, GPU, cluster, or Drive operation occurred.
