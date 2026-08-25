# CIW-DEEM multi-application v1

This experiment places the registered CIW-DEEM response method in the
reconstruction benchmark's application lanes without mixing prediction units.

## Executed adapters

`scripts/reconstruction_benchmark/run_ciw_external.py` fits exact CIW-DEEM on
each eligible external completed-response cell. The separate evaluator joins
the frozen opaque row IDs to the benchmark labels.

`scripts/reconstruction_benchmark/run_ciw_localization.py` combines frozen CIW
response ranks with the frozen token-IU29 step ranks. It does not train a new
token model.

`scripts/reconstruction_benchmark/run_ciw_prefix.py` rebuilds the CIW input
causally at each token budget, fits only on label-free calibration prefixes,
and scores held evaluation prefixes.

`scripts/reconstruction_benchmark/run_ciw_ragtruth_response.py` fits exact
CIW-DEEM on RAGTruth's label-free original-30 response features, freezes both
dev and test scores, verifies their hashes, and only then reads official
response labels.

## Result summary

- Registered 24-cell cell-macro/equal-family AUROC:
  `0.7820255514493354 / 0.7492330051057238`.
- RAGTruth test response AUROC/AUPRC: `0.771222 / 0.635797`.
- ProcessBench localization macro-F1: `0.309136`.
- PRMBench step AUROC: `0.581138`.
- Causal-prefix AUROC at 16/32/64/128/256 tokens:
  `0.563896 / 0.562387 / 0.587503 / 0.611073 / 0.646165`.

External response results are heterogeneous: CIW helps on GPQA and HLE but
does not improve ProcessBench, Evidence-Drop, or PRMBench relative to B3.

## Contract boundaries

The exact registered CIW method cannot be computed from EDIS v2's prepared
26-feature matrix because the partition-energy source is absent. RAG
sentence/token/span/claim tasks and white-box hidden-layer tasks have different
units or access contracts. They are listed as explicit compatibility gaps,
not silently filled with a different algorithm. Prefix detection is not a
stopping policy.

Canonical compact artifacts are under
`results/ciw_deem_multi_application_v1/`. Large score freezes remain under
`local_cache/ciw_multi_application_v1/`.
