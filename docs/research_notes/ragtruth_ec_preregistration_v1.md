# Evidence-Contrast U-PCR/DUFS-LIU on RAGTruth — preregistration v1

**Date:** 2026-08-09
**Status:** frozen before any test-split scoring. Round 1 data collection.

## Research question

Can the DUFS-gated Laplacian IU-PCR fusion core detect and localize unsupported claims in a
fixed, already-published RAG response, when retrieval evidence is treated as a controlled
intervention rather than as nuisance covariance to remove?

## Corpus

RAGTruth (Niu et al. 2024, ACL), vendored under `data/ragtruth_protocol/`, pinned commit
`c103204b9ce28d6bbad859304bf30de72b8ed8fe`. Full schema, row counts, and chunk-coverage
measurements in `data/ragtruth_protocol/PROVENANCE.md` — read that file before touching this
one; it records the empirical finding that **Summary documents have zero natural
sub-structure** (100% of 900 test-split Summary rows), so the leave-one-chunk-out condition
does not exist for that task type. This is the documented fallback case, not a bug, and it
means the evidence-graph and LOO-derived features have real coverage only on QA and Data2txt
(2/3 of the corpus) this round.

## Scorer

**Qwen2.5-1.5B-Instruct**, off-policy (none of RAGTruth's six response-generating models).
Chosen for protocol comparability with GASP, the closest direct competitor. See
`cluster/manifests/ragtruth_ec_v1.json` for the repetition-penalty handling this choice
requires.

## Split and round-1 scope

- **Primary**: full RAGTruth **test** split (2,700 responses; 16,200 (response, condition)
  items at mean 6.0 conditions/response, per `PROVENANCE.md`).
- **Dev slice**: 150 seeded `source_id`s from the **train** split (`group_split`), scored in a
  separate job, used only for any component selection this campaign needs (not for the frozen
  hyperparameters below, which are carried unchanged from the localization work).
- Round-1 does **not** include RAGTruth++/Enhance, TofuEval, RAGBench, or TRIVIA+. Those are
  confirmation/falsification stages, reopened only after the primary test-split result exists
  (`docs/research_notes/rag_localization_methods_and_benchmarks_2026.md`, "Recommended
  benchmark sequence").

## Evidence conditions and chunking (frozen, from `spectral_utils/ragtruth.py`)

- `full`: the original published prompt, unmodified.
- `noctx`: the evidence substring removed entirely.
- `loo_j`: chunk `j` removed from the evidence substring, everything else (instruction text,
  question, output cue) byte-identical to `full`.
- Chunk unit by task type: **QA** = one of the 3 `"passage N:"` blocks; **Data2txt** = one of
  9 top-level JSON fields (`name, address, city, state, categories, hours, attributes,
  business_stars, review_info`); **Summary** = the whole document (1 chunk always — see
  corpus note above). `distinct_conditions` drops any `loo_j` pass that would duplicate
  `noctx` (currently: every Summary row's `loo_0`).
- All conditions are built by **prompt surgery** on the original published prompt (locate the
  evidence text as an exact substring, edit only that substring) — never by reconstructing
  the surrounding instruction text. Verified as an exact substring for all 2,700/2,700 test
  rows before this document was written.

## Views and fusion (frozen)

- Per-token telemetry per condition: forced-token NLL (`token_spilled_energies`), entropy
  (`token_entropies`), top-50 logprobs, full-vocab `token_logsumexp` — the same four channels
  `candidate_quantities` already produces.
- Evidence-contrast Δ views (response-level and token-level): `dnll_noctx`, `dent_noctx`,
  `dnll_loo_max`, `dnll_loo_mean`, `dent_loo_max`, `dent_loo_mean`, per-token Jensen-Shannon
  divergence between the `full` and each ablated top-50 distribution (renormalized over the
  ID union with a tail bucket), `dlogsumexp_noctx`.
- **Evidence graph** (the new exogenous Laplacian construction): nodes = intervention views
  (`full`, `noctx`, each `loo_j`); edges = TF-IDF word-level cosine similarity between the
  corresponding chunk texts, deterministic and offline. This encodes redundant-evidence
  structure the feature covariance itself cannot see — the class of side information Steps
  228–230 showed static-matrix graphs cannot substitute for.
- Carried hyperparameters (unchanged from the localization work, not re-tuned here):
  `global_lambda=0.1`, `local_lambda=0.3`, `k=7`, DUFS seeds `{11, 23, 37}`, DUFS epochs `80`
  (`results/gl_liu_factorial_v2/RUN_DEFINITION.json`).

## Frozen arm roster (see `cluster/manifests/ragtruth_ec_v1.json` for the full list)

Six primary rows plus two quoted ceilings. Row 6 (EC views + naive average, no U-PCR/DUFS-LIU)
is the **fusion-isolation ablation** — since GASP already does evidence perturbation, the
entire novelty claim of this campaign rests on rows 5/5b beating row 6. This is registered as
a primary comparison, not a secondary table.

## Metrics

- **Response AUROC / AUPRC** (grouped bootstrap by `source_id`).
- **Token/span AUROC**, **span F1** at a fixed IoU threshold against `span_token_spans`.
- Failure tests (from `docs/research_notes/evidence_contrast_upcr_rag_direction.md`, carried
  forward): correct-from-memory-but-unsupported answers, redundant-chunk insensitivity,
  irrelevant-chunk-removal score drift, length/chunk-count confounds, retrieval-vs-generation
  failure conflation, same-source leakage across splits.

## Label boundary

Every Δ view, graph, and fusion score is fit and hashed **before** `response_label` /
`span_labels` are read. Splits and bootstrap intervals are grouped by `source_id`. No
threshold, hyperparameter, or arm selection may use test-split labels; the dev slice (from
train) is the only labeled surface available before the freeze.

## What this preregistration does NOT cover

- RAGTruth++/Enhance, TofuEval, RAGBench, TRIVIA+, L-CiteEval — later stages.
- A faithful GASP reproduction's exact scorer-matching protocol beyond "same scorer class" —
  finalize before `scripts/rag_ec_v1/gasp.py` is scored, and declare its fidelity level per
  the tailor-not-transplant convention (`feedback_tailor_not_transplant`).
- Multi-sample (K>1 generation) extensions — explicitly out of scope for round 1 per Omri's
  2026-08-09 instruction; teacher-forced rescoring under different evidence conditions is not
  repeated generation and was explicitly approved for round 1.
