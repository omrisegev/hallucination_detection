# Graph Geometry Selection Research V1 — final provenance audit

**Status: `PASS`; no leakage or artifact-integrity blocker found.**

This was an independent, read-only replay after external transfer. No frozen development source/artifact or transfer source/artifact was modified. The audit created only this report and its JSON companion.

## Verified chain

- The external isolation manifest is self-consistent (`3478fac1579c…10d76`; file SHA-256 `e8e7a86adcf5…88c60`). All 16 isolated files have the exact panel-specific member registry, all 50 members and file hashes verify, and the roster contains 21,141 samples.
- The transfer fit manifest is self-consistent (`f08f757d5295…ced35`; file SHA-256 `16de11565642…afaaf`). Its 15-source hash closure exactly matches the current files, including the transfer test.
- The development-to-transfer chain matches the frozen development fit (`08f7d41081b1…11652`), label-free selection (`a5068f9787fb…ebb7a`), transfer selection (`3a1a6bfbd49e…13b34`), and canonical selection (`ff0b6e824d01…d1aa3`).
- All 144 external score arrays (16 cells × IU plus eight frozen methods) were independently reconstructed from the isolated features and are exactly equal to the frozen arrays and registered hashes. All 16 feature/family registries and panel-specific row identifiers match.
- The prior canonical external bank reproduces exactly in all 16 cells for IU, canonical full, and canonical cross-only: 48/48 exact arrays. The reproduction artifact SHA-256 is `63e428a01f32…597a2`.

## Outcome boundary and row alignment

The frozen report source calls full fit/isolation/selection/score/canonical verification at line 1203 and first calls the label loader at line 1206. This audit followed the same order and reconstructed every score before opening labels. The fit manifest predates the outcome result. After label access, all 16 label vectors matched the isolated row identifiers; all 144 unique cell-method rows had exactly reproduced AUROC and AUPRC, and their deltas matched within `1e-13`.

The score-fit process was physically outcome-free. The preceding isolation process is narrower: it necessarily unpickled historically opened raw caches, which may materialize records containing outcomes, but indexed only the frozen telemetry, identifier, and non-target eligibility whitelist. Physical target isolation therefore begins at the score-fit NPZ inputs, not at raw-cache ingestion.

## Canonical historical correction

The canonical fit never indexed, decoded, or passed any `__labels` array into graph construction, calibration, or scoring, and its emitted state/score artifacts contain no target fields and verify against their frozen hashes. However, the fit received the full `results/dependency_fusion_raw/cells.npz`, which physically contains 24 `__labels.npy` members, and its provenance SHA read the archive bytes. Therefore `target_fields_received_by_fit=[]` and any unqualified claim that labels or the label-bearing archive were “unopened” are too strong. The historical boundary was logical member whitelisting, not physical input isolation. The present study repairs this by fitting only from the exact-whitelist sanitized archive; the label-bearing source is opened only by the outcome-report process after all candidate hashes verify.

## Retrospective claim boundary

The original eight-family LOFO analysis and all five external domains are retrospective analyses of historically opened outcomes. Outer LOFO is strict only for the new graph and selector stage conditional on the frozen mixed-v2 and confidence-orientation representation, which was previously outcome-informed. External transfer is a frozen stress test, not independent confirmation; confirmation requires a new sealed dataset family and preferably a new model family.

## Outcome artifact manifest

The JSON companion contains a canonical self-hashed manifest for `RESULT.json`, `REPORT.md`, `cell_metrics.csv`, and both transfer plot formats. Its manifest hash is `05478e5e8799…06f35`. Any change to those five files invalidates that manifest.

The audit payload hash is `d11ee4b0734d…e29fd`; the completed JSON file SHA-256 is `bf85c1fc410b…c666b`.
