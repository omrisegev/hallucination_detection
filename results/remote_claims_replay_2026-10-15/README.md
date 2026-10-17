# Independent PRM audit: preserved results and safe replay

The CSV/JSON results and the six figures are the original October 15 development
audit. They do not establish independent verification of the other four PRMs or
the current external L-SML/IU-PCR rows; see `COVERAGE.json`.

The October 17 integration correction changes only replay paths and output
handling. The exact original script is preserved at
`docs/reviews/integration_completion_20261017/sources/remote_claims_replay_20261015_original.py`.
The original `REPLAY_MANIFEST.json` remains unchanged and describes that original
script. All four recomputed CSV tables match the original tables exactly; see
`docs/reviews/integration_completion_20261017/REPLAY_PARITY.json`.

## Replay without overwriting the original evidence

Use the main checkout's virtual environment. Choose a new destination outside
this preserved package:

```sh
/Users/osegev/Desktop/hallucination_detection/.venv/bin/python \
  results/remote_claims_replay_2026-10-15/replay.py \
  --data-root /Users/osegev/Desktop/hallucination_detection \
  --inputs /Users/osegev/Desktop/hallucination_detection/results/remote_claims_replay_2026-10-15/inputs \
  --destination /tmp/prm-replay-new
```

An existing destination is refused. Every required data file is checked for
existence before any output is created. `--reported-csv` defaults to the tracked
v19 source table, not a temporary checkout.

`DRIVE_INPUTS.json` identifies the original Drive exports. A fresh clone needs
the `prm_boosting_v1`, `external_lr_prm_v1`, `monday_external_panel_v1` and
`monday_cont_baselines_v1` archive contents under `inputs/unpacked/results/`.
Gold JSON files are tracked here. The matching frozen advisor `population.npz`
and `prmbench_meta.json` are also required under the selected data root.
The integration archive index records cloud locations for the advisor payloads.

The replay still evaluates previously inspected development data with the
original definitions. It does not run token extraction, model inference or a
new feature-selection experiment.
