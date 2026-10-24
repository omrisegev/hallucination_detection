# GAP x AUTO x rank-one SML fit, PRMBench ablation (2026-10-24, Omri)

**Question (Omri):** with (1) the GAP flag rule, (2) AUTO placement (L-SML groups the PRM with the channels itself),
and (3) the off-diagonal rank-one SML fit (`fusion_utils.rank1_offdiag_fit`, v >= 0, "nnr1") instead of the eigenvector
of the zero-diagonal covariance, what PRMScore do we get, and what does each change add?

**Status:** code ready, smoke-tested on synthetic data (5 s). Not yet run on the real exports.

**Run** (CPU, minutes; needs the stage-1 exports of `../prm_boost_collection_v1`, local `hd_collection_data/` or AIRCC
`$R/prm_boost_gen_v1/ablation5_export_v1` laid out as `<dir>/prmbench/ABL5_{SCORES,CHANNELS}_<prm>.npz`):

    COLLECTION_DATA=<dir> python results/prm_boost_auto_gap_nnr1_v1/run.py

Outputs: `RESULTS.csv` (arm x rule: PRMScore, within-AUC, flag rate), `CONTRASTS.csv` (paired source-group bootstrap,
2,000 draws, ablation steps 1-4 plus every arm against the PRM), `DIAG.json` (AUTO groups, PRM weight share, parity).

**Parity gate:** `CHAIN_<bank>_signed` must reproduce the exported `score__CHAIN_<bank>` (max |diff| < 1e-9).
Metrics, rules (Q80, GAP, K2) and the bootstrap are imported unchanged from `evaluate_collection.py`.
PRM reference = the scale chosen under GAP (as `STATUS_MAIN.csv`): probability for the Qwen and PURE PRMs, logit for RLHFlow and Math-Shepherd.

**Caveat known in advance:** on the corrected template the rank-one fit lost to shipped L-SML on 7 of 10 telemetry-only
banks (`../sml_weight_estimator_v1/README.md` section 3), and on Socratic AUTO+nnr1 gave the PRM 0-4% of the weight.
