# HANDOFF: Simes answer gate (2026-11-15)

Status: DONE. Nothing running.

Reproduce (about 1 min CPU):
1. Data: hd_collection_data with prmbench_standalone, c4u_r1_inputs/auto_raw_plus_z_bundle, prm_standalone_banks,
   sub5_all_v1, c4u_r1_inputs/crop_ares_all/full_sa_*_processbench, crop_ares_v1/{scores,channels,gold} (sa processbench, 5 incl. math_shepherd), h2v, socratic.
   sub5_all.py reads <repo parent>/hd_collection_data, so put (or link) the data folder there.
2. `COLLECTION_DATA=<hd_collection_data> python -B run.py` then `python -B figures.py`.
   Needs results/cross_model_gate_check_v1/steps (in this branch).

Decisions: alpha = .05 fixed before the run (from the coordinator's description to Omri). Sweep is a diagnostic only.

Open (needs Omri): a fixed Simes alpha does not calibrate the gate (it opens about alpha of the answers).
Choosing the opening share needs a prevalence estimate, which is the same open problem as before.
