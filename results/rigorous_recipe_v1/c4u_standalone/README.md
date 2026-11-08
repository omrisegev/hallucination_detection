# C4.U standalone (recipe C4U-parisi-within-jaffe-cross-v1)

Run: `python run_c4u.py`. It runs the 8 cells (PRMBench R1, Socratic R0 × 4 PRMs) and checks them against
`expected_results.json`. Last run 2026-11-08: ALL PASS, rank bitwise identical in 8/8.

| file | what |
|---|---|
| `c4u_new.py` | the 8 functions that are not on the remote: `Cell`, `build_R1`, `lsml`, `votes`, `lw`, `load`, `posterior` and `c4u`, copied unchanged |
| `run_c4u.py` | runs the recipe and checks it against the expected results |
| `expected_results.json` | the frozen outputs per cell: input/label/flag/rank hashes, PRMScore, within-AUC, K, prevalence |
| `data/` | the 20 data files the run reads, as APFS clones of `hd_collection_data`, with `data/SHA256SUMS.txt` |

**Imported unchanged from files identical on the remote (checked 2026-11-07 against the fetched refs):**
- `prm_boost_collection_v1/evaluate_collection.py`;
- `prm_boost_math_rigor_v1/common.py`, `stage8_auto_forced_k.py` and `stage7_hygiene.py`;
- `spectral_utils/binary_view_estimation.py`;
- `spectral_utils/fusion_utils.py` (differs from the remote only in CRLF line endings);
- the gold/meta JSON files under `results/remote_claims_replay_2026-10-15/inputs/` and
  `results/advisor_completion_2026-10-11/inputs/`.

**Data.** `.npz` files are gitignored in this repo, so `data/` stays local (313 MB). The original source is Drive
(`cluster_results/auto_raw_plus_z_v1/bundle` for the INPUTS files) and the collection data. Hard2Verify is not
included: the frozen 8 cells do not use it.

`frozen_c4u/` is unchanged and remains the record and the drift check.
