# E1b self-review against PROTOCOL.json (before the real run)

The brief asked for an independent reviewer subagent; this worker cannot spawn one, so this is a self-review with
unit checks. Items checked:

| Spec item | Where | Status |
|---|---|---|
| Raw features before answer-z, parity with stored channels | `votes_e1b.raw_matrix` (assert < 1e-4) | implemented; a mismatch stops the cell |
| Pooled z, cubic position residualization, PRM included | `votes_e1b.clean` | implemented; same (j+.5)/n and pool as the corrected E1 |
| Stored orientation, vote = 1 above tau | `_votes`, `fuse_taus` | implemented |
| Quantile grid 0.50..0.98 step 0.02, coordinate descent from the median, <= 5 sweeps | `GRID`, `fuse_taus` | implemented |
| S-hat = Eq. 4 (paper index j1 < j2 < j3), clipped denominators | `shat` | implemented; deviation from the relay's all-pairs wording recorded in PROTOCOL |
| Incremental moment update equals full recomputation | `test_shat.py` | PASS (final S-hat equals recomputed to 1e-9; trace non-increasing) |
| S-hat discriminates TCI from shared nuisance | `test_shat.py` | PASS (0.0011 vs 0.072) |
| AUTO_NATIVE: PRM native flag reward < 0.5, fixed | `fuse_votes` (risk > 0 <=> reward < 0.5) | implemented, not cleaned or tuned |
| No labels before the diagnostics | `votes_e1b.py` reads only raw features, ids, offsets, PRM arm; `run_e1b.run_cell` reads labels after the fit | checked |
| Gate reuse | `D.group_votes`, `D.fit_hds`, `D.bin_design` from `ds_gate_v1b_fixed` (read-only import) | checked |
| Decision rules, metrics, verdict | `run_e1.metric`, `run_e1.pb_f1`, `run_e1.verdict` (read-only import) | checked |
| Missing raw banks | PRM-backbone PRMBench (AUTO, AUTO_NATIVE): FileNotFoundError -> reported, invalid | as declared |

No blocker found.
