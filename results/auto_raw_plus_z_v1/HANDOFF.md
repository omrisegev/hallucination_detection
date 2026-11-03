# Handoff: CAT34 (raw + answer-z AUTO), PRMBench (2026-11-03)

## Formulas

- Answer-z (unchanged, Z17): for each answer `a` and channel `j`, `z = (x - mean_a(x)) / sd_a(x)`, and 0 if
  `sd_a <= EPS` (`answer_standardize`). Digit channels use `masked_std` over active steps (0 when inactive).
  PRM view: `z(log((1-r)/r))`, where r is the PRM step reward.
- Raw view (new): `g(x) = Phi^-1((rank(x) - 0.5) / n)`, ranked over all n PRMBench steps of that PRM (scipy
  `rankdata`, average ties; `ndtri`). Digit channels: ranked over active steps, 0 elsewhere. The raw PRM view
  is `g(log((1-r)/r))`.
- Fusion: `spectral_utils.fusion_utils.lsml_continuous(*views)` with defaults (automatic grouping,
  K in 2..8, residual method). Sign: flip if corr(fused, z(PRM)) < 0.
- Q80: `flags = z_a(score) > quantile_0.8(z_a(score))` over all steps (`evaluate_amended.flags_for`).
- Metrics: within-AUC = mean over answers with an error of the step AUC; pooled AUC over all 94,203 steps;
  answer AUC = AUC of max step score vs "answer has an error step"; PRMScore = official `prmbench_evaluate`.
- Bootstrap: 1,000 multinomial reweightings (seed 20261103) of answers, and separately of source groups;
  CI = 2.5/97.5 percentiles of the difference vs Z17.

## Assumptions and deviations

- Inputs are the bank-sweep inputs on AIRCC (read-only from cycle2):
  `results/prm_boost_v1/bank_prm` (Qwen-7B) and `results/prm_boost_gen_v1/bank_<prm>` + `full_<prm>/prmbench_telemetry.pkl`.
  Code imports (evaluator, frozen adapter, `lsml_continuous`) come from `eval_mirror_prm_gen_v1`, as in the sweep.
- The raw values come from the same FEATURES.npz `values` that the answer-z is computed from (`Population.cell_bank(...)['raw']`).
- Digit channels in the raw view are ranked only over active steps. This mirrors `masked_std`, and it is
  the one choice not dictated by the request.
- The bootstrap uses 1,000 draws (the sweep used 2,000) and resamples answers, as requested. A source-group
  bootstrap is reported as well.
- Scope: RAW17 and ZMEAN34 were dropped by Omri's scope change. MathShepherd was excluded.
- Outputs went to `/shared/cycle2_tau_averbuch_prj/omrisegev1/auto_raw_plus_z_v1/` because the cycle3 home is
  read-only. The `cpu` partition only accepts `cycle4_tau_averbuch_spot_prj` (cycle3 was rejected).
- The per-step scores (`SCORES_<prm>.npz`, ~7 MB) are not in git. They are on AIRCC and on Drive.

## Rerun (other computer)

1. Copy `auto_raw_plus_z.py` and `run.sbatch` to `/shared/cycle2_tau_averbuch_prj/omrisegev1/auto_raw_plus_z_v1/`.
2. `export SLURM_CONF_SERVER=controller-primary; sbatch -A cycle4_tau_averbuch_spot_prj -p cpu run.sbatch`
   (about 1-2 minutes). The job stops with `HARD STOP: parity` if Z17 does not reproduce AUTO_N16.
3. Copy `out/` back here as `out_aircc/` and run `python make_tables.py`.
