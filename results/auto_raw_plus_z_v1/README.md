# AUTO with raw + answer-z channels (CAT34) vs current AUTO (Z17), PRMBench, 4 PRMs (2026-11-03)

Omri asked for this: feed L-SML both the answer-z channels it uses today and the same channels without the
per-answer z-score. AIRCC CPU job **304515** (`-p cpu`, account `cycle4_tau_averbuch_spot_prj`; 1 min 11 s, exit 0).
Log: `auto_raw_z_304515.out`. Script: `auto_raw_plus_z.py`. Tables: `make_tables.py` -> `TABLES.md`.
PRMBench development data (6,969 answers per PRM), PRM-backbone telemetry: the same inputs as the bank sweep
(`../prm_boosting_bank_sweep_v1`, job 288319). MathShepherd is excluded, as requested.

**Bottom line.** CAT34 beats Z17 under the pre-declared rule: PRMScore is higher, with a CI that excludes 0,
on 3 of 4 PRMs. The gains are +0.63 to +0.86. PURE goes the other way (-0.18, CI just excludes 0).
Within-AUC rises by +0.011 to +0.013 on 3 PRMs and by +0.001 on PURE (CI includes 0).
On Qwen-72B, the PRM's own score under the same Q80 rule (69.04) is still above CAT34 (68.69).

## Arms (nothing tuned on labels)

- **Z17**: current AUTO_N16. One L-SML with automatic grouping (`lsml_continuous` defaults) over z(PRM risk) and
  the 16 N16 channels, all answer-z. The fused score is oriented by its correlation with z(PRM).
- **CAT34**: the same L-SML call over Z17's 17 views plus the same 17 channels raw. Each raw channel goes through
  a GLOBAL rank -> normal-score transform over all 94,203 PRMBench steps of that PRM: `Phi^-1((rank - 0.5)/n)`,
  with tied values sharing their average rank. The raw PRM view is the logit risk `log((1-r)/r)`.
  The three digit channels are transformed over their active steps only and set to 0 elsewhere, which is what
  the answer-z (`masked_std`) does. The transform is increasing, so each raw view keeps the orientation of
  its z view. Same orientation anchor, same Q80 rule.
- References: **PRM_native** (the PRM's own step labels) and **PRM_Q80_PROB** (z(1 - reward) under Q80, as in
  `prm_boost_ablation5_v1`).
- RAW17 and ZMEAN34 were dropped by Omri's scope change, so they were not run.

## Parity gate: PASS

Z17 reproduces the stored `AUTO_N16` Q80 row of `SWEEP_RESULTS_<prm>.csv` exactly (difference < 1e-9) on all 4 PRMs:

| PRM | within-AUC | PRMScore |
|---|---|---|
| Qwen-7B | 0.8243 | 67.49 |
| Qwen-72B | 0.8278 | 67.83 |
| PURE | 0.7990 | 66.67 |
| RLHFlow | 0.7647 | 63.95 |

## Results

See `TABLES.md` for all tables, including the per-PRM CAT34 groups.

| PRM | Z17 PRMScore | CAT34 PRMScore | CAT34 - Z17 (95% CI, answer bootstrap) | within-AUC Z17 -> CAT34 | answer AUC Z17 -> CAT34 | PRM_Q80_PROB |
|---|---|---|---|---|---|---|
| Qwen-7B | 67.49 | 68.26 | +0.77 [+0.58, +0.95] | 0.8243 -> 0.8366 | 0.580 -> 0.769 | 67.32 |
| Qwen-72B | 67.83 | 68.69 | +0.86 [+0.68, +1.06] | 0.8278 -> 0.8391 | 0.568 -> 0.744 | 69.04 |
| PURE | 66.67 | 66.50 | -0.18 [-0.35, -0.00] | 0.7990 -> 0.8002 | 0.588 -> 0.680 | 64.72 |
| RLHFlow | 63.95 | 64.58 | +0.63 [+0.46, +0.82] | 0.7647 -> 0.7780 | 0.604 -> 0.708 | 56.54 |

A source-group bootstrap (answers of the same source question resampled together) gives the same conclusions;
both bootstraps are in `BOOTSTRAP.csv`.

## Diagnostics

- **raw_i and z_i are strongly correlated inside each step class.** |corr| is 0.59-0.98, the same for clean and
  error steps (Table 4). `realized_z` is 0.98, which means it is almost already answer-normalized.
  The raw copy is not new within-answer information. It adds the between-answer level (each answer's own
  mean and scale), which the z-score removes.
- **L-SML groups raw_i with z_i.** The pair lands in the same CAT34 group for 15, 17, 16 and 17 of the 17
  channels. The exceptions are `chosen_surprisal` (twice) and `true_tail50` (once).
- Pairs cannot get a group each. With the default K range (2..8), L-SML can form at most 8 groups out of 34
  views. In practice it forms 7-8 family groups, each holding both copies (Table 5). Nothing was changed.
- **The PRM stays in a group of its own (z and raw together) in 3 of 4 PRMs.** On PURE it shares a group with
  `true_tail50` and `energy_level`. In Z17 on PURE it was also merged into a telemetry group, which also held
  q15_H1, q15_VE1 and logprob_margin. PURE is also the one PRM where CAT34 does not help.
- **Leading-eigenvalue share** of the input correlation matrix: 0.31-0.35 for Z17 and 0.30-0.33 for CAT34.
  Adding the raw copies does not create one dominant direction.
- **Where the gain comes from.** Answer-level has-error AUC (answer score = max step score) jumps from
  0.57-0.60 to 0.68-0.77, and pooled step AUC rises by +0.02 to +0.03. So CAT34 restores the answer-level
  signal that answer-z throws away. The PRMScore gain is smaller than that jump because the Q80 rule
  answer-z's the fused score again before flagging.

## Caveats

- PRMBench is development data. The rule was declared before the run, but this is not external evidence.
- The PRMScore gains (+0.6 to +0.9) are smaller than the PRM-vs-Z17 gap on Qwen-72B. CAT34 does not beat
  that PRM's own Q80 score.
- The 34 views include 17 near-duplicate pairs. L-SML assumes conditionally independent groups; these
  pairs sit inside groups, so the within-group stage absorbs them, but the cross-group independence
  assumption is as approximate as it is in Z17.
