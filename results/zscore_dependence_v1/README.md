# Does the per-answer z-score create dependence between telemetry channels? (diagnostic, 2026-10-29)

Question (Omri): every fusion standardizes each channel inside each answer before L-SML. Does that add
dependence between the channels, and what happens to AUC without it?

Run: `python zscore_dependence.py` (CPU, a few minutes, no inference). Labels used only to split steps by class
and to score. The answer-z representation rebuilt from the raw features matches the stored channel files
(assert < 1e-4) in all 8 cells.

Representations of the same 16 N16 channels: **AZ** = per-answer z (today), **CENTER** = per-answer mean removed,
one pooled sd, **RAW** = one global mean/sd per channel. Dependence = mean |Pearson r| over the 120 channel pairs
inside one class (error steps, correct steps), pooled over steps; that is the class-conditional dependence SML
assumes is zero.

Cells: PRMBench with each PRM's base LM as standalone telemetry (raw banks restored from Drive
`cluster_results/prm_standalone_telemetry_v1/bank_<prm>` into `hd_collection_data/prm_standalone_banks`, not in git);
ProcessBench (prefix domain) with the 4 PRM backbones. **Not covered:** PRMBench with the PRM backbone (the AUTO
PRMBench setting): its raw banks are only on AIRCC, which was unreachable.

## Dependence (mean over the 4 PRMs; DEPENDENCE.csv)

| | AZ | CENTER | RAW |
|---|---|---|---|
| PRMBench, error steps | 0.312 | 0.293 | 0.288 |
| PRMBench, correct steps | 0.272 | 0.259 | 0.257 |
| ProcessBench, error steps | 0.246 | 0.241 | 0.235 |
| ProcessBench, correct steps | 0.275 | 0.250 | 0.268 |

The z-score adds a little dependence (about +0.02 mean |r| on PRMBench), almost all of it from the per-answer
division (CENTER is close to RAW). No pattern with answer length. Most of the dependence (~0.26-0.29) is there in
the raw channels.

## AUC (AUC.csv)

PRMBench within-answer AUC, AZ -> RAW: TELE16 .7934->.7977 (Qwen-7B), .7812->.7845 (Qwen-72B), .7704->.7732 (PURE),
.7877->.7846 (RLHFlow); MEAN16 better or equal without z on all 4. Changes are 0.003-0.008; no bootstrap was run, so
treat them as not established. Pooled step AUC rises without z (TELE16 +0.007 to +0.022), as expected: the answer-z
throws away the answer level.

ProcessBench within-AUC is mixed (AUTO RAW vs AZ: -0.001, -0.001, +0.089 PURE, +0.014 RLHFlow; TELE16 drops on
Qwen-72B). The PRM's ProcessBench within-AUC on RLHFlow (0.33) suggests its risk arm is inverted in that cell; it
is the same input in all three representations.
