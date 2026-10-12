# Count-structure check on PRMBench (2026-10-12)

Question: how much of the PRMScore gain from knowing the true number of errors per answer is answer-specific
information, and does the per-answer error count scale with answer length? Fixed ranking: N16 IU-PCR and
continuous L-SML scores from `results/thinkfix_rescore_v1/prmscore_inputs/bank_N16.npz` (corrected prefix,
unmasked export). Only the per-answer count k changes; the top-k steps by the fused score are flagged.
PRMScore pooled over the 6,211 noncontrols. Q80 reproduces v11 exactly (65.359 / 65.381).

Findings (see OUTPUT.txt):
- True counts are small and almost independent of length: mean 2.12, Spearman(true count, n_steps) = 0.092.
- Q80 flags a fraction of each answer (count grows with length); a proportional rule c*n is worse than Q80 even with
  the best c (64.5).
- A constant k=2 per answer (TOP_2) gives 67.90 / 67.92, MAE 0.863, versus Q80 65.36 / 65.38 (MAE 1.43) and
  Qwen2.5-Math-PRM-7B 65.46. k=1: 64.4; k=3: 67.0.
- Label-fitted oracles: count from length only 67.7; from error type only 68.9; type+length 69.0; true count 71.7.

Limits: TOP_2 is a historical post hoc choice (registered protocol says so), and k=2 matches how PRMBench inserts
errors (about two per answer). It is evidence about the benchmark's count structure, not a label-free method.
All oracles use labels. Development data. CPU only; labels for evaluation only.
