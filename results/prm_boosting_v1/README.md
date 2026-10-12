# prm_boosting_v1: label-free fusion of Qwen2.5-Math-PRM-7B with our N16 telemetry on PRMBench

Requested by Omri on 2026-10-12 ("unsupervised process reward model boosting"). Development evidence
on PRMBench (6,969 answers, all previously opened), not an untouched test.

## Pipeline
1. **PRM pass** (AIRCC job 282802, 15 min, `cluster/run_prmbench_prm_telemetry.py --mode prm`): one forward of
   Qwen2.5-Math-PRM-7B on the model-card template (one `<extra_0>` per step). From that forward we take the step
   rewards (identical to Step 242's `prmbench_prm.pkl`, max diff 0.0) and the token distribution: the
   checkpoint's `lm_head.weight` applied to the fine-tuned backbone's last hidden state. The PRM class never uses
   `lm_head`. It is bit-identical to Qwen2.5-Math-7B-Instruct's (`max_abs_diff 0.0`), so the backbone states are
   the only thing that differs from Instruct. Step tokens only; `<extra_0>` positions are not scored.
2. **Instruct reference** (job 282803): Qwen2.5-Math-7B-Instruct as a causal LM on the same rows, with the
   same system prompt and steps joined by `\n\n`. This is a comparison only.
3. **Banks** (jobs 282816 / 282805, `build_bank.py`): the frozen per-answer feature code
   (code_thinkfix_bank_553de79 `rebuild_source_bank_v4.one`), after the mask1 rule. The PRMBench rows of the
   think-fix mask1 source bank are replaced with the new features. Parity: rebuilding the Qwen3 cell reproduces
   the frozen bank exactly (max abs error 0.0, job 282775). In the PRM path, 3 answers have empty step text (zero
   tokens). Those steps are filled with the answer's column mean (MANIFEST `empty_step_fills`).
4. **Fusion** (`fuse.py`): the unchanged Monday adapter (`monday_external_panel_v1/score.py`). The N16 parity on
   Qwen3 reproduces `thinkfix_rescore_v1/prmscore_inputs/bank_N16.npz` to 3e-15. The PRM channel is
   `logit(1 - reward)`.
5. **Compositions** (`compose.py`) and **bootstrap** (`bootstrap.py`): paired source-group bootstrap, 2,000 draws
   over 707 groups.

Metrics:
- within-answer AUC on v3 labels;
- official PRMScore = (F1_valid + F1_error)/2 (`prmbench_evaluate`) under ANSWER_Z_Q80. PRM alone is also
  reported under its native rule (reward < 0.5).

## Results (COMPOSITIONS.csv)
| arm | within-AUC | PRMScore |
|---|---|---|
| PRM-7B reward, native 0.5 | 0.8012 | 65.46 |
| N16 Qwen3-8B (MEAN_MOM_ga) | 0.7767 | 65.23 |
| N16 Qwen2.5-Math-Instruct | 0.7885 | 65.93 |
| **N16 PRM-7B backbone** | **0.8131** | **67.20** |
| PRM as 17th channel in MEAN_MOM_ga (backbone bank) | 0.8231 | 67.58 |
| **½ z(PRM) + ½ z(N16 backbone)** | **0.8405** | **68.53** |
| ½ z(PRM) + ½ z(N16 Qwen3) | 0.8280 | 67.81 |
| ⅓ PRM + ⅓ backbone + ⅓ Qwen3 | 0.8318 | 68.25 |

Paired differences, with 95% CI:
- Two-view (PRM + backbone) vs PRM native: +0.039 [0.036, 0.043] AUC, +3.08 [2.47, 3.70] PRMScore.
- Two-view vs N16 backbone: +0.027 [0.024, 0.031] AUC, +1.33 [0.99, 1.67] PRMScore.
- N16 backbone vs PRM native: +0.012 [0.006, 0.018] AUC, +1.74 [1.06, 2.43] PRMScore.

## Caveats
- The equal two-view composition was chosen after seeing that PRM-as-one-channel is diluted (stage 1). It has
  no fitted weight, but the choice is post hoc on PRMBench.
- 3-view SML weights are degenerate: the backbone and Qwen3 views correlate at 0.86, and one loading estimate
  exceeds 1. The weights collapse onto one view, so SML is reported as failing, not as a method.
- PRM-7B was trained on labels (Qwen MC + judge). The fused score inherits that supervision. Only the fusion
  is label-free.
- Large outputs (raw telemetry ~1 GB per path, FEATURES.npz) are kept on AIRCC under `results/prm_boost_v1/`.
