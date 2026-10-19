#!/bin/bash
# Math-Shepherd digit-id fix (claude/prm-boosting 466ddc3d, HANDOFF_MS_DIGIT_FIX_2026-10-19.md), CPU only.
# Steps: G3 parity of the v2 builder on the Qwen3-8B cell, rebuild Math-Shepherd PRMBench + Socratic banks with
# Mistral-7B-v0.1 digit ids, then ablation5 (guarded L-SML): a no-op check on qwen_prm_7b, Math-Shepherd with the
# old bank (guard only) and with the digit-fixed bank. Nothing existing is overwritten (builders refuse existing dirs).
set -euo pipefail
S=/shared/cycle2_tau_averbuch_prj/omrisegev1; R=$S/results; O=$R/prm_boost_gen_v1; M=$S/eval_mirror_prm_gen_v1
C=$S/code_thinkfix_bank_553de79; X=$R/prm_boost_external_v1; H=$S/hf_cache/hub
TQ3=$(ls $H/models--Qwen--Qwen3-8B/snapshots/*/tokenizer.json | head -1)
TMS=$H/models--mistralai--Mistral-7B-v0.1/snapshots/27d67f1b5f57dc0953326b2601d68371d40ea8da/tokenizer.json
BASE=$R/thinkfix_rescore_v1/source_bank_thinkfix_mask1/FEATURES.npz
echo "[1] G3 parity, Qwen3-8B cell"; cd $C
python3 -B scripts/prm_boost_build_bank_v2.py --raw $R/prmbench_qwen3_8b_telemetry_thinkfix/prmbench_telemetry.pkl \
  --base $BASE --mask-code $S/code_thinkfix_rescore_v1 --tokenizer $TQ3 --out $R/prm_boost_v1/bank_parity_qwen3_8b_v2 --workers 16 --expect-equal
echo "[2] Math-Shepherd PRMBench bank, digit fix"
python3 -B scripts/prm_boost_build_bank_v2.py --raw $O/full_math_shepherd/prmbench_telemetry.pkl \
  --base $BASE --mask-code $S/code_prm_boost_v1/mask_fixed --tokenizer $TMS --out $O/bank_math_shepherd_digitfix --workers 16
echo "[3] Math-Shepherd Socratic bank, digit fix"
python3 -B scripts/prm_boost_build_bank_external_v2.py --raw $X/full_math_shepherd_socratic/external_telemetry.pkl \
  --mask-code $S/code_thinkfix_rescore_v1 --names-from $R/prm_boost_v1/bank_prm/FEATURES.npz --tokenizer $TMS \
  --out $X/full_math_shepherd_socratic/bank_v3 --workers 16
cd $M
echo "[4] guarded ablation5, qwen_prm_7b (must equal ablation5_v1)"
python3 -B $M/ablation5.py qwen_prm_7b $R/prm_boost_v1/bank_prm/FEATURES.npz $R/prm_boost_v1/full_prm/prmbench_telemetry.pkl $O/ablation5_guardcheck_v1
echo "[5] guarded ablation5, math_shepherd, old bank (guard only)"
python3 -B $M/ablation5.py math_shepherd $O/bank_math_shepherd/FEATURES.npz $O/full_math_shepherd/prmbench_telemetry.pkl $O/ablation5_guardonly_v1
echo "[6] guarded ablation5, math_shepherd, digit-fixed bank"
python3 -B $M/ablation5.py math_shepherd $O/bank_math_shepherd_digitfix/FEATURES.npz $O/full_math_shepherd/prmbench_telemetry.pkl $O/ablation5_digitfix_v1
echo ALL_DONE
