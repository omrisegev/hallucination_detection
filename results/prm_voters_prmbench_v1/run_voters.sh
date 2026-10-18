set -uo pipefail
S=/shared/cycle2_tau_averbuch_prj/omrisegev1; C=$S/code_integration_a1d2dee36; R=$S/results; V=$R/prmbench_voters_v1; G=$R/prm_boost_gen_v1
cd $C
PRMS="--prm prm72b=$G/full_qwen_prm_72b/prmbench_telemetry.pkl --prm pure=$G/full_pure_prm/prmbench_telemetry.pkl --prm rlhflow=$G/full_rlhflow_prm/prmbench_telemetry.pkl --prm mshep=$G/full_math_shepherd/prmbench_telemetry.pkl"
echo "== ALL VOTERS"
python3 -B scripts/run_prm_voters_prmbench.py --prm7b $R/prmbench_qwen25math7b_full/prmbench_prm.pkl $PRMS \
  --judge j8b=$V/uprm_qwen3_8b_full/prmbench_uprm_judge.pkl --judge j14b=$V/uprm_qwen25_14b_full/prmbench_uprm_judge.pkl \
  --critic c72b="$V/critic_qwen72b_full/prmbench_critic_shard*of6.pkl" --out $R/prm_voters_prmbench_v1/all_voters
echo "== PRMS ONLY"
python3 -B scripts/run_prm_voters_prmbench.py --prm7b $R/prmbench_qwen25math7b_full/prmbench_prm.pkl $PRMS --out $R/prm_voters_prmbench_v1/prms_only
