# Shared settings for the missing-benchmark collection (plan docs/experiments/PRM_BOOST_MISSING_BENCHMARKS_2026-10-26_HE.md).
# Source it; override by exporting first. Re-check every path in /preflight.
export S=/shared/cycle2_tau_averbuch_prj/omrisegev1; export R=$S/results H=$S/hf_cache/hub
export ACCT=${ACCT:-cycle3_tau_averbuch_prj} PART=${PART:-power-gpu} QOS=${QOS:-owner_940}
# One frozen code dir for both modes: the standalone run's tree (SYNC_COMMIT 41b63ac), which has --answers,
# --mode instruct --prompt-style and the 5 PRM adapters. The PRMBench parity jobs below check it against the frozen pkls.
export TEL_CODE=${TEL_CODE:-$S/code_standalone_v1}
export BANK_CODE=$S/code_thinkfix_bank_553de79 MASK_CODE=$S/code_thinkfix_rescore_v1 MIRROR=$S/eval_mirror_prm_gen_v1
export NAMES_FROM=$R/prm_boost_v1/bank_prm/FEATURES.npz
export DATA=$S/data/prm_boost_crop_ares_benchmarks_v1          # <bench>/answers.json, no labels
export O=$R/prm_boost_crop_ares_benchmarks_v1                  # every output of this experiment
export GDRIVE=gdrive:hallucination_detection/cluster_results/prm_boost_crop_ares_benchmarks_v1
export RC="$S/bin/rclone --config $S/.config/rclone/rclone.conf"   # $HOME moved to cycle3; the gdrive remote is in the old home

export ADAPTERS="qwen_prm pure_prm rlhflow_prm math_shepherd qwen_prm_72b"
export NEW_BENCHES=${NEW_BENCHES:-"crv_arith crv_gsm8k math_shepherd prm800k deltabench"}     # prm mode AND standalone
export SA_ONLY_BENCHES=${SA_ONLY_BENCHES-"processbench socratic hard2verify"}                  # PRM telemetry exists; standalone missing
declare -A ANSWERS=( [processbench]=$S/data/prm_boost_processbench_v1/answers.json
                     [socratic]=$S/data/lsml_external_20260924/socratic/answers.json
                     [hard2verify]=$S/data/lsml_external_20260924/hard2verify/answers.json )
for b in $NEW_BENCHES; do ANSWERS[$b]=$DATA/$b/answers.json; done

# Standalone base model of each PRM, as released (same table as prm_standalone_telemetry_v1/aircc/common.sh).
declare -A SA_MODEL=( [qwen_prm]=Qwen/Qwen2.5-Math-7B-Instruct [qwen_prm_72b]=Qwen/Qwen2.5-Math-72B-Instruct
                      [pure_prm]=Qwen/Qwen2.5-Math-7B [rlhflow_prm]=meta-llama/Llama-3.1-8B-Instruct
                      [math_shepherd]=mistralai/Mistral-7B-v0.1 )
declare -A SA_STYLE=( [qwen_prm]=qwen_cot_chat [qwen_prm_72b]=qwen_cot_chat [pure_prm]=qwen_cot_chat
                      [rlhflow_prm]=chat [math_shepherd]=plain )
declare -A SA_MAXTOK=( [qwen_prm]=16384 [qwen_prm_72b]=8192 [pure_prm]=16384 [rlhflow_prm]=16384 [math_shepherd]=16384 )
# Frozen PRMBench pkls for the parity jobs (first 40 rows must match exactly).
declare -A PRM_PKL=( [qwen_prm]=$R/prm_boost_v1/full_prm/prmbench_telemetry.pkl
                     [qwen_prm_72b]=$R/prm_boost_gen_v1/full_qwen_prm_72b/prmbench_telemetry.pkl
                     [pure_prm]=$R/prm_boost_gen_v1/full_pure_prm/prmbench_telemetry.pkl
                     [rlhflow_prm]=$R/prm_boost_gen_v1/full_rlhflow_prm/prmbench_telemetry.pkl
                     [math_shepherd]=$R/prm_boost_gen_v1/full_math_shepherd/prmbench_telemetry.pkl )
declare -A SA_PKL=( [qwen_prm]=$R/prm_standalone_telemetry_v1/full_qwen_prm_7b/prmbench_telemetry.pkl
                    [qwen_prm_72b]=$R/prm_standalone_telemetry_v1/full_qwen_prm_72b/prmbench_telemetry.pkl
                    [pure_prm]=$R/prm_standalone_telemetry_v1/full_pure_prm/prmbench_telemetry.pkl
                    [rlhflow_prm]=$R/prm_standalone_telemetry_v1/full_rlhflow_prm/prmbench_telemetry.pkl
                    [math_shepherd]=$R/prm_standalone_telemetry_v1/full_math_shepherd/prmbench_telemetry.pkl )

snapshot() {   # exactly one cached snapshot directory of a hub id, or fail loudly
  local d=$H/models--${1//\//--}/snapshots n
  n=$(ls -1 "$d" 2>/dev/null | wc -l)
  [ "$n" = 1 ] || { echo "snapshot: $1 has $n snapshots under $d (need exactly 1)" >&2; return 1; }
  echo "$d/$(ls -1 "$d")"
}
