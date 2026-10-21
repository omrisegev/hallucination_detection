# Sourced by every job of this experiment (plan docs/experiments/PRM_STANDALONE_TELEMETRY_2026-10-22.md).
# Paths follow results/prm_boost_collection_v1/aircc/common.sh (verified on AIRCC 2026-10-21); re-check in /preflight.
S=/shared/cycle2_tau_averbuch_prj/omrisegev1; R=$S/results; O=$R/prm_boost_gen_v1; M=$S/eval_mirror_prm_gen_v1
C=$S/code_thinkfix_bank_553de79; H=$S/hf_cache/hub
W=$R/prm_standalone_telemetry_v1        # every output of this experiment, nothing else is written
CODE_DIR=${CODE_DIR:-$S/code_standalone_v1}   # frozen copy of the synced tree (HANDOFF step 2)
ACCOUNT=cycle3_tau_averbuch_prj; QOS=owner_940

PRMS_SA="qwen_prm_7b qwen_prm_72b pure_prm rlhflow_prm math_shepherd"
# The base checkpoint of each PRM, as released (model cards checked 2026-10-22; RLHFlow and Math-Shepherd from their READMEs)
declare -A SA_MODEL=( [qwen_prm_7b]=Qwen/Qwen2.5-Math-7B-Instruct [qwen_prm_72b]=Qwen/Qwen2.5-Math-72B-Instruct
                      [pure_prm]=Qwen/Qwen2.5-Math-7B [rlhflow_prm]=meta-llama/Llama-3.1-8B-Instruct
                      [math_shepherd]=mistralai/Mistral-7B-v0.1 )
declare -A SA_STYLE=( [qwen_prm_7b]=qwen_cot_chat [qwen_prm_72b]=qwen_cot_chat [pure_prm]=qwen_cot_chat
                      [rlhflow_prm]=chat [math_shepherd]=plain )
declare -A SA_MAXTOK=( [qwen_prm_7b]=16384 [qwen_prm_72b]=8192 [pure_prm]=16384 [rlhflow_prm]=16384 [math_shepherd]=16384 )
declare -A SA_WALL=( [qwen_prm_7b]=02:00:00 [qwen_prm_72b]=06:00:00 [pure_prm]=02:00:00 [rlhflow_prm]=02:00:00 [math_shepherd]=02:00:00 )
# The PRM's own backbone run: its rows (ids, question, steps, order) feed --meta-pkl, its rewards feed ablation5.
declare -A PRM_PKL=( [qwen_prm_7b]=$R/prm_boost_v1/full_prm/prmbench_telemetry.pkl
                     [qwen_prm_72b]=$O/full_qwen_prm_72b/prmbench_telemetry.pkl [pure_prm]=$O/full_pure_prm/prmbench_telemetry.pkl
                     [rlhflow_prm]=$O/full_rlhflow_prm/prmbench_telemetry.pkl [math_shepherd]=$O/full_math_shepherd/prmbench_telemetry.pkl )

snapshot() {   # exactly one cached snapshot directory of a hub id, or fail loudly
  local d=$H/models--${1//\//--}/snapshots n
  n=$(ls -1 "$d" 2>/dev/null | wc -l)
  [ "$n" = 1 ] || { echo "snapshot: $1 has $n snapshots under $d (need exactly 1)" >&2; return 1; }
  echo "$d/$(ls -1 "$d")"
}
