# Sourced by every step of the rule C transfer jobs (H2V, DeltaBench) on AIRCC cycle4, 2026-11-21.
# Mirrors results/ssl_one_experiment_v1/aircc/common.sh (2026-11-18); only the paths change (cycle2 workspace is gone).
S=/shared/cycle4_tau_averbuch_spot_prj/users/omrisegev1/ws_c4; R=$S/results; H=$S/hf_cache/hub
W=$R/ssl_step_threshold_h2v_deltabench_v1                 # every output of this experiment
HD_ROOT=$S/code_rule_c_v2   # frozen code v2 = v1 + np2_alias_run.py (loader shim for the stored numpy-2 pickles)
X=$HD_ROOT/results/ssl_one_experiment_v1; Y=$HD_ROOT/results/rule_c_h2v_deltabench_v1
ACCT=cycle4_tau_averbuch_spot_prj; QOS=spot_700; PART=power-gpu
MODEL=Qwen/Qwen2.5-Math-7B-Instruct; REV=ef9926d75ab1d54532f6a30dd5e760355eb9aa4d
DRIVER_ARGS="--mode instruct --model $MODEL --prompt-style qwen_cot_chat --max-batch 1"
ROPE_FACTOR=3                                             # DeltaBench only, as the stored ext cell (merge_longctx.py)
BANK_CODE=$S/code_thinkfix_bank_553de79
PB_BUILD=$BANK_CODE/scripts/prm_boost_build_bank_external_v2.py
PB_MASK=$S/code_thinkfix_rescore_v1
PB_NAMES=$R/prm_boost_v1/bank_prm/FEATURES.npz
STORED=$R/prm_boost_crop_ares_benchmarks_v1               # stored standalone C cells (copied from Drive, md5 checked)
TOK=$H/models--Qwen--Qwen2.5-Math-7B-Instruct/snapshots/$REV/tokenizer.json
input_args() {   # bench V -> driver input flag
  case "$2" in
    V3|V4|preC_V3|preC_V4) echo "--answers $W/inputs/$1_$2_answers.json" ;;
    *) echo "bad V $2" >&2; return 1 ;;
  esac
}
