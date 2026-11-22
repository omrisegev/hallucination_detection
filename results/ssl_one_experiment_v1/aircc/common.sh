# Sourced by every step of the one SSL experiment on AIRCC (RC_HANDOFF.md sections B-D), 2026-11-18.
# Paths follow results/prm_standalone_telemetry_v1/aircc/common.sh and
# results/prm_boost_crop_ares_benchmarks_v1/code/env.sh (checked on AIRCC 2026-11-18).
S=/shared/cycle2_tau_averbuch_prj/omrisegev1; R=$S/results; H=$S/hf_cache/hub
W=$R/ssl_one_experiment_v1                      # every output of this experiment
HD_ROOT=$S/code_ssl_one_v1                      # frozen checkout of the pushed branch commit
X=$HD_ROOT/results/ssl_one_experiment_v1        # this folder inside the frozen checkout
ACCT=cycle4_tau_averbuch_spot_prj; QOS=spot_700; PART=power-gpu   # cycle2/cycle3 are not valid for jobs
MODEL=Qwen/Qwen2.5-Math-7B-Instruct             # one cached snapshot, revision ef9926d7 (checked by snapshot())
DRIVER_ARGS="--mode instruct --model $MODEL --prompt-style qwen_cot_chat --max-batch 1"

# Inputs: C = the inputs of the stored standalone runs; V1/V2 = make_variants.py, V3/V4 = make_variants_v34.py, V5 = make_variants_v5.py,
# V6/V7/V8 = make_variants_v678.py (synthetic benchmark v2, 2026-11-22; all dev only).
PB_C=$HD_ROOT/results/prm_boost_processbench_v1/inputs/answers.json
PRMB_C=$R/prm_boost_v1/full_prm/prmbench_telemetry.pkl

# PRMBench bank: the frozen builder of the stored standalone bank (prm_standalone_telemetry_v1/aircc/downstream.sbatch).
BANK_CODE=$S/code_thinkfix_bank_553de79         # cwd of both builders
PRMB_BUILD=$R/prm_standalone_telemetry_v1/code/build_bank.py
PRMB_DIGITS=$R/prm_standalone_telemetry_v1/code/digit_token_ids.py
PRMB_BASE=$R/thinkfix_rescore_v1/source_bank_thinkfix_mask1/FEATURES.npz
PRMB_MASK=$S/code_prm_boost_v1/mask_fixed
PRMB_FILL=$R/prm_standalone_telemetry_v1/full_qwen_prm_7b/prmbench_telemetry.pkl   # filler rows for skipped parents only
# ProcessBench bank: graft rewards + the external builder (prm_boost_crop_ares_benchmarks_v1/code/downstream.sbatch).
PB_BUILD=$BANK_CODE/scripts/prm_boost_build_bank_external_v2.py
PB_MASK=$S/code_thinkfix_rescore_v1
PB_NAMES=$R/prm_boost_v1/bank_prm/FEATURES.npz
PB_PRM_RAW=$R/prm_boost_processbench_v1/full_qwen_prm/external_telemetry.pkl       # rewards for the independence check only

snapshot() {   # exactly one cached snapshot directory of a hub id, or fail loudly
  local d=$H/models--${1//\//--}/snapshots n
  n=$(ls -1 "$d" 2>/dev/null | wc -l)
  [ "$n" = 1 ] || { echo "snapshot: $1 has $n snapshots under $d (need exactly 1)" >&2; return 1; }
  echo "$d/$(ls -1 "$d")"
}

input_args() {   # bench V -> the driver's input flag
  case "$1_$2" in
    processbench_C) echo "--answers $PB_C" ;;
    processbench_V[1-8]|processbench_preC_V[3-8]) echo "--answers $W/inputs/pb_$2_answers.json" ;;   # preC_V3/4: C rows of the
                                                         # parents of the first 12 V3-V5 rows / first 12 k >= 2 rows of V6-V8 (preflight, make_variants_v34.py / _v5.py / _v678.py)
    prmbench_C) echo "--meta-pkl $PRMB_C" ;;
    prmbench_V[1-8]|prmbench_preC_V[3-8]) echo "--meta-pkl $W/inputs/prmbench_$2_meta.pkl" ;;
    *) echo "bad bench/V $1 $2" >&2; return 1 ;;
  esac
}

raw_name() { [ "$1" = processbench ] && echo external_telemetry.pkl || echo prmbench_telemetry.pkl; }
