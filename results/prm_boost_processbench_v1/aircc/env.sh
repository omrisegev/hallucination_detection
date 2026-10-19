# Shared settings for every ProcessBench PRM-boosting AIRCC script. Source it; override by exporting first.
# Rediscover account/QoS with `sdata` before submitting (CLAUDE.md, 2026-09-13: owner_940 on power-gpu).
export S=/shared/cycle2_tau_averbuch_prj/omrisegev1
export ACCT=${ACCT:-cycle3_tau_averbuch_prj} PART=${PART:-power-gpu} QOS=${QOS:-owner_940}
# GPU telemetry: the exact code dir of the Socratic run (driver/adapters byte-identical to git HEAD; HANDOFF step 2 checks).
export TEL_CODE=${TEL_CODE:-$S/code_integration_7534b8bf4}
# Downstream: same frozen dirs as the Socratic downstream (results/prm_boost_external_v1/ext_down.sh).
export BANK_CODE=$S/code_thinkfix_bank_553de79 MASK_CODE=$S/code_thinkfix_rescore_v1 MIRROR=$S/eval_mirror_prm_gen_v1
export NAMES_FROM=$S/results/prm_boost_v1/bank_prm/FEATURES.npz
export DATA=$S/data/prm_boost_processbench_v1                  # answers.json, answers_timing12.json (no labels)
export O=$S/results/prm_boost_processbench_v1                  # all outputs; code/ holds this aircc/ folder
export ADAPTERS="qwen_prm pure_prm rlhflow_prm math_shepherd qwen_prm_72b"
export GDRIVE=gdrive:hallucination_detection/cluster_results/prm_boost_processbench_v1
export RC=$S/bin/rclone
# --- Pre-registration section 8 (PREREGISTRATION_HE.md): v2 bank builder with --tokenizer, guarded L-SML. ---
# The old frozen copies stay where they are and are never overwritten; the v2 files have new names.
export OLD_BUILDER=$BANK_CODE/scripts/prm_boost_build_bank_external.py NEW_BUILDER=$BANK_CODE/scripts/prm_boost_build_bank_external_v2.py
export OLD_BUILDER_SHA256=5c7afd61e1fd555114c3cbd93ee294af84f267d525b73d05d426b1f4ba72c62f   # repo a21e3fc4 build_bank_external.py
export HF=$S/hf_cache/hub
export BANK_DIR=${BANK_DIR:-bank_v3}   # v2-builder banks; bank_v2 (old builder, e.g. a timing smoke) is never reused
export PARITY_ADAPTERS=${PARITY_ADAPTERS:-"qwen_prm pure_prm rlhflow_prm qwen_prm_72b"}   # digits 15..24: v2 must equal the old builder
# Copy src to dst only when dst is absent; an existing dst must be byte-identical (never overwritten).
install_or_cmp() {
  if [ -e "$2" ]; then cmp "$2" "$1" || { echo "FAIL: $2 differs from $1 (not overwritten)" >&2; return 1; }
  else cp "$1" "$2" && echo "installed $2"; fi
}
# tokenizer.json of adapter $1's backbone in the offline HF cache. TOK_<adapter> overrides.
tok_for() {
  local v="TOK_$1" r; [ -n "${!v:-}" ] && { echo "${!v}"; return; }
  case $1 in
    qwen_prm) r="Qwen--Qwen2.5-Math-PRM-7B";;
    qwen_prm_72b) r="Qwen--Qwen2.5-Math-PRM-72B";;
    pure_prm) r="jinachris--Qwen2.5-Math-7B-PRM800K jinachris--PURE-PRM-7B";;
    rlhflow_prm) r="RLHFlow--Llama3.1-8B-PRM-Mistral-Data";;
    # Math-Shepherd ships tokenizer.model only; same sha256 as Mistral-7B-v0.1's (AIRCC job 288932, digitfix_v1/README.md)
    math_shepherd) echo "$HF/models--mistralai--Mistral-7B-v0.1/snapshots/27d67f1b5f57dc0953326b2601d68371d40ea8da/tokenizer.json"; return;;
    *) return 1;;
  esac
  for x in $r; do ls $HF/models--$x/snapshots/*/tokenizer.json 2>/dev/null; done | head -1
}
# Expected '0'..'9' ids: 15..24 for the Qwen2.5 and Llama-3 vocabularies; Mistral-7B-v0.1 ids verified in job 288932.
digits_for() {
  case $1 in math_shepherd) echo 28734,28740,28750,28770,28781,28782,28784,28787,28783,28774;; *) echo 15,16,17,18,19,20,21,22,23,24;; esac
}
# Stop unless adapter $1's tokenizer $2 resolves to the expected digit ids (digit_token_ids.resolve_digit_ids).
check_digits() {
  [ -f "$2" ] || { echo "FAIL: no tokenizer.json for $1 ('$2'); set TOK_$1" >&2; return 1; }
  python3 -B - "$BANK_CODE/scripts/digit_token_ids.py" "$2" "$(digits_for $1)" "$1" <<'PY'
import importlib.util, sys
s = importlib.util.spec_from_file_location('dt', sys.argv[1]); D = importlib.util.module_from_spec(s); s.loader.exec_module(D)
ids, rec = D.resolve_digit_ids(sys.argv[2], expected=[int(x) for x in sys.argv[3].split(',')])
print('digit ids OK', sys.argv[4], list(ids), rec['tokenizer_sha256'][:16], sys.argv[2])
PY
}
