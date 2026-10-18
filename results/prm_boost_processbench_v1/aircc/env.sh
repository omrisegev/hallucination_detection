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
