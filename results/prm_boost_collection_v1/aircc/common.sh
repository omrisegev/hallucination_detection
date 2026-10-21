# Sourced by the three collection jobs. Paths verified on AIRCC 2026-10-21 (all inputs under cycle2; cycle3 holds none).
S=/shared/cycle2_tau_averbuch_prj/omrisegev1; R=$S/results; O=$R/prm_boost_gen_v1; M=$S/eval_mirror_prm_gen_v1
X=$R/prm_boost_external_v1; C=$S/code_thinkfix_bank_553de79; H=$S/hf_cache/hub
TMS=$H/models--mistralai--Mistral-7B-v0.1/snapshots/27d67f1b5f57dc0953326b2601d68371d40ea8da/tokenizer.json
SS=$M/results/sml_weight_estimator_v1/socratic_sign
J=$R/prm_boost_collection_v1          # job scripts, logs, parity reports
PRMS_PB="qwen_prm_72b pure_prm rlhflow_prm"
export ABL5_EXPORT_SCORES=1
# Bare python3 on the cpu nodes has no numpy (job 290474 failed on import); earlier CPU jobs (288319, 288932) sourced this env.
set +u; . $S/pyenv_native_312_20261011/activate.sh; set -u
python3 -c 'import numpy, scipy, sklearn; print("PYENV", numpy.__version__, scipy.__version__)'
cd $M
