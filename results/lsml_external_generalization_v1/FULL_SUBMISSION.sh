#!/bin/bash
set -euo pipefail
export SLURM_CONF_SERVER=controller-primary
S=/shared/cycle2_tau_averbuch_prj/omrisegev1
CODE=$S/src/lsml_external_411316927
R=$S/results/lsml_external_generalization_v1
mkdir -p "$CODE"
tar -xzf "$S/src/code_411316927.tar.gz" -C "$CODE"
mkdir "$R/full_submission_411316927"
for cell in hard2verify_qwen3_8b socratic_qwen3_8b socratic_qwq32b; do
  case "$cell" in
    hard2verify_qwen3_8b) DATA=hard2verify; MODEL=Qwen/Qwen3-8B; REV=b968826d9c46dd6066d109eabc6255188de91218; LIMIT=00:30:00;;
    socratic_qwen3_8b) DATA=socratic; MODEL=Qwen/Qwen3-8B; REV=b968826d9c46dd6066d109eabc6255188de91218; LIMIT=00:30:00;;
    socratic_qwq32b) DATA=socratic; MODEL=Qwen/QwQ-32B; REV=976055f8c83f394f35dbd3ab09a285a984907bd0; LIMIT=01:00:00;;
  esac
  JOB=$(sbatch --parsable -A cycle3_tau_averbuch_prj -p power-gpu --qos=owner_940 --time="$LIMIT" --job-name="ext_full_$cell" --chdir="$CODE" --container-workdir="$CODE" "$CODE/cluster/submit_external_telemetry.sbatch" --answers "$S/data/lsml_external_20260924/$DATA/answers.json" --model "$MODEL" --revision "$REV" --out "$R/${cell}_full_411316927" --preflight "$CODE/results/lsml_external_generalization_v1/FULL_PREFLIGHT.json" --protocol "$CODE/results/lsml_external_generalization_v1/COLLECTION_PROTOCOL.json" --budget-decision "$CODE/results/lsml_external_generalization_v1/FULL_BUDGET_DECISION.json" --estimate "$CODE/results/lsml_external_generalization_v1/ALL_TELEMETRY_COMPUTE_ESTIMATE.json" --alignment-evidence "$CODE/results/lsml_external_generalization_v1/QWQ_ALIGNMENT_DIAGNOSTIC.json" --mode full)
  printf '%s %s %s\n' "$cell" "$JOB" "$LIMIT" | tee -a "$R/full_submission_411316927/JOBS.txt"
done
