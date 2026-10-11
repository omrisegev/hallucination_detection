#!/bin/bash
# Think-fix rerun of the two Qwen3-8B external telemetry cells (Omri, 2026-10-10).
# Mirrors results/lsml_external_generalization_v1/FULL_SUBMISSION.sh (commit 411316927) with the
# immutable snapshot of 153589e43, --think-prefix empty_block and new output names.
# socratic_qwq32b is NOT rerun: QwQ's template has no non-thinking mode.
set -euo pipefail
export SLURM_CONF_SERVER=controller-primary
S=/shared/cycle2_tau_averbuch_prj/omrisegev1
CODE=$S/code_tf_thinkfix_153589e
REC=$CODE/results/lsml_external_generalization_v1_thinkfix
R=$S/results/lsml_external_generalization_v1
cp $S/src/lsml_external_411316927/results/lsml_external_generalization_v1/QWQ_ALIGNMENT_DIAGNOSTIC.json $REC/
mkdir "$R/full_submission_thinkfix_153589e"
cp "$0" "$R/full_submission_thinkfix_153589e/SUBMISSION.sh"
cp $REC/*.json "$R/full_submission_thinkfix_153589e/"
REV=b968826d9c46dd6066d109eabc6255188de91218
for cell in hard2verify_qwen3_8b socratic_qwen3_8b; do
  DATA=${cell%%_qwen3_8b}
  OUT="$R/${cell}_full_thinkfix_153589e"
  test ! -e "$OUT"
  JOB=$(sbatch --parsable -A cycle3_tau_averbuch_prj -p power-gpu --qos=owner_940 --time=00:30:00 \
    --job-name="ext_full_${cell}_thinkfix" --chdir="$CODE" --container-workdir="$CODE" \
    "$CODE/cluster/submit_external_telemetry.sbatch" \
    --answers "$S/data/lsml_external_20260924/$DATA/answers.json" --model Qwen/Qwen3-8B --revision "$REV" \
    --out "$OUT" --preflight "$REC/FULL_PREFLIGHT.json" --protocol "$REC/COLLECTION_PROTOCOL.json" \
    --budget-decision "$REC/FULL_BUDGET_DECISION.json" --estimate "$REC/ALL_TELEMETRY_COMPUTE_ESTIMATE.json" \
    --alignment-evidence "$REC/QWQ_ALIGNMENT_DIAGNOSTIC.json" --mode full --think-prefix empty_block)
  printf '%s %s 00:30:00\n' "$cell" "$JOB" | tee -a "$R/full_submission_thinkfix_153589e/JOBS.txt"
done
