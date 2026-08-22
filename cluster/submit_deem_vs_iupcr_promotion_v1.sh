#!/usr/bin/env bash
set -euo pipefail
SHARED=/shared/cycle2_tau_averbuch_prj/omrisegev1
DECISION=$SHARED/results/deem_vs_iupcr_24cell_v1/evaluation/B199/DECISION.json
PARTITION=${OWNER_PARTITION:-power-gpu}
QOS=${OWNER_QOS:-owner_880}
SBATCH=cluster/submit_deem_vs_iupcr_24cell_v1.sbatch
if [ ! -f "$DECISION" ] || [ "$(jq -r '.eligible_for_B999' "$DECISION")" != "true" ]; then
  echo "B=999 forbidden: B=199 did not pass every frozen promotion gate" >&2; exit 2
fi
submit_after() { sbatch --dependency="afterok:$1" -p "$PARTITION" --qos="$QOS" "$SBATCH" "$2" | awk '{print $4}'; }
previous=$(sbatch -p "$PARTITION" --qos="$QOS" "$SBATCH" evaluate-999 | awk '{print $4}'); echo "evaluate-999 $previous"
previous=$(submit_after "$previous" report-999); echo "report-999 $previous"
previous=$(submit_after "$previous" resume-evaluate-999); echo "resume-evaluate-999 $previous"
previous=$(submit_after "$previous" fresh-evaluate-999); echo "fresh-evaluate-999 $previous"
previous=$(submit_after "$previous" finalize-rebuild-999); echo "finalize-rebuild-999 $previous"
