#!/usr/bin/env bash
# Frozen initial chain: B=999 is deliberately absent and needs promotion.
set -euo pipefail
PARTITION=${OWNER_PARTITION:-power-gpu}
QOS=${OWNER_QOS:-owner_880}
SBATCH=cluster/submit_deem_vs_iupcr_24cell_v1.sbatch
submit() {
  local dependency=$1 stage=$2 kind=${3:-afterany} output
  if [ -n "$dependency" ]; then
    output=$(sbatch --dependency="$kind:$dependency" -p "$PARTITION" --qos="$QOS" "$SBATCH" "$stage")
  else
    output=$(sbatch -p "$PARTITION" --qos="$QOS" "$SBATCH" "$stage")
  fi
  awk '{print $4}' <<<"$output"
}
previous=$(submit "" preflight); echo "preflight $previous"
previous=$(submit "$previous" bundles afterok); echo "bundles $previous"
for index in $(seq 1 12); do
  if [ "$index" -eq 1 ]; then previous=$(submit "$previous" stage-a afterok)
  else previous=$(submit "$previous" stage-a afterany); fi
  echo "stage-a-$index $previous"
done
previous=$(submit "$previous" evaluate-199 afterok); echo "evaluate-199 $previous"
previous=$(submit "$previous" report-199 afterok); echo "report-199 $previous"
previous=$(submit "$previous" resume-evaluate afterok); echo "resume-evaluate $previous"
for index in $(seq 1 12); do
  if [ "$index" -eq 1 ]; then previous=$(submit "$previous" fresh-stage-a afterok)
  else previous=$(submit "$previous" fresh-stage-a afterany); fi
  echo "fresh-stage-a-$index $previous"
done
previous=$(submit "$previous" fresh-evaluate afterok); echo "fresh-evaluate $previous"
previous=$(submit "$previous" finalize-rebuild afterok); echo "finalize-rebuild $previous"
