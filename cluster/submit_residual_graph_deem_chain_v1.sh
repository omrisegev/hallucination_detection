#!/usr/bin/env bash
# Submit the frozen pipeline.  The B=999 lane is deliberately not submitted here;
# it is legal only after B=199 DECISION.json declares eligible_for_B999=true.
set -euo pipefail

PARTITION=${OWNER_PARTITION:-power-gpu}
QOS=${OWNER_QOS:-owner_880}
SBATCH=cluster/submit_residual_graph_deem_24cell_v1.sbatch

submit() {
    local dependency=$1
    local stage=$2
    local output
    if [ -n "$dependency" ]; then
        output=$(sbatch --dependency="afterany:$dependency" -p "$PARTITION" --qos="$QOS" "$SBATCH" "$stage")
    else
        output=$(sbatch -p "$PARTITION" --qos="$QOS" "$SBATCH" "$stage")
    fi
    awk '{print $4}' <<<"$output"
}

previous=$(submit "" phase0)
echo "phase0 $previous"
bundle_output=$(sbatch --dependency="afterok:$previous" -p "$PARTITION" --qos="$QOS" "$SBATCH" bundles)
previous=$(awk '{print $4}' <<<"$bundle_output")
echo "bundles $previous"
for index in $(seq 1 12); do
    previous=$(submit "$previous" stage-a)
    echo "stage-a-$index $previous"
done
previous=$(submit "$previous" evaluate-199)
echo "evaluate-199 $previous"
previous=$(submit "$previous" report-199)
echo "report-199 $previous"
previous=$(submit "$previous" resume-evaluate)
echo "resume-evaluate $previous"
for index in $(seq 1 12); do
    previous=$(submit "$previous" fresh-stage-a)
    echo "fresh-stage-a-$index $previous"
done
previous=$(submit "$previous" fresh-evaluate)
echo "fresh-evaluate $previous"
previous=$(submit "$previous" finalize-rebuild)
echo "finalize-rebuild $previous"
