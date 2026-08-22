#!/usr/bin/env bash
# Submit the frozen pipeline.  The B=999 lane is deliberately not submitted here;
# it is legal only after B=199 DECISION.json declares eligible_for_B999=true.
set -euo pipefail

PARTITION=${OWNER_PARTITION:-power-gpu}
QOS=${OWNER_QOS:-owner_880}
SBATCH=cluster/submit_residual_graph_deem_24cell_v1.sbatch

# Scientific boundaries must be afterok.  afterany is satisfied by a
# *cancellation*, so when phase0 failed (AIRCC job 217597) bundles was
# correctly cancelled but all twelve Stage-A jobs stayed eligible and would
# have run against a phase0 that never produced output.  afterany is still
# correct *within* a Stage-A run, where consecutive jobs continue immutable
# checkpoints across the eight-hour wall and must survive a requeue.
submit() {
    local dependency=$1
    local stage=$2
    local kind=${3:-afterany}
    local output
    if [ -n "$dependency" ]; then
        output=$(sbatch --dependency="$kind:$dependency" -p "$PARTITION" --qos="$QOS" "$SBATCH" "$stage")
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
    if [ "$index" -eq 1 ]; then
        previous=$(submit "$previous" stage-a afterok)
    else
        previous=$(submit "$previous" stage-a)
    fi
    echo "stage-a-$index $previous"
done
previous=$(submit "$previous" evaluate-199 afterok)
echo "evaluate-199 $previous"
previous=$(submit "$previous" report-199 afterok)
echo "report-199 $previous"
previous=$(submit "$previous" resume-evaluate afterok)
echo "resume-evaluate $previous"
for index in $(seq 1 12); do
    if [ "$index" -eq 1 ]; then
        previous=$(submit "$previous" fresh-stage-a afterok)
    else
        previous=$(submit "$previous" fresh-stage-a)
    fi
    echo "fresh-stage-a-$index $previous"
done
previous=$(submit "$previous" fresh-evaluate afterok)
echo "fresh-evaluate $previous"
previous=$(submit "$previous" finalize-rebuild afterok)
echo "finalize-rebuild $previous"
