#!/usr/bin/env bash
# Submit B=999 only after the frozen B=199 decision authorizes promotion.
set -euo pipefail

SHARED=/shared/cycle2_tau_averbuch_prj/omrisegev1
DECISION=$SHARED/results/residual_graph_deem_24cell_v1/evaluation/B199/DECISION.json
PARTITION=${OWNER_PARTITION:-power-gpu}
QOS=${OWNER_QOS:-owner_880}
SBATCH=cluster/submit_residual_graph_deem_24cell_v1.sbatch

if [ ! -f "$DECISION" ] || [ "$(jq -r '.eligible_for_B999' "$DECISION")" != "true" ]; then
    echo "B=999 forbidden: B=199 did not pass every promotion gate" >&2
    exit 2
fi

submit_after() {
    local prior=$1
    local stage=$2
    sbatch --dependency="afterok:$prior" -p "$PARTITION" --qos="$QOS" "$SBATCH" "$stage" | awk '{print $4}'
}

first=$(sbatch -p "$PARTITION" --qos="$QOS" "$SBATCH" evaluate-999 | awk '{print $4}')
echo "evaluate-999 $first"
second=$(submit_after "$first" report-999)
echo "report-999 $second"
third=$(submit_after "$second" resume-evaluate-999)
echo "resume-evaluate-999 $third"
fourth=$(submit_after "$third" fresh-evaluate-999)
echo "fresh-evaluate-999 $fourth"
fifth=$(submit_after "$fourth" finalize-rebuild-999)
echo "finalize-rebuild-999 $fifth"
