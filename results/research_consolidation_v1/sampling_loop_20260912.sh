#!/bin/bash
# Execution-only loop: relaunch the unchanged sampling driver across invocation caps.
# Reads RUN_STATE.json only after the driver has exited (avoids the Windows replace race).
cd "C:/Users/omris/TAU/hallucination_detection" || exit 2
for i in 1 2 3 4 5 6 7 8; do
  python -B -u scripts/run_full_sampling_v3.py --phase run >> results/research_consolidation_v1/full_sampling_loop_20260912.log 2>&1
  code=$?
  phase=$(python -c "import json;print(json.load(open('results/localization_full_sampling_v3/RUN_STATE.json'))['phase'])")
  echo "[loop] invocation $i exit=$code phase=$phase $(date -u +%FT%TZ)" >> results/research_consolidation_v1/full_sampling_loop_20260912.log
  case "$phase" in
    CHECKPOINTED_INVOCATION_CAP) continue ;;
    COMPLETE_REVIEWED_FULL_SAMPLING|SCORES_COMPLETE_WAITING_FOR_SHORTLIST_REVIEW|SCORING_COMPLETE)
      python -B -u scripts/complete_research_consolidation_v1.py >> results/research_consolidation_v1/supervisor_v5_stage1_20260912.stdout.log 2>&1
      echo "[loop] supervisor exit=$? $(date -u +%FT%TZ)" >> results/research_consolidation_v1/full_sampling_loop_20260912.log; exit 0 ;;
    *) echo "[loop] stopping: phase=$phase" >> results/research_consolidation_v1/full_sampling_loop_20260912.log; exit 1 ;;
  esac
done
