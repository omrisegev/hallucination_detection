#!/bin/bash
# Stage-1 outputs -> gdrive:hallucination_detection/cluster_results/prm_boost_collection_v1/ (copy, idempotent)
RC="/shared/cycle2_tau_averbuch_prj/omrisegev1/bin/rclone --config /shared/cycle2_tau_averbuch_prj/omrisegev1/.config/rclone/rclone.conf"  # HOME moved to cycle3
 R=/shared/cycle2_tau_averbuch_prj/omrisegev1/results
D=gdrive:hallucination_detection/cluster_results/prm_boost_collection_v1
F="--transfers 4 --checkers 8 --drive-chunk-size 64M --stats-one-line --stats 60s -v"
$RC copy $R/prm_boost_gen_v1/ablation5_export_v1 $D/prmbench/ablation5_export_v1 $F
$RC copy $R/prm_boost_gen_v1/ablation5_signabs_v1 $D/prmbench/ablation5_signabs_v1 $F
for d in collect_socratic_v1 collect_socratic_signabs_v1 collect_h2v_v1 collect_h2v_signabs_v1; do
  $RC copy $R/prm_boost_external_v1/$d $D/external/$d $F; done
$RC copy $R/prm_boost_external_v1/full_math_shepherd_hard2verify/bank_v3 $D/external/full_math_shepherd_hard2verify/bank_v3 $F
$RC copy $R/prm_boost_collection_v1 $D/jobs $F
$RC check $R/prm_boost_gen_v1/ablation5_export_v1 $D/prmbench/ablation5_export_v1 --one-way && echo CHECK_EXPORT_OK
echo UPLOAD_DONE
