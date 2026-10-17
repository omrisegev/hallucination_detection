SH=/shared/cycle2_tau_averbuch_prj/omrisegev1; R=$SH/results
RC="$SH/bin/rclone --config $SH/.config/rclone/rclone.conf"
L=$SH/logs/upload_thinkfix_20261010.log
D=gdrive:hallucination_detection/cluster_results
nohup bash -c "
for d in prmbench_qwen3_8b_telemetry_thinkfix pb_qwen3_8b_thinkfix pb_qwen3_4b_thinkfix; do
  $RC copy $R/\$d $D/\$d --transfers 4 --checkers 8 --drive-chunk-size 64M --stats-one-line --stats 60s -v
done
for d in hard2verify_qwen3_8b_full_thinkfix_153589e socratic_qwen3_8b_full_thinkfix_153589e full_submission_thinkfix_153589e; do
  $RC copy $R/lsml_external_generalization_v1/\$d $D/lsml_external_generalization_v1/\$d --transfers 8 --checkers 8 --drive-chunk-size 64M --stats-one-line --stats 60s -v
done
echo UPLOAD_ALL_DONE" > $L 2>&1 &
echo started $!
