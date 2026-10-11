#!/bin/bash
# Copy (idempotent) + verify the think-fix outputs on Drive. Nothing is deleted anywhere.
SH=/shared/cycle2_tau_averbuch_prj/omrisegev1; R=$SH/results
RC="$SH/bin/rclone --config $SH/.config/rclone/rclone.conf"
D=gdrive:hallucination_detection/cluster_results
L=$SH/logs/upload_thinkfix_20261011.log
OUT=$SH/logs/drive_verify_20261011.tsv
DIRS=$(cd $R && ls -d *_thinkfix *_thinkfix_mask1 2>/dev/null; cd $R && ls -d lsml_external_generalization_v1/*thinkfix* 2>/dev/null; echo thinkfix_rescore_v1)
echo "[dirs]" $DIRS
while pgrep -u $USER -f 'rclone.*copy' > /dev/null; do echo "waiting for running rclone"; sleep 60; done
for d in $DIRS; do
  echo "=== copy $d" >> $L
  $RC copy "$R/$d" "$D/$d" --transfers 4 --checkers 8 --drive-chunk-size 64M --stats-one-line --stats 60s -v >> $L 2>&1
  echo "copy_exit $d $?" >> $L
done
printf 'dir\tlocal_files\tlocal_bytes\tdrive_files\tdrive_bytes\tcheck\n' > $OUT
for d in $DIRS; do
  lf=$(find "$R/$d" -type f | wc -l); lb=$(du -sb "$R/$d" | cut -f1)
  ds=$($RC size "$D/$d" --json 2>/dev/null); df=$(echo "$ds" | python3 -c 'import json,sys; print(json.load(sys.stdin)["count"])' 2>/dev/null); db=$(echo "$ds" | python3 -c 'import json,sys; print(json.load(sys.stdin)["bytes"])' 2>/dev/null)
  ck=$($RC check "$R/$d" "$D/$d" --one-way 2>&1 | grep -E 'differences found|matching files|ERROR' | tail -2 | tr '\n' ' ')
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$d" "$lf" "$lb" "$df" "$db" "$ck" >> $OUT
done
echo VERIFY_DONE >> $OUT
