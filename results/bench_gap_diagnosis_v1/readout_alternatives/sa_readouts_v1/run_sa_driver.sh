#!/bin/bash
# Sequential: download raw sa pickle, verify sha256, extract readouts with gate, delete raw on PASS.
WT=/c/Users/omris/TAU/wt_sa_readouts
HD=/c/Users/omris/TAU/hd_collection_data
OUT=$HD/readout_steps_sa_v1
TMP=$HD/_sa_raw_tmp
R=gdrive:hallucination_detection/cluster_results
mkdir -p $OUT $TMP
LOG=$OUT/run.log
cells=()
for b in socratic hard2verify processbench; do for p in qwen_prm qwen_prm_72b pure_prm rlhflow_prm; do cells+=("$b $p"); done; done
for p in qwen_prm_7b qwen_prm_72b pure_prm rlhflow_prm; do cells+=("prmbench $p"); done
for c in "${cells[@]}"; do
  set -- $c; b=$1; p=$2
  if [ $b = prmbench ]; then
    src=$R/prm_standalone_telemetry_v1/full_$p/prmbench_telemetry.pkl; man=$HD/prm_standalone_banks/bank_$p/MANIFEST.json
    bank=$HD/prm_standalone_banks/bank_$p/FEATURES.npz; name=$p
  else
    src=$R/prm_boost_crop_ares_benchmarks_v1/full_sa_${p}_$b/grafted.pkl; man=$HD/c4u_r1_inputs/crop_ares_all/full_sa_${p}_$b/bank_v3/MANIFEST.json
    bank=$HD/c4u_r1_inputs/crop_ares_all/full_sa_${p}_$b/bank_v3/FEATURES.npz
    name=$p; [ $p = qwen_prm ] && name=qwen_prm_7b
  fi
  out=$OUT/${b}__${name}.npz
  if [ -f $out ]; then echo "SKIP $b $name (exists)" >> $LOG; continue; fi
  raw=$TMP/${b}__${name}.pkl
  echo "=== $b $name $(date +%T)" >> $LOG
  rclone copyto "$src" "$raw" >> $LOG 2>&1 || { echo "MISSING $b $name" >> $LOG; continue; }
  want=$(python -c "import json;print(json.load(open(r'$(cygpath -w $man)'))['raw_sha256'])")
  got=$(sha256sum "$raw" | cut -d' ' -f1)
  if [ "$want" = "$got" ]; then echo "SHA OK $b $name" >> $LOG; else echo "SHA MISMATCH $b $name $got vs $want" >> $LOG; fi
  echo "$b $name sha_ok=$([ "$want" = "$got" ] && echo 1 || echo 0)" >> $OUT/sha.txt
  python $WT/results/bench_gap_diagnosis_v1/readout_alternatives/extract_readouts.py "$(cygpath -w $raw)" "$(cygpath -w $out)" --bank "$(cygpath -w $bank)" --workers 7 >> $LOG 2>&1
  d=$(tail -5 $LOG | grep "^GATE" | awk '{print $3}')
  if [ -n "$d" ] && python -c "import sys;sys.exit(0 if float('$d')<=2.5e-6 else 1)"; then rm -f "$raw"; echo "DELETED raw $b $name" >> $LOG
  else echo "KEPT raw (gate not PASS) $b $name" >> $LOG; [ -f $out ] && mv $out $out.fail; fi
done
echo ALLDONE >> $LOG
