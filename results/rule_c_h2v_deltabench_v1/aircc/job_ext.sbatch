#!/bin/bash
# One job of the rule C transfer (2026-11-21): BENCH in {hard2verify, deltabench}, V in {C, V3, V4}.
# Same steps as results/ssl_one_experiment_v1/aircc/job.sbatch (ProcessBench path), cycle4 paths:
#   V = C : step table of the STORED standalone C cell (extract_steps.py on the stored raw + bank_v3); hard2verify also
#           checks the builder environment (bank rebuilt from the stored grafted.pkl == stored bank_v3).
#   V = V3/V4 : 1. preflight 12 rows of preC_V (parents) and of V -> smoke_check_ext.py (rules unchanged); exit != 0 stops
#               2. full run (deltabench: native pass + RoPE x3 pass, merged as the stored ext cell)
#               3. zero-reward graft + external bank builder v2 (unchanged)  4. extract_steps.py (GATE PASS)  5. JOB json
# Deviation (reported): a full V run may end COMPLETE_WITH_FAILURES only for context_overflow rows (H2V has no RoPE run;
# its stored C cell has the same failures); any other failure stops the job.
#SBATCH --gpus=1
#SBATCH -c 16
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --requeue
#SBATCH --signal=B:TERM@900
#SBATCH --output=/shared/cycle4_tau_averbuch_spot_prj/users/omrisegev1/ws_c4/logs/%x_%j.out
#SBATCH --container-image=nvcr.io/nvidia/pytorch:25.01-py3
#SBATCH --container-mounts=/shared/cycle4_tau_averbuch_spot_prj/users/omrisegev1:/shared/cycle4_tau_averbuch_spot_prj/users/omrisegev1
set -uo pipefail
: "${BENCH:?}" "${V:?}"
source /shared/cycle4_tau_averbuch_spot_prj/users/omrisegev1/ws_c4/code_rule_c_v1/results/rule_c_h2v_deltabench_v1/aircc/common_c4.sh
export HD_ROOT HF_HOME=$S/hf_cache HF_DATASETS_CACHE=$S/hf_cache/datasets HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
       TRANSFORMERS_OFFLINE=1 PIP_CACHE_DIR=$S/pip_cache PYTHONUSERBASE=/tmp/pyuserbase_${SLURM_JOB_ID} \
       PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$PYTHONUSERBASE"
T=${BENCH}_${V}; J=$W/jobs/$T; P=$W/pre/$T; RUN=$W/runs/$T; BK=$W/banks/$T
mkdir -p $J $P $W/runs $W/banks $W/steps
cd $HD_ROOT
echo "[job] $SLURM_JOB_ID restart ${SLURM_RESTART_COUNT:-0} on $(hostname) $T $(date -u +%FT%TZ)"
python -m pip install -q --user --no-deps $S/wheelhouse/tf_thinkfix_20261010/*.whl   # compute nodes are offline
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader > $J/gpu.txt; cat $J/gpu.txt
( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> $J/gpumem.log; sleep 20; done ) &
MON=$!
CHILD=
trap '[ -n "$CHILD" ] && kill -TERM $CHILD' TERM
step() {
  local name=$1 t0=$(date +%s) rc; shift
  echo "[step] $name: $*"
  "$@" & CHILD=$!; wait $CHILD; rc=$?
  if kill -0 $CHILD 2>/dev/null; then wait $CHILD; rc=$?; fi
  CHILD=
  printf '%s\t%s\t%s\t%s\n' "$name" "$rc" "$(( $(date +%s) - t0 ))" "$(date -u +%FT%TZ)" >> $J/steps.tsv
  echo "[step] $name exit $rc"; return $rc
}
finish() { kill $MON 2>/dev/null; python -B $X/aircc/job_record.py $W $BENCH $V $J; echo "[job] exit $1"; exit $1; }
drv() { python cluster/run_prmbench_prm_telemetry.py $DRIVER_ARGS "$@"; }
drv_rope() { python cluster/run_prmbench_prm_telemetry_longctx.py $DRIVER_ARGS --rope-dynamic-factor $ROPE_FACTOR "$@"; }
run_any() {   # name input out [n]: hard2verify native only; deltabench native + RoPE pass merged
  local name=$1 inp=$2 out=$3 n=${4:-}
  if [ $BENCH = deltabench ]; then
    step ${name}_native drv $inp ${n:+--n-samples $n} --out ${out}_native || return 1
    step ${name}_rope drv_rope $inp ${n:+--n-samples $n} --out ${out}_rope || return 1
    step ${name}_merge python -B $Y/aircc/merge_rope.py ${out}_native ${out}_rope $out || return 1
  else
    step $name drv $inp ${n:+--n-samples $n} --out $out || return 1
  fi
}

# 0. model revision (must be the stored runs' snapshot)
[ "$(cat $H/models--Qwen--Qwen2.5-Math-7B-Instruct/refs/main)" = $REV ] || { echo "model revision mismatch"; finish 9; }

if [ $V = C ]; then
  RAW=$STORED/full_sa_qwen_prm_${BENCH}$([ $BENCH = deltabench ] && echo _ext)/grafted.pkl
  SB=$(dirname $RAW)/bank_v3/FEATURES.npz
  if [ $BENCH = hard2verify ] && [ ! -e $J/BANK_PARITY_PASS ]; then
    step bank_parity_build bash -c "cd $BANK_CODE && python -B $PB_BUILD --raw $RAW --mask-code $PB_MASK --names-from $PB_NAMES --tokenizer $TOK --out $P/bank_parity --workers 16" || finish 13
    step bank_parity python -B $Y/aircc/bank_parity.py $P/bank_parity/FEATURES.npz $SB || finish 14
    touch $J/BANK_PARITY_PASS
  fi
  step extract python -B $X/extract_steps.py $RAW $SB $W/steps/STEPS_$T.npz --workers 16 || finish 40
  finish 0
fi

# 1. preflight + smoke
if [ ! -e $P/SMOKE_PASS ]; then
  run_any pre_C "$(input_args $BENCH preC_$V)" $P/${BENCH}_C 12 || finish 10
  run_any pre_$V "$(input_args $BENCH $V)" $P/${BENCH}_$V 12 || finish 11
  step smoke python -B $Y/aircc/smoke_check_ext.py --runs $P --variants $W/inputs \
       --stored-readouts $HD_ROOT/data_export/readout_steps_sa_v1 --out $P/SMOKE.json || finish 12
  touch $P/SMOKE_PASS
fi

# 2. full run
run_any full "$(input_args $BENCH $V)" $RUN || finish 20
python -B -c "
import json,sys; m=json.load(open('$RUN/manifest.json')); f=m.get('failed_rows') or {}
bad=[k for k,v in f.items() if (v if isinstance(v,str) else v.get('failure','?'))!='context_overflow']
print('MANIFEST', m['status'], m['n_rows'], m['n_failed_rows'], 'non-overflow failures', len(bad)); sys.exit(0 if m['status'].startswith('COMPLETE') and not bad else 1)" || finish 21

# 3. bank (zero-reward graft + frozen external builder, unchanged)
RAW=$RUN/external_telemetry.pkl
if [ ! -e $BK/FEATURES.npz ]; then
  [ -e $BK ] && mv $BK $BK.partial_$SLURM_JOB_ID
  mkdir -p $BK
  step graft python -B $X/aircc/pb_graft.py $RAW $BK/grafted_zero_rewards.pkl || finish 30
  step bank bash -c "cd $BANK_CODE && python -B $PB_BUILD --raw $BK/grafted_zero_rewards.pkl --mask-code $PB_MASK --names-from $PB_NAMES --tokenizer $TOK --out $BK/builder --workers 16" || finish 31
  cp $BK/builder/FEATURES.npz $BK/FEATURES.npz.tmp && mv $BK/FEATURES.npz.tmp $BK/FEATURES.npz
fi

# 4. step table (gate: exact top10 == bank)
step extract python -B $X/extract_steps.py $RAW $BK/FEATURES.npz $W/steps/STEPS_$T.npz --workers 16 || finish 40
finish 0
