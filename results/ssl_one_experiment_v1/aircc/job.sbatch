#!/bin/bash
# One job of the one SSL experiment (RC_HANDOFF.md section D): BENCH in {processbench, prmbench}, V in {C, V1, V2, V3, V4}.
# V3/V4 (dev-only variants, 2026-11-18): the preflight C run uses the C rows of the parents of the first 12 V rows.
#   1. preflight: 12 rows of C (and of V for a V job) -> smoke_check.py; exit != 0 stops the job (no full run)
#      ProcessBench also: the bank's FEATURES must not depend on the grafted rewards (zero == PRM rewards, C rows)
#   2. full run -> $W/runs/<bench>_<V>/   (resumes from --out after preemption/requeue)
#   3. bank with the frozen builder of the stored standalone bank -> $W/banks/<bench>_<V>/FEATURES.npz
#   4. extract_steps.py -> $W/steps/STEPS_<bench>_<V>.npz (must print GATE PASS)
#   5. JOB_<bench>_<V>.json
# Everything runs inside the job's container (CPU partition start estimate was 2027-01 on 2026-11-18).
# Submit with submit.sh only. Steps are idempotent, so a requeued job continues where it stopped.
#SBATCH --gpus=1
#SBATCH -c 16
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --signal=B:TERM@900
#SBATCH --output=/shared/cycle2_tau_averbuch_prj/omrisegev1/logs/%x_%j.out
#SBATCH --container-image=nvcr.io/nvidia/pytorch:25.01-py3
#SBATCH --container-mounts=/shared/cycle2_tau_averbuch_prj/omrisegev1:/shared/cycle2_tau_averbuch_prj/omrisegev1
set -uo pipefail
: "${BENCH:?}" "${V:?}"
source /shared/cycle2_tau_averbuch_prj/omrisegev1/code_ssl_one_v1/results/ssl_one_experiment_v1/aircc/common.sh
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
step() {   # name cmd...: run in the background so SIGTERM reaches python; record exit code + seconds
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

# 1. preflight + smoke (skipped when an earlier attempt of this job already passed it)
if [ ! -e $P/SMOKE_PASS ]; then
  PREC=C; case $V in V3|V4) PREC=preC_$V ;; esac
  step pre_C drv $(input_args $BENCH $PREC) --n-samples 12 --out $P/${BENCH}_C || finish 10
  if [ $V != C ]; then step pre_$V drv $(input_args $BENCH $V) --n-samples 12 --out $P/${BENCH}_$V || finish 11; fi
  step smoke python -B $X/smoke_check.py --runs $P --variants $W/inputs \
       --stored-readouts $HD_ROOT/data_export/readout_steps_sa_v1 --out $P/SMOKE.json ${PREFIX_TOL:+--prefix-tol $PREFIX_TOL} || finish 12
  if [ $BENCH = processbench ]; then   # FEATURES must not depend on the dummy rewards
    PC=$P/${BENCH}_C/external_telemetry.pkl
    step graft_zero python -B $X/aircc/pb_graft.py $PC $P/graft_zero.pkl || finish 13
    step graft_prm python -B $X/aircc/pb_graft.py $PC $P/graft_prm.pkl --rewards-from $PB_PRM_RAW || finish 13
    TOK=$(snapshot $MODEL)/tokenizer.json
    for g in zero prm; do
      [ -e $P/bank_$g/FEATURES.npz ] || { [ -e $P/bank_$g ] && mv $P/bank_$g $P/bank_${g}.partial_$SLURM_JOB_ID
        step bank_$g bash -c "cd $BANK_CODE && python -B $PB_BUILD --raw $P/graft_$g.pkl --mask-code $PB_MASK --names-from $PB_NAMES --tokenizer $TOK --out $P/bank_$g --workers 16" || finish 13; }
    done
    step reward_independence python -B -c "
import json, sys, numpy as np
a, b = np.load('$P/bank_zero/FEATURES.npz'), np.load('$P/bank_prm/FEATURES.npz')
same = sorted(a.files) == sorted(b.files) and all(np.array_equal(a[k], b[k]) for k in a.files)
json.dump(dict(files=sorted(a.files), features_equal=bool(same)), open('$P/REWARD_INDEPENDENCE.json', 'w'))
print('REWARD INDEPENDENCE', 'PASS' if same else 'FAIL'); sys.exit(0 if same else 5)" || finish 14
  fi
  touch $P/SMOKE_PASS
fi

# 2. full run
step full drv $(input_args $BENCH $V) --out $RUN || finish 20
python -B -c "import json,sys; m=json.load(open('$RUN/manifest.json')); print('MANIFEST', m['status'], m['n_rows'], m['n_failed_rows']); sys.exit(0 if m['status']=='COMPLETE' and m['n_failed_rows']==0 else 1)" || finish 21

# 3. bank (frozen builders, unchanged)
RAW=$RUN/$(raw_name $BENCH); TOK=$(snapshot $MODEL)/tokenizer.json
if [ ! -e $BK/FEATURES.npz ]; then
  [ -e $BK ] && mv $BK $BK.partial_$SLURM_JOB_ID
  mkdir -p $BK
  if [ $BENCH = processbench ]; then
    step graft python -B $X/aircc/pb_graft.py $RAW $BK/grafted_zero_rewards.pkl || finish 30
    step bank bash -c "cd $BANK_CODE && python -B $PB_BUILD --raw $BK/grafted_zero_rewards.pkl --mask-code $PB_MASK --names-from $PB_NAMES --tokenizer $TOK --out $BK/builder --workers 16" || finish 31
    cp $BK/builder/FEATURES.npz $BK/FEATURES.npz.tmp && mv $BK/FEATURES.npz.tmp $BK/FEATURES.npz
  elif [ $V = C ]; then
    step bank bash -c "cd $BANK_CODE && DIGIT_TOKEN_IDS_PY=$PRMB_DIGITS python -B $PRMB_BUILD --raw $RAW --base $PRMB_BASE --mask-code $PRMB_MASK --tokenizer $TOK --out $BK/builder --workers 16" || finish 31
    cp $BK/builder/FEATURES.npz $BK/FEATURES.npz.tmp && mv $BK/FEATURES.npz.tmp $BK/FEATURES.npz
  else
    step bank_prepare python -B $X/aircc/prmb_variant_bank.py prepare $RAW $PRMB_FILL $PRMB_BASE $BK/builder_input.pkl $BK/MAP.json || finish 30
    step bank bash -c "cd $BANK_CODE && DIGIT_TOKEN_IDS_PY=$PRMB_DIGITS python -B $PRMB_BUILD --raw $BK/builder_input.pkl --base $PRMB_BASE --mask-code $PRMB_MASK --tokenizer $TOK --out $BK/builder --workers 16" || finish 31
    step bank_rename python -B $X/aircc/prmb_variant_bank.py rename $BK/builder/FEATURES.npz $BK/MAP.json $BK/FEATURES.npz.tmp.npz || finish 32
    mv $BK/FEATURES.npz.tmp.npz $BK/FEATURES.npz
  fi
fi

# 4. step table (gate: exact top10 == bank)
step extract python -B $X/extract_steps.py $RAW $BK/FEATURES.npz $W/steps/STEPS_$T.npz --workers 16 || finish 40
finish 0
