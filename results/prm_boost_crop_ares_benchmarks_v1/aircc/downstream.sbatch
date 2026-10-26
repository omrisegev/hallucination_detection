#!/bin/bash
# CPU, label-free: telemetry -> 56-channel bank -> ablation5 arms, for every finished full run of this experiment.
#   P0 parity first: rebuild the frozen Socratic qwen_prm bank + scores with this exact pipeline; must match exactly.
#   prm cells: bank from full_prm_<a>_<b>/external_telemetry.pkl.
#   sa cells:  graft the same answers' PRM rewards onto the standalone pkl (graft_rewards.py), then the same bank.
#              Report the telemetry-only arms (TELE_*, MEAN_*) for the "L-SML on token features only" tables.
# Digit token ids come from the standalone model's tokenizer.json (same tokenizer family as its PRM).
#   sbatch [--export=ALL,ONLY_MODES="prm sa",ONLY_PRMS="qwen_prm",ONLY_BENCHES="crv_gsm8k"] downstream.sbatch
#SBATCH -J cab_downstream
#SBATCH -p cpu
#SBATCH -A cycle3_tau_averbuch_prj
#SBATCH --qos=owner_940
#SBATCH --mem=128G
#SBATCH -c 16
#SBATCH -t 24:00:00
#SBATCH -o /shared/cycle2_tau_averbuch_prj/omrisegev1/logs/cab_downstream_%j.out
set -euo pipefail
source /shared/cycle2_tau_averbuch_prj/omrisegev1/results/prm_boost_crop_ares_benchmarks_v1/code/env.sh
set +u; . $S/pyenv_native_312_20261011/activate.sh; set -u
export ABL5_EXPORT_SCORES=1
K=$O/code; B2=$BANK_CODE/scripts/prm_boost_build_bank_external_v2.py; A5=$MIRROR/results/prm_boost_ablation5_v1/ablation5_socratic.py
SC=$O/scores; mkdir -p $SC
declare -A PRM_RAW_OLD=( [processbench]=$R/prm_boost_processbench_v1/full_%s/external_telemetry.pkl
                         [socratic]=$R/prm_boost_external_v1/full_%s_socratic/external_telemetry.pkl
                         [hard2verify]=$R/prm_boost_external_v1/full_%s_hard2verify/external_telemetry.pkl )

bank() {   # raw tokenizer out
  [ -e $3/FEATURES.npz ] && { echo "bank exists: $3"; return 0; }
  (cd $BANK_CODE && python3 -B $B2 --raw $1 --mask-code $MASK_CODE --names-from $NAMES_FROM --tokenizer $2 --out $3 \
     --workers ${SLURM_CPUS_PER_TASK:-16})
}
complete_ok() { python3 -c "import json,sys; m=json.load(open('$1')); bad={k:v.get('failure') for k,v in m['failed_rows'].items() if v.get('failure')!='context_overflow'}; print('$1', m['status'], m['n_rows'], 'failed', m['n_failed_rows'], 'overflow', m.get('n_rows_context_overflow')); sys.exit(0 if m['status'] in ('COMPLETE','COMPLETE_WITH_FAILURES') and not bad else 1)"; }

if [ ! -e $O/PARITY_P0.json ]; then
  echo "[P0] rebuild Socratic qwen_prm bank + scores"
  TQ=$(snapshot ${SA_MODEL[qwen_prm]})/tokenizer.json; P=$O/parity_p0
  bank $R/prm_boost_external_v1/full_qwen_prm_socratic/external_telemetry.pkl $TQ $P/bank
  (cd $MIRROR && python3 -B $A5 $P/bank $P/SCORES_qwen_prm.npz)
  python3 - $P/bank/FEATURES.npz $R/prm_boost_external_v1/full_qwen_prm_socratic/bank_v2/FEATURES.npz \
            $P/SCORES_qwen_prm.npz $R/prm_boost_external_v1/collect_socratic_v1/SCORES_qwen_prm.npz $O/PARITY_P0.json <<'PY'
import json, sys, numpy as np
fa, fb, sa, sb, out = sys.argv[1:6]
rep = {}
for tag, x, y in (('bank', fa, fb), ('scores', sa, sb)):
    A, B = np.load(x, allow_pickle=True), np.load(y, allow_pickle=True)
    common = sorted(set(A.files) & set(B.files)); gaps = {}
    for k in common:
        a, b = A[k], B[k]
        if a.dtype.kind in 'fc' and a.shape == b.shape: gaps[k] = float(np.nanmax(np.abs(a - b), initial=0))
        else: gaps[k] = 0.0 if (a.shape == b.shape and (a == b).all()) else 'DIFF'
    rep[tag] = {'common_keys': len(common), 'only_new': sorted(set(A.files) - set(B.files)),
                'only_frozen': sorted(set(B.files) - set(A.files)), 'gaps': gaps}
ok = all(v == 0.0 for t in rep.values() for v in t['gaps'].values()) and rep['scores']['common_keys'] > 0
rep['status'] = 'PASS' if ok else 'FAIL'
json.dump(rep, open(out, 'w'), indent=1); print('P0', rep['status'], {t: {k: v for k, v in r['gaps'].items() if v != 0.0} for t, r in rep.items() if t != 'status'})
sys.exit(0 if ok else 3)
PY
fi
echo "P0 PASS"

for mode in ${ONLY_MODES:-prm sa}; do
  benches=$NEW_BENCHES; [ $mode = sa ] && benches="$NEW_BENCHES $SA_ONLY_BENCHES"
  for a in ${ONLY_PRMS:-$ADAPTERS}; do
    tok=$(snapshot ${SA_MODEL[$a]})/tokenizer.json
    for b in ${ONLY_BENCHES:-$benches}; do
      d=$O/full_${mode}_${a}_${b}; out=$SC/SCORES_${mode}_${a}_${b}.npz
      [ -e $out ] && { echo "done: $out"; continue; }
      [ -e $d/manifest.json ] || { echo "SKIP (no full run yet): $d"; continue; }
      complete_ok $d/manifest.json || { echo "SKIP (not complete): $d"; continue; }
      raw=$d/external_telemetry.pkl
      if [ $mode = sa ]; then
        if [[ " $SA_ONLY_BENCHES " == *" $b "* ]]; then praw=$(printf "${PRM_RAW_OLD[$b]}" $a); else praw=$O/full_prm_${a}_${b}/external_telemetry.pkl; fi
        [ -s $praw ] || { echo "SKIP (no PRM rewards yet): $praw"; continue; }
        [ -e $d/grafted.pkl ] || python3 -B $K/graft_rewards.py $raw $praw $d/grafted.pkl
        raw=$d/grafted.pkl
      fi
      echo "[bank] $mode $a $b"; bank $raw $tok $d/bank_v3
      echo "[score] $mode $a $b"; (cd $MIRROR && python3 -B $A5 $d/bank_v3 $out)
    done
  done
done
( cd $SC && md5sum ./*.npz > $O/MD5_scores.txt ) 2>/dev/null || echo "no scores yet"
echo CAB_DOWNSTREAM_DONE
