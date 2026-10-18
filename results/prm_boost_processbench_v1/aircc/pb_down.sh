#!/bin/bash
# CPU downstream for every PRM, label-free: bank (frozen rebuild_source_bank_v4 via build_bank_external.py, mask1,
# empty steps dropped) -> frozen arms (score_external.py) -> CHAIN/TELE/PRMG for N13 and N16 (score_chain_pb.py).
# Mirrors results/prm_boost_external_v1/ext_down.sh. Stops on the first failure. No labels exist on AIRCC.
set -uo pipefail; source "$(dirname "$0")/env.sh"
# The AIRCC copies must be the repo versions that produced Socratic bank_v2/SCORES (shipped in code/ as *.repo.py).
cmp $BANK_CODE/scripts/prm_boost_build_bank_external.py $O/code/build_bank_external.repo.py || { echo "FAIL: bank builder differs" >&2; exit 1; }
cmp $MIRROR/prm_boost_score_external.py $O/code/score_external.repo.py || { echo "FAIL: score_external differs" >&2; exit 1; }
cp $O/code/score_chain_pb.py $MIRROR/prm_boost_score_chain_pb.py
RUN=${RUN:-full}; SC=$O/scores; [ "$RUN" = timing ] && SC=$O/scores_timing   # timing = pipeline smoke only
mkdir -p $SC
for a in ${ONLY:-$ADAPTERS}; do
  d=$O/${RUN}_${a}
  python3 - "$d/manifest.json" <<'PY' || exit 1
import json, sys
m = json.load(open(sys.argv[1]))
bad = {k: v.get('failure') for k, v in m['failed_rows'].items() if v.get('failure') != 'context_overflow'}
ok = m['status'] in ('COMPLETE', 'COMPLETE_WITH_FAILURES') and not bad
print(sys.argv[1], m['status'], 'rows', m['n_rows'], 'failed', m['n_failed_rows'], 'overflow', m['n_rows_context_overflow'],
      'step-text mismatch rows', m['n_rows_step_text_mismatch'])
sys.exit(0 if ok else 1)
PY
  [ -e $d/bank_v2/FEATURES.npz ] || (cd $BANK_CODE && python3 -B scripts/prm_boost_build_bank_external.py --raw $d/external_telemetry.pkl \
      --mask-code $MASK_CODE --names-from $NAMES_FROM --out $d/bank_v2 --workers ${SLURM_CPUS_PER_TASK:-8}) || exit 1
  (cd $MIRROR && python3 -B $MIRROR/prm_boost_score_external.py $d/bank_v2 $SC/SCORES_${a}.npz) || exit 1
  (cd $MIRROR && python3 -B $MIRROR/prm_boost_score_chain_pb.py $d/bank_v2 $SC/SCORES_CHAIN_${a}.npz) || exit 1
done
echo "DOWNSTREAM DONE"
