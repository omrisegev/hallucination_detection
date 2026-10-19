#!/bin/bash
# CPU downstream for every PRM, label-free: bank (v2 builder prm_boost_build_bank_external_v2.py --tokenizer, mask1,
# empty steps dropped, digit ids from the backbone's tokenizer) -> frozen arms with guarded L-SML
# (score_external_v2.py) -> CHAIN/TELE/PRMG + MEAN/HALF/EQ for N13 and N16, POS_ONLY, LEN_ONLY (score_chain_pb.py).
# PREREGISTRATION_HE.md section 8. Mirrors results/prm_boost_external_v1/ext_down.sh. Stops on the first failure.
# No labels exist on AIRCC.
set -uo pipefail; source "$(dirname "$0")/env.sh"
# The v2 files are shipped in code/ as *.repo.py. They get NEW names on AIRCC: installed when absent, otherwise they
# must be byte-identical (job 288932 already installed the builder and digit_token_ids.py). The old frozen
# builder and score_external copies are not touched.
install_or_cmp $O/code/build_bank_external_v2.repo.py $NEW_BUILDER || exit 1
install_or_cmp $O/code/digit_token_ids.repo.py $BANK_CODE/scripts/digit_token_ids.py || exit 1
install_or_cmp $O/code/score_external_v2.repo.py $MIRROR/prm_boost_score_external_v2.py || exit 1
(cd $MIRROR && python3 -B -c 'from spectral_utils.fusion_utils import lsml_continuous_guarded') \
  || { echo "FAIL: $MIRROR/spectral_utils/fusion_utils.py has no lsml_continuous_guarded (appended in job 288932)" >&2; exit 1; }
cp $O/code/score_chain_pb.py $MIRROR/prm_boost_score_chain_pb.py
RUN=${RUN:-full}; SC=$O/scores; [ "$RUN" = timing ] && SC=$O/scores_timing   # timing = pipeline smoke only
if [ "$RUN" = full ]; then   # section 8 step 5: bank parity (bank_parity.sh) must have passed on the ProcessBench cells
  python3 - $O/parity/PARITY_SUMMARY.json $PARITY_ADAPTERS <<'PY' || exit 1
import json, os, sys
p = sys.argv[1]
if not os.path.exists(p): sys.exit(f'FAIL: {p} missing; run bank_parity.sh (submit_parity.sh) first')
s = json.load(open(p)); need = [f'{k}_{a}' for a in sys.argv[2:] for k in ('socratic', 'pb')]
miss = [n for n in need if n not in s['pairs']]; bad = [n for n, r in s['pairs'].items() if not r['ok']]
print('parity', p, 'ok' if s['ok'] and not miss else 'FAIL', 'missing', miss, 'failed', bad)
sys.exit(0 if s['ok'] and not miss and not bad else 1)
PY
fi
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
  tk=$(tok_for $a); check_digits $a "$tk" || exit 1
  [ -e $d/$BANK_DIR/FEATURES.npz ] || (cd $BANK_CODE && python3 -B scripts/prm_boost_build_bank_external_v2.py --raw $d/external_telemetry.pkl \
      --mask-code $MASK_CODE --names-from $NAMES_FROM --tokenizer "$tk" --out $d/$BANK_DIR --workers ${SLURM_CPUS_PER_TASK:-8}) || exit 1
  python3 - $d/$BANK_DIR/MANIFEST.json "$(digits_for $a)" <<'PY' || exit 1
import json, sys
m = json.load(open(sys.argv[1])); got = (m.get('digit_ids') or {}).get('digit_ids')
ok = got == [int(x) for x in sys.argv[2].split(',')]
print(sys.argv[1], 'digit ids', got, 'OK' if ok else 'FAIL (not a v2-builder bank with this tokenizer)')
sys.exit(0 if ok else 1)
PY
  (cd $MIRROR && python3 -B $MIRROR/prm_boost_score_external_v2.py $d/$BANK_DIR $SC/SCORES_${a}.npz) || exit 1
  (cd $MIRROR && python3 -B $MIRROR/prm_boost_score_chain_pb.py $d/$BANK_DIR $SC/SCORES_CHAIN_${a}.npz $d/external_telemetry.pkl) || exit 1
done
echo "DOWNSTREAM DONE"
