#!/bin/bash
# Run on the Windows machine from the repo root (Git Bash): sends ONLY label-free inputs and the aircc/ code to AIRCC.
# Use a checkout with LF line endings: pb_down.sh compares the shipped *.repo.py byte for byte with the AIRCC copies.
# local/ (gold labels, evaluator) is never sent. Verifies sha256 on both ends against inputs/INPUT_MANIFEST.json.
#   bash results/prm_boost_processbench_v1/ship.sh [ssh_host]
set -euo pipefail
H=${1:-aircc}; S=/shared/cycle2_tau_averbuch_prj/omrisegev1
O=$S/results/prm_boost_processbench_v1; DATA=$S/data/prm_boost_processbench_v1
B=results/prm_boost_processbench_v1
for f in answers.json answers_timing12.json; do
  want=$(grep -A4 '"sha256"' $B/inputs/INPUT_MANIFEST.json | grep "\"$f\"" | sed 's/.*: "\([0-9a-f]*\)".*/\1/')
  got=$(sha256sum $B/inputs/$f | cut -d' ' -f1)
  [ "$want" = "$got" ] || { echo "FAIL local sha $f: $got != manifest $want" >&2; exit 1; }
  echo "local sha OK $f $got"
done
if grep -q '"label"' $B/inputs/answers.json; then echo "FAIL: answers.json contains a label field" >&2; exit 1; fi
stage=$(mktemp -d)
cp $B/aircc/* "$stage/"
# PREREGISTRATION_HE.md section 8: the v2 builder, its digit-id helper and the guarded score_external, under NEW
# names. The old build_bank_external.repo.py / score_external.repo.py already on AIRCC are not re-sent (not overwritten).
cp results/prm_boost_external_v1/build_bank_external_v2.py "$stage/build_bank_external_v2.repo.py"
cp spectral_utils/digit_token_ids.py "$stage/digit_token_ids.repo.py"
cp results/prm_boost_external_v1/score_external_v2.py "$stage/score_external_v2.repo.py"
git rev-parse HEAD > "$stage/SHIPPED_COMMIT"
ssh "$H" "mkdir -p $O/code $DATA"
tar cf - -C "$stage" . | ssh "$H" "tar xf - -C $O/code && chmod +x $O/code/*.sh"
tar cf - -C $B/inputs answers.json answers_timing12.json INPUT_MANIFEST.json | ssh "$H" "tar xf - -C $DATA"
ssh "$H" "cd $DATA && sha256sum answers.json answers_timing12.json && cat $O/code/SHIPPED_COMMIT && ls $O/code"
rm -rf "$stage"
echo "compare the two remote sha256 lines with the local ones above"
