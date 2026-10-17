set -e
SH=/shared/cycle2_tau_averbuch_prj/omrisegev1
WH=$SH/wheelhouse/native_312_20261011
ENV=$SH/pyenv_native_312_20261011
test ! -e $ENV
mkdir -p $ENV/site
# Wheels are zip archives; unpacking into one site dir on PYTHONPATH is equivalent to a --no-deps
# install for import purposes (no console scripts needed). System python3.12 on the CPU nodes.
for w in $WH/*.whl; do python3 -m zipfile -e "$w" $ENV/site/; done
ls $WH > $ENV/WHEELS.txt
sha256sum $WH/*.whl > $ENV/WHEELS.sha256
cat > $ENV/activate.sh <<'EOF'
export PYTHONPATH=/shared/cycle2_tau_averbuch_prj/omrisegev1/pyenv_native_312_20261011/site${PYTHONPATH:+:$PYTHONPATH}
export PYTHONNOUSERSITE=1
EOF
du -sh $ENV
