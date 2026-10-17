set -e
SH=/shared/cycle2_tau_averbuch_prj/omrisegev1
W=$SH/wheelhouse/tf_thinkfix_20261010
mkdir -p $W
PY=$(command -v python3)
echo "login python: $($PY --version)"
# Download for the container's interpreter (cp312, manylinux x86_64); resolution against the
# container's numpy 1.26.4 is enforced at install time by the constraint below.
$PY -m pip download -q -d $W --only-binary=:all: --python-version 3.12 --platform manylinux2014_x86_64 \
  --platform manylinux_2_17_x86_64 --platform manylinux_2_28_x86_64 --implementation cp \
  'transformers>=4.51,<5' sentencepiece accelerate 'datasets' 'pyarrow==25.0.1' scikit-learn huggingface_hub \
  'numpy==1.26.4' 'scipy'
ls $W | wc -l
ls $W | grep -E 'pyarrow|datasets|transformers|numpy'
