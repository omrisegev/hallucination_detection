# Offline install for the 2026-10-10 think-fix reruns. Compute nodes have no DNS, and pip's
# cache resolved pyarrow 26 (needs NumPy>=2; the NGC image ships 1.26.4) -> job 280417 died at
# `import datasets`. Pinned wheels, --no-deps, keep the container's torch/numpy/scipy/pandas.
WHEELS=/shared/cycle2_tau_averbuch_prj/omrisegev1/wheelhouse/tf_thinkfix_20261010
python -m pip install -q --user --no-deps --no-index --find-links="$WHEELS" \
  'transformers==4.57.6' 'tokenizers==0.22.2' 'safetensors==0.8.0' 'huggingface_hub==0.36.2' \
  'accelerate==1.15.0' 'datasets==5.0.1' 'pyarrow==25.0.1' 'dill==0.4.1' 'multiprocess==0.70.19' \
  'xxhash==4.0.1' 'fsspec==2026.6.0' 'httpx==0.28.1' 'httpcore==1.0.9' 'h11==0.16.0' 'anyio==4.15.1' \
  'sentencepiece==0.2.2' 'regex==2026.9.29' 'scikit_learn==1.9.1' 'threadpoolctl==3.7.0' 'joblib==1.6.0' \
  'aiohttp==3.14.4' 'aiosignal==1.4.0' 'frozenlist==1.8.0' 'multidict==6.9.1' 'yarl==1.25.1' \
  'propcache==0.5.4' 'aiohappyeyeballs==2.7.1' 'attrs==26.1.0' 'filelock==4.1.0' 'tqdm==4.70.1' \
  'requests==2.34.2' 'urllib3==2.8.0' 'certifi==2026.7.22' 'idna==3.20' 'charset_normalizer==3.5.2' \
  'packaging==26.3' 'pyyaml==6.0.3' 'typing_extensions==4.16.0' 'narwhals==2.27.1'
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python -c "import numpy, pyarrow, datasets, transformers, torch, sklearn.metrics, scipy, spectral_utils; print('[env] numpy', numpy.__version__, 'pyarrow', pyarrow.__version__, 'datasets', datasets.__version__, 'transformers', transformers.__version__, 'torch', torch.__version__)"
