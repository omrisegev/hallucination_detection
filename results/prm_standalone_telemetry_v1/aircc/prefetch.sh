#!/bin/bash
# LOGIN NODE (compute nodes are offline). Downloads the standalone checkpoints into $S/hf_cache, then lists what is there.
# Llama-3.1-8B-Instruct is gated: export HF_TOKEN=<a token with access> first. Qwen2.5-Math-72B-Instruct is about 145 GB:
# check free space first (HANDOFF step 1).
set -euo pipefail
source "$(dirname "$0")/common.sh"
set +u; . $S/pyenv_native_312_20261011/activate.sh; set -u
export HF_HOME=$S/hf_cache
for p in $PRMS_SA; do
  m=${SA_MODEL[$p]}
  if snapshot "$m" >/dev/null 2>&1; then echo "cached $m -> $(snapshot "$m")"; continue; fi
  echo "downloading $m"
  python3 - "$m" <<'EOF'
import sys
from huggingface_hub import snapshot_download
print(snapshot_download(sys.argv[1], allow_patterns=['*.json', '*.safetensors', 'tokenizer*', '*.model', '*.jinja', 'merges.txt', 'vocab.json']))
EOF
done
for p in $PRMS_SA; do d=$(snapshot "${SA_MODEL[$p]}"); echo "$p ${SA_MODEL[$p]} $(du -sh "$d"/ | cut -f1) $(ls "$d" | tr '\n' ' ')"; done
echo PREFETCH_DONE
