#!/bin/bash
# Fetch the public sources for make_inputs.py into $1 (default ./src). Run on a machine with internet.
set -euo pipefail; D=${1:-src}; mkdir -p $D/crv/arithmetic_expressions $D/crv/gsm8k_expressions $D/math_shepherd $D/prm800k $D/deltabench
HF=https://huggingface.co/datasets
for n in 3 5 7 10; do curl -sfL -o $D/crv/arithmetic_expressions/arith.nt$n.annotated.json $HF/facebook/crv/resolve/main/arithmetic_expressions/arith.nt$n.annotated.json; done
curl -sfL -o $D/crv/gsm8k_expressions/gsm8k.annotated.json $HF/facebook/crv/resolve/main/gsm8k_expressions/gsm8k.annotated.json
curl -sL $HF/peiyi9979/Math-Shepherd/resolve/main/math-shepherd.jsonl | head -n 10000 > $D/math_shepherd/math-shepherd.first10000.jsonl || true
G=https://media.githubusercontent.com/media/openai/prm800k/main/prm800k/data
curl -sfL -o $D/prm800k/phase2_test.jsonl $G/phase2_test.jsonl
curl -sL $G/phase2_train.jsonl | head -n 12000 > $D/prm800k/phase2_train.first12000.jsonl || true
curl -sfL -o $D/deltabench/Deltabench_v1.csv $HF/OpenStellarTeam/DeltaBench/resolve/main/Deltabench_v1.csv
wc -l $D/math_shepherd/*.jsonl $D/prm800k/*.jsonl
