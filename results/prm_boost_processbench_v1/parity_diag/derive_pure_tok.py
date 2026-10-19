"""Write PURE-PRM's own tokenizer (vocab.json+merges.txt in its HF snapshot) as tokenizer.json, for the digit-id check."""
import glob, json, sys
from transformers import AutoTokenizer
H = '/shared/cycle2_tau_averbuch_prj/omrisegev1/hf_cache/hub'
snap = glob.glob(f'{H}/models--jinachris--Qwen2.5-Math-7B-PRM800K/snapshots/*/')[0]
out = sys.argv[1]
tok = AutoTokenizer.from_pretrained(snap, use_fast=True)
tok.save_pretrained(out)
ref = json.load(open(glob.glob(f'{H}/models--Qwen--Qwen2.5-Math-7B-Instruct/snapshots/*/vocab.json')[0]))
own = json.load(open(snap + 'vocab.json'))
print('snapshot', snap)
print('digit ids', [tok.convert_tokens_to_ids(str(d)) for d in range(10)])
print('vocab.json identical to Qwen2.5-Math-7B-Instruct:', own == ref, len(own))
