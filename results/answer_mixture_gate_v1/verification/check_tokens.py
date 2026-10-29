"""Spot-check PB_QWEN_ANSWER_TOKENS.npz against the raw telemetry pickle (own code). Run with python -I."""
import json, pickle, sys
import numpy as np
raw = pickle.load(open('/Users/osegev/Desktop/hd_collection_data/processbench_raw/qwen_prm/external_telemetry.pkl', 'rb'))
T = np.load('/Users/osegev/Desktop/hallucination_detection/results/answer_mixture_gate_v1/trackT/PB_QWEN_ANSWER_TOKENS.npz')
tok = dict(zip(T['uids'].astype(str), T['answer_tokens'].astype(int)))
keys = list(raw.keys()); ex = raw[keys[0]]
info = {'n_records': len(raw), 'record_keys': sorted(map(str, ex.keys())), 'n_failure': sum('failure' in e for e in raw.values())}
mism, n = 0, 0
label_like = [k for k in ex.keys() if any(s in str(k).lower() for s in ('label', 'gold', 'correct', 'error'))]
for e in raw.values():
    if 'failure' in e: continue
    u = str(e['idx']); sp = e['step_token_spans']
    t = int(sum(max(int(hi) - int(lo), 0) for lo, hi in sp))
    n += 1; mism += int(tok.get(u, -1) != t)
info.update(n_checked=n, n_mismatch=mism, npz_n=len(tok), label_like_keys_in_record=label_like,
            min_tokens=int(min(tok.values())))
print(json.dumps(info, indent=1))
json.dump(info, open('/Users/osegev/Desktop/hallucination_detection/results/answer_mixture_gate_v1/verification/check_tokens.json', 'w'), indent=1)
