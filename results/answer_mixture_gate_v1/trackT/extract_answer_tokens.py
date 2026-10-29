"""Answer length in tokens for ProcessBench x qwen_prm, from the raw (pre-mask) step_token_spans of the downloaded
external_telemetry.pkl (as score_chain_pb.py's step_log_tokens). Run with python -I. No labels.
    python -I -B extract_answer_tokens.py <external_telemetry.pkl> <out.npz>
"""
import pickle, sys
import numpy as np

raw = pickle.load(open(sys.argv[1], 'rb'))
uids, ntok, nst, nne = [], [], [], []
for e in raw.values():
    if 'failure' in e:
        continue
    sp = e['step_token_spans']
    uids.append(str(e['idx'])); ntok.append(sum(hi - lo for lo, hi in sp if hi > lo))
    nst.append(len(sp)); nne.append(sum(hi > lo for lo, hi in sp))
np.savez(sys.argv[2], uids=np.array(uids), answer_tokens=np.array(ntok), n_steps=np.array(nst), n_nonempty=np.array(nne))
print('answers', len(uids), 'failed', sum('failure' in e for e in raw.values()), 'min tokens', min(ntok))
