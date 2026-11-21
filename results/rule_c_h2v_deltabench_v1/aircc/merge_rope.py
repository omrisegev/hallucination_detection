"""DeltaBench: merge a native-context run and a dynamic-NTK RoPE x3 run of the SAME answers, as the stored ext cell
(merge_longctx.py, 2026-11-08): keep the native row when it exists, else the RoPE row. No labels.
    python merge_rope.py NATIVE_DIR ROPE_DIR OUT_DIR"""
import json, pickle, sys, os
nat, rope, out = sys.argv[1:4]
A = pickle.load(open(os.path.join(nat, 'external_telemetry.pkl'), 'rb')); B = pickle.load(open(os.path.join(rope, 'external_telemetry.pkl'), 'rb'))
assert sorted(A) == sorted(B)
M = {}; n_ext = 0; fails = {}
for k in A:
    e = A[k]
    if 'failure' in e and e['failure'] == 'context_overflow':
        e = B[k]; n_ext += 1
    if 'failure' in e: fails[str(e.get('idx', k))] = e['failure']
    M[k] = e
os.makedirs(out, exist_ok=True)
pickle.dump(M, open(os.path.join(out, 'external_telemetry.pkl.tmp'), 'wb')); os.replace(os.path.join(out, 'external_telemetry.pkl.tmp'), os.path.join(out, 'external_telemetry.pkl'))
ma = json.load(open(os.path.join(nat, 'manifest.json'))); mb = json.load(open(os.path.join(rope, 'manifest.json')))
m = dict(driver='merge_rope.py', sources=[nat, rope], n_rows=len(M), n_rows_from_extension=n_ext, n_failed_rows=len(fails), failed_rows=fails,
         status='COMPLETE' if not fails else 'COMPLETE_WITH_FAILURES', elapsed_sec=ma['elapsed_sec'] + mb['elapsed_sec'],
         model_revision=ma.get('model_revision'), rope_extension=mb.get('rope_extension'), max_input_tokens=max(ma.get('max_input_tokens') or 0, mb.get('max_input_tokens') or 0))
json.dump(m, open(os.path.join(out, 'manifest.json'), 'w'), indent=1); print('MERGE', json.dumps({k: v for k, v in m.items() if k != 'failed_rows'}))
