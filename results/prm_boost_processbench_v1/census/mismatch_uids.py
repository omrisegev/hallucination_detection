"""Diagnostic: list rlhflow_prm ProcessBench rows whose adapter build reports a step-text mismatch."""
import json, sys
from transformers import AutoTokenizer
from spectral_utils.prm_adapters import get_adapter
ad = get_adapter('rlhflow_prm')
tok = AutoTokenizer.from_pretrained(ad.model_id, trust_remote_code=True)
rows = json.load(open(sys.argv[1], encoding='utf8'))
out = []
for r in rows:
    b = ad.build(tok, r['question'], r['steps'])
    if b['n_step_text_mismatch'] > 0:
        out.append({'uid': r['uid'], 'benchmark': r.get('benchmark'), 'n_steps': len(r['steps']),
                    'n_mismatch_steps': int(b['n_step_text_mismatch']),
                    'n_boundary_whitespace': int(b.get('n_boundary_whitespace', -1)),
                    'other_keys': sorted(k for k in b if k not in ('input_ids',))})
by = {}
for o in out: by[o['benchmark']] = by.get(o['benchmark'], 0) + 1
json.dump({'n_rows': len(out), 'by_subset': by, 'rows': out}, open(sys.argv[2], 'w'), indent=1)
print(len(out), by)
