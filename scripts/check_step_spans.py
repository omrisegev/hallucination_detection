"""Assert: thinkfix step spans == old (same text, same tokenizer), and mask1 spans == old shifted by the
mask1 rule ([lo,hi) -> [max(lo-1,0), hi-1]). Writes SPANS_CHECK.json; exits non-zero on any mismatch."""
import json, sys
from pathlib import Path
import numpy as np
d = Path(sys.argv[1])
z = {v: np.load(d/f'STEP_TOKEN_SPANS_{v.upper()}.npz') for v in ('old', 'thinkfix', 'thinkfix_mask1')}
o, t, m = (z[v]['step_spans'] for v in ('old', 'thinkfix', 'thinkfix_mask1'))
assert all(np.array_equal(z['old']['offsets'], z[v]['offsets']) for v in z)
exp = o.copy(); mapped = o[:, 0] >= 0
exp[mapped, 0] = np.maximum(o[mapped, 0]-1, 0); exp[mapped, 1] = o[mapped, 1]-1
res = {'thinkfix_equals_old': bool(np.array_equal(o, t)), 'thinkfix_mismatched_steps': int((o != t).any(1).sum()),
       'mask1_equals_old_shifted': bool(np.array_equal(exp, m)), 'mask1_mismatched_steps': int((exp != m).any(1).sum()),
       'steps': int(len(o)), 'step0_length_change_mask1': int(((o[:, 1]-o[:, 0]) - (m[:, 1]-m[:, 0])).sum())}
(d/'SPANS_CHECK.json').write_text(json.dumps(res, indent=1))
print(res)
sys.exit(0 if res['thinkfix_equals_old'] and res['mask1_equals_old_shifted'] else 1)
