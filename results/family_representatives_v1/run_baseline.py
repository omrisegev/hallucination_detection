"""Step 1 of spec section 12: label-free structure metrics on the original N56 bank."""
import json, sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).parent; sys.path.insert(0, str(HERE))
from metrics_a import structure_metrics, ROOT  # noqa: E402

D = ROOT / 'results/advisor_completion_2026-10-11/inputs'
Z = np.load(D / 'bank_V_N56.npz'); B = np.load(D / 'bank_N56.npz'); P = np.load(D / 'population.npz')
off = Z['offsets']; folds = np.repeat(P['fold'], np.diff(off))   # fold ids from source groups only
names = [str(c) for c in Z['columns']]
t = time.time(); res, score = structure_metrics(Z['V'], folds, names)
# parity with the shipped continuous L-SML score of this bank (S__ is answer-z of the raw fused score)
az = np.concatenate([(lambda s: (s - s.mean()) / s.std() if s.std() > 0 else s * 0)(score[a:b]) for a, b in zip(off[:-1], off[1:])])
ref = B['S__LSML_CONT']
res['parity_vs_shipped_LSML_CONT'] = dict(max_abs_diff=float(np.max(np.abs(az - ref))), corr=float(np.corrcoef(az, ref)[0, 1]))
res['seconds'] = round(time.time() - t, 1); res['labels_read'] = False
res['input'] = str((D / 'bank_V_N56.npz').relative_to(ROOT))
(HERE / 'METRICS_A_N56.json').write_text(json.dumps(res, indent=1))
print(json.dumps({k: v for k, v in res.items() if k not in ('groups', 'effective_weights')}, indent=1))
