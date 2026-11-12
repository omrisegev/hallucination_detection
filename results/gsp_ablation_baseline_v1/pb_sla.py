"""ProcessBench macro SLA (Mind the Gap metric: exact first-error step on the erroneous answers, mean over the 4
subsets) with 95% CI (pb_metrics, 2,000 stratified draws) for every arm of cells/processbench__<prm>*.npz and pbcp."""
import sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ablation as AB  # noqa: E402
T, E = AB.T, AB.E
rows = []
for prm in AB.PRMS:
    c, X, ab = AB.load_cell('processbench', prm)
    F = {k[6:]: S[k] for p in sorted((HERE / 'cells').glob(f'processbench__{prm}*.npz')) for S in [np.load(p)] for k in S.files if k.startswith('flags|')}; preds = {k: T.pb_predict(c, f) for k, f in F.items()}
    for p in (HERE / 'cells').glob(f'pbcp__{prm}__*.npy'):
        name = p.stem.split('__', 2)[2].replace('_', ' ').replace('pi = 12', 'pi = 1/2')
        f = np.zeros(c.off[-1], bool)
        for q, a in zip(np.load(p), c.off[:-1]):
            if q >= 0: f[a + q] = True
        preds[name] = T.pb_predict(c, f)
    for k, pr in preds.items():
        m = T.pb_metrics(c, pr)
        rows.append(dict(bench='processbench', prm=prm, arm=k, macro_SLA=m['macro SLA'], lo=m['macro SLA lo'],
                         hi=m['macro SLA hi'], avg_F1=m['average F1']))
    print(prm, 'done', flush=True); del c
E.write_csv(HERE / 'PB_SLA.csv', rows)
