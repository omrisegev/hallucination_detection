import json, sys
from pathlib import Path
import numpy as np
exec(open(Path(__file__).with_name('rt_c.py')).read().split("# ---------------- Null A")[0])
d = json.load(open(OUTD / 'rt_c_part1.json'))
print('saved tau - q80(OOF z):', json.dumps(d['saved_tau_vs_q80_rule_on_OOF']))
print('z vec vs loop:', d['answer_z_vectorized_vs_loop_maxabs'], 'flag flips:', d['flag_flips_vectorized_vs_loop'])
from scipy.stats import spearmanr, pearsonr
NCI = np.flatnonzero(nonc)
err_share = np.bincount(st_ans, lab.astype(float), n)[NCI] / ns[NCI]
for m in ['S_equal', 'B13_equal', 'realized_drv', 'PRM_z_q80', 'step_index']:
    fl = np.bincount(st_ans[prm_step & ~valid[m]], minlength=n)[NCI] / ns[NCI]
    print(m, 'corr(flag share, error share) over non-control answers: pearson %.4f spearman %.4f' % (pearsonr(fl, err_share)[0], spearmanr(fl, err_share)[0]),
          ' flag share sd %.4f' % fl.std())
# chance PRMScore as function of flag rate (random allocation)
pi = lab[nc_step].mean()
for f in (0.199, 0.2, 0.2014):
    f1e = 2 * pi * f / (pi + f); f1c = 2 * (1 - pi) * (1 - f) / ((1 - pi) + (1 - f)); print('random flags f=%.4f PRMScore %.5f' % (f, (f1e + f1c) / 2))
print('error prevalence nc', pi)
