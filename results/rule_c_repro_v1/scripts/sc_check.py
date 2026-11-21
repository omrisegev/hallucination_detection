import sys, numpy as np
WS='C:/Users/omris/TAU/rule_c_ws/run'; sys.path.insert(0, WS+'/agentX1')
import common as C
from scipy.stats import spearmanr
for b in ['processbench','prmbench']:
    d=C.load_dev(b, draws=10)
    LN=np.load(f'{WS}/agentX1/out/LN_SCORES_{b}.npz'); s=d['sC']; t=LN['sC']
    print(b, 'uids_equal', bool((LN['uids']==d['Cd'].uids).all()), 'max|dsC|', float(np.max(np.abs(s-t))), 'scale', float(np.std(t)), 'spearman', spearmanr(s,t)[0], 'rank_equal', bool(np.array_equal(np.argsort(-s,kind='stable'),np.argsort(-t,kind='stable'))))
    print('  LN keys', LN.files)
    import inspect
