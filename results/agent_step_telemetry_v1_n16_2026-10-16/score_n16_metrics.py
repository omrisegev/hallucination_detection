import numpy as np
from scipy.stats import rankdata
def auroc(y, s):
    y = np.asarray(y, bool); s = np.asarray(s, float); n1, n0 = y.sum(), (~y).sum()
    r = rankdata(s); return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
