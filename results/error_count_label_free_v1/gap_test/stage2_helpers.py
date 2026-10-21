"""auroc helper shared with posthoc.py (identical to the one in stage2_eval.py)."""
import numpy as np
from scipy.stats import rankdata


def auroc(pos, s):
    pos = np.asarray(pos, bool); s = np.asarray(s, float); ok = np.isfinite(s); pos, s = pos[ok], s[ok]
    n1 = pos.sum(); n0 = len(pos) - n1
    if not n1 or not n0:
        return np.nan
    return float((rankdata(s)[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
