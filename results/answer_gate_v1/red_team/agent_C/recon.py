"""Reconstruct fold-k zA for evaluation AND calibration answers (needed for a label-free calibrated gate), verify vs DECISIONS."""
from load import *
import pickle as pk
from spectral_utils.fusion_utils import lsml_continuous
from spectral_utils.streaming_utils import FEATURE_SIGNS, anchor_orient
from spectral_utils.upcr import upcr_fit

FIT = dict(loss='l2', exclusion=True, difficulty_gate=False, simple_avg_fallback=True, recompute_after_exclusion=True, g2_projection_k=1, scale_ratio=0.25)
GOOD_5 = ['epr', 'low_band_power', 'sw_var_peak', 'cusum_max', 'spectral_entropy']


def lsml_linear(Xfit):
    fused, m = lsml_continuous(*[Xfit[:, j] for j in range(Xfit.shape[1])])
    Wv = np.zeros(Xfit.shape[1]); gw = m['group_weights']; cw = np.asarray(m['cross_weights'], float)
    if len(gw) == 1: idx, w = gw[0]; Wv[idx] = w
    else:
        for g, (idx, w) in enumerate(gw): Wv[idx] += cw[g] * np.asarray(w, float)
    return Wv


DETS = ['D1_upcr_full', 'D2_lsml_cont_good5', 'D4_equal_full', 'D5_epr', 'D6_length']
zA = {d: {k: np.full(n, np.nan) for k in range(5)} for d in DETS}
e_ix = names.index('epr'); l_ix = names.index('trace_length'); g5 = [names.index(f) for f in GOOD_5]
sig5 = np.array([FEATURE_SIGNS[f] for f in GOOD_5], float)
polcheck = []
for cell in sorted(set(cells)):
    cm = cells == cell
    for k in range(5):
        c = (k + 1) % 5; fa = cm & ~np.isin(fold, [k, c]); use = cm & np.isin(fold, [k, c])
        med = np.nanmedian(X[fa], 0); Xall = np.where(np.isfinite(X), X, med)
        mu = Xall[fa].mean(0); sd = Xall[fa].std(0); keep = sd > 1e-5
        Z = np.zeros_like(Xall); Z[:, keep] = (Xall[:, keep] - mu[keep]) / sd[keep]
        anchor_fit = Z[fa, e_ix]; vals = {}
        Ff = Z[fa][:, keep].T
        probe = upcr_fit(Ff, **FIT); pol = np.sign(probe.rho_hat_full); pol[pol == 0] = 1.0
        res = upcr_fit(Ff * pol[:, None], **FIT); w = np.zeros(len(names)); w[np.flatnonzero(keep)] = res.w * pol
        vals['D1_upcr_full'] = w
        W5 = lsml_linear(Z[fa][:, g5] * sig5); w2 = np.zeros(len(names)); w2[g5] = W5 * sig5; vals['D2_lsml_cont_good5'] = w2
        kk = np.flatnonzero(keep); w4 = np.zeros(len(names)); w4[kk] = pol / len(kk); vals['D4_equal_full'] = w4
        w5 = np.zeros(len(names)); w5[e_ix] = 1.0; vals['D5_epr'] = w5
        w6 = np.zeros(len(names)); w6[l_ix] = 1.0; vals['D6_length'] = w6
        for d, wd in vals.items():
            sf = Z[fa] @ wd
            flipped = False if d == 'D6_length' else anchor_orient(sf, anchor_fit)[1]
            sgn = -1.0 if flipped else 1.0; sf = sgn * sf; m_, s_ = sf.mean(), max(sf.std(), 1e-12)
            zA[d][k][use] = (sgn * (Z[use] @ wd) - m_) / s_
        # polarity sanity: weight on epr in D1 (after sign) and D1's fit-fold corr with epr
        polcheck.append({'cell': cell, 'fold': k, 'upcr_w_epr_sign': float(np.sign(w[e_ix])), 'n_views_neg_pol': int((pol < 0).sum()),
                         'n_keep': int(keep.sum()), 'upcr_kept': int(res.keep.sum()), 'epr_kept_by_upcr': bool(res.keep[list(np.flatnonzero(keep)).index(e_ix)])})
# verify vs DECISIONS
for d in DETS:
    Aev = np.full(n, np.nan)
    for k in range(5): Aev[fold == k] = zA[d][k][fold == k]
    print(d, 'max |recon - saved A| =', float(np.max(np.abs(Aev.astype(np.float32) - DEC['A_' + d]))))
pk.dump({'zA': zA, 'polcheck': polcheck}, open('zA_recon.pkl', 'wb'))
print(pd.DataFrame(polcheck).to_string())
