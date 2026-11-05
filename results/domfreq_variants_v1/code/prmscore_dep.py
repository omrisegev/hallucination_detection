"""domfreq_variants_v1 stage 3: PRMScore of the 3 candidates (alone, and swapped for dominant_freq16 inside the
current Z17 L-SML), and class-conditional dependence of their errors on the TELE16 channels.

Q80 rule, lsml_continuous defaults and the z(PRM) anchor follow auto_raw_plus_z.py / reproduce_from_bundle.py.
PRMBench: official prmbench_evaluate (vendored copy in the bundle). Socratic: 50*(F1+negF1) on included steps
with the same Q80 flags (no 'correct' class there). Labels are used for evaluation only.

    python3 prmscore_dep.py <results dir>
"""
import json, sys
from pathlib import Path
import numpy as np
from scipy.stats import rankdata

S = Path('/shared/cycle2_tau_averbuch_prj/omrisegev1'); BUN = S / 'auto_raw_plus_z_v1/bundle'
sys.path.insert(0, str(BUN / 'code'))
from spectral_utils.fusion_utils import lsml_continuous   # noqa: E402
from spectral_utils.prmbench import prmbench_evaluate     # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
import metrics as MX                                        # noqa: E402  (load_cell, N16, DIG)

D = Path(sys.argv[1]); MX.D = D
CAND = {'JS_flat16': 'js|flat|w16|mc|last', 'JS_sent32': 'js|sent|w32|mc|last', 'SUR_sent8': 'surprisal|sent|w8|mc|last'}
PRMS = MX.PRMS; N16 = MX.N16; DIG = MX.DIG


def az(v, o):
    out = np.zeros(len(v))
    for a, b in zip(o[:-1], o[1:]):
        x = v[a:b]; sd = x.std()
        if sd > 1e-12: out[a:b] = (x - x.mean()) / sd
    return out


def az_masked(v, o, m):
    out = np.zeros(len(v))
    for a, b in zip(o[:-1], o[1:]):
        k = m[a:b]; x = v[a:b][k]
        if k.sum() > 1 and x.std() > 1e-12: out[a:b][k] = (x - x.mean()) / x.std()
    return out


def within(y, s, o):
    v = []
    for a, b in zip(o[:-1], o[1:]):
        yy = y[a:b]; n1 = yy.sum(); n0 = len(yy) - n1
        if n1 and n0: v.append((rankdata(s[a:b])[yy].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
    return float(np.mean(v))


def lsml(V):
    s, d = lsml_continuous(*[V[:, j] for j in range(V.shape[1])]); s = np.asarray(s, float)
    if np.corrcoef(s, V[:, 0])[0, 1] < 0: s = -s
    return s


def q80(s, o):
    z = az(s, o); return z > np.quantile(z, .8)


def main():
    out = {}
    signs = {}
    for bench in ('prmbench', 'socratic'):
        for tag in PRMS:
            c = MX.load_cell(bench, tag)                  # kept (nonempty) steps, FEATURES order
            o, V, names = c['o'], c['V'], c['names']
            Ch = c['Ch']; act = c['act']
            if bench == 'prmbench':
                B = np.load(BUN / 'inputs' / f'INPUTS_{tag}.npz'); C = c['C']; ki = C['keep_index']
                rows = np.concatenate([B['offsets'][k] + ki[o[k]:o[k + 1]] for k in range(len(o) - 1)])
                prm = B['raw17'][rows, 0].astype(float)
                # full step grid (official scoring needs every step): dropped empty steps get the answer mean
                fo = B['offsets']; ids = B['row_ids'].astype(str)
                eo, ef = B['error_steps_offsets'], B['error_steps_flat']
                meta = [{'idx': i, 'classification': str(cl), 'error_steps': ef[eo[j]:eo[j + 1]].tolist()}
                        for j, (i, cl) in enumerate(zip(ids, B['classification']))]
            else:
                F = np.load(MX.R / 'prm_boost_external_v1' / MX.SOC_DIR[tag] / 'bank_v2/REWARDS.npz')
                ne = np.load(MX.R / 'prm_boost_external_v1' / MX.SOC_DIR[tag] / 'bank_v2/FEATURES.npz')['nonempty'].astype(bool)
                r = np.clip(1 - F['rewards'][ne].astype(float), 1e-6, 1 - 1e-6); prm = np.log(r / (1 - r))
            y, inc = c['y'], c['include']
            sg = {k: -1.0 for k in CAND}                  # PRMBench-derived sign, -1 for all three in all PRMs
            vz = {k: az(sg[k] * V[:, names.index(nm)], o) for k, nm in CAND.items()}
            Z = np.column_stack([az(Ch[:, j], o) if N16[j] not in DIG else az_masked(Ch[:, j], o, act[:, DIG.index(N16[j])])
                                 for j in range(16)])
            zp = az(prm, o); jd = N16.index('dominant_freq16')
            base = np.column_stack([zp, Z])
            arms = {'PRM_alone': prm, 'dominant_freq16_alone': Ch[:, jd]}
            arms.update({k + '_alone': vz[k] for k in CAND})
            arms['Z17_current'] = lsml(base)
            arms['Z17_minus_domfreq'] = lsml(np.delete(base, 1 + jd, axis=1))
            for k in CAND:
                b2 = base.copy(); b2[:, 1 + jd] = vz[k]; arms[f'Z17_swap_{k}'] = lsml(b2)
            arms['EQ16_current'] = Z.mean(1)
            for k in CAND:
                z2 = Z.copy(); z2[:, jd] = vz[k]; arms[f'EQ16_swap_{k}'] = z2.mean(1)
            res = {}
            for nm, s in arms.items():
                f = q80(s, o)
                if bench == 'prmbench':
                    sf = np.full(fo[-1], np.nan); sf[rows] = s
                    ff = np.zeros(fo[-1], bool); ff[rows] = f
                    # dropped empty steps: not flagged (keep the answer's other decisions unchanged)
                    pred = [{'idx': i, 'labels': (~ff[a:b]).astype(int).tolist()} for i, a, b in zip(ids, fo[:-1], fo[1:])]
                    ev = prmbench_evaluate(pred, meta); t = ev['total']; bc = ev['by_classification']
                    ps = 50 * (t['f1'] + t['negative_f1'])
                    cat = {cl: 50 * (bc['f1'][cl] + bc['negative_f1'][cl]) for cl in sorted(bc['f1']) if cl != 'correct'}
                else:
                    yy, ff = y[inc], f[inc]
                    tp = (yy & ff).sum(); fp = (~yy & ff).sum(); fn = (yy & ~ff).sum(); tn = (~yy & ~ff).sum()
                    f1 = 2 * tp / (2 * tp + fp + fn); nf1 = 2 * tn / (2 * tn + fn + fp); ps = 50 * (f1 + nf1); cat = None
                yi = y[inc]; si = s[inc]
                oi = np.r_[0, np.cumsum(np.bincount(np.repeat(np.arange(len(o) - 1), np.diff(o))[inc], minlength=len(o) - 1))]
                res[nm] = {'prmscore': float(ps), 'within_auc': within(yi, si, oi), 'by_category': cat}
            # class-conditional dependence (answer-z values, included steps), and Q80 co-error lift
            dep = {}
            flags = {k: q80(vz[k], o)[inc] for k in CAND}
            chf = {ch: q80(Z[:, j], o)[inc] for j, ch in enumerate(N16)}
            yi = y[inc]
            for k in CAND:
                v = vz[k][inc]; d = {}
                for j, ch in enumerate(N16):
                    zc = Z[inc, j]; row = {}
                    for cls, m in (('error', yi), ('valid', ~yi)):
                        ra, rb = rankdata(v[m]), rankdata(zc[m])
                        row[f'rho_{cls}'] = float(np.corrcoef(ra, rb)[0, 1])
                    ma, mb = ~flags[k][yi], ~chf[ch][yi]            # misses on error steps
                    fa, fb = flags[k][~yi], chf[ch][~yi]            # false alarms on valid steps
                    row['miss_lift'] = float((ma & mb).mean() / max(ma.mean() * mb.mean(), 1e-12))
                    row['fa_lift'] = float((fa & fb).mean() / max(fa.mean() * fb.mean(), 1e-12))
                    d[ch] = row
                dep[k] = d
            out[f'{bench}/{tag}'] = {'arms': res, 'dependence': dep}
            print(bench, tag, {n: round(r['prmscore'], 2) for n, r in res.items()}, flush=True)
    json.dump(out, open(D / 'PRMSCORE_DEPENDENCE.json', 'w'), indent=1)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
