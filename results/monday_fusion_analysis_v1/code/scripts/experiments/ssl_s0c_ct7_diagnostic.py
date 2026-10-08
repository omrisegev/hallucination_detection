"""POST-EVALUATION DIAGNOSTIC for S0-C (not part of the frozen primary family).

Question: would the frozen comparators (CT7, token L-SML) ALSO gain PRMScore from the same
per-answer scale transforms?  The S0-C protocol keeps them untuned; this file exists so the
"gap to CT7 nearly closed" reading is not left unmatched.  Same nested threshold protocol as
the frozen run's reference path (scoring.py non-new branch): thresholds from the four training
folds' OOF scores, quantile selected on inner-held folds, frozen numeric threshold applied to k.
"""
from pathlib import Path
import json, pickle, sys, time
import numpy as np, pandas as pd
from scipy.stats import rankdata
ROOT = Path(__file__).resolve().parents[2]; MAIN = Path(r'C:\Users\omris\TAU\hallucination_detection')
R = MAIN / '.worktrees/readout-quickest-detection-v1/results/step_evidence_v1'; OUT = ROOT / 'results/ssl_pseudolabel_residual_v1/S0C'
sys.path.insert(0, str(MAIN)); from spectral_utils.prmbench import prmbench_evaluate
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def T_raw(s): return np.asarray(s, float)
def T_z(s): s = np.asarray(s, float); return (s - s.mean()) / max(s.std(), 1e-8)
def T_ecdf(s): s = np.asarray(s, float); return (rankdata(s, method='average') - .5) / len(s)
TR = {'C_RAW': T_raw, 'C_Z': T_z, 'C_ECDF': T_ecdf}
ans = pd.read_csv(R / 'OOF_ANSWERS.csv', encoding='utf-8-sig'); Z = np.load(R / 'OOF_STEP_SCORES.npz'); off = Z['offsets']; labels = Z['labels']; n = len(ans); ns = np.diff(off)
prm = ~ans.cell.str.startswith('pb_').to_numpy(); fold = ans.fold.to_numpy(); groups = ans.source_group.to_numpy(); ids = ans.id.to_numpy()
meta = {m['idx']: m for m in pickle.load(open(read(R / 'INPUT_FREEZE.json')['prm_metadata']['path'], 'rb')).values()}
prm_idx = np.flatnonzero(prm); noncontrol = np.array([prm[i] and meta[ids[i]]['classification'] != 'correct' for i in range(n)])
elig = np.repeat(noncontrol, ns); prm_steps = np.repeat(prm, ns); qgrid = np.round(np.linspace(.5, .99, 50), 2); saved = read(R / 'PRMSCORE.json')
def counts(valid, y, e):
    v = np.asarray(valid, bool)[..., e]; good = np.asarray(y)[e] == 0
    return (v & good).sum(-1), (v & ~good).sum(-1), (~v & ~good).sum(-1), (~v & good).sum(-1)
def score(tp, fp, tn, fn):
    def ratio(a, b): return np.divide(a, b, out=np.full(np.shape(a), -1., float), where=b != 0)
    p = ratio(tp, tp + fp); r = ratio(tp, tp + fn); f = ratio(2 * p * r, p + r); p = ratio(tn, tn + fn); r = ratio(tn, tn + fp); nf = ratio(2 * p * r, p + r); return (f + nf) / 2
def official(valid):
    r = prmbench_evaluate([{'idx': ids[i], 'labels': valid[off[i]:off[i+1]].astype(int).tolist()} for i in prm_idx], [meta[ids[i]] for i in prm_idx])['total']
    return .5 * (r['f1'] + r['negative_f1'])
g = groups[prm_idx]; G, ginv = np.unique(g, return_inverse=True)
def gcounts(valid):
    c = np.zeros((len(G), 4))
    for gi, i in zip(ginv, prm_idx):
        if not noncontrol[i]: continue
        a, b = off[i:i+2]; v = valid[a:b]; good = labels[a:b] == 0
        c[gi] += [(v & good).sum(), (v & ~good).sum(), (~v & ~good).sum(), (~v & good).sum()]
    return c
def boot(cA, cB, draws=100_000, seed=20260923):
    rng = np.random.default_rng(seed); d = np.empty(draws); pos = 0
    while pos < draws:
        w = rng.multinomial(len(G), np.full(len(G), 1 / len(G)), size=min(5000, draws - pos)).astype(float)
        d[pos:pos + len(w)] = score(*(w @ cA).T) - score(*(w @ cB).T); pos += len(w)
    return d
rows = []; dec = {}
t0 = time.perf_counter()
for m in ['ct7', 'token_lsml']:
    s0 = Z[m]
    for tname, T in TR.items():
        ts = np.full(int(off[-1]), np.nan)
        for i in prm_idx: a, b = off[i:i+2]; ts[a:b] = T(s0[a:b])
        valid = np.zeros(int(off[-1]), bool)
        for k in range(5):
            train = prm & (fold != k); trs = np.flatnonzero(np.repeat(train, ns)); thr = np.quantile(ts[trs], qgrid)
            inner_valid = np.zeros((50, len(trs)), bool); g2t = np.full(int(off[-1]), -1, int); g2t[trs] = np.arange(len(trs))
            for j in range(5):
                if j == k: continue
                fit = np.repeat(train & (fold != j), ns); held = np.repeat(train & (fold == j), ns); th = np.quantile(ts[fit], qgrid)
                inner_valid[:, g2t[np.flatnonzero(held)]] = ts[held][None, :] < th[:, None]
            grid = score(*counts(inner_valid, labels[trs], elig[trs])); best = int(np.argmax(grid))
            tst = np.repeat(prm & (fold == k), ns); valid[tst] = ts[tst] < thr[best]
        dec[f'{m}__{tname}'] = valid
        rows.append({'method': m, 'transform': tname, 'prmscore_inner': official(valid), 'original_run_prmscore_inner': saved[m]['inner_selected']['prmscore']})
        print(rows[-1], f'{time.perf_counter()-t0:.0f}s', flush=True)
res = pd.DataFrame(rows); res.to_csv(OUT / 'DIAGNOSTIC_COMPARATORS_TRANSFORMED.csv', index=False)
cont = []
for m in ['ct7', 'token_lsml']:
    for tname in ['C_Z', 'C_ECDF']:
        d = boot(gcounts(dec[f'{m}__{tname}']), gcounts(dec[f'{m}__C_RAW']))
        cont.append({'method': m, 'contrast': f'{tname} - C_RAW', 'delta': float(res[(res.method == m) & (res['transform'] == tname)].prmscore_inner.iloc[0] - res[(res.method == m) & (res['transform'] == 'C_RAW')].prmscore_inner.iloc[0]), 'ci95': [float(np.quantile(d, .025)), float(np.quantile(d, .975))]})
S = np.load(OUT / 'DECISIONS.npz')
for tname in ['C_Z', 'C_ECDF']:
    d = boot(gcounts(S[f'evidence__all__position__equal__{tname}__inner']), gcounts(dec[f'ct7__{tname}']))
    cont.append({'method': 'evidence__all__position__equal', 'contrast': f'{tname} position evidence - {tname} ct7 (matched transform)', 'delta': float(pd.read_csv(OUT / 'S0C_CALIBRATION.csv').set_index(['method', 'transform']).loc[('evidence__all__position__equal', tname)].prmscore_inner - res[(res.method == 'ct7') & (res['transform'] == tname)].prmscore_inner.iloc[0]), 'ci95': [float(np.quantile(d, .025)), float(np.quantile(d, .975))]})
Path(OUT / 'DIAGNOSTIC_COMPARATORS_BOOTSTRAP.json').write_text(json.dumps({'status': 'post-evaluation diagnostic, outside the frozen primary family', 'draws': 100000, 'seed': 20260923, 'contrasts': cont}, indent=2), encoding='utf8')
print(pd.DataFrame(cont).to_string()); print(f'done {time.perf_counter()-t0:.0f}s')
