"""Frozen C4U (C4U-parisi-within-jaffe-cross-v1) under the PRISM, SLIDER, CROP and Mind the Gap protocols.

The recipe is NOT changed: c4u_new.c4u is called as frozen (results/rigorous_recipe_v1/c4u_standalone/).
C4U produces a continuous rank and its own per-answer flag count (DS posterior). Use:
  * flag-based metrics (PRISM T2/T7 PRMScore, SLIDER label-free acc, ProcessBench F1, SLA): C4U's own flags;
  * threshold-based metrics (PRISM matched TPR 87.3, SLIDER LR threshold, CROP T6/T17): C4U's rank, under the
    same label-calibrated threshold procedure as every other row (compute.py functions, unchanged).

Inputs per benchmark:
  PRMBench           R1, exactly c4u_new.load('prmbench', prm, 'R1') (INPUTS_<prm>.npz bundle); hashes and
                     PRMScore checked against expected_results.json before anything else.
  ProcessBench, CRV-Arith, CRV-GSM8K, PRM800K
                     R1 built the same way as build_R1: [answer-z PRM risk, raw PRM error prob globally standardized,
                     answer-z of the GLOBAL rank->normal of the 16 raw N16 channels] (digit channels ranked over their
                     active steps only, 0 elsewhere; rank over all scored steps of the cell). Raw channels from the PRM
                     bank FEATURES.npz; gate: answer-z of the raw values must reproduce the stored N16 channels.
                     If a bank is missing the cell falls back to R0 (stored answer-z N16) and is labelled R0.
Protocol code: results/telemetry_only_vs_competitors_v1/compute.py (prmbench_core, crop_table6, crop_cross), MODE 'prm'.
ProcessBench: PRISM Table 6 (per-subset error / correct accuracy, F1) and Mind the Gap Table 3 SLA (exact first-error
match on the 2,221 error answers, macro over 4 subsets), prediction = first C4U-flagged step (-1 if none).
Parity: every PRM native row is gated against RESULTS_BACKBONE.csv (same protocol code, same cells).

    COLLECTION_DATA=<hd_collection_data> python c4u_competitors.py   # -> RESULTS.csv, PARITY.json, FIT.json
"""
import hashlib, json, os, sys
from pathlib import Path
import numpy as np
from scipy.special import ndtri
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / 'results/rigorous_recipe_v1/c4u_standalone'))
import c4u_new as N  # noqa: E402  (frozen recipe)
sys.path.insert(0, str(ROOT / 'results/telemetry_only_vs_competitors_v1'))
import compute as C  # noqa: E402
A, E, S8 = C.A, C.E, N.S8
EXP = json.loads((ROOT / 'results/rigorous_recipe_v1/c4u_standalone/expected_results.json').read_text())['cells']
R1DIR = Path(os.environ.get('C4U_R1_INPUTS', Path(E.DATA) / 'c4u_r1_inputs'))
PFILE = {'qwen_prm_7b': 'qwen_prm'}
N16 = ['q15_H1', 'q15_VE1', 'chosen_surprisal', 'logprob_margin', 'true_tail50', 'energy_level',
       'energy_innovation', 'top15_turnover', 'top50_js', 'dominant_freq16', 'bocpd_p0',
       'realized_z', 'realized_drv', 'digit_alternative', 'digit_spread', 'digit_alternative_innovation']
DIG = N16[13:]
SUBSETS = ('gsm8k', 'math', 'olympiadbench', 'omnimath')
REF = {}
for r in E.read_csv(ROOT / 'results/telemetry_only_vs_competitors_v1/RESULTS_BACKBONE.csv'):
    REF[tuple(r[k] for k in ('prm', 'benchmark', 'protocol', 'arm', 'metric', 'pool', 'rule', 'scope', 'grid'))] = float(r['value'])
FIT = {}

# C4U's own flags enter every flag-based metric through the rule name 'GAP' that compute.py uses for its arms
_flags0 = E.Cell._flags


def _flags(self, arm, rule):
    if arm == 'C4U':
        assert rule == 'GAP'; return self.c4u_flags
    return _flags0(self, arm, rule)


E.Cell._flags = _flags


def sha(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a); h.update(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes())
    return h.hexdigest()


def rank_gauss(x, mask=None):
    """auto_raw_plus_z.rank_gauss: global rank -> N(0,1) scores, ties averaged; outside mask -> 0."""
    out = np.zeros(len(x)); m = np.isfinite(x) if mask is None else (mask & np.isfinite(x))
    r = rankdata(x[m]); out[m] = ndtri((r - .5) / m.sum())
    return out


def bank_path(prm, bench):
    p = PFILE.get(prm, prm)
    if bench == 'processbench':
        return Path(E.DATA) / 'sub5_all_v1' / 'pb_prm' / f'bank_{p}' / 'FEATURES.npz'
    return R1DIR / 'crop_ares' / f'full_prm_{p}_{bench}' / 'bank_v3' / 'FEATURES.npz'


def raw16(c, prm, bench):
    """Raw N16 values and digit-active mask on c's scored steps, or None if the bank is not here."""
    p = bank_path(prm, bench)
    if not p.exists():
        return None, None
    F = np.load(p); names = list(F['names'].astype(str)); V = F['values']; act = F['digit_active'].astype(bool)
    uids = F['uids'].astype(str); off = F['offsets'].astype(np.int64); pos = {u: j for j, u in enumerate(uids)}
    rows = np.concatenate([np.arange(off[pos[u]], off[pos[u] + 1]) for u in c.ids])
    assert len(rows) == c.off[-1], (p, len(rows), c.off[-1])
    raw = np.column_stack([V[rows, names.index(n)] for n in N16]).astype(float)
    return raw, act[rows]


def answer_z_masked(x, off, m):
    z = np.zeros(len(x))
    for a, b in zip(off[:-1], off[1:]):
        ii = np.flatnonzero(m[a:b]) + a
        if not len(ii): continue
        xs = x[ii]; mu = xs.mean(); sd = xs.std()
        z[ii] = 0 if sd <= 1e-12 else (xs - mu) / sd
    return z


def build_X(c, prm, bench):
    """R1 (raw bank here) or R0 for a CAB cell; columns as c4u_new.load / build_R1."""
    zp = E.answer_z(c.arms['PRM'], c.off); raw = c.arms['PRMPROB']
    Z = E.channels(c, 'N16'); rv, act = raw16(c, prm, bench)
    head = [zp, (raw - raw.mean()) / raw.std()]
    if rv is None:
        return np.column_stack(head + [Z]), {1: raw > .5}, 'R0', None
    gate = []
    for j, n in enumerate(N16):
        m = act[:, DIG.index(n)] if n in DIG else np.ones(len(rv), bool)
        gate.append(float(np.abs(answer_z_masked(rv[:, j], c.off, m) - Z[:, j]).max()))
    assert max(gate) < 1e-4, (prm, bench, gate)             # the bank is the source of the stored channels
    G16 = np.column_stack([rank_gauss(rv[:, j], act[:, DIG.index(n)] if n in DIG else None) for j, n in enumerate(N16)])
    Zr = np.column_stack([E.answer_z(G16[:, j], c.off) for j in range(16)])
    return np.column_stack(head + [Zr]), {1: raw > .5}, 'R1', max(gate)


def run_c4u(c, X, ab, key):
    r = N.c4u(c, X, ab)
    c.arms['C4U'] = r['rank']; c.c4u_flags = r['flags']
    FIT[key] = dict(K=len(r['groups']), group_sizes=[len(g) for g in r['groups']], prevalence=r['prevalence'],
                    flag_rate=float(r['flags'].mean()), answers_zero_flags=int((r['k'] == 0).sum()),
                    unassigned=r['unassigned'], anchor_group=r['anchor_group'])
    return r


def ref_gate(name, got, key):
    if key in REF:
        C.gate(name, got, REF[key], 1e-6)


# ------------------------------------------------------------------------------------------- PRMBench (R1)
def prmbench(prm):
    c, ans_eval, X, ab, wk = N.load('prmbench', prm, 'R1')
    x = EXP[f'prmbench|{prm}|R1']
    r = run_c4u(c, X, ab, f'prmbench/{prm}/R1')
    wa = E.per_answer_auc(c.err, r['rank'], c.off, wk); wa[~ans_eval] = np.nan
    pt = S8.point(c, (wa, E.counts_by_answer(c, r['flags']) * ans_eval[:, None]))
    # input and flag hashes must be exact; the rank hash (rounded to 9 decimals) is reported only: this machine's BLAS
    # moves the rank by ~1e-14, which flips the 9th decimal of a few steps (flags, PRMScore, within-AUC stay exact)
    C.PAR.append(dict(check=f'{prm} PRMBench R1 input+flags hashes vs expected_results.json',
                      ok=bool(sha(X, ab[1], c.off, ans_eval, np.zeros(0)) == x['input_sha256'] and
                              sha(r['flags'].astype(np.uint8)) == x['flags_sha256']), abs_diff=0., tol=0.,
                      rank_r9_hash_equal=bool(sha(np.round(r['rank'], 9)) == x['rank_sha256_r9'])))
    C.gate(f'{prm} PRMBench R1 C4U PRMScore vs expected_results.json', pt['gap_prmscore'], x['prmscore'], 1e-9)
    C.gate(f'{prm} PRMBench R1 C4U within-AUC vs expected_results.json', pt['within_auc'], x['within_auc'], 1e-9)
    n0 = len(C.ROWS); C.prmbench_core(c, prm)
    for row in C.ROWS[n0:]:
        row['input'] = 'R1'
    # protocol parity: PRM native rows through the same code vs the stored backbone results
    for row in C.ROWS[n0:]:
        if row['arm'] in ('PRM native', 'PRM (same rule)', 'PRM (1-reward)', 'majority class'):
            k = (prm, 'prmbench', row['protocol'], row['arm'], row['metric'], row.get('pool', ''), row.get('rule', ''),
                 row.get('scope', ''), row.get('grid', ''))
            ref_gate(f"{prm} PRMBench {row['protocol']} {row['arm']} {row['metric']} {row.get('pool', '')}{row.get('scope', '')}{row.get('grid', '')} vs RESULTS_BACKBONE", row['value'], k)
    C.gate(f'{prm} PRMBench PRISM T2 C4U PRMScore = frozen PRMScore', next(
        r_['value'] for r_ in C.ROWS[n0:] if r_['protocol'] == 'PRISM T2' and r_['arm'] == 'C4U' and r_['metric'] == 'PRMScore'), x['prmscore'], 1e-6)
    del c


# ------------------------------------------------------------------------------------------- ProcessBench
def pb_predict(c, flags):
    pred = np.full(len(c.ids), -1)
    for j, (a, b) in enumerate(zip(c.off[:-1], c.off[1:])):
        f = np.flatnonzero(flags[a:b])
        if len(f): pred[j] = int(f[0])
    return pred


def pb_metrics(c, pred):
    lab = np.array([A.PB_GOLD[u]['label'] for u in c.ids]); sub = np.array([A.PB_GOLD[u]['subset'] for u in c.ids])
    out = {}
    for s in SUBSETS:
        e = (sub == s) & (lab != -1); k = (sub == s) & (lab == -1)
        ea = 100 * float((pred[e] == lab[e]).mean()); ca = 100 * float((pred[k] == -1).mean())
        out[s] = dict(TPR=ea, TNR=ca, F1=2 * ea * ca / (ea + ca) if ea + ca else 0., SLA=ea)
    out['average F1'] = float(np.mean([out[s]['F1'] for s in SUBSETS]))
    err = lab != -1
    out['macro SLA'] = float(np.mean([out[s]['SLA'] for s in SUBSETS]))
    out['overall SLA'] = 100 * float((pred[err] == lab[err]).mean())
    hit = (pred == lab).astype(float); idx = {s: np.flatnonzero((sub == s) & err) for s in SUBSETS}
    rng = np.random.default_rng(20261102)                    # as evaluate_sla_standalone: 2,000 stratified draws
    bm = np.array([np.mean([100 * hit[rng.choice(ix, len(ix))].mean() for ix in idx.values()]) for _ in range(2000)])
    out['macro SLA lo'], out['macro SLA hi'] = (float(v) for v in np.percentile(bm, [2.5, 97.5]))
    return out


def processbench(prm):
    c = C.cab_cell('processbench', prm)
    X, ab, rep, g = build_X(c, prm, 'processbench')
    FIT[f'processbench/{prm}/gate'] = g
    run_c4u(c, X, ab, f'processbench/{prm}/{rep}')
    for label, pred in (('PRM native', pb_predict(c, c.arms['PRM'] > 0)),   # eval_processbench PRM_native: risk > 0
                         ('C4U', pb_predict(c, c.c4u_flags))):
        m = pb_metrics(c, pred)
        for s in SUBSETS:
            for k in ('TPR', 'TNR', 'F1'):
                C.add(prm, 'processbench', 'PRISM PB T6', label, f'{s} {k}', m[s][k], rule='C4U own count' if label == 'C4U' else 'native', input=rep)
            C.add(prm, 'processbench', 'MtG T3 SLA', label, f'{s} SLA', m[s]['SLA'], input=rep)
        C.add(prm, 'processbench', 'PRISM PB T6', label, 'average F1', m['average F1'], rule='C4U own count' if label == 'C4U' else 'native', input=rep)
        for k in ('macro SLA', 'overall SLA', 'macro SLA lo', 'macro SLA hi'):
            C.add(prm, 'processbench', 'MtG T3 SLA', label, k, m[k], input=rep)
        if label == 'PRM native':
            ref_gate(f'{prm} ProcessBench PRM native average F1 vs RESULTS_BACKBONE', m['average F1'],
                     (prm, 'processbench', 'PRISM PB T6', 'PRM native', 'average F1', '', 'native', '', ''))
    n0 = len(C.ROWS); C.crop_table6(c, 'processbench', prm)
    for row in C.ROWS[n0:]:
        row['input'] = rep
        if row['arm'] == 'PRM (1-reward)':
            ref_gate(f"{prm} ProcessBench CROP T6 PRM {row['metric']} {row['grid']} vs RESULTS_BACKBONE", row['value'],
                     (prm, 'processbench', 'CROP T6', row['arm'], row['metric'], '', '', '', row['grid']))
    del c


# ------------------------------------------------------------------------------------------- CROP benchmarks
def cab_hook(c, bench, prm):
    X, ab, rep, g = build_X(c, prm, bench)
    FIT[f'{bench}/{prm}/gate'] = g; c.rep = rep
    run_c4u(c, X, ab, f'{bench}/{prm}/{rep}')


def crop(prm):
    for bench in ('crv_arith', 'crv_gsm8k', 'prm800k'):
        c = C.cab_cell(bench, prm)
        n0 = len(C.ROWS); C.crop_table6(c, bench, prm)
        for row in C.ROWS[n0:]:
            row['input'] = c.rep
            if row['arm'] == 'PRM (1-reward)':
                ref_gate(f"{prm} {bench} CROP T6 PRM {row['metric']} {row['grid']} vs RESULTS_BACKBONE", row['value'],
                         (prm, bench, 'CROP T6', row['arm'], row['metric'], '', '', '', row['grid']))
        del c
    for s, t in (('crv_arith', 'crv_gsm8k'), ('crv_gsm8k', 'crv_arith')):
        n0 = len(C.ROWS); C.crop_cross(s, t, prm)
        for row in C.ROWS[n0:]:
            row['input'] = '/'.join(sorted({FIT_rep(s, prm), FIT_rep(t, prm)}))


def FIT_rep(bench, prm):
    return next(k.split('/')[-1] for k in FIT if k.startswith(f'{bench}/{prm}/') and k.split('/')[-1] in ('R0', 'R1'))


def main():
    C.MODE, C.TELE = 'prm', ['C4U']
    prms = sys.argv[1:] or N.PRMS
    for prm in prms:
        C.CELL_HOOK = None
        prmbench(prm); processbench(prm)
        C.CELL_HOOK = cab_hook; crop(prm)
        print(prm, 'done', flush=True)
    for r in C.ROWS:
        r.pop('standalone_model', None)
    E.write_csv(HERE / 'RESULTS.csv', C.ROWS)
    fails = [p['check'] for p in C.PAR if not p['ok']]
    json.dump(dict(n_checks=len(C.PAR), failed=fails, checks=C.PAR), open(HERE / 'PARITY.json', 'w'), indent=1, default=float)
    json.dump(FIT, open(HERE / 'FIT.json', 'w'), indent=1, default=float)
    print('rows', len(C.ROWS), 'parity', len(C.PAR), 'failed', fails)
    if fails: raise SystemExit(3)


if __name__ == '__main__':
    main()
