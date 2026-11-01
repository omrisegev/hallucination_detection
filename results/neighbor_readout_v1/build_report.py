"""Collect every cell of the neighbor readout study (frozen run + post hoc arm extension) into report_data.json and
render REPORT.html from report_template.html. Each (arm, benchmark, PRM) cell is kept separate; nothing is averaged.

    python build_report.py
"""
import csv, json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run as R  # noqa: E402
E = R.E

ARMS = ['AUTO', 'SUB5', 'TELE_N16', 'CHAIN_N16']
SRC = {'AUTO': 'EXT', 'SUB5': 'EXT', 'TELE_N16': '', 'CHAIN_N16': ''}
CANDS = ['H1a', 'H1b', 'H1_NOPOSa', 'H1_NOPOSb', 'V1', 'G1']


def rd(name): return list(csv.DictReader(open(HERE / name, encoding='utf-8')))


def fnum(x):
    try:
        v = float(x); return v if np.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def main():
    T = {p: {k: rd(f'{p}{k}.csv') for k in ('CURVES', 'RESULTS', 'CONTRASTS')} for p in ('', 'EXT_')}
    truth = {}
    for b in R.BENCHES:
        for p in R.PRMS:
            c = E.load(b, p); m = np.add.reduceat(c.keep.astype(int), c.off[:-1]) > 0
            truth[(b, p)] = dict(step_error_rate=float(c.err[c.keep].mean()), mean_true_count=float(c.true_count[m].mean()),
                                 n_answers=int(len(c.ids)), n_steps=int(c.off[-1]))
    cells = {}
    for a in ARMS:
        pre = 'EXT_' if SRC[a] else ''; fits = HERE / ('fits_ext' if SRC[a] else 'fits')
        for b in R.BENCHES:
            sc = 1 if b == 'prmbench' else 100
            for p in R.PRMS:
                sel = lambda L: [r for r in L if r['arm'] == a and r['benchmark'] == b and r['prm'] == p]
                rows = {r['readout']: r for r in sel(T[pre]['RESULTS'])}
                if not rows: continue
                cv = sel(T[pre]['CURVES']); con = sel(T[pre]['CONTRASTS'])
                pack = lambda r: dict(official=fnum(r['official']) * sc, within=fnum(r['within_auc']), mae=fnum(r['count_mae']),
                                      flags=fnum(r['mean_flags']))
                ci = lambda name, met: next(([fnum(r['delta']) * (sc if met == 'official' else 1),
                                              fnum(r['ci_low']) * (sc if met == 'official' else 1),
                                              fnum(r['ci_high']) * (sc if met == 'official' else 1)]
                                             for r in con if r['contrast'] == name and r['metric'] == met), None)
                cand = {}
                for n in CANDS:
                    d = pack(rows[n]); d['fallback'] = rows[n].get('fallback_to_gap') == 'True'
                    d['d_off'] = ci(f'{n}[FIX] - {a}[GAP]', 'official'); d['d_w'] = ci(f'{n}[FIX] - {a}[GAP]', 'within_auc')
                    cand[n] = d
                curves = {}
                for form in ('SYM', 'LIFT', 'LAP', 'SYM_RECOUNT'):
                    curves[form] = [[fnum(r['param']), fnum(r['official']) * sc, fnum(r['within_auc']), fnum(r['count_mae'])]
                                    for r in cv if r['form'] == form]
                F = json.loads((fits / f'{b}__{p}__{a}.json').read_text())
                fit = dict(H1={k: F['H1'][k] for k in ('valid', 'reason', 'implied_share', 'rho', 'a', 'b', 'mu', 'sd')},
                           H1_NOPOS={k: F['H1_NOPOS'][k] for k in ('valid', 'reason', 'implied_share', 'rho')},
                           V1={k: F['V1'][k] for k in ('valid', 'reason', 'lam', 'w')}, G1=dict(lam=F['G1']['lam']))
                cells[f'{a}|{b}|{p}'] = dict(
                    ref={k: pack(rows[k]) for k in ('GAP', 'K2', 'ORACLE')}, curves=curves, cand=cand, fit=fit,
                    h1b_vs_k2=dict(off=ci(f'H1b[FIX] - {a}[K2]', 'official')),
                    pos_vs_nopos=dict(a_off=ci('H1a[FIX] - H1_NOPOSa[FIX]', 'official'), a_w=ci('H1a[FIX] - H1_NOPOSa[FIX]', 'within_auc'),
                                      b_off=ci('H1b[FIX] - H1_NOPOSb[FIX]', 'official')),
                    truth=truth[(b, p)])
    verdict = dict(frozen=json.loads((HERE / 'VERDICT.json').read_text()),
                   ext=json.loads((HERE / 'EXT_VERDICT.json').read_text()))
    data = dict(arms=ARMS, benches=R.BENCHES, prms=R.PRMS, cells=cells, verdict=verdict,
                parity=json.loads((HERE / 'independent_check' / 'PARITY.json').read_text()))
    (HERE / 'report_data.json').write_text(json.dumps(R.E_json(data)))
    html = (HERE / 'report_template.html').read_text(encoding='utf-8').replace('/*__DATA__*/null', json.dumps(R.E_json(data)))
    (HERE / 'REPORT.html').write_text(html, encoding='utf-8')
    print('cells', len(cells), 'REPORT.html', len(html) // 1024, 'KB')


if __name__ == '__main__':
    main()
