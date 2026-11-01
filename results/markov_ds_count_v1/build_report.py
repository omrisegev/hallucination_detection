"""Collect every (arm, benchmark, PRM) cell of the Markov DS count study into report_data.json and render REPORT.html
from report_template.html (style block shared with results/neighbor_readout_v1). Nothing is averaged.

    python build_report.py
"""
import csv, json, re, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run as R  # noqa: E402


def rd(name): return list(csv.DictReader(open(HERE / name, encoding='utf-8')))


def fnum(x):
    try:
        v = float(x); return v if v == v and abs(v) != float('inf') else None
    except (TypeError, ValueError):
        return None


def main():
    rows, con, mech = rd('RESULTS.csv'), rd('CONTRASTS.csv'), rd('MECHANISM.csv')
    post = {(r['benchmark'], r['prm'], r['arm']): r for r in json.loads((HERE / 'POSTHOC_POSTERIOR_WITHIN_AUC.json').read_text())}
    cells = {}
    for b, p, a in R.CELLS:
        sc = 1 if b == 'prmbench' else 100
        F = json.loads((HERE / 'fits' / f'{b}__{p}__{a}.json').read_text())
        sel = lambda L: [r for r in L if r['benchmark'] == b and r['prm'] == p and r['arm'] == a]
        rd_ = {}
        for r in sel(rows):
            v = r['valid']; rd_[r['readout']] = dict(mae=fnum(r['count_mae']), bias=fnum(r['count_bias']), sp=fnum(r['spearman_count']),
                                                     official=fnum(r['official']) * sc, valid=None if v == '' else v == 'True')
        cn = {}
        for r in sel(con):
            k = 1 if r['metric'] == 'count_mae' else sc
            cn.setdefault(r['contrast'], {})['mae' if r['metric'] == 'count_mae' else 'off'] = [fnum(r[x]) * k for x in ('delta', 'ci_low', 'ci_high')]
        m = sel(mech)[0]
        fit = dict(K=F['voters']['K'], n_long=F['n_long_fallback'],
                   **{nm: {x: F[nm].get(x) for x in ('valid', 'reason', 'implied_share', 'e', 'rho', 'delta')} for nm in ('MDS', 'IND', 'SINGLE', 'TOPK')})
        ph = post.get((b, p, a))
        cells[f'{a}|{b}|{p}'] = dict(readouts=rd_, contrasts=cn, fit=fit,
                                     truth=dict(step_share=fnum(m['true_step_share'])),
                                     mech={x: fnum(m.get(x)) for x in ('spearman_delta_auc', 'spearman_posterior_position', 'spearman_truth_position')},
                                     posthoc=dict(post=ph['within_auc_posterior'], fused=ph['within_auc_fused']) if ph else None)
    par = json.loads((HERE / 'independent_check' / 'PARITY.json').read_text())
    integ = (f'הקבצים הוקפאו לפי hash לפני שנקראה תווית כלשהי (<span class="mono">PROTOCOL.json</span>). '
             f'12 מבחנים סינתטיים עברו. על נתונים שנוצרו מהמודל עצמו: MAE ‏0.89 למודל, מול 1.17 ל-GAP ו-1.03 ל-SHARE. '
             f'הזרועות שוחזרו בלי תוויות: |corr| = 1 ל-AUTO ול-TELE16, והפרש מרבי 0 ל-SUB5. '
             f'מימוש נפרד, שנכתב מתוך הטקסט בלבד, הסכים על {par.get("n_match")} מתוך {par.get("n_compared")} ספירות '
             f'ב-Socratic Qwen-7B TELE16.')
    data = dict(arms=R.ARMS, benches=R.BENCHES, prms=R.PRMS, cells=cells,
                verdict=json.loads((HERE / 'VERDICT.json').read_text()), integrity=integ)
    (HERE / 'report_data.json').write_text(json.dumps(R.E_json(data)))
    nb = (R.ROOT / 'results/neighbor_readout_v1/report_template.html').read_text(encoding='utf-8')
    style = re.search(r'<style>\n(.*?)</style>', nb, re.S).group(1)
    html = (HERE / 'report_template.html').read_text(encoding='utf-8')
    html = html.replace('/*__STYLE__*/', style).replace('/*__DATA__*/null', json.dumps(R.E_json(data)))
    (HERE / 'REPORT.html').write_text(html, encoding='utf-8')
    print('cells', len(cells), 'REPORT.html', len(html) // 1024, 'KB')


if __name__ == '__main__':
    main()
