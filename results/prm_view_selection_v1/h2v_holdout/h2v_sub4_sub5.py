"""SUB4 and SUB5 on Hard2Verify, frozen sets, refitted per cell (PREDECLARATION_H2V_SUB4_SUB5.md).

    python h2v_sub4_sub5.py      # -> H2V_SUB4_SUB5.csv, H2V_SUB4_SUB5.json
"""
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(_k, '4')
import csv, json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'stage1')); sys.path.insert(0, str(HERE.parent / 'stage2'))
import stage1 as S1  # noqa: E402
import stage2 as S2  # noqa: E402
E = S1.E
ROOT = S1.ROOT
# STATUS_MAIN scores Hard2Verify on the 177-answer common set (collection_status.cells); load the same set here.
from collection_parity import h2v_common_uids  # noqa: E402
_COMMON, _load = h2v_common_uids(), E.load
E.load = lambda bench, prm, **kw: _load(bench, prm, **({'uids_keep': _COMMON} if bench == 'hard2verify' else {}), **kw)

NAMES = ['q15_H1', 'q15_VE1', 'chosen_surprisal', 'logprob_margin', 'true_tail50', 'energy_level', 'energy_innovation',
         'top15_turnover', 'top50_js', 'dominant_freq16', 'bocpd_p0', 'realized_z', 'realized_drv', 'digit_alternative',
         'digit_spread', 'digit_alternative_innovation']
SETS = {'SUB4': 'q15_VE1 + chosen_surprisal + energy_innovation + top15_turnover + realized_z + digit_alternative + '
                'digit_alternative_innovation',
        'SUB5': 'q15_H1 + chosen_surprisal + top15_turnover + dominant_freq16 + bocpd_p0 + digit_spread'}
BENCH = 'hard2verify'
PRMS = ['qwen_prm_7b', 'qwen_prm_72b', 'pure_prm', 'rlhflow_prm', 'math_shepherd']
B, SEED = 2000, 20261026
AUTO = {(r['bench'], r['prm']): r['A']['auto'] for r in
        json.loads((ROOT / 'results/prm_boost_math_rigor_v1/stage8/AUTO_FORCEDK_RESULTS.json').read_text())}


def gapf(c, s):
    return E.topk_flags(s, c.off, E.gap_counts(s, c.off))


def within(c, s):
    return float(np.nanmean(E.per_answer_auc(c.err, s, c.off, c.keep)))


def boot(c, fa, fb, rng_idx):
    Ca, Cb = E.counts_by_answer(c, fa), E.counts_by_answer(c, fb)
    d = [100 * (E.official_from_counts(Ca[i].sum(0), BENCH) - E.official_from_counts(Cb[i].sum(0), BENCH)) for i in rng_idx]
    return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


def main():
    rows, parity = [], []
    for prm in PRMS:
        s1 = S1.CellData(BENCH, prm); c = s1.c
        S2.PRM_ALONE[f'{BENCH}/{prm}'] = {'prm_exact': s1.ps_prm}
        t = S2.TeleCell().build(BENCH, prm)
        ref = S1.STATUS[(BENCH, prm, 'PRM same rule (better scale under GAP)')]
        parity.append(dict(prm=prm, prm_exact=s1.ps_prm, status=float(ref['GAP']), diff=abs(s1.ps_prm - float(ref['GAP']))))
        if parity[-1]['diff'] > 1e-6:
            raise SystemExit(f'PARITY FAIL {prm}: {parity[-1]}')
        fP = gapf(c, s1.Z[:, 0])
        rng = np.random.default_rng(SEED); idx = [rng.integers(0, len(c.off) - 1, len(c.off) - 1) for _ in range(B)]
        base = dict(prm=prm, n_answers=len(c.off) - 1, prm_ps=s1.ps_prm, prm_within=within(c, s1.Z[:, 0]))
        for arm in ('CHAIN_N16', 'TELE_N16'):
            base[f'{arm}_ps'] = float(ref and S1.STATUS[(BENCH, prm, arm)]['GAP'])
            base[f'{arm}_within'] = float(S1.STATUS[(BENCH, prm, arm)]['within_auc'])
        a = AUTO.get((BENCH, prm)); base['AUTO_N16_ps'] = a and a['gap_prmscore']; base['AUTO_N16_within'] = a and a['within_auc']
        for name, s in SETS.items():
            S = [NAMES.index(n) for n in s.split(' + ')]
            for tag, cell in (('+PRM', s1), ('alone', t)):
                o = cell.evaluate(S, want_detail=True)
                r = dict(base, set=name, version=tag, K=o['K'], valid=bool(o['valid']))
                if o['valid']:
                    ff, fm = gapf(c, o['fused']), gapf(c, o['mean'])
                    r.update(fused_ps=100 * float(E.official(c, ff)[0]), mean_ps=100 * float(E.official(c, fm)[0]),
                             fused_within=within(c, o['fused']), mean_within=within(c, o['mean']),
                             ci_vs_prm=boot(c, ff, fP, idx), ci_vs_mean=boot(c, ff, fm, idx),
                             prm_share=o.get('w_prm_share'))
                    r['vs_prm'] = r['fused_ps'] - r['prm_ps']; r['vs_mean'] = r['fused_ps'] - r['mean_ps']
                    r['vs_chain'] = r['fused_ps'] - base['CHAIN_N16_ps']; r['vs_tele'] = r['fused_ps'] - base['TELE_N16_ps']
                else:
                    r['reason'] = o.get('reason', 'K < 3')
                rows.append(r); print(prm, name, tag, {k: r.get(k) for k in ('K', 'fused_ps', 'mean_ps', 'prm_ps', 'vs_prm', 'vs_mean')}, flush=True)
    (HERE / 'H2V_SUB4_SUB5.json').write_text(json.dumps(dict(parity=parity, rows=rows), indent=1, default=float))
    keys = sorted({k for r in rows for k in r})
    with open(HERE / 'H2V_SUB4_SUB5.csv', 'w', newline='') as fh:
        w = csv.DictWriter(fh, keys); w.writeheader(); w.writerows(rows)


if __name__ == '__main__':
    main()
