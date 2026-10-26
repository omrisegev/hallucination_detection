"""Re-reads of existing outputs for the 8 cells without Math-Shepherd (Omri's decision 2026-10-26); no new fit.

- stage-1 and stage-2 successes on 8 cells, and which channels the stage-1 8-cell successes over-represent;
- HEM (stage 1 secondary, 300 subsets) on 8 cells;
- position: within-answer correlation of every view with relative position (label-free), the same within the valid
  class (labels as a diagnostic), and relative position's own within-answer AUC (labels).

    python existing_8cell.py   # -> EXISTING_8CELL.json, POSITION_CORR.csv
"""
import csv, json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
VS = HERE.parent
ROOT = VS.parents[1]
sys.path.insert(0, str(ROOT / 'results/prm_boost_collection_v1'))
import evaluate_collection as E  # noqa: E402

NAMES = ['q15_H1', 'q15_VE1', 'chosen_surprisal', 'logprob_margin', 'true_tail50', 'energy_level', 'energy_innovation',
         'top15_turnover', 'top50_js', 'dominant_freq16', 'bocpd_p0', 'realized_z', 'realized_drv', 'digit_alternative',
         'digit_spread', 'digit_alternative_innovation']
PRMS = ['qwen_prm_7b', 'qwen_prm_72b', 'pure_prm', 'rlhflow_prm']


def name_of(m):
    return '+'.join(NAMES[i] for i in range(16) if (int(m) >> i) & 1)


def main():
    out = {}
    D1, D2 = np.load(VS / 'stage1' / 'SEARCH.npz'), np.load(VS / 'stage2' / 'SEARCH.npz')
    P = json.loads((VS / 'stage1' / 'PARITY.json').read_text()); cells = list(D1['cells'])
    prm = np.array([P[c]['prm_exact'] for c in cells]); c8 = [i for i, c in enumerate(cells) if 'math_shepherd' not in c]
    m = D1['masks']; bits = np.array([[(int(x) >> i) & 1 for i in range(16)] for x in m], bool)
    v1 = D1['valid'][:, c8].all(1); f1, mu1 = D1['ps_fused'][:, c8], D1['ps_mean'][:, c8]
    s1 = v1 & (f1 > prm[c8]).all(1) & (f1 > mu1).all(1)
    v2 = D2['valid'][:, c8].all(1); f2, mu2 = D2['ps_fused'][:, c8], D2['ps_mean'][:, c8]
    s2 = v2 & (f2 > mu2).all(1)
    mm1 = np.minimum(f1 - prm[c8], f1 - mu1).min(1)
    out['stage1_8cell'] = dict(valid=int(v1.sum()), success=int(s1.sum()),
                               successes=[dict(subset=name_of(m[i]), min_margin=float(mm1[i]),
                                               socratic_prm_share=np.round(D1['w_prm_share'][i, [5, 6, 7, 8]], 3).tolist(),
                                               stage2_beats_own_mean_cells=int(((f2[i] > mu2[i]) & D2['valid'][i, c8]).sum()))
                                          for i in np.where(s1)[0][np.argsort(-mm1[s1])]])
    trio = bits[:, NAMES.index('top15_turnover')] & bits[:, NAMES.index('dominant_freq16')] & bits[:, NAMES.index('bocpd_p0')]
    out['stage1_8cell']['trio_t15_dom_bocpd'] = dict(in_successes=int((trio & s1).sum()), base_rate_valid=float(trio[v1].mean()))
    out['stage1_8cell']['channel_freq'] = {n: dict(successes=float(bits[s1, k].mean()), valid=float(bits[v1, k].mean()))
                                           for k, n in enumerate(NAMES)}
    out['stage2_8cell'] = dict(valid=int(v2.sum()), success=int(s2.sum()), joint=[name_of(x) for x in m[s1 & s2]],
                               share_beating_prm_by_cell=dict(zip([cells[j] for j in c8],
                                                                  np.round((f2[s2] > prm[c8]).mean(0), 3).tolist())))
    # HEM (stage 1 secondary) on 8 cells
    pos = {name_of(x): i for i, x in enumerate(m)}; n_ok = 0; best = -np.inf
    for r in csv.DictReader(open(VS / 'stage1' / 'HEM.csv')):
        i = pos['+'.join(r['subset'].split(' + '))]
        h = np.array([float(r[f'hem {cells[j]}']) for j in c8])
        marg = np.minimum(h - prm[c8], h - D1['ps_mean'][i, c8]).min(); best = max(best, marg)
        n_ok += bool(marg > 0)
    out['hem_8cell'] = dict(n_subsets=300, successes=n_ok, best_min_margin=float(best))
    # position
    rows = []
    for b in ('prmbench', 'socratic'):
        for p in PRMS:
            C = np.load(VS / 'stage1' / 'cache' / f'{b}__{p}.npz'); Z, off, pool = C['Z'], C['off'], C['pool']
            n = np.diff(off); rp = np.concatenate([(np.arange(k) + .5) / k for k in n])
            rc = rp - np.repeat(np.add.reduceat(rp, off[:-1]) / n, n)
            ok = C['yv_m'] & pool; err = C['nyv_m']
            for k, nm in enumerate(['PRM'] + NAMES):
                rows.append(dict(cell=f'{b}/{p}', view=nm, stage1_sign=int(C['sign'][k]),
                                 corr_position=float(np.corrcoef(Z[pool, k], rc[pool])[0, 1]),
                                 corr_position_valid_class=float(np.corrcoef(Z[ok, k], rc[ok])[0, 1])))
            if p == PRMS[0]:
                keep = None if b == 'prmbench' else (C['yv_m'] | C['nyv_m'])
                out[f'position_within_auc_{b}'] = float(np.nanmean(E.per_answer_auc(err, rp, off, keep)))
    E.write_csv(HERE / 'POSITION_CORR.csv', rows)
    (HERE / 'EXISTING_8CELL.json').write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != 'stage1_8cell'}, indent=1))
    print(json.dumps({k: out['stage1_8cell'][k] for k in ('valid', 'success', 'trio_t15_dom_bocpd')}, indent=1))


if __name__ == '__main__':
    main()
