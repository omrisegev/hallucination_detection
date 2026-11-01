"""POST HOC arm extension of the neighbor readout study (2026-11-01, after the CHAIN/TELE evaluation was read).

Omri: CHAIN_N16 is out of scope (it forces the PRM into its own group); the arms in scope are AUTO, SUB5 and TELE16.
TELE16 = TELE_N16, already in the frozen run. This script adds AUTO and SUB5 (backbone telemetry), running the FROZEN
run.py / neighbor_readout.py unchanged (hash-checked) on the new fused scores. No parameter, grid or rule changes.
  AUTO = stage8_auto_forced_k.auto(N16 channels, answer-z PRM)   (as in ds_gate_v1b_fixed/run_e1.py: arm_inputs)
  SUB5 = stage2.TeleCell.evaluate(SUB5)['fused'], telemetry only  (as in prm_sub5_all_benchmarks_v1: sub5_scores)
Standalone-telemetry variants are not on this machine and are not run.

    python arms_extension.py scores     # label-free fused scores -> fits_ext/scores/<bench>__<prm>.npz
    python arms_extension.py fit [w]    # frozen label-free fits -> fits_ext/
    python arms_extension.py evaluate   # -> EXT_CURVES.csv, EXT_RESULTS.csv, EXT_CONTRASTS.csv, EXT_VERDICT.json
"""
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_k] = '1'
import json, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import run as R  # noqa: E402  (frozen runner)
E = R.E
NEW_ARMS = ['AUTO', 'SUB5']
SC = HERE / 'fits_ext' / 'scores'
R.FITS = HERE / 'fits_ext'
R.ARMS = NEW_ARMS
R.CELLS = [(b, p, a) for b in R.BENCHES for p in R.PRMS for a in NEW_ARMS]

_orig_load = E.load


def _load_with_new_arms(bench, prm, **kw):
    c = _orig_load(bench, prm, **kw)
    f = SC / f'{bench}__{prm}.npz'
    if f.exists():
        D = np.load(f); assert np.array_equal(D['off'], c.off)
        for a in NEW_ARMS:
            if f'score__{a}' in D: c.arms[a] = D[f'score__{a}'].astype(float)
    return c


E.load = _load_with_new_arms


def scores():
    sys.path.insert(0, str(ROOT / 'results/prm_boost_math_rigor_v1')); sys.path.insert(0, str(ROOT / 'results/prm_view_selection_v1/stage2'))
    sys.path.insert(0, str(ROOT / 'results/prm_view_selection_v1/stage1'))
    import stage8_auto_forced_k as S8
    import stage2 as S2
    SC.mkdir(parents=True, exist_ok=True)
    for b in R.BENCHES:
        for p in R.PRMS:
            out = SC / f'{b}__{p}.npz'
            if out.exists(): continue
            t0 = time.time(); c = _orig_load(b, p)
            X = E.channels(c, 'N16'); zp = E.answer_z(c.arms['PRM_Q80' if b == 'prmbench' else 'PRM'], c.off)
            auto, ai = S8.auto(X, zp)
            S2.PRM_ALONE.setdefault(f'{b}/{p}', {'prm_exact': float('nan')})
            t = S2.TeleCell().build(b, p); assert np.array_equal(t.off, c.off)
            o = t.evaluate(SUB5_IDX, want_detail=True)
            d = dict(off=c.off, score__AUTO=np.asarray(auto, float))
            if o['valid']: d['score__SUB5'] = np.asarray(o['fused'], float)
            np.savez(out, **d)
            print(b, p, f'AUTO K={ai["K"]} prm_alone={ai["prm_alone"]}', f'SUB5 K={o["K"]} valid={o["valid"]}',
                  f'{time.time() - t0:.0f}s', flush=True)


N16N = ['q15_H1', 'q15_VE1', 'chosen_surprisal', 'logprob_margin', 'true_tail50', 'energy_level', 'energy_innovation',
        'top15_turnover', 'top50_js', 'dominant_freq16', 'bocpd_p0', 'realized_z', 'realized_drv', 'digit_alternative',
        'digit_spread', 'digit_alternative_innovation']
SUB5_IDX = [N16N.index(n) for n in ['q15_H1', 'chosen_surprisal', 'top15_turnover', 'dominant_freq16', 'bocpd_p0', 'digit_spread']]


def _cells_available():
    ok = []
    for b, p, a in R.CELLS:
        f = SC / f'{b}__{p}.npz'
        if f.exists() and f'score__{a}' in np.load(f): ok.append((b, p, a))
    return ok


def fit(workers):
    R.check_hashes(); R.FITS.mkdir(exist_ok=True)
    import multiprocessing as mp
    cells = _cells_available()
    with mp.get_context('fork').Pool(workers) as pool:
        for name, st in pool.imap_unordered(R.fit_cell, cells):
            print(time.strftime('%H:%M:%S'), name, st, flush=True)


def evaluate():
    R.check_hashes()
    cells = _cells_available(); curves, rows, con = [], [], []
    for cell in cells:
        t0 = time.time(); cv, rw, cn = R.eval_cell(cell); curves += cv; rows += rw; con += cn
        print(time.strftime('%H:%M:%S'), R.tag(cell), f'{time.time() - t0:.0f}s', flush=True)
    E.write_csv(HERE / 'EXT_CURVES.csv', curves); E.write_csv(HERE / 'EXT_RESULTS.csv', rows)
    E.write_csv(HERE / 'EXT_CONTRASTS.csv', con)
    missing = sorted(set(R.CELLS) - set(cells))
    try:
        v = R.verdict(curves, rows, con)
    except AssertionError as e:                                 # a missing (invalid) cell breaks the 4-PRM rules
        v = {'error': f'verdict incomplete: {str(e)[:200]}'}
    v['missing_cells'] = [list(m) for m in missing]
    json.dump(R.E_json(v), open(HERE / 'EXT_VERDICT.json', 'w'), indent=1)
    print('written EXT_* ; missing cells:', missing)


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'scores': scores()
    elif cmd == 'fit': fit(int(sys.argv[2]) if len(sys.argv) > 2 else 8)
    elif cmd == 'evaluate': evaluate()
    else: raise SystemExit(cmd)
