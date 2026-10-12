"""PRM boosting: fuse Qwen2.5-Math-PRM-7B step rewards with an N16 telemetry bank on PRMBench,
label-free, with the unchanged Monday fitting adapter (results/monday_external_panel_v1/score.py).

PRM channel (declared before any metric): prm_risk = logit(1 - reward), p clipped to [1e-6, 1-1e-6].
Banks: N16, and N16P = N16 + prm_risk (PRM as one more scorer in the same fusion).
Metrics: within-answer AUC (mean over PRMBench answers with both classes, v3 labels) and official
PRMScore (spectral_utils.prmbench.prmbench_evaluate; (F1_valid + F1_error)/2 over non-control rows)
with the deployed ANSWER_Z_Q80 rule; PRM alone also with its native reward<0.5 rule.
Labels are read only for evaluation.

    python fuse.py [FEATURES.npz] [tag]
Default FEATURES = the Qwen3-8B think-fix mask1 source bank; for that input the N16 scores must
reproduce results/thinkfix_rescore_v1/prmscore_inputs/bank_N16.npz (gate 1e-6).
"""
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_k] = '2'
import importlib.util, json, pickle, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata

MAIN = Path(r'C:\Users\omris\TAU\hallucination_detection'); OUT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('ext_score', MAIN / 'results/monday_external_panel_v1/score.py')
ext = importlib.util.module_from_spec(spec); spec.loader.exec_module(ext)
sys.path.insert(0, str(MAIN))
from spectral_utils.prmbench import prmbench_evaluate  # noqa: E402

SE = MAIN / '.worktrees/readout-quickest-detection-v1/results/step_evidence_v1'
FEAT = Path(sys.argv[1]) if len(sys.argv) > 1 else MAIN / 'scratch/external_lr_prm_v1/inputs/src8b_FEATURES.npz'
TAG = sys.argv[2] if len(sys.argv) > 2 else 'qwen3_8b_mask1'
PARITY = MAIN / 'results/thinkfix_rescore_v1/prmscore_inputs/bank_N16.npz'
PRM_PKL = MAIN / 'dataset_cache/four_localization/prmbench_qwen25math7b_full/prmbench_prm.pkl'
N16 = json.loads((MAIN / 'results/monday_nested_banks_v2/BANKS.json').read_text())['N16']
SCORERS = ['MEAN_MOM_ga', 'HEM_MOM_ga', 'MEAN_MOM_cc', 'HEM_MOM_cc', 'TCI_ONLY_ga', 'ALL_EQ', 'DS_FILTER_EQ', 'MEAN_EQ', 'HEM_EQ']
CELL = 'prmbench_qwen3_8b'


def auc(y, s):
    n1 = int(y.sum()); n0 = len(y) - n1
    return np.nan if not n1 or not n0 else (rankdata(s)[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def main():
    t0 = time.time()
    A = pd.read_csv(SE / 'OOF_ANSWERS.csv', encoding='utf-8-sig')
    with np.load(SE / 'OOF_STEP_SCORES.npz') as z:
        OFF = z['offsets'].astype(np.int64); LAB = z['labels'].astype(bool)
    F = dict(np.load(FEAT))
    if [str(u) for u in F['uids']] != A.uid.astype(str).tolist() or not np.array_equal(F['offsets'], OFF):
        raise SystemExit('source order')
    prm = {r['idx']: r for r in pickle.load(open(PRM_PKL, 'rb')).values()}
    cells = A.cell.to_numpy(); ids = A.id.astype(str).to_numpy()
    ans = np.flatnonzero(cells == CELL); S = int(OFF[-1])
    risk = np.zeros(S)
    for i in ans:
        r = np.asarray(prm[ids[i]]['rewards'], float)
        if len(r) != OFF[i + 1] - OFF[i]: raise SystemExit('step count ' + ids[i])
        p = np.clip(1 - r, 1e-6, 1 - 1e-6); risk[OFF[i]:OFF[i + 1]] = np.log(p / (1 - p))
    F['values'] = np.column_stack([F['values'], risk]); F['names'] = np.r_[F['names'].astype(str), ['prm_risk']]
    ext.BN = dict(ext.BN, N16=N16, N16P=N16 + ['prm_risk'])
    pop = ext.Population(F, ['N16', 'N16P']); print('population', round(time.time() - t0), flush=True)
    meta = [{'idx': ids[i], 'classification': prm[ids[i]]['classification'], 'error_steps': prm[ids[i]]['error_steps']} for i in ans]
    lab = [LAB[OFF[i]:OFF[i + 1]] for i in ans]
    o = np.r_[0, np.cumsum([len(y) for y in lab])]

    def evaluate(step_score, flags):
        """step_score / flags concatenated over ans; flags True = predicted error."""
        au = np.array([auc(lab[j], step_score[o[j]:o[j + 1]]) for j in range(len(ans))])
        preds = [{'idx': ids[i], 'labels': (~flags[o[j]:o[j + 1]]).astype(int).tolist()} for j, i in enumerate(ans)]
        ev = prmbench_evaluate(preds, meta); t = ev['total']
        out = {'within_auc': float(np.nanmean(au)), 'n_auc': int(np.isfinite(au).sum()),
               'prmscore': 100 * (t['f1'] + t['negative_f1']) / 2, 'f1_valid': t['f1'], 'f1_error': t['negative_f1'],
               'flag_rate': float(flags.mean())}
        bc = ev['by_classification']
        for c in sorted({m['classification'] for m in meta} - {'correct'}):
            out['prmscore__' + c] = 100 * (bc['f1'][c] + bc['negative_f1'][c]) / 2
        return out

    rows = []
    rP = np.concatenate([risk[OFF[i]:OFF[i + 1]] for i in ans])
    native = np.concatenate([np.asarray(prm[ids[i]]['labels']) == 0 for i in ans])
    rows.append({'bank': 'PRM', 'scorer': 'PRM_REWARD', 'rule': 'NATIVE_0.5', **evaluate(rP, native)})
    parity, diag, scores = {}, {}, {}
    for bi, bk in enumerate(['N16', 'N16P']):
        bank = pop.cell_bank(bk, ans)
        if bk == 'N16P':      # PRM alone under the same deployed decision rule as our scorers
            j = bank['names'].index('prm_risk'); dd, _ = ext.decisions(bank['V'][:, j], bank, np.array([j]))
            rows.insert(1, {'bank': 'PRM', 'scorer': 'PRM_REWARD', 'rule': 'ANSWER_Z_Q80', **evaluate(rP, dd['ANSWER_Z_Q80'])})
        sc, surv, d = ext.fit_cell(pop, bank, 1000 * bi); diag[bk] = d
        if bk == 'N16' and TAG == 'qwen3_8b_mask1':
            with np.load(PARITY, allow_pickle=True) as z:
                if list(z['idx'].astype(str)) != list(ids[ans]): raise SystemExit('parity order')
                az = lambda v: ext.answer_standardize(v[:, None], o)[:, 0]
                parity = {s: float(np.max(np.abs(az(sc[s]) - z['S__' + s]))) for s in SCORERS if 'S__' + s in z.files and s != 'TCI_ONLY_ga'}
            print('parity', parity, flush=True)
            if max(parity.values()) > 1e-6: raise SystemExit('PARITY FAIL')
        for s in SCORERS:
            dd, _ = ext.decisions(sc[s], bank, surv); scores[f'{bk}__{s}'] = sc[s]
            rows.append({'bank': bk, 'scorer': s, 'rule': 'ANSWER_Z_Q80', **evaluate(sc[s], dd['ANSWER_Z_Q80'])})
        print(bk, 'survivors', d['survivors'], 'groups', d['groups'], 'prevalence', round(d['prevalence_hat'], 3),
              round(time.time() - t0), flush=True)
    df = pd.DataFrame(rows); df.to_csv(OUT / f'RESULTS_{TAG}.csv', index=False)
    np.savez_compressed(OUT / f'SCORES_{TAG}.npz', idx=ids[ans], offsets=o, prm_risk=rP, **scores)
    json.dump({'features': str(FEAT), 'tag': TAG, 'parity_vs_frozen_bank_N16': parity, 'fit_diagnostics': diag,
               'seconds': round(time.time() - t0, 1)},
              open(OUT / f'DIAGNOSTICS_{TAG}.json', 'w'), indent=1,
              default=lambda v: v.item() if isinstance(v, np.generic) else str(v))
    pd.set_option('display.width', 220)
    print(df[['bank', 'scorer', 'rule', 'within_auc', 'prmscore', 'f1_valid', 'f1_error', 'flag_rate']].round(4).to_string())


if __name__ == '__main__':
    main()
