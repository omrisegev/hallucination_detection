"""Amendment A2 supplement (development): ALL_EQ_ORIENTED control, N13/N16 parity on all 9 cells, failure-reason check.
No refit of the methods; only the bank construction of the validated adapter is reused."""
from __future__ import annotations
import hashlib, importlib.util, json, re, time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata

OUT = Path(__file__).resolve().parent; MAIN = OUT.parents[1]; EXT = MAIN / 'results/monday_external_panel_v1'
spec = importlib.util.spec_from_file_location('dev_run', OUT / 'dev_run.py'); dr = importlib.util.module_from_spec(spec); spec.loader.exec_module(dr)
ext = dr.ext; NB = dr.NB; BANKS = dr.BANKS
PANEL = MAIN / 'results/monday_candidate_panel_v1/run_20261004'; TCIR = MAIN / 'results/mom_filter_within_diagnostic_20261004/tci_wide_20261005'
GROUPING = re.compile(r'groups before merge: minimum is 3|Fewer than 3 groups')


def main():
    t0 = time.time()
    cov = json.loads((OUT / 'COVERAGE.json').read_text(encoding='utf8'))
    bad = [f for f in cov['failures'] if not GROUPING.search(f['reason'])]
    if bad: raise SystemExit('non-registered failures: ' + json.dumps(bad))
    A = pd.read_csv(dr.SE / 'OOF_ANSWERS.csv', encoding='utf-8-sig'); cells = A.cell.to_numpy()
    F = np.load(ext.FE / 'source/FEATURES.npz'); pop = ext.Population(F, BANKS)
    PA = np.load(OUT / 'PER_ANSWER.npz')
    with np.load(dr.SE / 'OOF_STEP_SCORES.npz') as z: OFF = z['offsets'].astype(np.int64); LAB = z['labels'].astype(bool)
    # --- oriented plain average, per bank x cell (label-free sign vs LEVEL mean)
    N = len(A); ori = {bk: {'auc': np.full(N, np.nan), 'argmax': np.full(N, -9, int), 'top1err': np.full(N, np.nan)} for bk in BANKS}; signs = {}
    for bk in BANKS:
        for ce in sorted(set(cells)):
            ans = np.flatnonzero(cells == ce); bank = pop.cell_bank(bk, ans); V = bank['V']; lev = pop.level[bank['rows']]
            sg = np.array([np.sign(np.corrcoef(V[:, j], lev)[0, 1]) or 1. if V[:, j].std() > 0 else 1. for j in range(V.shape[1])])
            signs[f'{bk}/{ce}'] = {bank['names'][j]: float(sg[j]) for j in range(len(sg)) if sg[j] < 0}
            s = (V * sg).mean(1); off = bank['off']
            for j, i in enumerate(ans):
                ss = s[off[j]:off[j + 1]]; am = int(np.flatnonzero(ss >= ss.max() - 8 * np.finfo(float).eps)[0]); ori[bk]['argmax'][i] = am
                if ce == dr.PRM:
                    y = LAB[OFF[i]:OFF[i + 1]]; n1 = y.sum(); n0 = len(y) - n1
                    if n1 and n0: ori[bk]['auc'][i] = (rankdata(ss)[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
                    if n1: ori[bk]['top1err'][i] = float(y[am])
        print(bk, 'oriented', round(time.time() - t0), flush=True)
    # aggregate with the same definitions (point estimates + paired contrasts vs ALL_EQ / methods)
    prm_idx = np.flatnonzero(cells == dr.PRM); pb_idx = np.flatnonzero(cells != dr.PRM); target = A.target.to_numpy()
    pb_cells = sorted(set(cells[pb_idx])); groups = A.source_group.to_numpy()
    pg, pinv = np.unique(groups[prm_idx], return_inverse=True); bg, binv = np.unique(groups[pb_idx], return_inverse=True)
    rng = np.random.default_rng(dr.SEED)
    Wp = np.vstack([np.ones(len(pg)), rng.multinomial(len(pg), np.full(len(pg), 1 / len(pg)), size=dr.DRAWS)]).astype(float)
    Wb = np.vstack([np.ones(len(bg)), rng.multinomial(len(bg), np.full(len(bg), 1 / len(bg)), size=dr.DRAWS)]).astype(float)
    def gs(X, inv, G): o = np.zeros((G, X.shape[1])); np.add.at(o, inv, X); return o
    def prm_auc(a):
        a = a[prm_idx]; S = Wp @ gs(np.column_stack([np.nan_to_num(a), np.isfinite(a)]).astype(float), pinv, len(pg)); return S[:, 0] / S[:, 1]
    def pb_hit(am):
        hit = am[pb_idx] == target[pb_idx]; err = target[pb_idx] >= 0; cols = []
        for c in pb_cells: ck = cells[pb_idx] == c; cols += [hit & err & ck, err & ck]
        S = (Wb @ gs(np.column_stack(cols).astype(float), binv, len(bg))).reshape(len(Wb), len(pb_cells), 2); return np.nanmean(S[:, :, 0] / S[:, :, 1], 1)
    ci = lambda x: (float(x[0]), float(np.quantile(x[1:], .025)), float(np.quantile(x[1:], .975)))
    rows, cons = [], []
    for bk in BANKS:
        o_auc, o_hit = prm_auc(ori[bk]['auc']), pb_hit(ori[bk]['argmax'])
        rows.append({'bank': bk, 'scorer': 'ALL_EQ_ORIENTED', 'prmb_within_auc': o_auc[0], 'pb_argmax_hit': o_hit[0],
                     'prmb_top1_on_error': float(np.nanmean(ori[bk]['top1err'][prm_idx])), 'n_flipped_channels_prmbench': len(signs[f'{bk}/{dr.PRM}'])})
        for name in dr.METHODS + ['ALL_EQ', 'DS_FILTER_EQ']:
            key = f'{bk}__{name}__auc'
            if key not in PA.files or (PA.get(f'{bk}__{name}__failed') is not None and PA[f'{bk}__{name}__failed'].any()): continue
            for metric, mine, ref in (('prmb_within_auc', prm_auc(PA[key]), o_auc), ('pb_argmax_hit', pb_hit(PA[f'{bk}__{name}__argmax']), o_hit)):
                pt, lo, hi = ci(mine - ref); cons.append({'bank': bk, 'method': name, 'reference': 'ALL_EQ_ORIENTED', 'metric': metric, 'delta': pt, 'lo95': lo, 'hi95': hi})
    pd.DataFrame(rows).to_csv(OUT / 'ORIENTED_CONTROL.csv', index=False); pd.DataFrame(cons).to_csv(OUT / 'ORIENTED_CONTRASTS.csv', index=False)
    (OUT / 'ORIENTED_SIGNS.json').write_text(json.dumps(signs, indent=1), encoding='utf8')
    # --- parity N13/N16 vs frozen B13/B16 (all 9 cells): re-fit is not needed; compare argmax/auc arrays implied by frozen scores
    par = []
    for nb, ob in (('N13', 'B13'), ('N16', 'B16')):
        assert NB[nb] == json.loads((MAIN / 'results/estimator_input_v1/run_20261004_resume/INPUT_MANIFEST.json').read_text(encoding='utf8'))['banks'][ob]
        for ce in sorted(set(cells)):
            ans = np.flatnonzero(cells == ce); off = np.r_[0, np.cumsum(np.diff(OFF)[ans])]
            for variant, tag in (('groups_answer', 'ga'), ('channels_cell', 'cc')):
                with np.load(PANEL / 'candidate_scores' / f'{ce}__{ob}__{variant}.npz') as z:
                    for r in ('MEAN_MOM', 'HEM_MOM'):
                        s = z[f'{r}__score']; am = np.array([int(np.flatnonzero(s[a:b] >= s[a:b].max() - 8 * np.finfo(float).eps)[0]) for a, b in zip(off[:-1], off[1:])])
                        mism = int((am != PA[f'{nb}__{r}_{tag}__argmax'][ans]).sum())
                        par.append({'nested': nb, 'frozen': ob, 'cell': ce, 'scorer': f'{r}_{tag}', 'argmax_mismatches': mism, 'answers': len(ans)})
            with np.load(TCIR / 'scores' / f'{ob}__{ce}.npz') as z:
                for nm, key in (('DS_FILTER_EQ', 'plain'), ('TCI_ONLY_ga', 'TCI_ONLY_ga')):
                    s = z[key]; am = np.array([int(np.flatnonzero(s[a:b] >= s[a:b].max() - 8 * np.finfo(float).eps)[0]) for a, b in zip(off[:-1], off[1:])])
                    par.append({'nested': nb, 'frozen': ob, 'cell': ce, 'scorer': nm, 'argmax_mismatches': int((am != PA[f'{nb}__{nm}__argmax'][ans]).sum()), 'answers': len(ans)})
    P = pd.DataFrame(par); P.to_csv(OUT / 'PARITY_N13_N16.csv', index=False)
    summ = P.groupby('scorer').argmax_mismatches.sum().to_dict()
    # A2: per-benchmark completeness -> PRMBench rows for banks incomplete only on ProcessBench; highlight by PRMBench completeness
    M4 = ['HEM_MOM_cc', 'MEAN_MOM_cc', 'MEAN_MOM_ga', 'TCI_ONLY_ga']; prm_rows = []
    for bk in BANKS:
        for name in dr.METHODS + ['ALL_EQ', 'DS_FILTER_EQ', 'MEAN_EQ', 'HEM_EQ']:
            fk = f'{bk}__{name}__failed'
            failed_prm = bool(PA[fk][prm_idx].any()) if fk in PA.files else True
            failed_pb = bool(PA[fk][pb_idx].any()) if fk in PA.files else True
            row = {'bank': bk, 'scorer': name, 'prmbench_complete': not failed_prm, 'processbench_complete': not failed_pb}
            if not failed_prm:
                a = prm_auc(PA[f'{bk}__{name}__auc']); row['prmb_within_auc'] = a[0]; row['prmb_top1_on_error'] = float(np.nanmean(PA[f'{bk}__{name}__top1err'][prm_idx]))
            prm_rows.append(row)
    PR = pd.DataFrame(prm_rows); PR.to_csv(OUT / 'PER_BENCHMARK_COMPLETENESS.csv', index=False)
    elig = [bk for bk in BANKS if PR[(PR.bank == bk) & PR.scorer.isin(M4)].prmbench_complete.all()]
    mean4 = PR[PR.bank.isin(elig) & PR.scorer.isin(M4)].groupby('bank').prmb_within_auc.mean().sort_values(ascending=False)
    a1 = json.loads((OUT / 'HIGHLIGHT.json').read_text(encoding='utf8'))
    if not (OUT / 'HIGHLIGHT_A1.json').exists(): (OUT / 'HIGHLIGHT_A1.json').write_text(json.dumps(a1, indent=1), encoding='utf8')
    if (OUT / 'SCORING_STATUS.json').exists(): raise SystemExit('external already scored: highlight cannot be re-decided')
    hl = {'rule': 'A2: highest mean PRMBench within-answer AUC over the 4 registered methods among banks complete on PRMBench', 'highlight': str(mean4.index[0]),
          'mean_auc_by_bank': mean4.to_dict(), 'a1_highlight': a1['highlight'], 'external_scoring_status_absent_at_decision': True,
          'previously_seen_externally': str(mean4.index[0]) in ('N13', 'N16')}
    (OUT / 'HIGHLIGHT.json').write_text(json.dumps(hl, indent=1), encoding='utf8')
    # longest-step share (definition: token-length longest step, first maximum; erroneous ProcessBench answers; cell-pooled)
    with np.load(json.loads((dr.SE / 'INPUT_FREEZE.json').read_text(encoding='utf-8-sig'))['tokens']['path']) as z: spans = z['step_spans']
    tl = spans[:, 1] - spans[:, 0]; err = pb_idx[target[pb_idx] >= 0]
    longest = np.array([int(np.argmax(tl[OFF[i]:OFF[i + 1]])) for i in err])
    share = {f'{bk}/{n}': float((PA[f'{bk}__{n}__argmax'][err] == longest).mean()) for bk in BANKS for n in M4 if f'{bk}__{n}__argmax' in PA.files and not PA[f'{bk}__{n}__failed'][err].any()}
    share['MTG'] = float((PA['MTG__argmax'][err] == longest).mean())
    (OUT / 'LONGEST_STEP_SHARE.json').write_text(json.dumps({'definition': 'share of erroneous ProcessBench answers (pooled over the 8 cells) whose top-ranked step is the token-longest step (first maximum)', 'share': share}, indent=1), encoding='utf8')
    ext_scored = (OUT / 'SCORING_STATUS.json').exists()
    (OUT / 'A2_STATUS.json').write_text(json.dumps({'non_registered_failures': 0, 'registered_failures': cov['failures'], 'parity_argmax_mismatches_by_scorer': summ,
        'parity_note': 'All non-TCI scorers must have 0 mismatches; TCI differs by design (seed by bank position).',
        'highlight': hl['highlight'], 'highlight_previously_seen_externally': hl['highlight'] in ('N13', 'N16'),
        'external_scored_before_highlight_check': ext_scored, 'runtime_s': round(time.time() - t0)}, indent=1), encoding='utf8')
    print(json.dumps(summ), 'highlight', hl['highlight'])


if __name__ == '__main__':
    main()
