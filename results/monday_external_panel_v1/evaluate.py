"""monday_external_panel_v1 evaluation: seal predictions, then open labels and evaluate.

Reuses the reviewed external helpers (confusion / summary / bootstrap / within_auc, official evaluator replay).
Usage: python evaluate.py [--seal-only]
"""
from __future__ import annotations
import os
for _k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_k] = '1'
import argparse, ast, hashlib, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import recall_score

MAIN = Path(r'C:\Users\omris\TAU\hallucination_detection'); OUT = Path(__file__).resolve().parent
SSL = MAIN / '.worktrees/ssl-pseudolabel-residual-v1'; sys.path.insert(0, str(SSL))
from spectral_utils.external_generalization.contracts import Answer, overlap_manifest  # noqa: E402
from spectral_utils.external_generalization.evaluation import confusion, within_auc  # noqa: E402
from spectral_utils.family_external_metrics import summary, bootstrap  # noqa: E402

INPUTS = MAIN / 'scratch/external_generalization_private/inputs'
CELLS = {'hard2verify_qwen3_8b': 'hard2verify', 'socratic_qwen3_8b': 'socratic', 'socratic_qwq32b': 'socratic'}
BANKS = ['B13', 'B16', 'B20', 'B23', 'B32', 'B35', 'B51', 'B54']; HL = 'B16'
METHODS = ['HEM_MOM_cc', 'MEAN_MOM_cc', 'MEAN_MOM_ga', 'TCI_ONLY_ga']
SCORERS = ['HEM_MOM_cc', 'MEAN_MOM_cc', 'MEAN_MOM_ga', 'HEM_MOM_ga', 'TCI_ONLY_ga', 'ALL_EQ', 'DS_FILTER_EQ', 'MEAN_EQ', 'HEM_EQ']
DRAWS, SEED = 5000, 20261006


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''): h.update(b)
    return h.hexdigest()


def load(p): return json.loads(Path(p).read_text(encoding='utf8'))


def pinned(path, names, ns):
    nodes = [n for n in ast.parse(path.read_text(encoding='utf8')).body if isinstance(n, ast.FunctionDef) and n.name in names]
    if {n.name for n in nodes} != set(names): raise ValueError('official functions missing')
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), ns); return ns


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--seal-only', action='store_true'); a = ap.parse_args()
    st = load(OUT / 'SCORING_STATUS.json')
    if st['external_labels_opened'] is not False or st['protocol_sha256'] != sha(OUT / 'PROTOCOL.json') or st['score_script_sha256'] != sha(OUT / 'score.py'):
        raise SystemExit('protocol / scoring script changed or labels opened')
    seals = {}
    for cell in CELLS:
        s = {'predictions_sha256': sha(OUT / cell / 'PREDICTIONS.npz'), 'scoring_status_sha256': sha(OUT / 'SCORING_STATUS.json'),
             'protocol_sha256': sha(OUT / 'PROTOCOL.json'), 'evaluator_sha256': sha(Path(__file__))}
        if s['predictions_sha256'] != st['cells'][cell]['predictions_sha256']: raise SystemExit('predictions changed after scoring')
        if (OUT / cell / 'SEAL.json').exists():
            old = load(OUT / cell / 'SEAL.json')
            if old['predictions_sha256'] != s['predictions_sha256']: raise SystemExit('sealed predictions changed')
        else:
            (OUT / cell / 'SEAL.json').write_text(json.dumps(s, indent=1), encoding='utf8')
        seals[cell] = s
    if a.seal_only:
        print('sealed', {c: v['predictions_sha256'][:12] for c, v in seals.items()}); return
    # ------------------------------------------------------------ labels are opened only below this line
    answers = {b: [Answer(**{**r, 'steps': tuple(r['steps'])}) for r in load(INPUTS / b / 'answers.json')] for b in set(CELLS.values())}
    gold = {b: {g['uid']: g for g in load(INPUTS / 'evaluator_only' / (b + '.json'))} for b in set(CELLS.values())}
    groups_map = overlap_manifest(answers)['groups']
    src = MAIN / 'scratch/external_generalization_private/sources'
    official = {'hard': pinned(src / 'hard2verify/utils.py', {'calculate_metrics'}, {'recall_score': recall_score}),
                'soc': pinned(src / 'prmeval_classified_task.py', {'evaluate_function', 'eval_on_hallucination_step'}, {})}
    rows, cons, replay, aucstore = [], [], {}, {}
    for cell, bench in CELLS.items():
        Z = np.load(OUT / cell / 'PREDICTIONS.npz', allow_pickle=False)
        uids = [str(u) for u in Z['uids']]; starts = Z['starts']; ne = Z['nonempty'].astype(bool)
        scn = [str(x) for x in Z['score_names']]; prn = [str(x) for x in Z['prediction_names']]; refn = [str(x) for x in Z['ref_names']]
        S = np.column_stack([Z['scores'], Z['ref_scores']]); scn_all = scn + refn
        Pm = np.column_stack([Z['predictions'], Z['ref_predictions']]); prn_all = prn + refn
        if set(gold[bench]) != set(uids): raise SystemExit('gold population mismatch ' + cell)
        groups = [groups_map[u] for u in uids]
        counts = {p: np.zeros((len(uids), 4), np.int64) for p in prn_all}
        aucs = {s: np.full(len(uids), np.nan) for s in scn_all}
        for i, u in enumerate(uids):
            g = gold[bench][u]; a0, b0 = starts[i], starts[i + 1]; include = np.asarray(g['include'], bool)
            if len(include) != b0 - a0: raise SystemExit('label alignment ' + u)
            valid = include & ne[a0:b0]
            for k, p in enumerate(prn_all): counts[p][i] = confusion(g['correct'], Pm[a0:b0, k].astype(int), include)
            for k, s in enumerate(scn_all):
                v = within_auc(g['correct'], S[a0:b0, k], valid); aucs[s][i] = np.nan if v is None else v
        # official replay for the highlighted bank + references (exact equality of the primary metric)
        rep = {}
        for p in [x for x in prn_all if x.startswith(HL + '__') or x in refn or x.startswith('STEP_INDEX')]:
            k = prn_all.index(p)
            if bench == 'hard2verify':
                y = [int(v) for u in uids for v, keep in zip(gold[bench][u]['correct'], gold[bench][u]['include']) if keep]
                pr = [int(v) for i, u in enumerate(uids) for v, keep in zip(Pm[starts[i]:starts[i + 1], k], gold[bench][u]['include']) if keep]
                off_v = official['hard']['calculate_metrics'](pr, y)['balanced_f1_score']
                ours = round(100 * summary(counts[p], bench)['metric'], 2); rep[p] = (off_v, ours)
                if off_v != ours: raise SystemExit(f'Hard2Verify official mismatch {p}: {off_v} vs {ours}')
            else:
                meta, preds = [], []
                for i, u in enumerate(uids):
                    g = gold[bench][u]; errs = [j + 1 for j, c in enumerate(g['correct']) if not c] + g.get('out_of_range_error_indices', [])
                    meta.append(dict(idx=u, classification=g['category'], error_steps=errs))
                    preds.append(dict(idx=u, scores={'step_level_validity_labels': Pm[starts[i]:starts[i + 1], k].astype(int).tolist()}))
                o = official['soc']['evaluate_function'](preds, meta)['total_hallucination_results']
                comp = summary(counts[p], bench); err = max(abs(o['f1'] - comp['f1_correct']), abs(o['negative_f1'] - comp['f1_error']))
                rep[p] = err
                if err > 1e-12: raise SystemExit(f'Socratic official mismatch {p}: {err}')
        replay[cell] = {'arms_checked': len(rep), 'pass': True}
        boot_arms = [x for x in prn_all if x.startswith(HL + '__') or x in refn or x.startswith('STEP_INDEX')]   # CIs for the highlight + references only (memory)
        names, samples, _ = bootstrap({x: counts[x] for x in boot_arms}, groups, bench, DRAWS, SEED)
        for p in prn_all:
            sm = summary(counts[p], bench); scorer = p.rsplit('__', 1)[0] if '__' in p else p; rule = p.rsplit('__', 1)[1] if '__' in p else 'sealed_reference'
            bk = scorer.split('__')[0] if scorer.split('__')[0] in BANKS else '-'; sname = scorer.split('__', 1)[1] if bk != '-' else scorer
            au = aucs.get(scorer, aucs.get(p))
            col = samples[:, names.index(p)] if p in names else np.array([np.nan]); col = col[np.isfinite(col)] if np.isfinite(col).any() else np.array([np.nan])
            rows.append({'cell': cell, 'benchmark': bench, 'bank': bk, 'scorer': sname, 'rule': rule, 'official_metric': sm['metric'],
                         'ci95_lo': float(np.quantile(col, .025)), 'ci95_hi': float(np.quantile(col, .975)),
                         'f1_correct': sm['f1_correct'], 'f1_error': sm['f1_error'], 'recall_correct': sm['recall_correct'], 'recall_error': sm['recall_error'],
                         'within_auc': float(np.nanmean(au)) if au is not None and np.isfinite(au).any() else np.nan})
        # paired contrasts at the highlighted bank (descriptive 95%)
        ug, gi = np.unique(groups, return_inverse=True)
        rng = np.random.default_rng(SEED); W = rng.multinomial(len(ug), np.full(len(ug), 1 / len(ug)), size=DRAWS).astype(float)
        def auc_delta(x, y):
            d = aucs[x] - aucs[y]; ok = np.isfinite(d); s = np.bincount(gi[ok], weights=d[ok], minlength=len(ug)); c = np.bincount(gi[ok], minlength=len(ug)).astype(float)
            bs = (W @ s) / (W @ c); return float(s.sum() / c.sum()), float(np.quantile(bs, .025)), float(np.quantile(bs, .975))
        for m in METHODS:
            for ref, rule in ((f'{HL}__DS_FILTER_EQ', 'ANSWER_Z_Q80'), ('frozen_lsml', None), ('ct7', None)):
                lp = f'{HL}__{m}__ANSWER_Z_Q80'; rp = f'{ref}__{rule}' if rule else ref
                d = samples[:, names.index(lp)] - samples[:, names.index(rp)]; d = d[np.isfinite(d)]
                pt = summary(counts[lp], bench)['metric'] - summary(counts[rp], bench)['metric']
                cons.append({'cell': cell, 'method': m, 'versus': ref, 'metric': 'official (ANSWER_Z_Q80)', 'delta': pt, 'lo95': float(np.quantile(d, .025)), 'hi95': float(np.quantile(d, .975))})
            for ref in (f'{HL}__DS_FILTER_EQ', 'STEP_INDEX', 'frozen_lsml'):
                pt, lo, hi = auc_delta(f'{HL}__{m}', ref)
                cons.append({'cell': cell, 'method': m, 'versus': ref, 'metric': 'within_auc', 'delta': pt, 'lo95': lo, 'hi95': hi})
        np.savez_compressed(OUT / cell / 'ANSWER_AUC.npz', **{k: v for k, v in aucs.items()}, uids=np.array(uids), groups=np.array(groups))
        print(cell, 'evaluated', flush=True)
    R = pd.DataFrame(rows); R.to_csv(OUT / 'METRICS.csv', index=False); pd.DataFrame(cons).to_csv(OUT / 'CONTRASTS.csv', index=False)
    # ------------------------------------------------------------ development vs external transfer
    dev = pd.read_csv(MAIN / 'results/monday_presentation_panels_v1/run_20261006/PANEL_A_ABSOLUTE.csv')
    tr = []
    for cell in CELLS:
        ext = R[(R.cell == cell) & (R.rule == 'ANSWER_Z_Q80') & (R.bank != '-')][['bank', 'scorer', 'within_auc', 'official_metric']]
        j = dev.merge(ext, on=['bank', 'scorer'])
        rho_auc = spearmanr(j.prmb_within_auc, j.within_auc).statistic; rho_off = spearmanr(j.prmb_within_auc, j.official_metric).statistic
        g = j.merge(j[j.scorer == 'DS_FILTER_EQ'][['bank', 'prmb_within_auc', 'within_auc']], on='bank', suffixes=('', '_ref'))
        g = g[g.scorer.isin(METHODS + ['HEM_MOM_ga'])]
        agree = int((np.sign(g.prmb_within_auc - g.prmb_within_auc_ref) == np.sign(g.within_auc - g.within_auc_ref)).sum())
        hl = j[(j.bank == HL) & j.scorer.isin(METHODS)]; rank = j.within_auc.rank(ascending=False)
        tr.append({'cell': cell, 'configs': len(j), 'spearman_dev_auc_vs_ext_auc': rho_auc, 'spearman_dev_auc_vs_ext_official': rho_off,
                   'gain_sign_agreement': f'{agree}/{len(g)}', 'B16_methods_ext_auc_rank_best': int(rank[hl.index].min()), 'B16_methods_ext_auc_rank_worst': int(rank[hl.index].max())})
        j.to_csv(OUT / cell / 'DEV_VS_EXTERNAL.csv', index=False)
    pd.DataFrame(tr).to_csv(OUT / 'TRANSFER.csv', index=False)
    (OUT / 'EVALUATION_PROVENANCE.json').write_text(json.dumps({'official_replay': replay, 'draws': DRAWS, 'seed': SEED, 'evaluator_sha256': sha(Path(__file__)),
        'labels_sha256': {b: sha(INPUTS / 'evaluator_only' / (b + '.json')) for b in set(CELLS.values())}, 'seals': seals}, indent=1), encoding='utf8')
    print(pd.DataFrame(tr).to_string(index=False))


if __name__ == '__main__':
    main()
