#!/usr/bin/env python
"""Frozen two-bank, five-source-fold development comparison; CPU only."""
from __future__ import annotations
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import hashlib, html, json, sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from spectral_utils.advisor_lr import fit_labelfree, as_dict
from spectral_utils.clustered_iu_prmbench import fit_clustered_iu
from spectral_utils.advisor_controls import (answer_z, metrics, confusion_by_group,
    score_from_confusion, bootstrap_answer_mean)
from spectral_utils.advisor_statistics import answer_auc
from sklearn.metrics import roc_auc_score, average_precision_score

FROZEN = ROOT / 'results/advisor_completion_2026-10-11'
OUT = ROOT / 'results/clustered_iu_prmbench_v1_2026-10-12'
METHODS = ('IU_PCR', 'LSML_CONT', 'CLUSTER_IU')
BANKS = ('N16', 'N56')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''): h.update(block)
    return h.hexdigest()


def clean(x):
    if isinstance(x, dict): return {k: clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [clean(v) for v in x]
    if isinstance(x, np.ndarray): return clean(x.tolist())
    if isinstance(x, np.generic): return clean(x.item())
    if isinstance(x, float) and not np.isfinite(x): return None
    return x


def dump(path, x):
    Path(path).write_text(json.dumps(clean(x), indent=2, allow_nan=False) + '\n')


def cut(off, answers):
    rows = np.concatenate([np.arange(off[i], off[i+1]) for i in answers])
    return rows, np.r_[0, np.cumsum(np.diff(off)[answers])]


def protocol():
    return {'version': 1, 'status': 'preregistered_before_new_fit',
        'banks': list(BANKS), 'methods': list(METHODS), 'primary_metric': 'mean within-answer AUC',
        'population': 'corrected thinkfix mask1 advisor v11; 6969 answers / 94203 steps',
        'split': 'unchanged FOLDS_V2.prmbench.outer; five source-disjoint folds',
        'clustering': 'exact LSML_CONT donor partition; default residual/unit loading; no K retuning',
        'normalization': 'existing answer-z V; train-only population mean/SD at each IU level',
        'keep': 'rho_hat >= max(0.05*var_y, rho_hat_max/3); once per level; refit survivors',
        'IU': 'IU_FIT_DEFAULTS; exclusion False during probe and survivor refit',
        'small_groups': 'one: passthrough; two: signed continuous L-SML; applies after deletion too',
        'zero_survivors': 'explicit failed fold, no restoration or fabricated score',
        'filtering_small_groups': 'unidentified rho: skip selection for initial singleton/pair',
        'orientation': 'inherited bank convention, higher = error; no label-based sign repair',
        'readout': 'Q80: train global .8 quantile of answer-z fused scores; test answer-z >= tau',
        'secondary': ['fold-mean AUROC', 'fold-mean error AUPRC', 'top1 error hit',
                      'noncontrol pooled Q80 macro-F1 (PRMScore)', 'type breakdown'],
        'uncertainty': '5000 shared source-group bootstrap draws; paired differences',
        'multiplicity': 'four primary bank/baseline contrasts; Bonferroni intervals plus pointwise 95%',
        'claim_boundary': 'previously inspected development population; no external confirmation'}


def fit_all():
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / 'predictions').mkdir(); (OUT / 'checkpoints').mkdir()
    dump(OUT / 'PROTOCOL.json', protocol())
    sources = ['scripts/run_clustered_iu_prmbench_v1.py', 'spectral_utils/clustered_iu_prmbench.py',
               'spectral_utils/advisor_lr.py', 'spectral_utils/fusion_utils.py',
               'spectral_utils/upcr.py', 'spectral_utils/laplacian_upcr.py',
               'spectral_utils/advisor_controls.py', 'spectral_utils/advisor_statistics.py']
    inputs = [FROZEN/'inputs'/n for n in ('population.npz', 'bank_V_N16.npz', 'bank_V_N56.npz', 'FOLDS_V2.json')]
    identity = {'code': {p: sha(ROOT/p) for p in sources},
                'inputs': {str(p.relative_to(ROOT)): sha(p) for p in inputs},
                'frozen_manifest': sha(FROZEN/'PACKAGE_MANIFEST.json')}
    dump(OUT / 'INPUT_FREEZE.json', identity)
    # Explicit allowlist: labels and classification are not read during fitting.
    with np.load(FROZEN/'inputs/population.npz', allow_pickle=False) as z:
        pop = {k: z[k] for k in ('idx', 'offsets', 'group', 'fold')}
    off = pop['offsets']; sf = np.repeat(pop['fold'], np.diff(off))
    folds = json.loads((FROZEN/'inputs/FOLDS_V2.json').read_text())['prmbench']['outer']
    for g, f in zip(pop['group'], pop['fold']):
        if int(folds[str(g)]) != int(f): raise ValueError('Fold identity mismatch')
    for bank in BANKS:
        with np.load(FROZEN/f'inputs/bank_V_{bank}.npz', allow_pickle=False) as z:
            X, names = z['V'], z['columns'].astype(str).tolist()
            if not np.array_equal(off, z['offsets']): raise ValueError('Offset identity mismatch')
        predictions = dict(pop)
        for method in METHODS:
            predictions[method] = np.full(len(X), np.nan)
            predictions[method+'__flags'] = np.zeros(len(X), bool)
        for f in range(5):
            started = time.time(); tr_ans = np.flatnonzero(pop['fold'] != f); te_ans = np.flatnonzero(pop['fold'] == f)
            if set(pop['group'][tr_ans]) & set(pop['group'][te_ans]): raise ValueError('Source leakage')
            tr, toff = cut(off, tr_ans); te, eoff = cut(off, te_ans)
            checkpoint = OUT/'checkpoints'/f'{bank}_fold{f}'; checkpoint.mkdir()
            diag = {'bank': bank, 'fold': f, 'names': names, 'train_groups': sorted(set(pop['group'][tr_ans])),
                    'test_groups': sorted(set(pop['group'][te_ans])), 'models': {}, 'failures': {}}
            models = {}
            print('FIT', bank, f, flush=True)
            for method in ('LSML_CONT', 'IU_PCR'):
                models[method] = fit_labelfree(X[tr], method)
            try:
                models['CLUSTER_IU'] = fit_clustered_iu(X[tr], models['LSML_CONT'].diagnostic['group'])
            except (ValueError, RuntimeError) as exc:
                diag['failures']['CLUSTER_IU'] = repr(exc)
            old = np.load(FROZEN/f'lr/checkpoints/{bank}/outer_{f}/SCORES.npz', allow_pickle=False)
            if not np.array_equal(old['rows'], te): raise ValueError('Baseline row mismatch')
            for method, m in models.items():
                train_score, test_score = m.predict(X[tr]), m.predict(X[te])
                if not np.isfinite(test_score).all(): raise ValueError('Non-finite holdout scores')
                tau = float(np.quantile(answer_z(train_score, toff), .8))
                predictions[method][te] = test_score
                predictions[method+'__flags'][te] = answer_z(test_score, eoff) >= tau
                diag['models'][method] = as_dict(m); diag['models'][method]['Q80_tau'] = tau
                if method in ('LSML_CONT', 'IU_PCR'):
                    error = float(np.max(np.abs(test_score - old['LSML_FULL' if method == 'LSML_CONT' else 'IU_FULL'])))
                    diag['models'][method]['baseline_saved_score_max_abs'] = error
                    if error > 1e-9: raise ValueError(f'Baseline replay failed: {error}')
            dump(checkpoint/'MODELS.json', diag)
            np.savez_compressed(checkpoint/'SCORES.npz', rows=te, **{m: predictions[m][te] for m in METHODS})
            print('DONE', bank, f, round(time.time()-started, 2),
                  models.get('CLUSTER_IU').diagnostic['n_features_active_final'] if 'CLUSTER_IU' in models else diag['failures'], flush=True)
        np.savez_compressed(OUT/'predictions'/f'{bank}.npz', **predictions)
    files = {str(p.relative_to(OUT)): sha(p) for p in OUT.rglob('*') if p.is_file()}
    dump(OUT/'PREDICTION_SEAL.json', {'status': 'FROZEN_BEFORE_EVALUATION', 'files': files})
    if any(sha(ROOT/p) != h for p, h in identity['inputs'].items()): raise ValueError('Frozen input mutated')
    if sha(FROZEN/'PACKAGE_MANIFEST.json') != identity['frozen_manifest']: raise ValueError('Frozen manifest mutated')


def evaluate():
    seal = json.loads((OUT/'PREDICTION_SEAL.json').read_text())
    for p, h in seal['files'].items():
        if sha(OUT/p) != h: raise ValueError('Seal mismatch: '+p)
    # Evaluation-only label access starts after all ten fits and prediction seal.
    with np.load(FROZEN/'inputs/population.npz', allow_pickle=False) as z:
        y, cls = z['y'].astype(bool), z['classification'].astype(str)
    result, contrasts, types, telemetry = [], [], [], []
    for bank in BANKS:
        p = np.load(OUT/'predictions'/f'{bank}.npz', allow_pickle=False)
        off, groups, af = p['offsets'], p['group'], p['fold']; sf = np.repeat(af, np.diff(off))
        unique, inv = np.unique(groups, return_inverse=True); sg = np.repeat(inv, np.diff(off))
        W = np.random.default_rng(20261012).multinomial(len(unique), np.full(len(unique), 1/len(unique)), size=5000).astype(float)
        aucs, draws, prm_draws = {}, {}, {}
        for method in METHODS:
            score = p[method]; covered = np.array([np.isfinite(score[a:b]).all() for a,b in zip(off[:-1],off[1:])])
            safe = np.where(np.isfinite(score), score, 0.)
            auc, _ = answer_auc(y, safe, off); auc[~covered] = np.nan; aucs[method] = auc
            draws[method] = bootstrap_answer_mean(auc, inv, covered, W)
            top = []
            for i,(a,b) in enumerate(zip(off[:-1],off[1:])):
                if covered[i] and y[a:b].any():
                    at = np.flatnonzero(score[a:b] >= score[a:b].max()-8*np.finfo(float).eps)[0]
                    top.append(y[a+at])
            fm = []
            for f in range(5):
                mask = (sf == f) & np.isfinite(score)
                if len(np.unique(y[mask])) == 2:
                    fm.append({'fold': f, 'auc': roc_auc_score(y[mask], score[mask]),
                               'error_auprc': average_precision_score(y[mask], score[mask])})
            selected = np.repeat(covered & (cls != 'correct'), np.diff(off))
            flag = p[method+'__flags']; confusion = confusion_by_group(y, flag, sg, selected, len(unique))
            prm_draws[method] = score_from_confusion(confusion, W)
            row = {'bank': bank, 'method': method, 'complete': bool(covered.all()),
                   'n_answers_scored': int(covered.sum()), 'n_steps_scored': int(np.isfinite(score).sum()),
                   'n_auc_answers': int(np.isfinite(auc).sum()), 'within_answer_auc': float(np.nanmean(auc)),
                   'auc_lo95': np.quantile(draws[method], .025), 'auc_hi95': np.quantile(draws[method], .975),
                   'fold_mean_auroc': np.mean([r['auc'] for r in fm]),
                   'fold_mean_error_auprc': np.mean([r['error_auprc'] for r in fm]),
                   'top1_on_error': np.mean(top), 'q80_noncontrol_prmscore': metrics(y[selected], flag[selected])['prm_score'],
                   'fold_metrics': fm}
            result.append(row)
            for typ in np.unique(cls):
                take = covered & (cls == typ); valid = take & np.isfinite(auc); sm = np.repeat(take, np.diff(off))
                types.append({'bank': bank, 'method': method, 'type': typ, 'n_answers': int(take.sum()),
                              'n_auc_answers': int(valid.sum()), 'auc': np.mean(auc[valid]) if valid.any() else None,
                              'q80_macro_f1': metrics(y[sm], flag[sm])['prm_score']})
        for baseline in ('IU_PCR', 'LSML_CONT'):
            delta = aucs['CLUSTER_IU'] - aucs[baseline]
            d = bootstrap_answer_mean(delta, inv, np.isfinite(delta), W)
            contrasts.append({'bank': bank, 'baseline': baseline, 'delta_auc': np.nanmean(delta),
                'lo95': np.quantile(d, .025), 'hi95': np.quantile(d, .975),
                'lo_simultaneous': np.quantile(d, .05/8), 'hi_simultaneous': np.quantile(d, 1-.05/8),
                'delta_q80_prmscore': next(r['q80_noncontrol_prmscore'] for r in result if r['bank']==bank and r['method']=='CLUSTER_IU') - next(r['q80_noncontrol_prmscore'] for r in result if r['bank']==bank and r['method']==baseline),
                'q80_delta_lo95': np.quantile(prm_draws['CLUSTER_IU']-prm_draws[baseline], .025),
                'q80_delta_hi95': np.quantile(prm_draws['CLUSTER_IU']-prm_draws[baseline], .975)})
        for f in range(5):
            m = json.loads((OUT/'checkpoints'/f'{bank}_fold{f}'/'MODELS.json').read_text())
            if 'CLUSTER_IU' in m['models']:
                d = m['models']['CLUSTER_IU']['diagnostic']
                telemetry.append({'bank': bank, 'fold': f, 'K': d['K'],
                    'features_kept_within': d['n_features_retained_within'], 'clusters_kept': d['n_clusters_retained'],
                    'features_active_final': d['n_features_active_final'], 'affine_max_abs': d['affine_max_abs'],
                    'inner_modes': [r['model']['diagnostic']['refit_mode'] for r in d['inner']],
                    'outer_mode': d['outer']['diagnostic']['refit_mode']})
    dump(OUT/'RESULTS.json', {'metrics': result, 'contrasts': contrasts, 'types': types, 'telemetry': telemetry})
    def table(rows, cols):
        def fmt(v):
            if isinstance(v,(float,np.floating)): return f'{v:.6f}'
            return html.escape(str(v))
        return '<table><thead><tr>'+''.join('<th>'+html.escape(k)+'</th>' for k in cols)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+fmt(r.get(k))+'</td>' for k in cols)+'</tr>' for r in rows)+'</tbody></table>'
    page = '<!doctype html><meta charset="utf-8"><title>Clustered IU-PCR on PRMBench</title><style>body{font:16px system-ui;max-width:1250px;margin:40px auto;padding:0 20px;color:#202733}table{border-collapse:collapse;width:100%;margin:20px 0;font-size:14px}th,td{padding:9px;text-align:left;border-bottom:1px solid #ddd}th{background:#f2f4f6}details{margin:25px 0}</style>'
    page += '<h1>L-SML clustering → filtered IU-PCR → filtered IU-PCR</h1><p>PRMBench N16 and N56. Corrected thinkfix mask1 inputs. 6,969 answers / 94,203 steps / five source-group folds. Previously inspected development data; no external confirmation.</p>'
    page += '<p>Primary: mean within-answer AUC on answers with both error and valid steps. Baselines replayed against saved advisor v11 scores to 1e-9. Label-free donor fitting; all predictions sealed before evaluation. Q80 uses a donor-fitted global threshold on answer-z fused scores. PRMScore here is pooled noncontrol macro-F1, matching the count-study convention; category/type breakdown is separate.</p>'
    page += table(result, ['bank','method','complete','within_answer_auc','auc_lo95','auc_hi95','fold_mean_auroc','fold_mean_error_auprc','top1_on_error','q80_noncontrol_prmscore'])
    page += '<h2>Paired differences: hybrid minus baseline</h2><p>5,000 shared source-group bootstrap draws. Simultaneous primary intervals use Bonferroni across the four bank/baseline contrasts. Q80 intervals are descriptive pointwise 95%.</p>'
    page += table(contrasts,['bank','baseline','delta_auc','lo95','hi95','lo_simultaneous','hi_simultaneous','delta_q80_prmscore','q80_delta_lo95','q80_delta_hi95'])
    page += '<h2>Filtering and small-cluster exceptions</h2><p>Keep rho_hat ≥ max(0.05 Var(Y), max(rho_hat)/3). One selection pass then survivor refit. No top-three restoration. Singletons pass through; pairs use signed continuous L-SML. Initially undersized groups cannot estimate rho and skip selection. Exceptions also apply after deletion.</p>'
    page += table(telemetry,['bank','fold','K','features_kept_within','clusters_kept','features_active_final','inner_modes','outer_mode','affine_max_abs'])
    page += '<details><summary>Type breakdown</summary>'+table(types,['bank','method','type','n_answers','n_auc_answers','auc','q80_macro_f1'])+'</details>'
    page += '<p>Reproduction: main repository .venv/bin/python scripts/run_clustered_iu_prmbench_v1.py. Existing output directories are refused. Inspect PROTOCOL.json, INPUT_FREEZE.json, PREDICTION_SEAL.json, RESULTS.json and checkpoints/*/MODELS.json for full models, rho estimates, thresholds and survivor masks.</p>'
    (OUT/'index.html').write_text(page)
    print(json.dumps(clean({'metrics': result, 'contrasts': contrasts}), indent=2), flush=True)


if __name__ == '__main__':
    fit_all()
    evaluate()
