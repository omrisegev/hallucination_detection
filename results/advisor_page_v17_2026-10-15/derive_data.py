"""Derive presentation metrics from saved predictions; no fitting or inference."""
from pathlib import Path
import csv, json, hashlib, importlib.util, pickle
import numpy as np
from scipy.stats import spearmanr, rankdata

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE = ROOT / 'results/advisor_completion_2026-10-11'

def read_csv(path):
    return list(csv.DictReader(Path(path).open()))

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

def main():
    metric = load_module('v17_official_prmbench', ROOT / 'spectral_utils/prmbench.py')
    helpers = load_module('v17_controls', ROOT / 'spectral_utils/advisor_controls.py')
    pop = np.load(BASE / 'inputs/population.npz', allow_pickle=False)
    ids, offsets, types = pop['idx'], pop['offsets'], pop['classification']
    meta = json.loads((BASE / 'inputs/prmbench_meta.json').read_text())['rows']
    meta_by_id = {r['idx']: r for r in meta}
    assert set(ids) == set(meta_by_id)
    meta = [meta_by_id[str(i)] for i in ids]
    lengths = np.diff(offsets)
    noncontrol = types != 'correct'
    source_paths = [BASE / 'inputs/population.npz', BASE / 'inputs/prmbench_meta.json',
                    ROOT / 'spectral_utils/prmbench.py', ROOT / 'spectral_utils/advisor_controls.py']

    def evaluate(flag):
        preds = [{'idx': str(i), 'labels': (~flag[a:b]).astype(int).tolist()}
                 for i, a, b in zip(ids, offsets[:-1], offsets[1:])]
        ev = metric.prmbench_evaluate(preds, meta)
        assert ev['n_predictions_scored'] == 6969
        bc, total = ev['by_classification'], ev['total']
        by_type = {t: 100 * (bc['f1'][t] if t == 'multi_solutions' else
                  (bc['f1'][t] + bc['negative_f1'][t]) / 2) for t in set(types)}
        counts = np.array([int(flag[a:b].sum()) for a,b in zip(offsets[:-1],offsets[1:])])
        return {'prmscore': 100 * (total['f1'] + total['negative_f1']) / 2,
                'f1_valid': 100 * total['f1'], 'f1_error': 100 * total['negative_f1'],
                'by_type': by_type, 'flag_rate_all': float(flag.mean()),
                'flag_rate_noncontrol': float(flag[np.repeat(noncontrol, lengths)].mean()),
                'counts': counts.tolist()}

    auc_rows = read_csv(BASE / 'controls/ALL_BANK_TOP1_AUC.csv')
    lr_rows = read_csv(BASE / 'lr/METRICS.csv')
    banks = ['N14','N16','N25','N43','N56']
    methods = ['ALL_EQ','DS_FILTER_EQ','LSML_CONT','IU_PCR','LR']
    scores, normalized, bank_metrics = {}, {}, {}
    for bank in banks:
        path = BASE / f'inputs/bank_{bank}.npz'
        z = np.load(path, allow_pickle=False)
        assert np.array_equal(z['idx'], ids) and np.array_equal(z['offsets'], offsets)
        lr_path = BASE / f'lr/OOF_{bank}.npz'
        lr = np.load(lr_path, allow_pickle=False)['LR_FULL']
        source_paths += [path, lr_path]
        bank_metrics[bank] = {}
        for method in methods:
            score = lr if method == 'LR' else z['S__' + method]
            assert np.isfinite(score).all()
            az = helpers.answer_z(score, offsets)
            cut = float(np.quantile(az, .8))
            row = evaluate(az > cut)
            if method == 'LR':
                rank = next(r for r in lr_rows if r['bank'] == bank and r['arm'] == 'LR_FULL')
                auc_check,top1_check=[],[]
                for a,b in zip(offsets[:-1],offsets[1:]):
                    y=pop['y'][a:b];v=score[a:b];n_error=int(y.sum());n_valid=int((~y).sum())
                    if n_error:top1_check.append(float(y[np.argmax(v)]))
                    if n_error and n_valid:auc_check.append(float((rankdata(v)[y].sum()-n_error*(n_error+1)/2)/(n_error*n_valid)))
                assert len(auc_check)==6030 and len(top1_check)==6035
                assert abs(np.mean(auc_check)-float(rank['within_answer_auc']))<1e-12
                assert abs(np.mean(top1_check)-float(rank['top1_on_error']))<1e-12
                row.update(auc=float(rank['within_answer_auc']), top1=100*float(rank['top1_on_error']),
                           provenance='fresh source-group OOF LR; same saved prediction for both metrics')
            else:
                rank = next(r for r in auc_rows if r['bank']==bank and r['method']==method
                            and r['classification']=='ALL_NONCONTROL')
                row.update(auc=float(rank['within_auc']), top1=100*float(rank['top1_on_error']),
                           provenance='corrected thinkfix + mask1 saved fusion scores')
            row['threshold'] = cut
            bank_metrics[bank][method] = row
            if bank == 'N16':
                scores[method], normalized[method] = score, az

    frozen = json.loads((BASE / 'controls/DIAGNOSTIC_CEILINGS.json').read_text())
    for method in methods[:-1]:
        assert abs(bank_metrics['N16'][method]['prmscore']/100 - frozen[method]['frozen']['prm_score']) < 1e-12
    source_paths += [BASE / 'controls/DIAGNOSTIC_CEILINGS.json', BASE / 'controls/ALL_BANK_TOP1_AUC.csv', BASE / 'lr/METRICS.csv']
    fixed = {m: evaluate(normalized[m] > .5) for m in ['LSML_CONT','IU_PCR']}
    curves = read_csv(BASE / 'controls/BUDGET_CURVES.csv')
    rules = read_csv(BASE / 'controls/EXACT_DECISION_RULES.csv')
    counts = read_csv(BASE / 'controls/PER_ANSWER_COUNTS.csv')
    true_counts = {r['idx']: int(r['true_count']) for r in counts if r['method']=='IU_PCR' and r['rule']=='ANSWER_Z_Q80'}
    truth = np.array([true_counts[str(i)] for i in ids])
    count_series = {'True annotated errors': truth,
                    'IU-PCR Q80': np.array(bank_metrics['N16']['IU_PCR']['counts']),
                    'IU-PCR fixed z > 0.5': np.array(fixed['IU_PCR']['counts'])}
    prm_pickle = BASE / 'inputs/prmbench_prm.pkl'
    prm = {r['idx']: r for r in pickle.load(prm_pickle.open('rb')).values()}
    native_flags = np.concatenate([np.array(prm[str(i)]['labels']) == 0 for i in ids])
    native = evaluate(native_flags)
    native_auc, native_top1 = [], []
    for i,a,b in zip(ids,offsets[:-1],offsets[1:]):
        y = pop['y'][a:b]
        risk = 1 - np.asarray(prm[str(i)]['rewards'])
        if y.any():
            native_top1.append(float(y[np.argmax(risk)]))
        n_error, n_valid = int(y.sum()), int((~y).sum())
        if n_error and n_valid:
            native_auc.append(float((rankdata(risk)[y].sum()-n_error*(n_error+1)/2)/(n_error*n_valid)))
    assert len(native_auc)==6030 and len(native_top1)==6035
    native.update(auc=float(np.mean(native_auc)),top1=100*float(np.mean(native_top1)))
    assert abs(native['prmscore']/100-frozen['PRM_NATIVE']['frozen']['prm_score'])<1e-12
    count_series['PRM native reward < 0.5'] = np.array(native['counts'])
    bins = [(1,7),(8,9),(10,11),(12,13),(14,15),(16,18),(19,99999)]
    length_data = {'labels':['≤7','8–9','10–11','12–13','14–15','16–18','≥19'],
                   'n':[int(((lengths>=a)&(lengths<=b)&noncontrol).sum()) for a,b in bins], 'series':{}}
    for name, values in count_series.items():
        length_data['series'][name] = {'values':[float(values[(lengths>=a)&(lengths<=b)&noncontrol].mean()) for a,b in bins],
            'rho': float(spearmanr(lengths[noncontrol], values[noncontrol]).statistic)}

    gallery_path = ROOT / 'docs/reviews/signal_visualization_options_20261012/DATA.json'
    gallery = json.loads(gallery_path.read_text())
    feature_csv = ROOT / 'results/signal_gallery_extensions_2026-10-12/FEATURE_PRMSCORE.csv'
    channel_scores = read_csv(feature_csv)
    selected = ['ct7_ve0','energy_level','prefix_innovation','digit_spread','STEP_INDEX']
    available = {r['feature'] for r in channel_scores}
    # Match actual exported spellings rather than inventing a feature name.
    if 'prefix_innovation' not in available:
        selected[2] = next(f['id'] for f in gallery['features'] if 'prefix' in f['id'])
    assert set(selected[:-1]) <= available
    channels = {name:{r['type']: float(r['value'])*100 for r in channel_scores if r['feature']==name} for name in selected}
    step_index = np.concatenate([np.arange(int(n),dtype=float) for n in lengths])
    step_z = helpers.answer_z(step_index,offsets)
    step_folds = np.repeat(pop['fold'],lengths)
    step_flags = np.zeros(len(step_z),dtype=bool)
    step_cuts = []
    for fold in range(5):
        train,test = step_folds!=fold,step_folds==fold
        assert not set(pop['group'][pop['fold']==fold]) & set(pop['group'][pop['fold']!=fold])
        cut = float(np.quantile(step_z[train],.8))
        step_flags[test] = step_z[test] >= cut
        step_cuts.append(cut)
    step_eval = evaluate(step_flags)
    channels['STEP_INDEX'] = step_eval['by_type']
    examples = json.loads((BASE / 'controls/DATASET_EXAMPLES.json').read_text())
    ex = examples[0]
    j = list(ids).index(ex['idx']); a,b=offsets[j:j+2]
    ex['lsml_risk'] = normalized['LSML_CONT'][a:b].tolist()
    ex['q80_cut'] = bank_metrics['N16']['LSML_CONT']['threshold']
    ex['lsml_top1_step'] = int(np.argmax(scores['LSML_CONT'][a:b])) + 1

    bank_def_path = BASE / 'sources/research/results/monday_nested_banks_v2/BANKS.json'
    bank_defs = json.loads(bank_def_path.read_text())
    transitions = [{'from':a,'to':b,'added':sorted(set(bank_defs[b])-set(bank_defs[a])),
                    'removed':sorted(set(bank_defs[a])-set(bank_defs[b]))} for a,b in zip(banks,banks[1:])]
    assert all(not t['removed'] for t in transitions)
    digits={'digit_alternative','digit_spread','digit_alternative_innovation'}
    assert all(digits <= set(bank_defs[b]) for b in banks)
    assert all(len(bank_defs[b])==int(b[1:]) for b in banks)
    fourway=read_csv(BASE / 'lr/FOURWAY_SUMMARY.csv')
    paper=read_csv(BASE / 'report/paper_table3.csv')
    local_copies={p.name:read_csv(p) for p in (HERE/'sources').glob('*.csv')}
    ceilings=[bank_metrics['N16']['LSML_CONT']['prmscore'], frozen['LSML_CONT']['global_threshold']['prm_score']*100,
       float(next(r for r in rules if r['method']=='LSML_CONT' and r['rule']=='TOP_2_DIAGNOSTIC' and r['classification']=='ALL_NONCONTROL')['prm_score'])*100,
       68.9, frozen['LSML_CONT']['true_count']['prm_score']*100, frozen['LSML_CONT']['perfect_ranking_fixed_counts']['prm_score']*100]
    paired_path = BASE / 'report/legacy/PAIRED_DECISION_ANALYSIS.json'
    paired = json.loads(paired_path.read_text())
    data={'banks':banks,'methods':methods,'bank_metrics':bank_metrics,'prm_native':native,'fixed_05':fixed,
          'length_data':length_data,'budget_curves':curves,'ceilings':ceilings,
          'global_cut_diagnostics':{m:{'cut':frozen[m]['global_threshold_cut'],'prmscore':frozen[m]['global_threshold']['prm_score']*100} for m in ['LSML_CONT','IU_PCR']},
          'ceilings_type_count_note':'68.9: rounded v16 value from count_structure_check_v1, commit 17aa1e4; no new replay',
          'gallery':gallery,'selected_features':selected,'feature_prmscore':channels,'example':ex,
          'step_index_prmscore_derivation':{'rule':'positive step index, per-answer z, source-fold donor Q80; >= boundary matches gallery protocol','thresholds':step_cuts,'overall':step_eval['prmscore']},
          'transitions':transitions,'fourway':fourway,'paper':paper,'external_sources':local_copies,'paired_contrasts':paired['contrasts'],
          'population':{'answers_all':6969,'answers_noncontrol':6211,'steps_all':94203,'steps_noncontrol':83371,'auc_answers':6030,'top1_answers':6035},
          'source_v16_commit':'2cf1209cda253a50b53897fa14c99b4431cb0fcd',
          'derivations':{'LR':'Saved LR_FULL OOF predictions, answer-z, one pooled 80th percentile, official evaluator. No refit. Historical LR AUC is not spliced into this result.',
                        'threshold_05':'Exact z > 0.5 on saved N16 fusion scores; same evaluator and population as Q80.',
                        'length':'Noncontrol rows, identical bins, same fixed-0.5 predictions as metric evaluation.'}}
    source_paths += [prm_pickle,gallery_path,feature_csv,bank_def_path,paired_path,BASE/'controls/EXACT_DECISION_RULES.csv',BASE/'controls/BUDGET_CURVES.csv',BASE/'controls/PER_ANSWER_COUNTS.csv',BASE/'controls/DATASET_EXAMPLES.json',BASE/'lr/FOURWAY_SUMMARY.csv',BASE/'report/paper_table3.csv']
    source_paths += list((HERE/'sources').glob('*.csv'))
    source_paths += [ROOT/'docs/reviews/advisor_v16_walkthrough_20261013/source_v16_2cf1209c.html',
                    ROOT/'docs/reviews/advisor_v16_walkthrough_20261013/advisor_walkthrough_draft.html',
                    ROOT/'docs/reviews/advisor_v16_walkthrough_20261013/prefix_token_diagnostic.png',
                    ROOT/'results/signal_gallery_extensions_2026-10-12/PROTOCOL.json']
    source_paths += list((HERE/'sources').glob('*.md'))
    receipts=[]
    for p in source_paths:
        receipts.append({'path':str(p.relative_to(ROOT)),'size':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    (HERE/'DATA.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    (HERE/'SOURCE_MANIFEST.json').write_text(json.dumps({'sources':receipts,'no_fitting':True,'remote_csv_ref':data['source_v16_commit']},indent=2)+'\n')
    print('Derived DATA.json; N16 LR Q80:',bank_metrics['N16']['LR']['prmscore'])
    print('Exact threshold 0.5:',{m:r['prmscore'] for m,r in fixed.items()})

if __name__=='__main__':
    main()
