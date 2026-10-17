"""Review historical score preservation and corrected raw-label evidence."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from collections import Counter
from copy import deepcopy
import importlib.util
from pathlib import Path
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
def module(path,name):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
d=module(ROOT/'scripts/bridge_localization_history_v3.py','review_history_driver')
OUT=d.OUT
load,save,sha=d.load,d.save,d.sha
ind=module(ROOT/'scripts/review_fusion_explicit_fallback_v1.py','history_independent_metrics')


def explicit_intervals(rows,left,right,include_fixed):
    pp=[r for r in rows if r['cell'].startswith('prm') and r['valid'][left] and r['valid'][right]]
    bb=[r for r in rows if r['cell'].startswith('pb')]
    def strata(records):
        out={}
        for r in records:out.setdefault(r['cell'],{}).setdefault(r['group_id'],[]).append(r)
        return out
    def sample(groups,rng):
        result=[]
        for cell in groups.values():
            keys=sorted(cell)
            for j in rng.integers(len(keys),size=len(keys)):result.extend(cell[keys[j]])
        return result
    ps,bs=strata(pp),strata(bb);rng=np.random.default_rng(2026090706)
    values={k:[] for k in ('prm_common_valid','prm_within_answer_common_valid','pb_all_population')}
    if include_fixed:values['pb_common_iu_gate_all_population']=[]
    for _ in range(1000):
        if pp:
            chosen=sample(ps,rng);a,b=ind.prm(chosen,left),ind.prm(chosen,right)
            for metric,key in [('auroc','prm_common_valid'),('within_answer_auc','prm_within_answer_common_valid')]:
                if a[metric] is not None and b[metric] is not None:values[key].append(a[metric]-b[metric])
        chosen=sample(bs,rng)
        for fixed,key in [(False,'pb_all_population')]+([(True,'pb_common_iu_gate_all_population')] if include_fixed else []):
            a,b=[ind.pb(chosen,arm,fixed)['macro_f1'] for arm in (left,right)]
            if a is not None and b is not None:values[key].append(a-b)
    result={}
    for key,series in values.items():
        result[key+'_ci95']=np.quantile(series,[.025,.975]).tolist() if series else None
        result[key+'_valid_draws']=len(series)
    return result


def decomposition(records,arm,kind):
    targets=np.concatenate([r['target'] for r in records]);values=np.concatenate([r['methods'][arm][kind] for r in records])
    answer=np.concatenate([np.full(len(r['target']),i) for i,r in enumerate(records)])
    pos,neg=values[targets==1],values[targets==0];pa,na=answer[targets==1],answer[targets==0]
    wins=(pos[:,None]>neg)+.5*(pos[:,None]==neg);same=pa[:,None]==na
    n=wins.size;nw=int(same.sum());nc=n-nw
    within=[ind.auc_value(r['target'],r['methods'][arm][kind]) for r in records];within=[v for v in within if v is not None]
    return {'total_pairs':n,'within_pairs':nw,'cross_pairs':nc,'cross_pair_fraction':nc/n if n else None,
        'pooled_auc':float(wins.mean()) if n else None,'pair_weighted_within_auc':float(wins[same].mean()) if nw else None,
        'cross_answer_auc':float(wins[~same].mean()) if nc else None,'mean_within_answer_auc':float(np.mean(within)) if within else None,'mixed_answers':len(within)}


def main():
    start=time.monotonic();m=d.verify();registry=load(OUT/'REGISTRY.json');execution=load(OUT/'CONTRAST_EXECUTION.json')
    assert execution['status']=='COMPLETE' and execution['comparisons']==199
    amendment=load(OUT/'EXECUTION_AMENDMENT.json');recovery=load(OUT/'EXECUTION_RECOVERY.json')
    assert recovery['status']=='COMPLETE' and not amendment['scientific_driver_or_sources_changed']
    assert amendment['runner_sha256']==recovery['runner_sha256']==sha(ROOT/'scripts/resume_localization_history_bridge_v3.py')
    assert amendment['frozen_manifest_sha256']==sha(OUT/'MANIFEST.json')
    anchor=load(d.LABELS/'original58_EVALUATION_V3.json');anchors={r['uid']:r for r in anchor['rows']}
    official=module(ROOT/'spectral_utils/prmbench.py','history_official_port');raw_targets={};counts=Counter()
    # Read original raw annotations; the old derived NPZ is not the authority.
    for cell,path in d.io.RAW_FILES.items():
        with path.open('rb') as f:raw=d.io.metadata.MetadataUnpickler(f).load()
        index={(r['idx'] if cell.startswith('prm') else cell.split('_')[1]+'::'+str(r['id'])):r for r in raw.values()}
        for rec in (r for r in anchors.values() if r['cell']==cell):
            source=index[rec['row_id']];assert rec['tokens']==len(source['gen_token_ids']) and rec['steps']==len(source['steps'])
            target=(1-np.asarray(official.eval_on_hallucination_step(source['error_steps'],[1]*rec['steps'])['total_step_acc_list'])).tolist() if cell.startswith('prm') else int(source['label'])
            assert target==rec['target'];raw_targets[rec['uid']]=target;counts['independent_raw_label_joins']+=1
        del raw,index
    assert len(raw_targets)==58
    checks=[];lane_reports={};equivalence={};source_files=[]
    for lane,info in m['lanes'].items():
        source=Path(info['directory']);old=load(source/'EVALUATION.json');e=load(OUT/(lane+'_EVALUATION_V3.json'))
        previous={r['uid']:r for r in old['rows']};assert set(previous)==set(anchors)
        assert e['source_sha256']==sha(source/'EVALUATION.json') and e['manifest_sha256']==sha(OUT/'MANIFEST.json')
        assert e['group_map_sha256']==sha(d.GROUPS/'CANONICAL_GROUPS.json')
        for row in e['rows']:
            uid=row['uid'];prior=previous[uid];baseline=anchors[uid]
            assert row['target']==raw_targets[uid] and row['group_id']==baseline['group_id']
            assert row['bridge_previous_group_id']==prior['group_id']
            restored=deepcopy(row);restored.pop('bridge_previous_group_id');restored['target']=prior['target'];restored['group_id']=prior['group_id']
            assert restored==prior;counts['exact_unchanged_row_payloads']+=1
            for k in ('cell','row_id','tokens','steps','input_sha256'):assert row[k]==baseline[k]
            meta=load(source/'scores'/(uid+'.json'));assert meta['array_sha256']==sha(source/'scores'/(uid+'.npz'))
            details=meta['report']['methods'] if lane=='representation' else meta['methods']
            with np.load(source/'scores'/(uid+'.npz'),allow_pickle=False) as arrays:
                assert len(arrays['step_starts'])==len(arrays['step_ends'])==row['steps']
                for arm in info['arms']:
                    detail=details.get(arm,{});valid=bool(detail.get('valid',False));assert row['valid'][arm]==valid,(lane,uid,arm)
                    pred=detail.get('readout',{}).get('prediction') if lane=='representation' else detail.get('prediction')
                    assert row['predictions'][arm]==pred
                    assert row['decision_valid'][arm]==bool(valid and pred is not None)
                    if valid or arm in row['scores']:
                        key=arm+('__step' if lane=='representation' else '__risk')
                        np.testing.assert_array_equal(row['scores'][arm],arrays[key]);assert len(row['scores'][arm])==row['steps']
                        if not valid:
                            assert lane=='representation' and detail['status'] in ('FINITE_UNCONVERGED_DESCRIPTIVE','FIT_DIAGNOSTIC_ONLY')
                            assert not row['decision_valid'][arm]
                            counts['descriptive_invalid_scores_preserved']+=1
                    else:assert arm not in row['scores']
                    counts['source_array_and_prediction_records']+=1
        for arm,bundle in e['metrics'].items():
            expected={'prm':ind.prm(e['rows'],arm),'pb':ind.pb(e['rows'],arm)}
            if 'pb_common_iu_gate' in bundle:expected['pb_common_iu_gate']=ind.pb(e['rows'],arm,True)
            ind.check_equal(bundle,expected);assert bundle['pb']==old['metrics'][arm]['pb'];counts['independent_metric_bundles']+=1
        pairfile=OUT/(lane+'_CONTRASTS_V3.json');c=load(pairfile);assert c['status']=='COMPLETE' and c['evaluation_sha256']==sha(OUT/(lane+'_EVALUATION_V3.json'))
        old_pairs=old['paired'] if lane=='representation' else load(source/'CONTRASTS.json')['pairs']
        assert set(c['pairs'])==set(old_pairs)
        for key,pair in c['pairs'].items():
            rows=e['rows'];common=[r for r in rows if r['valid'][pair['left']] and r['valid'][pair['right']]]
            assert pair['selected_ids']==[r['uid'] for r in rows] and pair['common_valid_ids']==[r['uid'] for r in common]
            for side in ('left','right'):
                ind.check_equal(pair[side+'_prm'],ind.prm(common,pair[side]));ind.check_equal(pair[side+'_pb'],ind.pb(rows,pair[side]))
                assert pair[side+'_pb']==old_pairs[key][side+'_pb']
                if lane=='context':ind.check_equal(pair[side+'_pb_common_iu_gate'],ind.pb(rows,pair[side],True))
            assert pair['uncertainty']['fixed_gate_included']==(lane=='context')
            counts['paired_scope_and_point_replays']+=1
        pair=next(iter(c['pairs'].values()));actual=pair['uncertainty']
        for k,v in explicit_intervals(e['rows'],pair['left'],pair['right'],lane=='context').items():ind.check_equal(actual[k],v)
        checks.append({'lane':lane,'left':pair['left'],'right':pair['right'],'draws':1000,'status':'MATCH'})
        lane_reports[lane]={'arms':len(info['arms']),'pairs':len(c['pairs']),'answers':58}
        source_files += [OUT/(lane+'_EVALUATION_V3.json'),pairfile]
        print('Reviewed corrected historical lane:',lane,flush=True)
    for arm,bundle in anchor['metrics'].items():
        ind.check_equal(bundle['prm'],ind.prm(anchor['rows'],arm));ind.check_equal(bundle['pb'],ind.pb(anchor['rows'],arm))
        counts['external_anchor_metric_replays']+=1
    assert len(registry['entries'])==156
    for entry in registry['entries']:
        original=anchor if entry['lane']=='fallback_references' else load(OUT/(entry['lane']+'_EVALUATION_V3.json'))
        assert entry['metric']==original['metrics'][entry['arm']]
    gate=load(OUT/'gate_DIAGNOSTICS_V3.json');old_gate=load(d.GATE/'EVALUATION.json');oldrows={r['uid']:r for r in old_gate['rows']}
    for row in gate['rows']:
        uid=row['uid'];prior=oldrows[uid];assert row['target']==raw_targets[uid] and row['group_id']==anchors[uid]['group_id']
        restored=deepcopy(row);restored.pop('bridge_previous_group_id');restored['target']=prior['target'];restored['group_id']=prior['group_id'];assert restored==prior
        with np.load(d.GATE/'measurements'/(uid+'.npz'),allow_pickle=False) as arrays:
            for arm,detail in row['methods'].items():
                if detail['valid']:
                    for kind in ('risk','origin_projection'):np.testing.assert_array_equal(detail[kind],arrays[arm+'__'+kind]);counts['gate_diagnostic_array_replays']+=1
        counts['gate_rows_unchanged']+=1
    for arm,summary in gate['summaries'].items():
        selected=[r for r in gate['rows'] if r['cell'].startswith('prm') and r['methods'][arm]['valid']]
        for kind in ('risk','origin_projection'):ind.check_equal(summary['prm'][kind],decomposition(selected,arm,kind));counts['independent_auc_decompositions']+=1
        for k in set(summary)-{'prm'}:assert summary[k]==old_gate['summaries'][arm][k]
    dependencies=[Path(__file__),Path(ind.__file__),ROOT/'spectral_utils/prmbench.py',ROOT/'scripts/audit_localization_source_groups.py',ROOT/'scripts/resume_localization_history_bridge_v3.py']
    files=[OUT/'MANIFEST.json',OUT/'REGISTRY.json',OUT/'gate_DIAGNOSTICS_V3.json',OUT/'CONTRAST_EXECUTION.json',OUT/'EXECUTION_AMENDMENT.json',OUT/'EXECUTION_RECOVERY.json']+source_files
    save(OUT/'REVIEW.json',{'status':'PASS','counts':dict(counts),'lanes':lane_reports,'explicit_bootstrap_checks':checks,
        'hashes':{str(p):sha(p) for p in files},'dependencies':{str(p):sha(p) for p in dependencies},'seconds':time.monotonic()-start,
        'review_notes':['Review initially rejected archived diagnostic scores from invalid fits, then allowed only one of the two source status names. Source inspection confirmed3 FINITE_UNCONVERGED_DESCRIPTIVE and9 FIT_DIAGNOSTIC_ONLY records intentionally retain arrays while valid/decision_valid are false. Review now verifies both source-defined statuses, exact arrays and continued exclusion. No scientific source, prediction, target or validity changed.'],
        'scope':'Same-session raw-annotation/official-port review, exact old row and stored-array preservation, direct pairwise AUC/PB metrics, flattened AUC decompositions and five explicit-row1000-draw bootstraps. Shared raw metadata reader, official port and original frozen scores disclosed. No refits or external review.'})
    print('REVIEW PASS',dict(counts),flush=True)


if __name__=='__main__':main()
