"""Raw-source and official-port review of the Step313 label-only bridge."""
from collections import Counter
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def module(path,name):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


driver=module(ROOT/'scripts/repair_localization_prm_labels_v3.py','label_bridge_driver')
OUT=driver.OUT
load,save,sha=driver.load,driver.save,driver.sha


def main():
    start=time.monotonic();manifest=driver.verify();counts=Counter()
    audit=load(OUT/'LABEL_AUDIT.json');v3=load(OUT/'RELEASE_V3.json');v2=load(driver.SOURCE/'RELEASE_V2.json')
    assert audit['manifest_sha256']==sha(OUT/'MANIFEST.json')
    assert v3['label_audit_sha256']==sha(OUT/'LABEL_AUDIT.json')
    assert v3['folds_sha256']==sha(driver.SOURCE/'FOLDS_V2.json')
    assert v3['canonical_groups_sha256']==sha(driver.SOURCE/'CANONICAL_GROUPS.json')
    official=module(ROOT/'spectral_utils/prmbench.py','official_prm_contract_review')
    metrics=module(ROOT/'scripts/review_fusion_explicit_fallback_v1.py','label_bridge_independent_metrics')
    with driver.RAW.open('rb') as f:raw=driver.io_module.metadata.MetadataUnpickler(f).load()
    index={v['idx']:v for v in raw.values()};saved={r['row_id']:r for r in audit['rows']}
    assert set(index)==set(saved) and len(index)==6969
    totals=Counter()
    with np.load(v2['cells']['prmbench_qwen3_8b']['label_path'],allow_pickle=False) as old,np.load(v3['cells']['prmbench_qwen3_8b']['label_path'],allow_pickle=False) as new:
        for field in ('row_ids','classification','step_flag_offsets'):np.testing.assert_array_equal(old[field],new[field])
        for i,rid in enumerate(new['row_ids'].astype(str)):
            row=index[rid];rec=saved[rid];n=len(row['steps']);a,b=new['step_flag_offsets'][i:i+2]
            expected=1-np.array(official.eval_on_hallucination_step(row['error_steps'],np.ones(n,dtype=int))['total_step_acc_list'])
            np.testing.assert_array_equal(expected,new['step_error_flags'][a:b]);np.testing.assert_array_equal(expected,rec['corrected_flags'])
            np.testing.assert_array_equal(old['step_error_flags'][a:b],rec['previous_flags'])
            assert rec['raw_error_steps_onebased']==row['error_steps'];counts['raw_and_official_port_label_replays']+=1
            before=np.array(rec['previous_flags']);totals['answers']+=1;totals['steps']+=n
            totals['changed_answers']+=int(np.any(expected!=before));totals['changed_step_labels']+=int(np.sum(expected!=before))
            totals['old_positive_steps']+=int(before.sum());totals['corrected_positive_steps']+=int(expected.sum())
            totals['out_of_range_annotations']+=sum(not 1<=s<=n for s in row['error_steps'])
            totals['rows_with_out_of_range_annotations']+=int(any(not 1<=s<=n for s in row['error_steps']))
            totals['old_all_correct_new_has_error']+=int(not before.any() and expected.any())
            totals['old_error_new_all_correct']+=int(before.any() and not expected.any())
    assert dict(totals)==audit['counts']
    for cell,info in v2['cells'].items():
        if cell.startswith('pb'):
            assert info==v3['cells'][cell];assert sha(info['label_path'])==info['label_opaque_sha256'];counts['unchanged_pb_label_files']+=1
        else:
            before=deepcopy(info);after=deepcopy(v3['cells'][cell])
            for k in ('label_path','label_opaque_sha256'):before.pop(k);after.pop(k)
            assert before==after
    bootstrap_checks=[]
    for cohort,path in driver.COHORTS.items():
        previous=load(path);e=load(OUT/(cohort+'_EVALUATION_V3.json'));c=load(OUT/(cohort+'_CONTRASTS_V3.json'))
        assert e['source_evaluation_sha256']==sha(path);assert c['status']=='COMPLETE'
        assert c['evaluation_sha256']==sha(OUT/(cohort+'_EVALUATION_V3.json'))
        assert len(e['rows'])==len(previous['rows']);changed=[]
        for before,after in zip(previous['rows'],e['rows']):
            assert before['uid']==after['uid']
            for k in before:
                if k!='target':assert before[k]==after[k],(cohort,before['uid'],k)
            if after['cell'].startswith('prm'):
                assert after['target']==saved[after['row_id']]['corrected_flags']
                if before['target']!=after['target']:changed.append(after['uid'])
            else:assert before['target']==after['target']
            counts['exact_unchanged_row_fields']+=1
            counts['exact_method_prediction_score_replays']+=len(after['scores'])
        assert changed==e['changed_target_uids']
        assert e['previous_metrics']==previous['metrics']
        for arm,bundle in e['metrics'].items():
            metrics.check_equal(bundle['prm'],metrics.prm(e['rows'],arm))
            metrics.check_equal(bundle['pb'],metrics.pb(e['rows'],arm))
            for k in ('pb_common_iu_gate','pb_fixed_iu_gate'):
                if k in bundle:metrics.check_equal(bundle[k],metrics.pb(e['rows'],arm,True))
            assert bundle['pb']==previous['metrics'][arm]['pb'];counts['independent_metric_bundles']+=1
        expected_keys={driver.key(p) for p in manifest['contrasts'][cohort]};assert set(c['pairs'])==expected_keys
        for k,p in c['pairs'].items():
            if p['scope']=='all':rows=e['rows']
            else:
                mode,kind=p['scope'].split(':')
                rows=[r for r in e['rows'] if r['augmentation']['native_joint_valid'][kind] and (mode!='native_and_original' or r['augmentation']['original_joint_valid'])]
            assert [r['uid'] for r in rows]==p['selected_ids']
            common=[r for r in rows if r['valid'][p['left']] and r['valid'][p['right']]]
            for side in ('left','right'):
                metrics.check_equal(p[side+'_prm'],metrics.prm(common,p[side]))
                metrics.check_equal(p[side+'_pb'],metrics.pb(rows,p[side]))
            counts['paired_point_scope_replays']+=1
        requested=[('dual__cond100_graph010','dual__iu','all'),('ar1__iu','dual__iu','all'),
                   ('ar1__graph010','dual__cond100_graph010','native_and_original:ar1')] if cohort=='current110' else [
                   ('single__joint0','moment__iu','all'),('dual__joint0','single__joint0','all')]
        for left,right,scope in requested:
            k=driver.key(dict(left=left,right=right,scope=scope));assert k in c['pairs'],k
            rows=driver.select(e['rows'],scope);explicit=metrics.explicit_bootstrap(rows,left,right)
            for field,value in explicit.items():metrics.check_equal(c['pairs'][k]['uncertainty'][field],value)
            bootstrap_checks.append(dict(cohort=cohort,pair=k,draws=1000));print('Explicit interval replay',cohort,k,flush=True)
    dependencies=[Path(__file__),ROOT/'scripts/review_fusion_explicit_fallback_v1.py',ROOT/'spectral_utils/prmbench.py']
    artifacts=['MANIFEST.json','LABEL_AUDIT.json','RELEASE_V3.json','BRIDGE.json','prmbench_qwen3_8b_labels_v3.npz']
    for cohort in driver.COHORTS:artifacts += [cohort+'_EVALUATION_V3.json',cohort+'_CONTRASTS_V3.json']
    save(OUT/'REVIEW.json',{'status':'PASS','counts':dict(counts),'bootstrap_checks':bootstrap_checks,
        'scope':'Same-session review: direct raw-pickle join and existing official-port conversion, pairwise AUC and explicit-row bootstrap independent of the production cached-count evaluator. Shared metadata reader and port are disclosed; no external reviewer.',
        'hashes':{str(OUT/n):sha(OUT/n) for n in artifacts},'review_dependencies':{str(p):sha(p) for p in dependencies},
        'seconds':time.monotonic()-start})
    print('Review PASS',dict(counts),flush=True)


if __name__=='__main__':main()
