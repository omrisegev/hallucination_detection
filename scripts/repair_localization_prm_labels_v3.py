"""Versioned raw-label repair; no inference, no refit, no old-file mutation."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
import argparse
from collections import Counter
from copy import deepcopy
import importlib.util
import io
from pathlib import Path
import sys
import time
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'local_cache/short_cycle01_code'))
import spectral_utils
spectral_utils.__path__.append(str(ROOT/'spectral_utils'))
from spectral_utils.prm_label_contract import prm_error_flags
from spectral_utils.fusion_benchmark_bootstrap import paired_source_group_intervals


def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


io_module=module(ROOT/'scripts/audit_fusion_localization_forensics_v1.py','prm_bridge_io')
load,save,sha=io_module.load,io_module.save,io_module.sha
OUT=ROOT/'results/localization_prm_label_audit_v1'
SOURCE=ROOT/'results/localization_source_group_audit_v1'
CURRENT=ROOT/'results/fusion_prediction_quality_v1'
GRAPH=ROOT/'results/fusion_graph_conditioning_v1'
RAW=io_module.RAW_FILES['prmbench_qwen3_8b']
RELEASE_ID='localization-cached-v3-prm-onebased-20260907'
COHORTS={'current110':CURRENT/'EVALUATION.json','original58':SOURCE/'EVALUATION_V2.json'}


def key(p):return p['left']+' minus '+p['right']+' ['+p['scope']+']'


def roster():
    current=load(CURRENT/'MANIFEST.json')['contrasts']
    current += [{'left':a,'right':b,'scope':'all'} for a,b in load(GRAPH/'MANIFEST.json')['contrasts']]
    old=[{'left':p['left'],'right':p['right'],'scope':'all'} for p in load(SOURCE/'CONTRASTS_V2.json')['pairs'].values()]
    return {'current110':list({key(p):p for p in current}.values()),'original58':old}


def tests():
    suite=unittest.defaultTestLoader.loadTestsFromModule(module(ROOT/'tests/test_prm_label_contract.py','prm_contract_tests'))
    stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    save(OUT/'TESTS.json',{'passed':result.wasSuccessful(),'count':result.testsRun,'output':stream.getvalue()})
    print(stream.getvalue());assert result.wasSuccessful()


def prepare():
    assert not (OUT/'MANIFEST.json').exists()
    assert load(OUT/'TESTS.json')['passed']
    v2=load(SOURCE/'RELEASE_V2.json')
    paths=[Path(__file__),ROOT/'spectral_utils/prm_label_contract.py',ROOT/'tests/test_prm_label_contract.py',
           ROOT/'spectral_utils/prmbench.py',ROOT/'spectral_utils/fusion_benchmark_bootstrap.py',
           ROOT/'docs/experiments/LOCALIZATION_PRM_LABEL_AUDIT_V1.md',
           ROOT/'scripts/run_answer_localization_v2.py',ROOT/'scripts/audit_fusion_localization_forensics_v1.py',
           ROOT/'scripts/audit_localization_source_groups.py',ROOT/'cluster/run_prmbench_teacher_forced.py',
           Path(r'C:\Users\omris\TAU\hd_jlsml_v2_wt\scripts\joint_lsml_optimization_v2\run_v2.py'),
           RAW,SOURCE/'RELEASE_V2.json',SOURCE/'FOLDS_V2.json',SOURCE/'CANONICAL_GROUPS.json',
           SOURCE/'CONTRASTS_V2.json',CURRENT/'MANIFEST.json',GRAPH/'MANIFEST.json',
           ROOT/'results/fusion_localization_forensics_v1/FAILURE.json']+list(COHORTS.values())
    paths += [Path(v['label_path']) for v in v2['cells'].values()]
    save(OUT/'MANIFEST.json',{'release_id':RELEASE_ID,'status':'RAW_LABEL_REPAIR_NO_NEW_MODEL',
        'hashes':{str(p):sha(p) for p in paths},'cohorts':{k:str(v) for k,v in COHORTS.items()},
        'contrasts':roster(),'created_unix':time.time(),'labels_previously_seen':True,
        'official_source':'https://github.com/ssmisya/PRMBench/blob/main/mr_eval/tasks/prmtest_classified/task.py'})
    print('Frozen score bridges and contrasts:',{k:len(v) for k,v in roster().items()},flush=True)


def verify():
    m=load(OUT/'MANIFEST.json')
    for p,h in m['hashes'].items():assert sha(p)==h,p
    return m


def labels():
    m=verify();start=time.monotonic();v2=load(SOURCE/'RELEASE_V2.json')
    with RAW.open('rb') as f:data=io_module.metadata.MetadataUnpickler(f).load()
    index={v['idx']:v for v in data.values()};assert len(index)==len(data)==6969
    info=v2['cells']['prmbench_qwen3_8b'];rows=[];flags=[];offsets=[0];counts=Counter()
    with np.load(info['label_path'],allow_pickle=False) as old:
        ids=old['row_ids'].copy();classes=old['classification'].copy()
        for i,row_id in enumerate(ids.astype(str)):
            raw=index[row_id];n=len(raw['steps']);assert n==len(raw['step_token_spans'])
            assert raw['classification']==str(classes[i])
            a,b=old['step_flag_offsets'][i:i+2];assert b-a==n
            before=old['step_error_flags'][a:b];buggy=np.zeros(n,dtype=np.int64)
            for step in raw['error_steps']:
                if 0<=step<n:buggy[step]=1
            np.testing.assert_array_equal(before,buggy)
            after=prm_error_flags(raw['error_steps'],n)
            rows.append({'row_id':row_id,'steps':n,'classification':raw['classification'],
                'raw_error_steps_onebased':raw['error_steps'],'previous_flags':before.tolist(),'corrected_flags':after.tolist()})
            counts['answers']+=1;counts['steps']+=n;counts['changed_answers']+=int(np.any(before!=after))
            counts['changed_step_labels']+=int(np.sum(before!=after));counts['old_positive_steps']+=int(before.sum())
            counts['corrected_positive_steps']+=int(after.sum());counts['out_of_range_annotations']+=sum(not 1<=s<=n for s in raw['error_steps'])
            counts['rows_with_out_of_range_annotations']+=int(any(not 1<=s<=n for s in raw['error_steps']))
            counts['old_all_correct_new_has_error']+=int(not before.any() and after.any())
            counts['old_error_new_all_correct']+=int(before.any() and not after.any())
            flags.extend(after.tolist());offsets.append(len(flags))
    label_path=OUT/'prmbench_qwen3_8b_labels_v3.npz'
    np.savez_compressed(label_path,row_ids=ids,classification=classes,step_error_flags=np.array(flags,dtype=np.int64),step_flag_offsets=np.array(offsets,dtype=np.int64))
    save(OUT/'LABEL_AUDIT.json',{'status':'COMPLETE','manifest_sha256':sha(OUT/'MANIFEST.json'),
        'counts':dict(counts),'rows':rows,'corrected_label_sha256':sha(label_path),'seconds':time.monotonic()-start})
    v3=deepcopy(v2);v3.update(release_id=RELEASE_ID,predecessor_release_id=v2['release_id'],
        source_group_release_id=v2['release_id'],scoring_namespace='localization-cached-v1-20260907',
        status='CORRECTED_PRMB_ONEBASED_LABELS',prm_label_contract='one-based raw error_steps -> index step-1; out-of-range inert',
        folds_path=str(SOURCE/'FOLDS_V2.json'),folds_sha256=sha(SOURCE/'FOLDS_V2.json'),
        canonical_groups_path=str(SOURCE/'CANONICAL_GROUPS.json'),canonical_groups_sha256=sha(SOURCE/'CANONICAL_GROUPS.json'),
        label_audit_path=str(OUT/'LABEL_AUDIT.json'),label_audit_sha256=sha(OUT/'LABEL_AUDIT.json'))
    v3['cells']['prmbench_qwen3_8b'].update(label_path=str(label_path),label_opaque_sha256=sha(label_path))
    save(OUT/'RELEASE_V3.json',v3);print('All raw labels repaired:',dict(counts),flush=True)


def metric_module():return module(ROOT/'scripts/run_answer_localization_v2.py','prm_bridge_metrics')
def fixed(rows):return [{**r,'decision_valid':r['fixed_iu_valid'],'predictions':r['fixed_iu_predictions']} for r in rows]


def bridge():
    m=verify();start=time.monotonic();audit=load(OUT/'LABEL_AUDIT.json');index={r['row_id']:r for r in audit['rows']};mm=metric_module()
    inventory={}
    for cohort,path in COHORTS.items():
        previous=load(path);rows=deepcopy(previous['rows']);changed=[]
        for row in rows:
            if row['cell'].startswith('prm'):
                raw=index[row['row_id']];assert raw['previous_flags']==row['target'];assert raw['steps']==row['steps']
                if row['target']!=raw['corrected_flags']:changed.append(row['uid'])
                row['target']=raw['corrected_flags']
        metrics={};fx=fixed(rows)
        for arm,old in previous['metrics'].items():
            metric=deepcopy(old);metric['prm']=mm.prm_metric(rows,arm)
            assert mm.pb_metric(rows,arm)==old['pb'],arm
            for k in ('pb_common_iu_gate','pb_fixed_iu_gate'):
                if k in old:assert mm.pb_metric(fx,arm)==old[k],(arm,k)
            metrics[arm]=metric
        save(OUT/(cohort+'_EVALUATION_V3.json'),{'status':'CORRECTED_LABEL_SCORE_BRIDGE_DEVELOPMENT',
            'release_id':RELEASE_ID,'source_evaluation_sha256':sha(path),'label_audit_sha256':sha(OUT/'LABEL_AUDIT.json'),
            'rows':rows,'metrics':metrics,'previous_metrics':previous['metrics'],'changed_target_uids':changed,
            'scores_predictions_validity_groups_unchanged':True})
        inventory[cohort]={'answers':len(rows),'methods':len(metrics),'changed_prm_answers':len(changed)}
        print(cohort,inventory[cohort],flush=True)
        for arm in ['dual__iu','dual__cond100_graph010','ar1__iu','ar1__graph010','context__iu']:
            if arm in metrics:print(arm,'old/new PRM',previous['metrics'][arm]['prm']['auroc'],metrics[arm]['prm']['auroc'],'PB',metrics[arm]['pb']['macro_f1'],flush=True)
    save(OUT/'BRIDGE.json',{'status':'COMPLETE','cohorts':inventory,'seconds':time.monotonic()-start})


def select(rows,scope):
    if scope=='all':return rows
    mode,kind=scope.split(':');assert mode in ('native','native_and_original')
    return [r for r in rows if r['augmentation']['native_joint_valid'][kind]
            and (mode!='native_and_original' or r['augmentation']['original_joint_valid'])]


def contrasts():
    m=verify();mm=metric_module();start=time.monotonic()
    for cohort,roster_ in m['contrasts'].items():
        epath=OUT/(cohort+'_EVALUATION_V3.json');evaluation=load(epath);destination=OUT/(cohort+'_CONTRASTS_V3.json')
        state=load(destination) if destination.exists() else {'evaluation_sha256':sha(epath),'pairs':{}}
        assert state['evaluation_sha256']==sha(epath)
        for p in roster_:
            k=key(p)
            if k in state['pairs']:continue
            rows=select(evaluation['rows'],p['scope']);common=[r for r in rows if r['valid'].get(p['left']) and r['valid'].get(p['right'])]
            state['pairs'][k]={**p,'selected_ids':[r['uid'] for r in rows],
                'left_prm':mm.prm_metric(common,p['left']),'right_prm':mm.prm_metric(common,p['right']),
                'left_pb':mm.pb_metric(rows,p['left']),'right_pb':mm.pb_metric(rows,p['right']),
                'uncertainty':paired_source_group_intervals(rows,p['left'],p['right'])}
            save(destination,state)
            if len(state['pairs'])%25==0:print(cohort,len(state['pairs']),'/',len(roster_),flush=True)
        assert len(state['pairs'])==len(roster_);state['status']='COMPLETE';save(destination,state)
    save(OUT/'CONTRAST_EXECUTION.json',{'status':'COMPLETE','seconds':time.monotonic()-start});print('All registered label bridges complete.',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--phase',required=True,choices=['tests','prepare','labels','bridge','contrasts'])
    globals()[parser.parse_args().phase]()
