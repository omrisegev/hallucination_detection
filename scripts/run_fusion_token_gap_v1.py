"""One bounded reparameterization; all corrected historical anchors retained."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
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
from spectral_utils.fusion_token_gap import NEW_ARMS,ANCHORS,CORES,score_gap
from spectral_utils.fusion_benchmark_bootstrap import paired_source_group_intervals
OUT=ROOT/'results/fusion_token_gap_v1'
ORIGINAL=ROOT/'results/fusion_replication_v1'
PARENT=ROOT/'results/localization_prm_label_audit_v1'
EVALUATION=PARENT/'current110_EVALUATION_V3.json'


def module(path,name):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


io_module=module(ROOT/'scripts/audit_fusion_localization_forensics_v1.py','token_gap_io')
sha,load,save=io_module.sha,io_module.load,io_module.save


def key(p):return p['left']+' minus '+p['right']+' ['+p['scope']+']'


def pairs():
    out=[]
    def add(a,b,scope='all'):
        p=dict(left=a,right=b,scope=scope)
        if key(p) not in {key(x) for x in out}:out.append(p)
    for core in CORES:add('gap__'+core,ANCHORS[core])
    for left,right in [('iu','equal'),('joint0','iu'),('joint0','equal'),('graph010','joint0'),('graph010','graph_perm'),
                       ('graph010','iu'),('graph010','equal_graph010'),('equal_graph010','equal'),('equal_graph010','equal_graph_perm')]:add('gap__'+left,'gap__'+right)
    add('gap_scalar','surprisal_scalar');add('gap__iu','gap_scalar');add('gap__graph010','gap_scalar')
    for left in ('gap__iu','gap__graph010'):
        for right in ('dual__iu','dual__equal_graph_perm'):add(left,right)
    add('gap__graph010','gap__iu','native_gap');add('gap__graph010','gap__equal_graph010','native_gap')
    add('gap__graph010','dual__cond100_graph010','native_gap_and_original')
    assert len(out)==25
    return out


def tests():
    suite=unittest.defaultTestLoader.loadTestsFromModule(module(ROOT/'tests/test_fusion_token_gap.py','token_gap_tests'))
    stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    save(OUT/'TESTS.json',{'passed':result.wasSuccessful(),'tests':result.testsRun,'output':stream.getvalue()})
    print(stream.getvalue());assert result.wasSuccessful()


def prepare():
    assert not (OUT/'MANIFEST.json').exists();assert load(OUT/'TESTS.json')['passed']
    original=load(ORIGINAL/'MANIFEST.json');release=load(PARENT/'RELEASE_V3.json')
    # The parent method roster is in a target-free manifest; do not decode the
    # corrected evaluation until score freezing has completed.
    parent_manifest=ROOT/'results/fusion_prediction_quality_v1/MANIFEST.json'
    arms=load(parent_manifest)['arms'];assert len(arms)==98
    paths=[Path(__file__),ROOT/'spectral_utils/fusion_token_gap.py',ROOT/'tests/test_fusion_token_gap.py',
        ROOT/'docs/experiments/FUSION_TOKEN_GAP_V1.md',ROOT/'cluster/backfill_views.py',ROOT/'cluster/run_teacher_forced.py',
        ROOT/'cluster/run_prmbench_teacher_forced.py',ROOT/'spectral_utils/token_feature_views.py',
        ROOT/'scripts/run_answer_localization_v2.py',ROOT/'scripts/audit_fusion_localization_forensics_v1.py',
        ORIGINAL/'MANIFEST.json',parent_manifest,EVALUATION,PARENT/'RELEASE_V3.json',PARENT/'REVIEW.json']
    for name,m in list(sys.modules.items()):
        if name.startswith('spectral_utils') and getattr(m,'__file__',None):paths.append(Path(m.__file__).resolve())
    for rec in original['selected']:
        uid=rec['uid'];paths.append(ORIGINAL/'inputs'/(uid+'.npz'))
        paths.extend(ORIGINAL/'scores'/(uid+ext) for ext in ('.json','.npz'))
    paths += [Path(info['label_path']) for cell,info in release['cells'].items() if cell in {r['cell'] for r in original['selected']}]
    paths += list(io_module.RAW_FILES.values())
    save(OUT/'MANIFEST.json',{'status':'DEVELOPMENT_PROVIDED_TOKEN_GAP','release_id':release['release_id'],
        'scoring_namespace':original['scoring_namespace'],'selected':original['selected'],'arms':arms+list(NEW_ARMS),
        'new_arms':list(NEW_ARMS),'external_arms':arms,'contrasts':pairs(),'hashes':{str(p):sha(p) for p in paths},
        'created_unix':time.time(),'labels_decoded_for_scoring':False,'workers':3,'score_seconds_cap':1200})
    print('Frozen107 arms:98 exact anchors +9 new outputs;25 registered comparisons.',flush=True)


def verify():
    m=load(OUT/'MANIFEST.json')
    for p,h in m['hashes'].items():assert sha(p)==h,p
    return m


def one(rec,manifest_hash,namespace):
    start=time.monotonic();uid=rec['uid'];jp=OUT/'scores'/(uid+'.json');npz=OUT/'scores'/(uid+'.npz')
    if jp.exists():
        old=load(jp);assert old['manifest_sha256']==manifest_hash and sha(npz)==old['array_sha256'];return uid
    with np.load(ORIGINAL/'inputs'/(uid+'.npz'),allow_pickle=False) as a:raw=a['raw'];ss=a['step_starts'];ee=a['step_ends']
    with np.load(ORIGINAL/'scores'/(uid+'.npz'),allow_pickle=False) as a:original={k:a[k] for k in a.files}
    meta=load(ORIGINAL/'scores'/(uid+'.json'))
    # Keep the original permutation identity: a feature change must not also
    # change the randomized graph control's seed.
    identity=namespace+'/'+rec['cell']+'/'+rec['row_id']+'/moments27_local8'
    arrays,methods,diagnostics=score_gap(raw,ss,ee,original,meta,identity)
    npz.parent.mkdir(parents=True,exist_ok=True);tmp=npz.with_suffix('.npz.tmp')
    with tmp.open('wb') as f:np.savez_compressed(f,**arrays)
    tmp.replace(npz)
    save(jp,{**rec,'methods':methods,'diagnostics':diagnostics,'routing':meta['routing'],
        'original_selected_joint_valid':meta['methods'][diagnostics['bank']+'__joint0']['valid'],
        'array_sha256':sha(npz),'manifest_sha256':manifest_hash,'labels_used':False,'seconds':time.monotonic()-start})
    return uid


def scores():
    m=verify();start=time.monotonic();digest=sha(OUT/'MANIFEST.json');remaining=iter(m['selected']);completed=[]
    with ProcessPoolExecutor(max_workers=3) as executor:
        pending={}
        def submit():
            try:r=next(remaining)
            except StopIteration:return
            pending[executor.submit(one,r,digest,m['scoring_namespace'])]=r['uid']
        for _ in range(3):submit()
        while pending:
            done,_=wait(pending,return_when=FIRST_COMPLETED)
            for future in done:
                completed.append(future.result());pending.pop(future)
                if time.monotonic()-start<m['score_seconds_cap']:submit()
            if len(completed)%10==0:print('Completed',len(completed),'/110',flush=True)
    assert len(completed)==110,'Scoring cap reached; existing per-answer checkpoints retained.'
    paths=[p for rec in m['selected'] for p in (OUT/'scores'/(rec['uid']+'.json'),OUT/'scores'/(rec['uid']+'.npz'))]
    save(OUT/'SCORES_FROZEN.json',{'status':'COMPLETE','files':{str(p):sha(p) for p in paths},
        'manifest_sha256':digest,'labels_decoded':False,'seconds':time.monotonic()-start})
    print('All9 new outputs frozen for110 answers.',flush=True)


def fixed(rows):return [{**r,'predictions':r['fixed_iu_predictions'],'decision_valid':r['fixed_iu_valid']} for r in rows]
def metrics_module():return module(ROOT/'scripts/run_answer_localization_v2.py','gap_metric_reference')


def evaluate():
    m=verify();freeze=load(OUT/'SCORES_FROZEN.json');assert freeze['status']=='COMPLETE' and not freeze['labels_decoded']
    assert freeze['manifest_sha256']==sha(OUT/'MANIFEST.json')
    for p,h in freeze['files'].items():assert sha(p)==h,p
    previous=load(EVALUATION);rows=deepcopy(previous['rows']);release=load(PARENT/'RELEASE_V3.json')
    for cell in {r['cell'] for r in rows}:
        with np.load(release['cells'][cell]['label_path'],allow_pickle=False) as labels:
            index={str(v):i for i,v in enumerate(labels['row_ids'])}
            for row in (r for r in rows if r['cell']==cell):
                uid=row['uid'];i=index[row['row_id']]
                if cell.startswith('prm'):
                    a,b=labels['step_flag_offsets'][i:i+2];np.testing.assert_array_equal(row['target'],labels['step_error_flags'][a:b])
                else:assert row['target']==int(labels['first_error'][i])
                meta=load(OUT/'scores'/(uid+'.json'));assert meta['routing']==row['routing']
                row['token_gap']={'bank':meta['diagnostics']['bank'],'joint_valid':meta['diagnostics']['shared']['joint_valid'],
                                  'original_joint_valid':meta['original_selected_joint_valid']}
                with np.load(OUT/'scores'/(uid+'.npz'),allow_pickle=False) as a:
                    for arm,detail in meta['methods'].items():
                        for k in ('valid','decision_valid','fixed_iu_valid'):row[k][arm]=detail[k]
                        for dst,src in [('predictions','prediction'),('fixed_iu_predictions','fixed_iu_prediction'),('peaks','peak')]:row[dst][arm]=detail.get(src)
                        row['sources'][arm]=detail.get('source_arm')
                        if detail['valid']:row['scores'][arm]=a[arm+'__risk'].tolist()
    mm=metrics_module();fx=fixed(rows);metrics={arm:{'prm':mm.prm_metric(rows,arm),'pb':mm.pb_metric(rows,arm),
        'pb_common_iu_gate':mm.pb_metric(fx,arm)} for arm in m['arms']}
    for arm in m['external_arms']:assert metrics[arm]==previous['metrics'][arm],arm
    save(OUT/'EVALUATION.json',{'status':'DEVELOPMENT_TOKEN_GAP_QUALITY','release_id':m['release_id'],
        'scores_sha256':sha(OUT/'SCORES_FROZEN.json'),'rows':rows,'metrics':metrics,'parent_metrics':previous['metrics']})
    for arm in NEW_ARMS:print(arm,'PRM',metrics[arm]['prm']['auroc'],'PB',metrics[arm]['pb']['macro_f1'],flush=True)


def select(rows,scope):
    if scope=='all':return rows
    assert scope in ('native_gap','native_gap_and_original')
    return [r for r in rows if r['token_gap']['joint_valid'] and (scope!='native_gap_and_original' or r['token_gap']['original_joint_valid'])]


def contrasts():
    m=verify();e=load(OUT/'EVALUATION.json');path=OUT/'CONTRASTS.json';digest=sha(OUT/'EVALUATION.json');start=time.monotonic();mm=metrics_module()
    state=load(path) if path.exists() else {'evaluation_sha256':digest,'pairs':{}};assert state['evaluation_sha256']==digest
    for p in m['contrasts']:
        k=key(p)
        if k in state['pairs']:continue
        rows=select(e['rows'],p['scope']);common=[r for r in rows if r['valid'][p['left']] and r['valid'][p['right']]]
        state['pairs'][k]={**p,'selected_ids':[r['uid'] for r in rows],
            'left_prm':mm.prm_metric(common,p['left']),'right_prm':mm.prm_metric(common,p['right']),
            'left_pb':mm.pb_metric(rows,p['left']),'right_pb':mm.pb_metric(rows,p['right']),
            'uncertainty':paired_source_group_intervals(rows,p['left'],p['right'])};save(path,state)
    assert len(state['pairs'])==25;state.update(status='COMPLETE',seconds=time.monotonic()-start);save(path,state)
    print('All25 comparisons complete.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',required=True,choices=['tests','prepare','scores','evaluate','contrasts']);globals()[p.parse_args().phase]()
